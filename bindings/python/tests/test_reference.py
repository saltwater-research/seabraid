# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Reference provider: known value, every validation branch of src/batch.rs, and native parity.

Checker for seabraid.reference. The parity test needs SEABRAID_LIBRARY (the built native library) and is skipped otherwise.
"""

from __future__ import annotations

import logging
import os
import unittest
import warnings
from unittest.mock import patch

from seabraid import SeabraidError, Runtime
from seabraid._model import Shape
from seabraid.constants import BUSY_OR_STATE, INVALID, SourceModality
from seabraid.reference import ReferenceSession

SHAPE = dict(hidden=2, experts=3, max_tokens=2, top_k=2)


def reference_runtime() -> Runtime:
    """A Runtime on the reference provider even when SEABRAID_LIBRARY is set in the environment."""
    with patch.dict(os.environ):
        os.environ.pop("SEABRAID_LIBRARY", None)
        # The notice itself is checked in ProviderChoiceTests; here it would only be stderr noise.
        with warnings.catch_warnings(), patch.object(logging.getLogger("seabraid"), "disabled", True):
            warnings.simplefilter("ignore", RuntimeWarning)
            return Runtime()


class KnownValueTests(unittest.TestCase):
    def test_documented_example(self) -> None:
        with reference_runtime() as runtime:
            prepared = runtime.prepare(**SHAPE)
            self.assertEqual(prepared.weave([1.0, 2.0], [0, 2], [0.25, 0.75]), [2.5, 5.0])
            self.assertEqual(runtime.provider, "reference")

    def test_two_tokens_and_session_reuse(self) -> None:
        with reference_runtime() as runtime:
            prepared = runtime.prepare(hidden=2, experts=4, max_tokens=3, top_k=3)
            output = prepared.weave(
                [2.0, -4.0, 0.5, 8.0], [0, 2, 1, 1, 3, 0], [0.25, 0.5, 0.25, 0.5, 0.375, 0.125],
            )
            self.assertEqual(output, [4.5, -9.0, 1.3125, 21.0])
            self.assertEqual(prepared.weave([1.0, 2.0], [1, 0, 3], [1.0, 0.0, 0.0]), [2.0, 4.0])

    def test_inputs_round_to_float32_like_native_packing(self) -> None:
        with reference_runtime() as runtime:
            prepared = runtime.prepare(hidden=1, experts=1, max_tokens=1, top_k=1)
            # 0.1 is not representable; the c_float path stores 0.100000001490116...
            self.assertEqual(prepared.weave([1.0], [0], [0.1]), [0.10000000149011612])

    def test_status_counts_and_modalities(self) -> None:
        with reference_runtime() as runtime:
            prepared = runtime.prepare(**SHAPE)
            origins = SourceModality.AUDIO | SourceModality.VIDEO
            prepared.weave([1.0, 2.0], [0, 2], [0.25, 0.75], source_modalities=origins)
            status = runtime.status()
            self.assertEqual((status.submitted, status.collected, status.outstanding), (1, 1, False))
            self.assertEqual(status.last_source_modalities, origins)
            self.assertEqual(runtime.capabilities().source_modalities & origins, origins)


class ProviderChoiceTests(unittest.TestCase):
    def test_no_library_warns_exactly_once_and_logs(self) -> None:
        with patch.dict(os.environ), warnings.catch_warnings(record=True) as caught:
            os.environ.pop("SEABRAID_LIBRARY", None)
            warnings.simplefilter("always")
            with self.assertLogs("seabraid", "WARNING") as logs:
                runtime = Runtime()
            runtime.close()
        self.assertEqual(len(caught), 1)
        self.assertIs(caught[0].category, RuntimeWarning)
        self.assertIn("pure-Python reference provider", str(caught[0].message))
        self.assertEqual(len(logs.records), 1)
        self.assertEqual(caught[0].filename, __file__)  # blamed on the caller, not on the library

    def test_missing_explicit_library_raises_without_the_path_and_never_falls_back(self) -> None:
        secret = "definitely-missing-library-name.so"
        with self.assertRaises(OSError) as raised:
            Runtime(secret)
        self.assertNotIn(secret, str(raised.exception))


class ValidationTests(unittest.TestCase):
    def session(self, **shape: int) -> ReferenceSession:
        session = ReferenceSession()
        session.prepare(Shape(**(SHAPE | shape)))
        return session

    def assert_invalid(self, operation: str, call, status: int = INVALID) -> None:
        with self.assertRaises(SeabraidError) as raised:
            call()
        self.assertEqual((raised.exception.operation, raised.exception.status), (operation, status))

    def test_config_validation_matches_config_validate(self) -> None:
        for shape in (dict(top_k=4), dict(experts=0), dict(max_tokens=0), dict(hidden=0), dict(top_k=0),
                      dict(hidden=2**32 - 1, max_tokens=2**32 - 1)):
            with self.subTest(shape=shape):
                self.assert_invalid("seabraid_prepare", lambda: ReferenceSession().prepare(Shape(**(SHAPE | shape))))

    def test_prepare_twice_is_a_state_error(self) -> None:
        session = self.session()
        self.assert_invalid("seabraid_prepare", lambda: session.prepare(Shape(**SHAPE)), BUSY_OR_STATE)

    def test_batch_geometry_and_expert_ids(self) -> None:
        session = self.session()
        good = ([1.0, 2.0], [0, 2], [0.25, 0.75])
        cases = {
            "zero tokens": (0, [], [], [], 0),
            "too many tokens": (3, [0.0] * 6, [0] * 6, [0.5] * 6, 0),
            "short activations": (1, [1.0], good[1], good[2], 0),
            "short ids": (1, good[0], [0], good[2], 0),
            "long weights": (1, good[0], good[1], [0.25, 0.5, 0.25], 0),
            "id at experts": (1, good[0], [0, 3], good[2], 0),
            "negative id": (1, good[0], [0, -1], [0.5, 0.0], 0),
            "infinite weight": (1, good[0], good[1], [0.25, float("inf")], 0),
            "nan weight": (1, good[0], good[1], [0.25, float("nan")], 0),
            "unknown origin": (1, good[0], good[1], good[2], 1 << 31),
        }
        for name, (tokens, acts, ids, weights, origins) in cases.items():
            with self.subTest(name):
                self.assert_invalid(
                    "seabraid_submit_routed" if origins else "seabraid_submit",
                    lambda: session.submit(tokens, acts, ids, weights, origins),
                )
        self.assertEqual(session.submit(1, *good, 0) > 0, True)  # rejected batches stranded nothing

    def test_skip_requires_partial_routes_and_zero_weight(self) -> None:
        # The four cases of the Rust unit test skips_require_provider_support_and_zero_weight.
        shape = Shape(hidden=1, experts=2, max_tokens=1, top_k=2)

        def attempt(ids: list[int], weights: list[float], partial: bool) -> bool:
            session = ReferenceSession(partial_routes=partial)
            session.prepare(shape)
            try:
                session.submit(1, [2.0], ids, weights, 0)
            except SeabraidError as error:
                self.assertEqual(error.status, INVALID)
                return False
            return True

        self.assertFalse(attempt([0, -1], [0.5, 0.0], False))
        self.assertFalse(attempt([0, -1], [0.5, 0.1], True))
        self.assertFalse(attempt([0, -2], [0.5, 0.0], True))
        self.assertTrue(attempt([0, -1], [0.5, 0.0], True))

    def test_one_ticket_at_a_time_and_collect_rules(self) -> None:
        session = self.session()
        self.assert_invalid("seabraid_collect", lambda: session.collect(1, 0, 2), BUSY_OR_STATE)
        ticket = session.submit(1, [1.0, 2.0], [0, 2], [0.25, 0.75], 0)
        self.assert_invalid("seabraid_submit", lambda: session.submit(1, [1.0, 2.0], [0, 2], [0.5, 0.5], 0),
                            BUSY_OR_STATE)
        self.assert_invalid("seabraid_collect", lambda: session.collect(ticket + 1, 0, 2))
        self.assert_invalid("seabraid_collect", lambda: session.collect(ticket, 0, 3))
        self.assertEqual(session.collect(ticket, 0, 2), [2.5, 5.0])
        self.assert_invalid("seabraid_collect", lambda: session.collect(ticket, 0, 2), BUSY_OR_STATE)

    def test_submit_before_prepare_is_a_state_error(self) -> None:
        self.assert_invalid("seabraid_submit", lambda: ReferenceSession().submit(1, [1.0], [0], [1.0], 0), BUSY_OR_STATE)


PARITY_BATCHES = [
    ([1.0, 2.0], [0, 2], [0.25, 0.75]),
    ([0.1, -3.3], [1, 1], [0.7, 0.2]),  # repeated id within a row: both refuse it
    ([1e30, 1e30], [2, 1], [1e10, 1e10]),  # overflows float32: both providers must give inf
    ([2.0, 3.0, 4.0, 5.0], [0, 1, 2, 0], [0.3, 0.6, 0.9, 0.1]),
    ([1.0, 2.0], [0, 3], [0.5, 0.5]),  # invalid id
    ([1.0, 2.0], [0, 2], [0.5, float("nan")]),  # invalid weight
    ([1.0, 2.0, 3.0], [0, 2], [0.5, 0.5]),  # ragged rows
]


@unittest.skipUnless(os.environ.get("SEABRAID_LIBRARY"), "set SEABRAID_LIBRARY for native parity")
class NativeParityTests(unittest.TestCase):
    def run_batches(self, runtime: Runtime) -> list[object]:
        prepared = runtime.prepare(**SHAPE)
        results: list[object] = []
        for batch in PARITY_BATCHES:
            try:
                results.append(prepared.weave(*batch))
            except (SeabraidError, ValueError) as error:
                # WHY no operation name: the native binding submits through seabraid_session_submit while the
                # pure-Python reference keeps the v1 entry names, so only the error class and status must agree.
                results.append((type(error).__name__, getattr(error, "status", None)))
        return results

    def test_same_batches_give_same_results_and_errors(self) -> None:
        with reference_runtime() as reference, Runtime(os.environ["SEABRAID_LIBRARY"], reference=True) as native:
            self.assertEqual((reference.provider, native.provider), ("reference", "native"))
            self.assertEqual(self.run_batches(reference), self.run_batches(native))
