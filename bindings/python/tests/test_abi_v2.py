# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Batch v2 / provider v3 semantics on the reference session and, with SEABRAID_LIBRARY, in parity with the engine: placement epochs,
F16/BF16 bytes, the compact_rows construction rule.

Checker for seabraid.reference's v2 mirror (src/session.rs epochs, src/batch.rs formats) and for the provider
trampoline's raw-bytes view. Native parity cases run the same scripts on both sessions.
"""

from __future__ import annotations

import os
import struct
import unittest
from array import array
from unittest.mock import Mock, patch

from seabraid import Runtime, SeabraidError
from seabraid._model import BatchInfo, Shape
from seabraid.constants import BUSY_OR_STATE, FORMAT_ROUTED_BF16, FORMAT_ROUTED_F16, FORMAT_ROUTED_F32, INVALID

from ._v2_support import (
    BF16, HALVES, LIBRARY, SHAPE, V3_SPEC, EpochScript, echo, half_provider, outcome, reference_runtime, view_kind,
)


class ReferenceEpochTests(unittest.TestCase):
    def test_stale_zero_after_nonzero_and_a_rejected_batch_leaves_the_mark(self) -> None:
        with reference_runtime() as runtime:
            story = EpochScript().run(runtime)
        self.assertEqual(story[2], ("SeabraidError", BUSY_OR_STATE))
        self.assertEqual(story[3], ("SeabraidError", INVALID))
        self.assertEqual(story[4], ("SeabraidError", INVALID))
        for index in (0, 1, 5, 6):
            self.assertIsInstance(story[index], list, index)
        self.assertEqual(story[0], story[1])  # an epoch tag never changes the numbers

    def test_epoch_is_validated_as_an_unsigned_64_bit_integer(self) -> None:
        with reference_runtime() as runtime:
            prepared = runtime.prepare(**SHAPE)
            for bad in (-1, 1 << 64):
                with self.subTest(bad), self.assertRaises(ValueError):
                    prepared.weave([1.0, 2.0], [0, 1], [0.5, 0.5], placement_epoch=bad)
            self.assertIsInstance(prepared.weave([1.0, 2.0], [0, 1], [0.5, 0.5], placement_epoch=(1 << 64) - 1), list)


class ReferenceFormatTests(unittest.TestCase):
    def test_undeclared_formats_are_refused_before_the_provider_runs(self) -> None:
        for kwargs in ({"provider": echo}, {"reference": True}):
            for fmt in (FORMAT_ROUTED_F16, FORMAT_ROUTED_BF16):
                with self.subTest(kwargs=sorted(kwargs), fmt=fmt), reference_runtime(**kwargs) as runtime:
                    prepared = runtime.prepare(**SHAPE)
                    with self.assertRaises(SeabraidError) as raised:
                        prepared.weave(HALVES[:4], [0, 1], [0.5, 0.5], activation_format=fmt)
                    self.assertEqual(raised.exception.status, INVALID)

    def test_declared_f16_and_bf16_bytes_reach_the_callable_exactly_and_unconverted(self) -> None:
        for fmt, payload in ((FORMAT_ROUTED_F16, HALVES), (FORMAT_ROUTED_BF16, BF16)):
            seen: list = []
            with self.subTest(fmt=fmt), reference_runtime(provider=half_provider(seen), accepts=(fmt,)) as runtime:
                prepared = runtime.prepare(**SHAPE)
                out = prepared.weave(payload[:4], [0, 1], [0.5, 0.5], placement_epoch=3, activation_format=fmt)
            (record,) = seen
            self.assertEqual(record[:3], (fmt, 3, 4))
            self.assertEqual(record[3], "raw")  # raw bytes, not a float view
            self.assertEqual(record[4], payload[:4])
            self.assertEqual(out, [3.0, -4.0])  # the callable decoded 1.5 and -2.0 itself and doubled them

    def test_any_buffer_of_16_bit_elements_is_accepted_and_a_sequence_is_not(self) -> None:
        seen: list = []
        with reference_runtime(provider=half_provider(seen), accepts=(FORMAT_ROUTED_F16,)) as runtime:
            prepared = runtime.prepare(**SHAPE)
            as_halves = array("H", struct.unpack("<2H", HALVES[:4]))
            prepared.weave(as_halves, [0, 1], [0.5, 0.5], activation_format=FORMAT_ROUTED_F16)
            prepared.weave(memoryview(bytearray(HALVES[:4])), [0, 1], [0.5, 0.5], activation_format=FORMAT_ROUTED_F16)
            with self.assertRaisesRegex(ValueError, "bytes-like"):
                prepared.weave([1.0, 2.0], [0, 1], [0.5, 0.5], activation_format=FORMAT_ROUTED_F16)
            with self.assertRaisesRegex(ValueError, "multiple of 2"):
                prepared.weave(b"\0\0\0", [0, 1], [0.5, 0.5], activation_format=FORMAT_ROUTED_F16)
        self.assertEqual([r[4] for r in seen], [HALVES[:4], HALVES[:4]])

    def test_f32_still_works_on_a_provider_that_declared_halves(self) -> None:
        def serve(operation: int, config: Shape, acts: object, ids: object, weights: object, out: object,
                  info: BatchInfo) -> None:
            self.assertEqual(info.activation_format, FORMAT_ROUTED_F32)
            self.assertEqual(view_kind(acts), "float")
            out[0], out[1] = acts[0], acts[1]

        with reference_runtime(provider=serve, accepts=(FORMAT_ROUTED_F16,)) as runtime:
            self.assertEqual(runtime.prepare(**SHAPE).weave([1.0, 2.0], [0, 1], [0.5, 0.5]), [1.0, 2.0])

    def test_construction_refuses_what_the_engine_would_refuse(self) -> None:
        with self.assertRaises(ValueError):
            reference_runtime(provider=echo, accepts=(99,))
        with self.assertRaisesRegex(ValueError, "require provider"):
            reference_runtime(accepts=(FORMAT_ROUTED_F16,))
        with self.assertRaisesRegex(ValueError, "only F32"):
            reference_runtime(reference=True, accepts=(FORMAT_ROUTED_F16,))


class CompactRowsValidationTests(unittest.TestCase):
    def test_compact_rows_without_partial_routes_is_the_package_error_before_any_c_call(self) -> None:
        library = Mock(spec=V3_SPEC)
        library.seabraid_abi_epoch.return_value = 1
        with patch("seabraid._abi.ct.CDLL", return_value=library):
            for kwargs in ({"provider": echo}, {"reference": True}):
                with self.subTest(sorted(kwargs)), self.assertRaises(SeabraidError) as raised:
                    Runtime(__file__, compact_rows=True, **kwargs)
                self.assertEqual(raised.exception.status, INVALID)
                self.assertIn("compact_rows requires partial_routes", str(raised.exception))
        library.seabraid_session_open.assert_not_called()
        with self.assertRaises(SeabraidError):
            reference_runtime(provider=echo, compact_rows=True)
        with self.assertRaises(SeabraidError):
            reference_runtime(reference=True, compact_rows=True)

    def test_compact_rows_or_partial_routes_need_something_to_apply_to(self) -> None:
        with self.assertRaisesRegex(ValueError, "require provider"):
            reference_runtime(compact_rows=True)


@unittest.skipUnless(LIBRARY, "set SEABRAID_LIBRARY for native parity")
class NativeParityTests(unittest.TestCase):
    """Every v2 behavior above, run on the engine and on the reference session, compared class for class."""

    def pair(self, **kwargs: object) -> tuple[Runtime, Runtime]:
        return reference_runtime(**kwargs), Runtime(LIBRARY, **kwargs)

    def test_epoch_story_has_the_same_outcomes_with_a_python_provider_and_with_the_builtin(self) -> None:
        for kwargs in ({"provider": echo}, {"reference": True}):
            with self.subTest(sorted(kwargs)):
                ref, native = self.pair(**kwargs)
                with ref, native:
                    expected, found = EpochScript().run(ref), EpochScript().run(native)
                self.assertEqual(found, expected)
                self.assertEqual(found[2], ("SeabraidError", BUSY_OR_STATE))
                self.assertEqual(found[3], ("SeabraidError", INVALID))

    def test_undeclared_format_is_refused_alike_and_declared_bytes_reach_python_exactly(self) -> None:
        ref, native = self.pair(reference=True)
        with ref, native:
            results = [outcome(lambda r=r: r.prepare(**SHAPE).weave(HALVES[:4], [0, 1], [0.5, 0.5],
                                                                        activation_format=FORMAT_ROUTED_F16))
                       for r in (ref, native)]
        self.assertEqual(results, [("SeabraidError", INVALID)] * 2)
        for fmt, payload in ((FORMAT_ROUTED_F16, HALVES), (FORMAT_ROUTED_BF16, BF16)):
            seen_ref: list = []
            seen_native: list = []
            with self.subTest(fmt=fmt):
                ref = reference_runtime(provider=half_provider(seen_ref), accepts=(fmt,))
                native = Runtime(LIBRARY, provider=half_provider(seen_native), accepts=(fmt,))
                with ref, native:
                    a = ref.prepare(**SHAPE).weave(payload[:4], [0, 1], [0.5, 0.5], placement_epoch=6,
                                                   activation_format=fmt)
                    b = native.prepare(**SHAPE).weave(payload[:4], [0, 1], [0.5, 0.5], placement_epoch=6,
                                                      activation_format=fmt)
                self.assertEqual(a, b)
                self.assertEqual(seen_native, seen_ref)
                self.assertEqual(seen_native[0][4], payload[:4])
                self.assertEqual(seen_native[0][:2], (fmt, 6))

    def test_session_status_counts_rows_alike(self) -> None:
        ref, native = self.pair(provider=echo, partial_routes=True)
        with ref, native:
            seen = []
            for runtime in (ref, native):
                prepared = runtime.prepare(**SHAPE)
                prepared.weave([1.0, 2.0, 3.0, 4.0], [0, 1, -1, -1], [0.5, 0.5, 0.0, 0.0])
                status = runtime.status()
                seen.append((status.submitted, status.collected, status.rows_submitted, status.rows_shipped))
        self.assertEqual(seen, [(1, 1, 2, 2)] * 2)  # uncompacted: every row ships


if __name__ == "__main__":
    unittest.main()
