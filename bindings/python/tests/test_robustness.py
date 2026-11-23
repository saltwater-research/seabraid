# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Robustness of the Python session: pickling, trampoline guards, thread serialisation, fork.

Checker for seabraid.errors (pickle), seabraid._provider (struct_size and count guards), seabraid.runtime (the lock)
and seabraid._native (the fork guard). Mocked library, no linker; the real-fork case runs where os.fork exists.
"""

from __future__ import annotations

import copy
import ctypes as ct
import gc
import os
import pickle
import threading
import unittest
import weakref
from unittest.mock import Mock, patch

from seabraid import Runtime, SeabraidError, _abi, _native, _provider
from seabraid.constants import FORMAT_ROUTED_F16, FORMAT_ROUTED_F32, INVALID, OK, TIMEOUT

from ._v2_support import HALVES, V1_SPEC, V3_SPEC, echo


class ErrorPickleTests(unittest.TestCase):
    def test_every_field_survives_pickle_and_copy(self) -> None:
        for detail in (None, "native text"):
            error = SeabraidError("seabraid_collect", TIMEOUT, detail)
            clones = (("pickle", pickle.loads(pickle.dumps(error))), ("copy", copy.copy(error)),
                      ("deepcopy", copy.deepcopy(error)))
            for name, clone in clones:
                with self.subTest(name, detail=detail):
                    self.assertIsInstance(clone, SeabraidError)
                    self.assertEqual((clone.operation, clone.status, clone.detail), ("seabraid_collect", TIMEOUT, detail))
                    self.assertEqual(str(clone), str(error))


class TrampolineGuardTests(unittest.TestCase):
    """Drive the real v3 CFUNCTYPE with batches an engine should never send; the callable must not run."""

    def setUp(self) -> None:
        self.calls: list = []

        def record(operation: int, config: object, acts: object, ids: object, weights: object, out: object) -> None:
            self.calls.append(bytes(acts))
            out[0] = 1.0

        self.provider = _provider.PythonProvider(record)
        self.config = _abi._ConfigV1(1, 2, 3, 4, 2)  # hidden 2, experts 3, max_tokens 4, top_k 2
        self.acts = (ct.c_float * 4)(1.0, 2.0, 3.0, 4.0)
        self.ids = (ct.c_int32 * 4)(0, 1, 2, 0)
        self.weights = (ct.c_float * 4)(0.5, 0.5, 0.25, 0.75)
        self.out = (ct.c_float * 4)()

    def batch(self, **overrides: object) -> _abi._BatchV2:
        batch = _abi._BatchV2(2, 80, 0, 0, 0, 2, FORMAT_ROUTED_F32, ct.cast(self.acts, ct.c_void_p), 4, self.ids, 4,
                              self.weights, 4)
        for name, value in overrides.items():
            setattr(batch, name, value)
        return batch

    def run_execute(self, batch: _abi._BatchV2, output_count: int = 4) -> int:
        descriptor = self.provider.struct_v3(self.config)
        return descriptor.execute(None, ct.byref(self.config), ct.byref(batch), self.out, output_count)

    def test_a_well_formed_batch_reaches_the_callable(self) -> None:
        self.assertEqual(self.run_execute(self.batch()), OK)
        self.assertEqual(len(self.calls), 1)

    def test_a_struct_shorter_than_the_mirror_is_refused_not_read_past(self) -> None:
        with self.assertLogs("seabraid", "ERROR") as logs:
            self.assertEqual(self.run_execute(self.batch(struct_size=48)), INVALID)
        self.assertIn("shorter", logs.output[0])
        self.assertEqual(self.calls, [])

    def test_counts_that_disagree_with_tokens_are_refused_before_any_view_is_mapped(self) -> None:
        cases = {
            "activations": {"activation_count": 6}, "expert_ids": {"expert_id_count": 3},
            "weights": {"weight_count": 5}, "tokens over max_tokens": {"tokens": 5},
        }
        for name, overrides in cases.items():
            with self.subTest(name):
                self.assertEqual(self.run_execute(self.batch(**overrides)), INVALID)
        with self.subTest("output count"):
            self.assertEqual(self.run_execute(self.batch(), output_count=6), INVALID)
        self.assertEqual(self.calls, [])

    def test_a_compacted_batch_is_checked_against_its_own_token_count(self) -> None:
        # COMPACT_ROWS: one kept row of two submitted; every count follows the library's smaller `tokens`.
        batch = self.batch(tokens=1, activation_count=2, expert_id_count=2, weight_count=2)
        self.assertEqual(self.run_execute(batch, output_count=2), OK)

    def test_16_bit_activations_map_two_bytes_per_element_never_four(self) -> None:
        seen: list = []

        def serve(operation: int, config: object, acts: object, ids: object, weights: object, out: object,
                  info: object) -> None:
            seen.append(bytes(acts))

        provider = _provider.PythonProvider(serve, accepts=(FORMAT_ROUTED_F16,))
        payload = (ct.c_uint16 * 4).from_buffer_copy(HALVES)
        batch = _abi._BatchV2(2, 80, 0, 0, 0, 2, FORMAT_ROUTED_F16, ct.cast(payload, ct.c_void_p), 4, self.ids, 4,
                              self.weights, 4)
        descriptor = provider.struct_v3(self.config)
        self.assertEqual(descriptor.execute(None, ct.byref(self.config), ct.byref(batch), self.out, 4), OK)
        self.assertEqual(seen, [HALVES])  # 8 bytes for 4 elements: an over-map would show 16


class V3PrepareTests(unittest.TestCase):
    def test_a_v3_session_reports_zeros_until_prepare_opens_it(self) -> None:
        library = Mock(spec=V3_SPEC)
        library.seabraid_abi_epoch.return_value = 1
        library.seabraid_close.return_value = 0
        with patch("seabraid._abi.ct.CDLL", return_value=library), Runtime(__file__, provider=echo) as runtime:
            status = runtime.status()
            self.assertEqual((status.submitted, status.rows_submitted, status.outstanding), (0, 0, False))
            library.seabraid_session_open.assert_not_called()
            library.seabraid_session_status.assert_not_called()


class MockedV1Library:
    """A v1 mock whose collect can be held open, to race it against close() from another thread."""

    def __init__(self) -> None:
        self.library = Mock(spec=V1_SPEC)
        self.library.seabraid_abi_epoch.return_value = 1
        self.library.seabraid_open_reference.side_effect = self._open
        self.library.seabraid_prepare.return_value = 0
        self.library.seabraid_submit.side_effect = self._submit
        self.library.seabraid_collect.side_effect = self._collect
        self.library.seabraid_close.return_value = 0
        self.in_collect, self.release = threading.Event(), threading.Event()
        self.events: list[str] = []

    @staticmethod
    def _open(out: object) -> int:
        ct.cast(out, ct.POINTER(ct.c_void_p))[0] = 37
        return 0

    @staticmethod
    def _submit(_session: object, _batch: object, out: object) -> int:
        ct.cast(out, ct.POINTER(ct.c_uint64))[0] = 5
        return 0

    def _collect(self, _session: object, _ticket: int, _timeout: int, output: ct.Array, count: int) -> int:
        self.in_collect.set()
        self.release.wait(10)
        self.events.append("collect-done")
        return 0

    def patch(self) -> object:
        return patch("seabraid._abi.ct.CDLL", return_value=self.library)


class RuntimeLockTests(unittest.TestCase):
    def test_close_from_another_thread_waits_for_the_collect_in_flight(self) -> None:
        fake = MockedV1Library()
        fake.library.seabraid_close.side_effect = lambda _session: fake.events.append("close") or 0
        with fake.patch():
            runtime = Runtime(__file__, reference=True)
            prepared = runtime.prepare(hidden=1, experts=1, max_tokens=1, top_k=1)
            weaver = threading.Thread(target=lambda: prepared.weave([1.0], [0], [1.0]))
            closer = threading.Thread(target=runtime.close)
            weaver.start()
            self.assertTrue(fake.in_collect.wait(10))
            closer.start()
            closer.join(0.3)
            self.assertTrue(closer.is_alive(), "close() ran while a collect was in flight")
            fake.library.seabraid_close.assert_not_called()
            fake.release.set()
            weaver.join(10)
            closer.join(10)
        self.assertEqual(fake.events, ["collect-done", "close"])

    def test_concurrent_weaves_serialise_instead_of_corrupting_the_pending_batch(self) -> None:
        fake = MockedV1Library()
        fake.release.set()
        errors: list[BaseException] = []

        def weave(prepared: object) -> None:
            try:
                for _ in range(50):
                    prepared.weave([1.0], [0], [1.0])
            except BaseException as error:  # the test reports whatever escaped
                errors.append(error)

        with fake.patch(), Runtime(__file__, reference=True) as runtime:
            prepared = runtime.prepare(hidden=1, experts=1, max_tokens=1, top_k=1)
            threads = [threading.Thread(target=weave, args=(prepared,)) for _ in range(4)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(30)
        self.assertEqual(errors, [])  # unlocked, a second weave sees "batch is pending"


class ForkGuardTests(unittest.TestCase):
    @staticmethod
    def open_session() -> Runtime:
        runtime = Runtime(__file__, reference=True)
        runtime.prepare(hidden=1, experts=1, max_tokens=1, top_k=1)
        return runtime

    def test_disowning_means_no_close_no_finalizer_close_and_a_clear_error_on_use(self) -> None:
        fake = MockedV1Library()
        fake.release.set()
        with fake.patch():
            runtime = self.open_session()
            session = runtime._session
            with patch.object(_native, "_LIVE", weakref.WeakSet([session._owned])):
                _native._disown_after_fork()
            with self.assertRaisesRegex(RuntimeError, "parent process"):
                runtime.status()
            runtime.close()
            del runtime, session
            gc.collect()
        fake.library.seabraid_close.assert_not_called()

    @unittest.skipUnless(hasattr(os, "fork"), "no fork on this platform")
    def test_a_real_forked_child_never_closes_the_parents_session(self) -> None:
        fake = MockedV1Library()
        fake.release.set()
        with fake.patch():
            runtime = self.open_session()
            pid = os.fork()
            if pid == 0:  # child: report through the exit code only, never raise into the test runner
                code = 2
                try:
                    runtime.close()
                    gc.collect()
                    code = 0 if fake.library.seabraid_close.call_count == 0 else 3
                finally:
                    os._exit(code)
            _, status = os.waitpid(pid, 0)
            self.assertEqual(os.waitstatus_to_exitcode(status), 0, "the child closed the parent's native handle")
            runtime.close()
            self.assertEqual(fake.library.seabraid_close.call_count, 1)  # the parent still owns and closes it


if __name__ == "__main__":
    unittest.main()
