# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""A Python callable hosted as the provider: reference path, the real C trampoline, a mocked library, and native.

Checker for seabraid._provider and the provider= argument of seabraid.Runtime. Native cases need SEABRAID_LIBRARY.
"""

from __future__ import annotations

import ctypes as ct
import os
import unittest
import warnings
from unittest.mock import Mock, patch

from seabraid import SeabraidError, Runtime, _abi, _provider
from seabraid._model import Shape
from seabraid._views import ProviderView
from seabraid.constants import INVALID, OK, PROVIDER

SHAPE = dict(hidden=2, experts=3, max_tokens=2, top_k=2)
LIBRARY = os.environ.get("SEABRAID_LIBRARY")


def double(operation: int, config: Shape, acts: object, ids: object, weights: object, out: object) -> None:
    """out = activations * 2 + operation: simple, visible, and independent of routing."""
    for i in range(len(out)):
        out[i] = acts[i] * 2 + operation


def explode(*_args: object) -> None:
    raise KeyError("secret tensor values 1.5 2.5")


def silent_runtime(**kwargs: object) -> Runtime:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with patch.dict(os.environ):
            os.environ.pop("SEABRAID_LIBRARY", None)
            with patch.object(_provider.logging.getLogger("seabraid"), "disabled", True):
                return Runtime(**kwargs)


class ReferencePathTests(unittest.TestCase):
    def test_callable_serves_the_batch_and_gets_the_documented_arguments(self) -> None:
        seen = []

        def record(operation: int, config: Shape, acts: object, ids: object, weights: object, out: object) -> None:
            seen.append((operation, config, list(acts), list(ids), list(weights), len(out)))
            double(operation, config, acts, ids, weights, out)

        with silent_runtime(provider=record) as runtime:
            result = runtime.prepare(**SHAPE).weave([1.0, 2.5], [0, 2], [0.25, 0.75])
        self.assertEqual(result, [2.0, 5.0])
        self.assertEqual(seen, [(0, Shape(2, 3, 2, 2), [1.0, 2.5], [0, 2], [0.25, 0.75], 2)])

    def test_views_are_provider_views_without_numpy_and_expire_after_the_call(self) -> None:
        kept = []

        def keep(operation: int, config: Shape, acts: object, ids: object, weights: object, out: object) -> None:
            kept.extend((acts, out))
            self.assertIsInstance(acts, ProviderView)
            self.assertTrue(acts.readonly and ids.readonly and weights.readonly and not out.readonly)
            self.assertEqual((acts.format, ids.format, out.format), ("f", "i", "f"))
            out[0], out[1] = 7.0, 8.0

        with patch.object(_provider, "_numpy_or_none", return_value=None):
            with silent_runtime(provider=keep) as runtime:
                self.assertEqual(runtime.prepare(**SHAPE).weave([1.0, 2.0], [0, 1], [1.0, 0.0]), [7.0, 8.0])
        with self.assertRaises(ValueError):
            kept[0].tolist()  # released: a stashed view fails loudly instead of reading freed memory

    def test_no_derived_window_can_outlive_the_call_without_numpy(self) -> None:
        """R7 MED 2: out[0:4] / acts[0:4] / casts used to survive release(); now none can be made."""
        kept, refused = {}, []

        def stash(operation: int, config: Shape, acts: object, ids: object, weights: object, out: object) -> None:
            for name, view in (("acts", acts), ("out", out)):
                try:
                    kept[name] = view[0:1]
                except TypeError:
                    refused.append(name)
                self.assertFalse(hasattr(view, "cast"))
                with self.assertRaises(TypeError):
                    memoryview(view)  # no buffer protocol: no numpy.frombuffer / memoryview re-export either
            kept["iterator"] = iter(out)
            out[0:2] = [3.0, 4.0]  # writing a range copies INTO the memory and keeps nothing

        with patch.object(_provider, "_numpy_or_none", return_value=None):
            with silent_runtime(provider=stash) as runtime:
                self.assertEqual(runtime.prepare(**SHAPE).weave([1.0, 2.0], [0, 1], [1.0, 0.0]), [3.0, 4.0])
        self.assertEqual(sorted(refused), ["acts", "out"])
        self.assertEqual(list(kept), ["iterator"])
        with self.assertRaises(ValueError):
            next(kept["iterator"])

    def test_numpy_views_when_numpy_is_importable(self) -> None:
        numpy = _provider._numpy_or_none()
        if numpy is None:
            self.skipTest("numpy not installed")

        def check(operation: int, config: Shape, acts: object, ids: object, weights: object, out: object) -> None:
            self.assertEqual((acts.dtype, ids.dtype, out.dtype), (numpy.float32, numpy.int32, numpy.float32))
            self.assertFalse(acts.flags.writeable)
            out[:] = acts * 3

        with silent_runtime(provider=check) as runtime:
            self.assertEqual(runtime.prepare(**SHAPE).weave([1.0, 2.0], [0, 1], [1.0, 0.0]), [3.0, 6.0])

    def test_operation_ids_select_and_are_bounded(self) -> None:
        with silent_runtime(provider=double, operations=2) as runtime:
            prepared = runtime.prepare(**SHAPE)
            args = ([1.0, 2.0], [0, 1], [1.0, 0.0])
            self.assertEqual(prepared.weave(*args), [2.0, 4.0])
            self.assertEqual(prepared.weave(*args, operation=1), [3.0, 5.0])
            with self.assertRaises(SeabraidError) as raised:
                prepared.weave(*args, operation=2)
            self.assertEqual((raised.exception.operation, raised.exception.status), ("seabraid_submit_operation", INVALID))

    def test_callable_exception_is_one_log_record_and_a_provider_error(self) -> None:
        with silent_runtime(provider=explode) as runtime:
            prepared = runtime.prepare(**SHAPE)
            with self.assertLogs("seabraid", "ERROR") as logs, self.assertRaises(SeabraidError) as raised:
                prepared.weave([1.0, 2.0], [0, 1], [1.0, 0.0])
            self.assertEqual((raised.exception.operation, raised.exception.status), ("seabraid_collect", PROVIDER))
            self.assertEqual(len(logs.records), 1)
            self.assertIn("KeyError", logs.output[0])
            self.assertNotIn("secret", logs.output[0])  # type only, never the message or tensor data

    def test_session_is_reusable_after_a_failed_callable(self) -> None:
        calls = []

        def flaky(*args: object) -> None:
            calls.append(1)
            (explode if len(calls) == 1 else double)(*args)

        with silent_runtime(provider=flaky) as runtime:
            prepared = runtime.prepare(**SHAPE)
            with self.assertLogs("seabraid", "ERROR"), self.assertRaises(SeabraidError):
                prepared.weave([1.0, 2.0], [0, 1], [1.0, 0.0])
            self.assertEqual(prepared.weave([1.0, 2.0], [0, 1], [1.0, 0.0]), [2.0, 4.0])

    def test_argument_errors(self) -> None:
        with self.assertRaisesRegex(TypeError, "callable"):
            silent_runtime(provider=42)
        with self.assertRaisesRegex(ValueError, "require provider"):
            silent_runtime(operations=2)
        with self.assertRaisesRegex(ValueError, "require provider"):
            silent_runtime(partial_routes=True)
        with self.assertRaisesRegex(ValueError, "operations must be between"):
            silent_runtime(provider=double, operations=0)

    def test_partial_routes_accepts_skipped_picks_only_when_declared(self) -> None:
        args = ([1.0, 2.0], [-1, 1], [0.0, 0.5])
        with silent_runtime(provider=double, partial_routes=True) as runtime:
            self.assertEqual(runtime.prepare(**SHAPE).weave(*args), [2.0, 4.0])
        with silent_runtime(provider=double) as runtime:
            with self.assertRaises(SeabraidError):
                runtime.prepare(**SHAPE).weave(*args)

    def test_provider_notice_is_a_single_warning(self) -> None:
        with self.assertWarnsRegex(RuntimeWarning, "provider callable runs inside the pure-Python"):
            with patch.dict(os.environ):
                os.environ.pop("SEABRAID_LIBRARY", None)
                with patch.object(_provider.logging.getLogger("seabraid"), "disabled", True):
                    Runtime(provider=double).close()


class TrampolineTests(unittest.TestCase):
    """Drive the real CFUNCTYPE object exactly as the engine's worker thread would."""

    def call(self, provider: _provider.PythonProvider, operation: int = 0) -> tuple[int, list[float]]:
        acts, ids, wts = (ct.c_float * 2)(1.5, 2.0), (ct.c_int32 * 2)(0, 1), (ct.c_float * 2)(1.0, 0.0)
        batch = _abi._BatchV1(1, 1, acts, 2, ids, 2, wts, 2)
        config = _abi._ConfigV1(1, 2, 3, 2, 2)
        out = (ct.c_float * 2)(-1.0, -1.0)
        status = provider.struct().execute(None, operation, ct.byref(config), ct.byref(batch), out, 2)
        return status, list(out)

    def test_struct_fields_follow_the_header(self) -> None:
        provider = _provider.PythonProvider(double, operations=3, partial_routes=True)
        struct = provider.struct()
        self.assertEqual((struct.abi_version, struct.operation_count, struct.flags), (2, 3, 1))

    def test_ok_writes_through_the_runtime_owned_output(self) -> None:
        self.assertEqual(self.call(_provider.PythonProvider(double, 2), 1), (OK, [4.0, 5.0]))

    def test_exception_never_crosses_the_boundary(self) -> None:
        with self.assertLogs("seabraid", "ERROR") as logs:
            self.assertEqual(self.call(_provider.PythonProvider(explode))[0], PROVIDER)
        self.assertEqual(len(logs.records), 1)

    def test_system_exit_is_also_contained(self) -> None:
        def leave(*_args: object) -> None:
            raise SystemExit(3)

        with self.assertLogs("seabraid", "ERROR"):
            self.assertEqual(self.call(_provider.PythonProvider(leave))[0], PROVIDER)

    def test_stashed_array_is_a_provider_error(self) -> None:
        if _provider._numpy_or_none() is None:
            self.skipTest("numpy not installed: plain memoryviews are released even when stashed")
        kept = []

        def stash(operation: int, config: Shape, acts: object, ids: object, weights: object, out: object) -> None:
            kept.append(acts)
            out[:] = 0

        with self.assertLogs("seabraid", "ERROR") as logs:
            self.assertEqual(self.call(_provider.PythonProvider(stash))[0], PROVIDER)
        self.assertIn("kept a reference", logs.output[0])

    def test_stashed_slices_and_casts_never_reach_engine_memory_after_the_call(self) -> None:
        """Without numpy: the slice is refused inside the call, so nothing aliasing the buffers can be stashed.
        With numpy: a stashed slice keeps the buffer exported, which is a provider error with one ERROR record."""
        kept, refused = {}, []

        def stash(operation: int, config: Shape, acts: object, ids: object, weights: object, out: object) -> None:
            for name, view in (("acts", acts), ("out", out)):
                try:
                    kept[name] = view[0:1]
                except TypeError:
                    refused.append(name)
            for i in range(len(out)):
                out[i] = 0.0

        with patch.object(_provider, "_numpy_or_none", return_value=None):
            provider = _provider.PythonProvider(stash)
        self.assertEqual(self.call(provider)[0], OK)
        self.assertEqual((sorted(refused), kept), (["acts", "out"], {}))
        numpy = _provider._numpy_or_none()
        if numpy is None:
            return
        kept.clear()
        with self.assertLogs("seabraid", "ERROR") as logs:
            self.assertEqual(self.call(_provider.PythonProvider(stash))[0], PROVIDER)
        self.assertEqual(sorted(kept), ["acts", "out"])
        self.assertIn("kept a reference", logs.output[0])

    def test_bad_operation_and_null_pointers_are_invalid_not_crashes(self) -> None:
        provider = _provider.PythonProvider(double, 1)
        self.assertEqual(self.call(provider, 5)[0], INVALID)
        self.assertEqual(provider.struct().execute(None, 0, None, None, None, 0), INVALID)


class MockedLibraryTests(unittest.TestCase):
    """NativeSession wiring (open_multi, submit_operation, collect out=) without a linker."""

    def setUp(self) -> None:
        self.library = Mock(spec=[
            "seabraid_open", "seabraid_open_reference", "seabraid_open_multi", "seabraid_abi_epoch", "seabraid_prepare", "seabraid_submit", "seabraid_submit_operation",
            "seabraid_collect", "seabraid_close",
        ])
        self.opened: list[_abi._ProviderV2] = []
        self.library.seabraid_open_multi.side_effect = self._open
        self.library.seabraid_abi_epoch.return_value = 1
        self.library.seabraid_prepare.return_value = 0
        self.library.seabraid_submit_operation.side_effect = self._submit
        self.library.seabraid_collect.side_effect = self._collect
        self.library.seabraid_close.return_value = 0
        loader = patch("seabraid._abi.ct.CDLL", return_value=self.library)
        loader.start()
        self.addCleanup(loader.stop)

    def _open(self, provider: object, out: object) -> int:
        self.opened.append(ct.cast(provider, ct.POINTER(_abi._ProviderV2)).contents)
        ct.cast(out, ct.POINTER(ct.c_void_p))[0] = 5
        return 0

    def _submit(self, session: object, operation: int, batch: object, origins: int, out: object) -> int:
        self.assertEqual((operation, origins), (1, 0))
        ct.cast(out, ct.POINTER(ct.c_uint64))[0] = 9
        return 0

    def _collect(self, session: object, ticket: int, timeout: int, output: object, count: int) -> int:
        for i in range(count):
            output[i] = i + 0.5
        return 0

    def test_provider_is_bound_with_open_multi_and_operations_use_submit_operation(self) -> None:
        with Runtime(__file__, provider=double, operations=2) as runtime:
            prepared = runtime.prepare(**SHAPE)
            (struct,) = self.opened
            self.assertEqual((struct.operation_count, struct.flags), (2, 0))
            out = (ct.c_float * 2)()
            self.assertIs(prepared.weave([1.0, 2.0], [0, 1], [1.0, 0.0], operation=1, out=out), out)
            self.assertEqual(list(out), [0.5, 1.5])
        self.library.seabraid_submit.assert_not_called()

    def test_old_library_without_open_multi_says_so(self) -> None:
        del self.library.seabraid_open_multi
        with self.assertRaisesRegex(RuntimeError, "multi-operation providers"):
            Runtime(__file__, provider=double)

    def test_callback_object_outlives_open_and_is_dropped_after_close(self) -> None:
        runtime = Runtime(__file__, provider=double)
        session = runtime._session
        self.assertIsNotNone(session._provider)
        runtime.close()
        self.assertIsNone(session._provider)


@unittest.skipUnless(LIBRARY, "set SEABRAID_LIBRARY for native integration")
class NativeProviderTests(unittest.TestCase):
    def test_callable_runs_on_the_worker_thread_through_the_real_engine(self) -> None:
        import threading

        threads = []

        def serve(operation: int, config: Shape, acts: object, ids: object, weights: object, out: object) -> None:
            threads.append(threading.current_thread() is not threading.main_thread())
            double(operation, config, acts, ids, weights, out)

        with Runtime(LIBRARY, provider=serve, operations=2) as runtime:
            prepared = runtime.prepare(**SHAPE)
            self.assertEqual(runtime.provider, "native")
            self.assertEqual(prepared.weave([1.0, 2.0], [0, 1], [1.0, 0.0]), [2.0, 4.0])
            self.assertEqual(prepared.weave([1.0, 2.0], [0, 1], [1.0, 0.0], operation=1), [3.0, 5.0])
            with self.assertRaises(SeabraidError) as raised:
                prepared.weave([1.0, 2.0], [0, 1], [1.0, 0.0], operation=2)
            self.assertEqual(raised.exception.status, INVALID)
        self.assertEqual(threads, [True, True])

    def test_callable_exception_surfaces_as_provider_status_with_one_python_log_record(self) -> None:
        with Runtime(LIBRARY, provider=explode) as runtime:
            prepared = runtime.prepare(**SHAPE)
            with self.assertLogs("seabraid", "ERROR") as logs, self.assertRaises(SeabraidError) as raised:
                prepared.weave([1.0, 2.0], [0, 1], [1.0, 0.0])
            self.assertEqual(raised.exception.status, PROVIDER)
            python_records = [r for r in logs.records if "provider callable raised KeyError" in r.getMessage()]
            self.assertEqual(len(python_records), 1)

    def test_partial_routes_flag_reaches_the_engine(self) -> None:
        with Runtime(LIBRARY, provider=double, partial_routes=True) as runtime:
            prepared = runtime.prepare(**SHAPE)
            self.assertEqual(prepared.weave([1.0, 2.0], [-1, 1], [0.0, 0.5]), [2.0, 4.0])


if __name__ == "__main__":
    unittest.main()
