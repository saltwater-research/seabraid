# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Layout-epoch refusal, null-provider refusal, the GC finalizer ordering and bounded close.

Checker for seabraid._abi._check_epoch, seabraid._native (finalizer, close) and Runtime(reference=, close(timeout_ms)).
Mocked-library cases run anywhere; native cases need SEABRAID_LIBRARY.
"""

from __future__ import annotations

import ctypes as ct
import gc
import os
import threading
import unittest
import weakref
from unittest.mock import Mock, patch

from seabraid import Runtime, SeabraidError, _abi

LIBRARY = os.environ.get("SEABRAID_LIBRARY")
SPEC = ["seabraid_open", "seabraid_open_reference", "seabraid_open_multi", "seabraid_abi_epoch",
        "seabraid_close_timeout", "seabraid_prepare", "seabraid_submit", "seabraid_collect", "seabraid_close"]
SHAPE = dict(hidden=1, experts=1, max_tokens=1, top_k=1)


def noop(operation: int, config: object, acts: object, ids: object, weights: object, out: object) -> None:
    for i in range(len(out)):
        out[i] = 0.0


class MockedBase(unittest.TestCase):
    epoch: int | None = 1

    def setUp(self) -> None:
        spec = SPEC if self.epoch is not None else [name for name in SPEC if name != "seabraid_abi_epoch"]
        self.library = Mock(spec=spec)
        if self.epoch is not None:
            self.library.seabraid_abi_epoch.return_value = self.epoch
        self.library.seabraid_open_multi.side_effect = self._open
        self.library.seabraid_open_reference.side_effect = self._open_one
        self.library.seabraid_close.return_value = 0
        self.library.seabraid_close_timeout.return_value = 0
        loader = patch("seabraid._abi.ct.CDLL", return_value=self.library)
        loader.start()
        self.addCleanup(loader.stop)

    @staticmethod
    def _open(_provider: object, out: object) -> int:
        ct.cast(out, ct.POINTER(ct.c_void_p))[0] = 77
        return 0

    @staticmethod
    def _open_one(out: object) -> int:
        ct.cast(out, ct.POINTER(ct.c_void_p))[0] = 77
        return 0


class EpochTests(MockedBase):
    epoch = 2

    def test_a_library_with_another_layout_epoch_is_refused_and_not_remembered(self) -> None:
        before = _abi.active_library()
        with self.assertRaisesRegex(OSError, "epoch 2, this binding needs 1"):
            Runtime(__file__, reference=True)
        self.assertIs(_abi.active_library(), before)
        self.library.seabraid_open_reference.assert_not_called()


class MissingEpochTests(MockedBase):
    epoch = None

    def test_a_library_without_the_symbol_predates_the_epoch(self) -> None:
        with self.assertRaisesRegex(OSError, "predates layout epoch 1"):
            Runtime(__file__, reference=True)


class NullProviderTests(MockedBase):
    def test_a_native_library_needs_a_provider_or_an_explicit_reference(self) -> None:
        with self.assertRaisesRegex(ValueError, "reference=True"):
            Runtime(__file__)
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            Runtime(__file__, provider=noop, reference=True)
        self.library.seabraid_open.assert_not_called()
        self.library.seabraid_open_reference.assert_not_called()

    def test_reference_classmethod_opens_through_open_reference(self) -> None:
        with Runtime.reference(__file__):
            pass
        self.library.seabraid_open_reference.assert_called_once()
        self.library.seabraid_open.assert_not_called()


class FinalizerTests(MockedBase):
    def test_a_dropped_runtime_closes_the_session_before_its_provider_is_released(self) -> None:
        runtime = Runtime(__file__, provider=noop)
        provider_ref = weakref.ref(runtime._session._provider)
        seen: list[bool] = []

        def close(session: ct.c_void_p) -> int:
            seen.append(provider_ref() is not None and session.value == 77)
            return 0

        self.library.reset_mock()  # the mock keeps call arguments, which would hold the provider alive
        self.library.seabraid_close.side_effect = close
        del runtime
        gc.collect()
        self.assertEqual(seen, [True], "closed once, while the provider thunk was still alive")
        self.assertIsNone(provider_ref(), "the provider is released only after the close")

    def test_an_explicit_close_is_not_repeated_by_the_finalizer(self) -> None:
        runtime = Runtime(__file__, provider=noop)
        runtime.close()
        del runtime
        gc.collect()
        self.library.seabraid_close.assert_called_once()


class BoundedCloseTests(MockedBase):
    def test_timeout_raises_leaks_the_provider_and_still_closes_the_runtime(self) -> None:
        self.library.seabraid_close_timeout.return_value = 3
        runtime = Runtime(__file__, provider=noop)
        provider_ref = weakref.ref(runtime._session._provider)
        with self.assertRaisesRegex(SeabraidError, "seabraid_close_timeout: timeout"):
            runtime.close(timeout_ms=50)
        self.assertEqual(self.library.seabraid_close_timeout.call_args.args[1], 50)
        gc.collect()
        self.assertIsNotNone(provider_ref(), "a stuck worker may still call the thunk, so it is kept")
        runtime.close()  # idempotent
        self.library.seabraid_close.assert_not_called()
        with self.assertRaisesRegex(ValueError, "timeout_ms must be between"):
            Runtime(__file__, provider=noop).close(timeout_ms=-1)


class CloseWithoutTimeoutSupportTests(MockedBase):
    """R7 LOW 9: close(timeout_ms) on a library without seabraid_close_timeout left the runtime marked closed."""

    def test_the_runtime_stays_open_and_a_plain_close_still_works(self) -> None:
        del self.library.seabraid_close_timeout
        runtime = Runtime(__file__, provider=noop)
        with self.assertRaisesRegex(RuntimeError, "does not expose a bounded close"):
            runtime.close(timeout_ms=10)
        self.library.seabraid_close.assert_not_called()
        runtime.close()  # the retry the caller would make
        self.library.seabraid_close.assert_called_once()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            runtime.status()


class V1EntryPointTests(MockedBase):
    """R7 LOW 16: the deprecated v1 symbols are optional (removed at 1.0); using one that is absent is a clear error."""

    def test_a_library_with_only_the_non_deprecated_calls_loads(self) -> None:
        for name in ("seabraid_open", "seabraid_open_reference", "seabraid_prepare", "seabraid_submit"):
            delattr(self.library, name)
        self.library.seabraid_session_open = Mock(return_value=0)  # a v3-only library
        self.assertIsNotNone(_abi.load_library(__file__))

    def test_the_v1_open_and_prepare_say_what_is_missing(self) -> None:
        del self.library.seabraid_open_reference
        with self.assertRaisesRegex(RuntimeError, "does not expose seabraid_open_reference"):
            Runtime(__file__, reference=True)

    def test_prepare_and_submit_say_what_is_missing(self) -> None:
        runtime = Runtime(__file__, reference=True)
        del self.library.seabraid_prepare
        with self.assertRaisesRegex(RuntimeError, "does not expose seabraid_prepare"):
            runtime.prepare(**SHAPE)
        runtime.close()

@unittest.skipUnless(LIBRARY, "set SEABRAID_LIBRARY for native integration")
class NativeSafetyTests(unittest.TestCase):
    def stuck_runtime(self) -> tuple[Runtime, threading.Event]:
        release = threading.Event()

        def wait(operation: int, config: object, acts: object, ids: object, weights: object, out: object) -> None:
            release.wait(10)
            for i in range(len(out)):
                out[i] = 1.0

        runtime = Runtime(LIBRARY, provider=wait)
        prepared = runtime.prepare(**SHAPE)
        with self.assertRaises(SeabraidError):
            prepared.weave([1.0], [0], [1.0], timeout_ms=0)  # leaves the ticket pending, worker blocked
        return runtime, release

    def test_close_with_timeout_abandons_a_stuck_provider(self) -> None:
        runtime, release = self.stuck_runtime()
        with self.assertRaisesRegex(SeabraidError, "timeout"):
            runtime.close(timeout_ms=50)
        release.set()  # let the leaked worker finish so the test process exits cleanly

    def test_dropping_a_runtime_with_a_pending_ticket_is_safe(self) -> None:
        runtime, release = self.stuck_runtime()
        threading.Timer(0.2, release.set).start()
        del runtime
        gc.collect()  # the finalizer waits for the worker, then frees the thunk; no crash is the assertion

    def test_the_real_library_reports_the_epoch_this_binding_expects(self) -> None:
        library = _abi.load_library(LIBRARY)
        self.assertEqual(library.seabraid_abi_epoch(), 1)

    def test_a_null_provider_is_refused_by_the_engine(self) -> None:
        library = _abi.load_library(LIBRARY)
        out = ct.c_void_p()
        self.assertEqual(library.seabraid_open(None, ct.byref(out)), 1)
        self.assertFalse(out)


if __name__ == "__main__":
    unittest.main()
