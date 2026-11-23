# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Edge cases of the Python binding itself, each pinned on the reference session and (with SEABRAID_LIBRARY) natively.

Checker for seabraid._abi.load_library (bad paths), runtime.Runtime (close, use after close, re-entry from a provider,
close from another thread), runtime.PreparedMoE.collect (KeyboardInterrupt), reference/native submit (Ctrl-C inside a
provider), and the buffer rules for `out` aliasing an input and for very large shapes. Every backend runs the same
assertions, so a difference between reference and native is a failure here, not a note.
"""

from __future__ import annotations

import os
import signal
import sys
import tempfile
import threading
import time
import unittest
from array import array
from unittest.mock import patch

from seabraid import Runtime, SeabraidError, _abi
from seabraid.constants import INVALID, PROVIDER, TIMEOUT

from ._v2_support import LIBRARY, reference_runtime

SHAPE = dict(hidden=2, experts=3, max_tokens=4, top_k=2)
GOOD = ([1.0, 2.0], [0, 1], [0.5, 0.5])  # one row; the built-in transform gives [1.5, 3.0]


def backends() -> list[tuple[str, object]]:
    """(name, make(**kwargs) -> Runtime) for the reference session and, when a library is configured, the native one."""
    found = [("reference", reference_runtime)]
    if LIBRARY:
        found.append(("native", lambda **kwargs: Runtime(LIBRARY, **kwargs)))
    return found


def fill(value: float):
    def serve(operation: int, config: object, acts: object, ids: object, weights: object, out: object) -> None:
        for index in range(len(out)):
            out[index] = value

    return serve


def _foreign_library() -> str | None:
    """A real shared library that is not Seabraid, so the epoch check (not the loader) is what refuses it."""
    if sys.platform == "win32":
        return os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "kernel32.dll")
    # WHY the maps file: a glob can find a 32-bit copy; the mapped libc is by construction one this process can load.
    try:
        with open("/proc/self/maps") as maps:
            # WHY the basename test: a bare "/libc" substring also matches this package's own library once its
            # name starts with "c" (the rename rehearsal hit it), and that one is not foreign.
            paths = {path for path in (line.split()[-1] for line in maps)
                     if os.path.basename(path).startswith(("libc.", "libc-"))}
    except OSError:
        return None
    return sorted(paths)[0] if paths else None


class LibraryPathTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.text = os.path.join(self.dir.name, "notalib.so")
        with open(self.text, "w") as handle:
            handle.write("not a shared library")
        self.empty = os.path.join(self.dir.name, "empty.so")
        open(self.empty, "wb").close()

    def assert_refused(self, path: object, *words: str) -> None:
        for how in ("argument", "environment"):
            with self.subTest(how=how, path=str(path)[-12:]), patch.dict(os.environ):
                os.environ.pop("SEABRAID_LIBRARY", None)
                call = (lambda: Runtime(path, reference=True))
                if how == "environment":
                    os.environ["SEABRAID_LIBRARY"] = str(path)
                    call = (lambda: Runtime(reference=True))
                with self.assertRaises(OSError) as caught:
                    call()
                message = str(caught.exception)
                self.assertIn("could not be loaded", message)
                self.assertNotIn(self.dir.name, message)  # the caller's path is never echoed
                for word in words:
                    self.assertIn(word, message)

    def test_a_missing_path_is_an_oserror_without_the_path(self) -> None:
        self.assert_refused(os.path.join(self.dir.name, "nope.so"))

    def test_a_directory_is_named_as_one(self) -> None:
        self.assert_refused(self.dir.name, "directory")

    def test_a_file_that_is_not_a_library_is_refused(self) -> None:
        self.assert_refused(self.text)
        self.assert_refused(self.empty)

    def test_a_null_byte_in_the_path_is_an_oserror_not_a_valueerror(self) -> None:
        with self.assertRaises(OSError):
            _abi.load_library("seabraid\0.so")

    def test_a_loadable_library_that_is_not_seabraid_is_refused_by_the_epoch_check(self) -> None:
        path = _foreign_library()
        if path is None or not os.path.exists(path):
            self.skipTest("no system shared library to use as a stand-in")
        with self.assertRaises(OSError) as caught:
            _abi.load_library(path)
        self.assertIn("seabraid_abi_epoch", str(caught.exception))

    def test_a_path_of_the_wrong_type_is_a_typeerror(self) -> None:
        with self.assertRaises(TypeError):
            Runtime(123, reference=True)


class LifecycleTests(unittest.TestCase):
    def test_close_twice_is_a_no_op_and_every_call_after_close_is_refused(self) -> None:
        for name, make in backends():
            with self.subTest(name):
                runtime = make(reference=True)
                prepared = runtime.prepare(**SHAPE)
                runtime.close()
                runtime.close()
                runtime.close(timeout_ms=10)
                for label, call in (("weave", lambda: prepared.weave(*GOOD)), ("collect", prepared.collect),
                                    ("status", runtime.status), ("capabilities", runtime.capabilities),
                                    ("prepare", lambda: runtime.prepare(**SHAPE)), ("enter", runtime.__enter__)):
                    with self.subTest(name, call=label), self.assertRaises(RuntimeError):
                        call()

    def test_close_with_a_batch_left_pending_by_a_timeout_releases_it(self) -> None:
        for name, make in backends():
            with self.subTest(name):
                gate = threading.Event()
                if name == "reference":
                    gate.set()  # the reference provider runs inside submit, so it can never be what times out

                def serve(operation: int, config: object, acts: object, ids: object, weights: object,
                          out: object) -> None:
                    gate.wait(30)
                    for index in range(len(out)):
                        out[index] = 1.0

                runtime = make(provider=serve)
                prepared = runtime.prepare(**SHAPE)
                if name == "reference":  # no wait to time out here, so inject the engine's TIMEOUT answer
                    def timed_out(*args: object, **kwargs: object) -> object:
                        raise SeabraidError("seabraid_collect", TIMEOUT)

                    with patch.object(runtime._session, "collect", timed_out), \
                            self.assertRaises(SeabraidError) as caught:
                        prepared.weave(*GOOD)
                else:  # native: the provider really is still running when the 1 ms wait ends
                    with self.assertRaises(SeabraidError) as caught:
                        prepared.weave(*GOOD, timeout_ms=1)
                self.assertEqual(caught.exception.status, TIMEOUT)
                self.assertTrue(runtime.status().outstanding, "the batch is still pending")
                with self.assertRaises(RuntimeError):
                    prepared.weave(*GOOD)  # refused: a batch is pending
                gate.set()
                runtime.close()  # must release the pending batch, not hang or leak it
                with self.assertRaises(RuntimeError):
                    runtime.status()
                with self.assertRaises(RuntimeError):
                    prepared.collect()

    def test_a_with_block_closes_on_an_exception(self) -> None:
        for name, make in backends():
            with self.subTest(name), self.assertRaises(KeyError):
                with make(reference=True) as runtime:
                    raise KeyError("x")
            with self.assertRaises(RuntimeError):
                runtime.status()


class ProviderCallbackTests(unittest.TestCase):
    def test_a_provider_cannot_call_back_into_its_runtime(self) -> None:
        for name, make in backends():
            with self.subTest(name):
                seen: dict = {}
                holder: dict = {}

                def serve(operation: int, config: object, acts: object, ids: object, weights: object,
                          out: object) -> None:
                    runtime, prepared = holder["runtime"], holder["prepared"]
                    for label, call in (("close", runtime.close), ("status", runtime.status),
                                        ("capabilities", runtime.capabilities), ("collect", prepared.collect),
                                        ("weave", lambda: prepared.weave(*GOOD, timeout_ms=50)),
                                        ("prepare", lambda: runtime.prepare(**SHAPE))):
                        try:
                            call()
                        except RuntimeError as error:
                            seen[label] = str(error)
                    for index in range(len(out)):
                        out[index] = 4.0

                with make(provider=serve) as runtime:
                    holder.update(runtime=runtime, prepared=runtime.prepare(**SHAPE))
                    started = time.monotonic()
                    self.assertEqual(holder["prepared"].weave(*GOOD), [4.0, 4.0])
                    self.assertLess(time.monotonic() - started, 5.0, "re-entry must fail fast, never wait out a timeout")
                    self.assertEqual(sorted(seen), ["capabilities", "close", "collect", "prepare", "status", "weave"])
                    self.assertTrue(all("own Runtime" in text for text in seen.values()), seen)
                    self.assertEqual(runtime.status().submitted, 1)  # the callback's close() did not close it

    def test_a_provider_may_use_a_different_runtime(self) -> None:
        for name, make in backends():
            with self.subTest(name):
                inner = reference_runtime(reference=True)
                inner_prepared = inner.prepare(**SHAPE)
                results: list = []

                def serve(operation: int, config: object, acts: object, ids: object, weights: object,
                          out: object) -> None:
                    results.append(inner_prepared.weave(*GOOD))
                    for index in range(len(out)):
                        out[index] = 1.0

                with make(provider=serve) as runtime:
                    runtime.prepare(**SHAPE).weave(*GOOD)
                inner.close()
                self.assertEqual(results, [[1.5, 3.0]])

    def test_close_from_another_thread_waits_for_the_callback_and_the_result_is_intact(self) -> None:
        for name, make in backends():
            with self.subTest(name):
                started = threading.Event()

                def slow(operation: int, config: object, acts: object, ids: object, weights: object,
                         out: object) -> None:
                    started.set()
                    time.sleep(0.3)
                    for index in range(len(out)):
                        out[index] = 7.0

                runtime = make(provider=slow)
                prepared = runtime.prepare(**SHAPE)
                result: dict = {}
                worker = threading.Thread(target=lambda: result.update(value=prepared.weave(*GOOD)))
                worker.start()
                self.assertTrue(started.wait(5))
                began = time.monotonic()
                runtime.close()
                self.assertGreaterEqual(time.monotonic() - began, 0.05)  # it waited for the batch in flight
                worker.join(5)
                self.assertEqual(result["value"], [7.0, 7.0])
                with self.assertRaises(RuntimeError):
                    prepared.weave(*GOOD)

    def test_what_a_misbehaving_provider_can_do(self) -> None:
        def returns_a_list(op: int, cfg: object, acts: object, ids: object, weights: object, out: object) -> list:
            for index in range(len(out)):
                out[index] = 2.0
            return [9.0] * 100  # ignored: only the output buffer counts

        def overruns(op: int, cfg: object, acts: object, ids: object, weights: object, out: object) -> None:
            out[len(out)] = 1.0

        def writes_nan(op: int, cfg: object, acts: object, ids: object, weights: object, out: object) -> None:
            for index in range(len(out)):
                out[index] = float("nan")

        for name, make in backends():
            with self.subTest(name), self.assertLogs("seabraid", "ERROR"):
                with make(provider=returns_a_list) as runtime:
                    self.assertEqual(runtime.prepare(**SHAPE).weave(*GOOD), [2.0, 2.0])
                with make(provider=overruns) as runtime:
                    prepared = runtime.prepare(**SHAPE)
                    with self.assertRaises(SeabraidError) as caught:
                        prepared.weave(*GOOD)
                    self.assertEqual(caught.exception.status, PROVIDER)
                    self.assertEqual((runtime.status().provider_failures, runtime.status().collected), (1, 0))
                    with self.assertRaises(SeabraidError):  # the session is still usable afterwards
                        prepared.weave(*GOOD)
                with make(provider=writes_nan) as runtime:
                    result = runtime.prepare(**SHAPE).weave(*GOOD)
                    self.assertTrue(all(value != value for value in result))


class InterruptTests(unittest.TestCase):
    def test_ctrl_c_inside_a_reference_provider_propagates_and_changes_nothing(self) -> None:
        for exception in (KeyboardInterrupt, SystemExit):
            with self.subTest(exception.__name__):
                interrupt = [True]

                def serve(operation: int, config: object, acts: object, ids: object, weights: object,
                          out: object) -> None:
                    if interrupt[0]:
                        raise exception()
                    for index in range(len(out)):
                        out[index] = 3.0

                with reference_runtime(provider=serve) as runtime:
                    prepared = runtime.prepare(**SHAPE)
                    with self.assertRaises(exception):
                        prepared.weave(*GOOD, placement_epoch=9)
                    status = runtime.status()
                    self.assertEqual((status.submitted, status.collected, status.provider_failures,
                                      status.outstanding), (0, 0, 0, False))
                    interrupt[0] = False
                    self.assertEqual(prepared.weave(*GOOD, placement_epoch=1), [3.0, 3.0])  # the epoch 9 never landed

    def test_ctrl_c_after_the_engine_finished_forgets_the_lost_batch(self) -> None:
        with reference_runtime(provider=fill(5.0)) as runtime:
            prepared = runtime.prepare(**SHAPE)
            real = runtime._session.collect

            def interrupted(*args: object, **kwargs: object) -> object:
                real(*args, **kwargs)  # the ticket is consumed ...
                raise KeyboardInterrupt  # ... and the signal is handled only now

            with patch.object(runtime._session, "collect", interrupted), self.assertRaises(KeyboardInterrupt):
                prepared.weave(*GOOD)
            self.assertEqual(prepared.weave(*GOOD), [5.0, 5.0])  # not "a batch is pending"

    def test_ctrl_c_before_the_engine_finished_keeps_the_batch_collectable(self) -> None:
        with reference_runtime(provider=fill(6.0)) as runtime:
            prepared = runtime.prepare(**SHAPE)

            def interrupted(*args: object, **kwargs: object) -> object:
                raise KeyboardInterrupt  # before the wait consumed anything

            with patch.object(runtime._session, "collect", interrupted), self.assertRaises(KeyboardInterrupt):
                prepared.weave(*GOOD)
            self.assertEqual(prepared.collect(), [6.0, 6.0])

    @unittest.skipUnless(LIBRARY, "set SEABRAID_LIBRARY for the native interrupt")
    def test_a_native_interrupt_during_collect_leaves_the_session_usable(self) -> None:
        import _thread

        # WHY the signal is raised here and not from the provider after a sleep: Python runs a signal handler only
        # once the native wait returns, so "SIGINT while the main thread waits" is observably "SIGINT the instant the
        # wait returns". Raising it at exactly that instant, on the main thread, needs no timing guess (the old
        # 0.1 s sleep lost the race under load) and still goes through the real signal-delivery path.
        # WHY the handler is installed: a test run started from a background shell (`cmd &`, nohup, some CI runners)
        # inherits SIGINT as ignored, Python then installs no handler, and interrupt_main() does nothing, so the test
        # failed with "KeyboardInterrupt not raised" for a reason that has nothing to do with the binding.
        if signal.getsignal(signal.SIGINT) is not signal.default_int_handler:
            self.addCleanup(signal.signal, signal.SIGINT, signal.getsignal(signal.SIGINT))
            signal.signal(signal.SIGINT, signal.default_int_handler)
        with Runtime(LIBRARY, provider=fill(5.0)) as runtime:
            prepared = runtime.prepare(**SHAPE)
            real = runtime._session.collect
            armed = [True]

            def collect_then_signal(*args: object, **kwargs: object) -> object:
                result = real(*args, **kwargs)  # the native wait ran to completion: the ticket is consumed
                if armed[0]:
                    _thread.interrupt_main()  # pending signal, handled at the next eval-breaker check ...
                    for _ in range(1_000_000):  # ... which a backward jump is
                        pass
                return result

            with patch.object(runtime._session, "collect", collect_then_signal), self.assertRaises(KeyboardInterrupt):
                prepared.weave(*GOOD)
            armed[0] = False
            self.assertEqual(prepared.weave(*GOOD), [5.0, 5.0])  # the lost batch was forgotten, not "pending"
            self.assertEqual(runtime.status().submitted, 2)


class BufferEdgeTests(unittest.TestCase):
    def run_all(self, check: object) -> None:
        for name, make in backends():
            for kind, kwargs in (("built-in", {"reference": True}),
                                 ("provider", {"provider": lambda op, c, a, i, w, out: [
                                     out.__setitem__(k, a[k] * 2.0) for k in range(len(out))]})):
                with self.subTest(name, kind=kind), make(**kwargs) as runtime:
                    check(runtime.prepare(hidden=2, experts=3, max_tokens=4, top_k=2), make, kwargs)

    def test_out_may_be_the_activations_array_itself(self) -> None:
        def check(prepared: object, make: object, kwargs: dict) -> None:
            ids, weights = [0, 1, 2, 0], [0.5, 0.5, 0.25, 0.75]
            with make(**kwargs) as other:
                want = other.prepare(**SHAPE).weave([1.0, 2.0, 3.0, 4.0], ids, weights)
            acts = array("f", [1.0, 2.0, 3.0, 4.0])
            self.assertIs(prepared.weave(acts, ids, weights, out=acts), acts)
            self.assertEqual(list(acts), want)

        self.run_all(check)

    def test_out_may_overlap_the_activations_partially(self) -> None:
        def check(prepared: object, make: object, kwargs: dict) -> None:
            ids, weights = [0, 1, 2, 0], [0.5, 0.5, 0.25, 0.75]
            with make(**kwargs) as other:
                want = other.prepare(**SHAPE).weave([1.0, 2.0, 3.0, 4.0], ids, weights)
            buffer = array("f", [1.0, 2.0, 3.0, 4.0, 0.0, 0.0])
            view = memoryview(buffer)
            prepared.weave(view[0:4], ids, weights, out=view[2:6])  # the engine copied its inputs before writing out
            self.assertEqual(list(buffer), [1.0, 2.0, *want])

        self.run_all(check)

    def test_out_may_be_the_weights_buffer_when_the_sizes_match(self) -> None:
        def check(prepared: object, make: object, kwargs: dict) -> None:
            weights = array("f", [0.5, 0.5])  # hidden == top_k == 2, one row
            with make(**kwargs) as other:
                want = other.prepare(**SHAPE).weave([1.0, 2.0], [0, 1], [0.5, 0.5])
            prepared.weave([1.0, 2.0], [0, 1], weights, out=weights)
            self.assertEqual(list(weights), want)

        self.run_all(check)

    def test_a_shape_with_max_tokens_at_the_uint32_limit_prepares_without_allocating_for_it(self) -> None:
        limit = (1 << 32) - 1
        for name, make in backends():
            with self.subTest(name), make(reference=True) as runtime:
                prepared = runtime.prepare(hidden=1, experts=2, max_tokens=limit, top_k=1)
                self.assertEqual(prepared.weave([1.0, 2.0, 3.0], [0, 1, 1], [0.5, 0.5, 2.0]), [0.5, 2.0, 12.0])

    def test_a_large_batch_at_max_tokens_agrees_between_backends_and_one_more_row_is_refused(self) -> None:
        tokens, hidden = 1 << 13, 8
        acts = array("f", [(index % 97) * 0.125 - 3.0 for index in range(tokens * hidden)])
        ids = array("i", [(index // 2 + index % 2) % 3 if index % 2 == 0 else (index // 2 + 1) % 3 for index in
                          range(tokens * 2)])
        weights = array("f", [0.25 + (index % 5) * 0.125 for index in range(tokens * 2)])
        results = []
        for name, make in backends():
            with make(reference=True) as runtime:
                prepared = runtime.prepare(hidden=hidden, experts=3, max_tokens=tokens, top_k=2)
                out = array("f", bytes(4 * tokens * hidden))
                prepared.weave(acts, ids, weights, out=out)
                results.append(out.tobytes())
                over = array("f", bytes(4 * (tokens + 1) * hidden))
                with self.assertRaises(SeabraidError) as caught:
                    prepared.weave(over, array("i", [0, 1] * (tokens + 1)), array("f", [0.5] * (2 * (tokens + 1))))
                self.assertEqual(caught.exception.status, INVALID)
        self.assertEqual(len(set(results)), 1)


if __name__ == "__main__":
    unittest.main()
