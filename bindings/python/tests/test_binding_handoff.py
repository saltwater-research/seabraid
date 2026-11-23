# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Hand-offs that an interrupt, a fork or a refused close can break: submit to pending, lock after fork, close refusal.

Checker for seabraid.runtime.PreparedMoE.weave (the ticket survives a Ctrl-C between the native submit and `_pending`),
seabraid.runtime._renew_locks (a forked child never inherits a held lock) and seabraid._native._close_owned (a refused
close keeps the handle). Every backend runs the same assertions; the native ones need SEABRAID_LIBRARY.
"""

from __future__ import annotations

import gc
import os
import signal
import threading
import unittest
import weakref
from unittest.mock import patch

from seabraid import Runtime, SeabraidError
from seabraid.constants import BUSY_OR_STATE, INVALID, TIMEOUT

from ._v2_support import LIBRARY
from .test_abi_safety import MockedBase, noop
from .test_binding_edges import GOOD, SHAPE, backends, fill


class SubmitInterruptTests(unittest.TestCase):
    def test_ctrl_c_right_after_the_engine_took_the_batch_keeps_the_ticket_collectable(self) -> None:
        for name, make in backends():
            with self.subTest(name), make(provider=fill(4.0)) as runtime:
                prepared = runtime.prepare(**SHAPE)
                real = runtime._session.submit

                def submitted_then_interrupted(*args: object, **kwargs: object) -> int:
                    real(*args, **kwargs)  # the engine now holds the ticket ...
                    raise KeyboardInterrupt  # ... and the signal is handled before `_pending` could be set

                with patch.object(runtime._session, "submit", submitted_then_interrupted), \
                        self.assertRaises(KeyboardInterrupt):
                    prepared.weave(*GOOD)
                self.assertTrue(runtime.status().outstanding)
                self.assertEqual(prepared.collect(), [4.0, 4.0])  # not "no batch is pending", not BUSY_OR_STATE
                self.assertEqual(prepared.weave(*GOOD), [4.0, 4.0])

    def test_a_submit_the_engine_refused_leaves_nothing_pending(self) -> None:
        for name, make in backends():
            with self.subTest(name), make(provider=fill(4.0)) as runtime:
                prepared = runtime.prepare(**SHAPE)
                with self.assertRaises(SeabraidError):
                    prepared.weave([1.0, 2.0], [0, 9], [0.5, 0.5])  # expert 9 does not exist
                self.assertEqual(prepared.weave(*GOOD), [4.0, 4.0])


@unittest.skipUnless(hasattr(os, "fork"), "no fork on this platform")
class ForkLockTests(unittest.TestCase):
    def test_a_child_forked_while_another_thread_holds_the_lock_does_not_hang(self) -> None:
        for name, make in backends():
            with self.subTest(name), make(reference=True) as runtime:
                runtime.prepare(**SHAPE)
                held, release = threading.Event(), threading.Event()

                def hold() -> None:
                    with runtime._lock:
                        held.set()
                        release.wait(30)

                holder = threading.Thread(target=hold)
                holder.start()
                held.wait(30)
                pid = os.fork()
                if pid == 0:  # child: only the exit code speaks; a hang is turned into a SIGALRM death
                    code = 2
                    try:
                        signal.alarm(10)
                        try:
                            runtime.status()
                            code = 0 if name == "reference" else 3
                        except RuntimeError as error:  # native: the intended "belongs to the parent process"
                            code = 0 if name == "native" and "parent process" in str(error) else 4
                    finally:
                        os._exit(code)
                release.set()
                holder.join()
                _, status = os.waitpid(pid, 0)
                self.assertEqual(os.waitstatus_to_exitcode(status), 0,
                                 "the child hung on (or mishandled) the lock a parent thread held")


class RefusedCloseTests(MockedBase):
    def test_a_refused_close_keeps_the_handle_and_the_provider_and_a_retry_closes_once(self) -> None:
        self.library.seabraid_close.return_value = BUSY_OR_STATE
        runtime = Runtime(__file__, provider=noop)
        provider_ref = weakref.ref(runtime._session._provider)
        with self.assertRaises(SeabraidError) as caught:
            runtime.close()
        self.assertEqual(caught.exception.status, BUSY_OR_STATE)
        gc.collect()
        self.assertIsNotNone(provider_ref(), "the engine kept the handle, so its provider thunk must stay alive")
        self.assertEqual(runtime._session._owned.session.value, 77)
        self.assertTrue(runtime._open, "a refused close leaves the runtime open so it can be retried")
        self.library.seabraid_close.return_value = 0
        runtime.close()
        self.assertEqual(self.library.seabraid_close.call_count, 2)
        self.assertEqual({call.args[0].value for call in self.library.seabraid_close.call_args_list}, {77})
        runtime.close()  # now idempotent
        del runtime
        gc.collect()
        self.assertEqual(self.library.seabraid_close.call_count, 2, "the finalizer must not close a third time")

    def test_a_bounded_close_refused_the_same_way_is_also_retryable(self) -> None:
        self.library.seabraid_close_timeout.return_value = BUSY_OR_STATE
        runtime = Runtime(__file__, provider=noop)
        with self.assertRaises(SeabraidError):
            runtime.close(timeout_ms=5)
        self.library.seabraid_close_timeout.return_value = 0
        runtime.close(timeout_ms=5)
        self.assertEqual(self.library.seabraid_close_timeout.call_count, 2)

    def test_a_session_collected_after_a_refused_close_leaks_its_provider_instead_of_freeing_it(self) -> None:
        self.library.seabraid_close.return_value = BUSY_OR_STATE
        runtime = Runtime(__file__, provider=noop)
        provider_ref = weakref.ref(runtime._session._provider)
        with self.assertRaises(SeabraidError):
            runtime.close()
        del runtime
        gc.collect()  # the finalizer retries once, is refused again, and must keep the thunk the worker may call
        self.assertEqual(self.library.seabraid_close.call_count, 2)
        self.assertIsNotNone(provider_ref())

    def test_a_close_that_says_the_handle_is_not_live_is_not_retried(self) -> None:
        self.library.seabraid_close.return_value = INVALID
        runtime = Runtime(__file__, provider=noop)
        with self.assertRaises(SeabraidError):
            runtime.close()
        runtime.close()
        del runtime
        gc.collect()
        self.assertEqual(self.library.seabraid_close.call_count, 1, "a dead handle is never closed again")

    def test_a_timeout_still_consumes_the_handle(self) -> None:
        self.library.seabraid_close_timeout.return_value = TIMEOUT
        runtime = Runtime(__file__, provider=noop)
        with self.assertRaises(SeabraidError):
            runtime.close(timeout_ms=5)
        runtime.close()
        self.library.seabraid_close.assert_not_called()


@unittest.skipUnless(LIBRARY, "set SEABRAID_LIBRARY for the native close refusal")
class NativeRefusedCloseTests(unittest.TestCase):
    def test_a_close_from_the_provider_thread_is_refused_and_the_runtime_still_closes_cleanly(self) -> None:
        seen: dict = {}
        holder: dict = {}
        # WHY an event: the provider thread calls NativeSession.close directly, which bypasses the Runtime lock (the
        # Runtime serialises every call; this one is on purpose outside it to reach the engine's refusal). `_close_owned`
        # takes the handle out of the owner for the length of that call and puts it back when the engine refuses, so a
        # main thread that read `_session` in that window got a null handle ("seabraid_collect: session is null") - a
        # flake that only showed when the machine was loaded enough for the provider thread to run first. The main thread
        # now collects only after the provider's attempt has ended, so what is asserted (the refusal, the restored
        # handle, a working session afterwards) no longer depends on which thread the scheduler runs first.
        attempted = threading.Event()

        def serve(operation: int, config: object, acts: object, ids: object, weights: object, out: object) -> None:
            try:  # NativeSession.close directly: Runtime.close would be refused before reaching the engine
                holder["runtime"]._session.close()
            except SeabraidError as error:
                seen["status"] = error.status
                seen["detail"] = error.detail
            finally:
                attempted.set()
            for index in range(len(out)):
                out[index] = 2.0

        runtime = Runtime(LIBRARY, provider=serve)
        holder["runtime"] = runtime
        real_collect = runtime._session.collect

        def collect_after_the_attempt(*args: object, **kwargs: object) -> object:
            self.assertTrue(attempted.wait(30), "the provider never reached its close attempt")
            return real_collect(*args, **kwargs)

        prepared = runtime.prepare(**SHAPE)
        with patch.object(runtime._session, "collect", collect_after_the_attempt):
            self.assertEqual(prepared.weave(*GOOD), [2.0, 2.0])
            self.assertEqual(seen["status"], BUSY_OR_STATE)
            self.assertIn("provider callback", seen["detail"])
            self.assertTrue(runtime._session._finalizer.alive, "the backstop is re-armed after a refusal")
            attempted.clear()
            self.assertEqual(prepared.weave(*GOOD), [2.0, 2.0])  # the session survived the refused close
            self.assertEqual(seen["status"], BUSY_OR_STATE)  # refused again, by the engine, not by luck
        seen.clear()
        runtime.close()  # no refusal now; this is the one real close
        self.assertEqual(seen, {})
        with self.assertRaises(RuntimeError):
            runtime.status()


if __name__ == "__main__":
    unittest.main()
