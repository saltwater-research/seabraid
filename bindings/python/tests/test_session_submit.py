# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""NativeSession.submit uses seabraid_session_submit when the library has it (a library without it is covered by
test_mock_runtime, whose mock library has only the older submit calls)."""

from __future__ import annotations

import ctypes as ct
import unittest
from unittest.mock import Mock, patch

from seabraid import Runtime, SeabraidError, SourceModality

TICKET = (1 << 48) + 5


class SessionSubmitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.library = Mock(spec=[
            "seabraid_open", "seabraid_open_reference", "seabraid_abi_epoch", "seabraid_prepare", "seabraid_submit",
            "seabraid_session_submit", "seabraid_collect", "seabraid_close",
        ])
        self.library.seabraid_abi_epoch.return_value = 1
        self.library.seabraid_open_reference.side_effect = self._open
        self.library.seabraid_prepare.return_value = 0
        self.library.seabraid_collect.side_effect = self._collect
        self.library.seabraid_close.return_value = 0
        loader = patch("seabraid._abi.ct.CDLL", return_value=self.library)
        loader.start()
        self.addCleanup(loader.stop)

    def _open(self, out: object) -> int:
        ct.cast(out, ct.POINTER(ct.c_void_p))[0] = 41
        return 0

    def _collect(self, _session: object, _ticket: int, _timeout: int, output: ct.Array, count: int) -> int:
        for n in range(count):
            output[n] = 1.5
        return 0

    def test_a_v2_batch_carries_operation_origins_format_and_data(self) -> None:
        seen = {}

        def record(session: ct.c_void_p, pointer: object, out: object) -> int:
            batch = ct.cast(pointer, self.library.seabraid_session_submit.argtypes[1]).contents
            seen.update(
                version=batch.abi_version, size=batch.struct_size, operation=batch.operation_id,
                origins=batch.source_modalities, epoch=batch.placement_epoch, tokens=batch.tokens,
                format=batch.activation_format, counts=(batch.activation_count, batch.expert_id_count,
                                                        batch.weight_count),
                activations=ct.cast(batch.activations, ct.POINTER(ct.c_float))[:2], ids=batch.expert_ids[:1],
            )
            ct.cast(out, ct.POINTER(ct.c_uint64))[0] = TICKET
            return 0

        self.library.seabraid_session_submit.side_effect = record
        with Runtime(__file__, reference=True) as runtime:
            prepared = runtime.prepare(hidden=2, experts=3, max_tokens=2, top_k=1)
            self.assertEqual(prepared.weave([2.0, 4.0], [1], [0.5], source_modalities=SourceModality.IMAGE),
                             [1.5, 1.5])
        self.assertEqual(
            seen,
            dict(version=2, size=80, operation=0, origins=int(SourceModality.IMAGE), epoch=0, tokens=1, format=1,
                 counts=(2, 1, 1), activations=[2.0, 4.0], ids=[1]),
        )
        self.library.seabraid_submit.assert_not_called()

    def test_a_native_failure_is_named_after_the_v2_call(self) -> None:
        self.library.seabraid_session_submit.return_value = 1
        with Runtime(__file__, reference=True) as runtime:
            prepared = runtime.prepare(hidden=1, experts=1, max_tokens=1, top_k=1)
            with self.assertRaisesRegex(SeabraidError, "seabraid_session_submit: invalid argument"):
                prepared.weave([1.0], [0], [1.0])

    def test_a_python_precheck_names_the_call_the_session_would_make_with_detail_and_one_log_record(self) -> None:
        """R7 LOW 8: the precheck said seabraid_submit on a v3 session, with no detail and no log record."""
        with Runtime(__file__, reference=True) as runtime:
            prepared = runtime.prepare(hidden=2, experts=3, max_tokens=2, top_k=1)
            cases = (
                ([1.0] * 6, [0, 1, 2], [0.5] * 3, "tokens 3 outside 1..=2"),
                ([1.0] * 4, [0], [0.5, 0.5], "expert_id_count 1 != tokens 2 * top_k 1"),
                ([1.0] * 4, [0, 1], [0.5], "weight_count 1 != tokens 2 * top_k 1"),
            )
            for acts, ids, wts, detail in cases:
                with self.subTest(detail), self.assertLogs("seabraid", "WARNING") as logs:
                    with self.assertRaises(SeabraidError) as raised:
                        prepared.weave(acts, ids, wts)
                    error = raised.exception
                    self.assertEqual((error.operation, error.status, error.detail),
                                     ("seabraid_session_submit", 1, detail))
                    self.assertEqual(logs.output, [f"WARNING:seabraid:seabraid_session_submit: {detail}"])
        self.library.seabraid_session_submit.assert_not_called()

    def test_a_native_refusal_does_not_leave_the_callers_arrays_export_locked(self) -> None:
        """R7 LOW 10 through NativeSession.submit: its frame holds the packed ctypes arrays."""
        import gc
        from array import array
        gc.disable()
        self.addCleanup(gc.enable)
        self.library.seabraid_session_submit.return_value = 1
        acts, ids, wts = array("f", [1.0, 2.0]), array("i", [0, 1]), array("f", [0.5, 0.5])
        saved = None
        with Runtime(__file__, reference=True) as runtime:
            prepared = runtime.prepare(hidden=2, experts=3, max_tokens=2, top_k=2)
            try:
                prepared.weave(acts, ids, wts)
            except SeabraidError as error:
                saved = error
            self.library.reset_mock()  # the mock records its call arguments, which would hold the arrays itself
            for buffer in (acts, ids, wts):
                buffer.append(buffer[0])
        self.assertIsNotNone(saved)


if __name__ == "__main__":
    unittest.main()
