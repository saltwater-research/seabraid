# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Check Python marshalling and recovery without requiring a native linker."""

from __future__ import annotations

import ctypes as ct
import unittest
from array import array
from collections.abc import Iterator
from itertools import product
from unittest.mock import Mock, patch

from seabraid import SeabraidError, Runtime, SourceModality


class RuntimeMockTests(unittest.TestCase):
    def setUp(self) -> None:
        self.library = Mock(spec=[
            "seabraid_open", "seabraid_open_reference", "seabraid_abi_epoch", "seabraid_close_timeout", "seabraid_prepare",
            "seabraid_submit", "seabraid_collect", "seabraid_close", "seabraid_query_capabilities", "seabraid_submit_routed", "seabraid_session_status",
        ])
        self.library.seabraid_abi_epoch.return_value = 1
        self.library.seabraid_open_reference.side_effect = self._open
        self.library.seabraid_prepare.return_value = 0
        self.library.seabraid_submit.side_effect = self._submit
        self.library.seabraid_collect.side_effect = self._collect
        self.library.seabraid_close.return_value = 0
        # The test file supplies an existing path; only CDLL loading is replaced.
        loader = patch("seabraid._abi.ct.CDLL", return_value=self.library)
        loader.start()
        self.addCleanup(loader.stop)

    def _open(self, out: object) -> int:
        ct.cast(out, ct.POINTER(ct.c_void_p))[0] = 37
        return 0

    def _submit(self, session: ct.c_void_p, _batch: object, out: object) -> int:
        self.assertEqual(session.value, 37)
        ct.cast(out, ct.POINTER(ct.c_uint64))[0] = (1 << 48) + 91
        return 0

    def _collect(
        self, session: ct.c_void_p, ticket: int, timeout: int, output: ct.Array, count: int,
    ) -> int:
        self.assertEqual(session.value, 37)
        self.assertEqual(ticket, (1 << 48) + 91)
        if timeout == 0:
            return 3
        self.assertEqual(count, len(output))
        for n in range(count):
            output[n] = n + 0.25
        return 0

    def test_marshals_config_and_batch_and_returns_output_from_native_buffer(self) -> None:
        def record_config(session: ct.c_void_p, pointer: object) -> int:
            self.assertEqual(session.value, 37)
            words = ct.cast(pointer, ct.POINTER(ct.c_uint32 * 5)).contents
            self.assertEqual(list(words), [1, 2, 6, 4, 3])
            return 0

        def record_batch(session: ct.c_void_p, pointer: object, out: object) -> int:
            # Use the configured ctypes function argument to read the actual call storage.
            batch = ct.cast(pointer, self.library.seabraid_submit.argtypes[1]).contents
            self.assertEqual((batch.abi_version, batch.tokens), (1, 2))
            self.assertEqual((batch.activation_count, batch.expert_id_count, batch.weight_count), (4, 6, 6))
            self.assertEqual(batch.activations[:4], [2.0, 5.5, -1.0, 4.0])
            self.assertEqual(batch.expert_ids[:6], [1, 5, 2, 4, 0, 3])
            self.assertEqual(batch.weights[:6], [0.75, 0.125, 0.125, 0.5, 0.25, 0.25])
            return self._submit(session, pointer, out)

        self.library.seabraid_prepare.side_effect = record_config
        self.library.seabraid_submit.side_effect = record_batch
        with Runtime(__file__, reference=True) as runtime:
            prepared = runtime.prepare(hidden=2, experts=6, max_tokens=4, top_k=3)
            result = prepared.weave(
                [2.0, 5.5, -1.0, 4.0], [1, 5, 2, 4, 0, 3], [0.75, 0.125, 0.125, 0.5, 0.25, 0.25],
            )
        self.assertEqual(result, [0.25, 1.25, 2.25, 3.25])

    def test_timeout_retries_same_ticket_without_resubmitting(self) -> None:
        with Runtime(__file__, reference=True) as runtime:
            prepared = runtime.prepare(hidden=1, experts=1, max_tokens=1, top_k=1)
            with self.assertRaisesRegex(SeabraidError, "seabraid_collect: timeout"):
                prepared.weave([7.0], [0], [1.0], timeout_ms=0)
            with self.assertRaisesRegex(RuntimeError, "batch is pending"):
                prepared.weave([5.0], [0], [1.0])
            self.assertEqual(prepared.collect(timeout_ms=1), [0.25])
        self.library.seabraid_submit.assert_called_once()
        self.assertEqual(self.library.seabraid_collect.call_count, 2)

    def test_invalid_prepared_shape_is_rejected_before_packing(self) -> None:
        class LengthOnly:
            def __init__(self, count: int) -> None:
                self.count = count

            def __len__(self) -> int:
                return self.count

            def __iter__(self) -> Iterator[float]:
                raise AssertionError("Invalid batch must be rejected before iteration")

        cases = [(6, 6, 6), (4, 3, 4), (4, 4, 5)]
        operations = [(SourceModality(0), "seabraid_submit"), (SourceModality.IMAGE, "seabraid_submit_routed")]
        with Runtime(__file__, reference=True) as runtime:
            prepared = runtime.prepare(hidden=2, experts=2, max_tokens=2, top_k=2)
            for (origins, operation), (rows, ids, weights) in product(operations, cases):
                with self.subTest(operation=operation, counts=(rows, ids, weights)):
                    with self.assertRaises(SeabraidError) as raised:
                        prepared.weave(LengthOnly(rows), LengthOnly(ids), LengthOnly(weights),
                                       source_modalities=origins)
                    self.assertEqual((raised.exception.operation, raised.exception.status), (operation, 1))
            self.library.seabraid_submit.assert_not_called()
            self.library.seabraid_submit_routed.assert_not_called()
            activations = memoryview(array("f", [2.0, 3.0, 4.0, 5.0])).toreadonly()
            result = prepared.weave(activations, (0, 1, 0, 1), (1.0, 0.0, 0.0, 1.0))
            self.assertEqual(result, [0.25, 1.25, 2.25, 3.25])

    def test_close_error_still_invalidates_consumed_handle(self) -> None:
        self.library.seabraid_close.return_value = 4
        runtime = Runtime(__file__, reference=True)
        with self.assertRaisesRegex(SeabraidError, "seabraid_close: provider failure"):
            runtime.close()
        runtime.close()
        self.library.seabraid_close.assert_called_once()
        with self.assertRaisesRegex(RuntimeError, "runtime is closed"):
            runtime.prepare(hidden=1, experts=1, max_tokens=1, top_k=1)

    def test_integer_bounds_do_not_let_ctypes_wrap_inputs(self) -> None:
        with Runtime(__file__, reference=True) as runtime:
            with self.assertRaisesRegex(ValueError, "hidden must be between"):
                runtime.prepare(hidden=(1 << 32) + 1, experts=1, max_tokens=1, top_k=1)
            prepared = runtime.prepare(hidden=1, experts=1, max_tokens=1, top_k=1)
            with self.assertRaisesRegex(ValueError, "expert id must be between"):
                prepared.weave([2.0], [1 << 32], [1.0])
            with self.assertRaisesRegex(ValueError, "timeout_ms must be between"):
                prepared.weave([2.0], [0], [1.0], timeout_ms=-1)
            self.library.seabraid_submit.assert_not_called()
            self.assertEqual(prepared.weave([2.0], [0], [1.0]), [0.25])


    def test_capabilities_and_mixed_origins_use_the_native_abi(self) -> None:
        def describe(out: object) -> int:
            words = ct.cast(out, ct.POINTER(ct.c_uint32 * 6)).contents
            self.assertEqual(list(words[:2]), [1, 24])  # abi_version and struct_size are set by the caller
            words[:] = [1, 24, 1, 1, 2, 31]
            return 0

        def submit(session: ct.c_void_p, batch: object, origins: int, out: object) -> int:
            self.assertEqual(origins, 12)
            return self._submit(session, batch, out)

        def snapshot(session: ct.c_void_p, out: object) -> int:
            self.assertEqual(session.value, 37)
            info = ct.cast(out, self.library.seabraid_session_status.argtypes[1]).contents
            self.assertEqual((info.abi_version, info.struct_size), (1, 64))
            info.submitted = 1
            info.collected = 1
            info.last_source_modalities = 12
            return 0

        self.library.seabraid_session_status.side_effect = snapshot
        self.library.seabraid_query_capabilities.side_effect = describe
        self.library.seabraid_submit_routed.side_effect = submit
        with Runtime(__file__, reference=True) as runtime:
            supported = runtime.capabilities().source_modalities
            origins = SourceModality.AUDIO | SourceModality.VIDEO
            self.assertEqual(supported & origins, origins)
            prepared = runtime.prepare(hidden=1, experts=1, max_tokens=1, top_k=1)
            self.assertEqual(prepared.weave([3.0], [0], [1.0], source_modalities=origins), [0.25])
            observed = runtime.status()
            self.assertEqual(observed.last_source_modalities, origins)
            self.assertEqual(observed.submitted, 1)
            self.assertFalse(observed.outstanding)
        self.library.seabraid_submit_routed.assert_called_once()
        self.library.seabraid_submit.assert_not_called()

    def test_untagged_work_remains_compatible_with_older_libraries(self) -> None:
        del self.library.seabraid_query_capabilities
        del self.library.seabraid_submit_routed
        with Runtime(__file__, reference=True) as runtime:
            with self.assertRaisesRegex(RuntimeError, "capability discovery"):
                runtime.capabilities()
            prepared = runtime.prepare(hidden=1, experts=1, max_tokens=1, top_k=1)
            with self.assertRaisesRegex(RuntimeError, "does not expose seabraid_submit_routed"):
                prepared.weave([3.0], [0], [1.0], source_modalities=SourceModality.IMAGE)
            self.assertEqual(prepared.weave([3.0], [0], [1.0]), [0.25])
