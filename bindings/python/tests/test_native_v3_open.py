# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""The native session's open choice (seabraid_session_open, provider v3) and its v3 trampoline, against mocked libraries.

Checker for seabraid._native and seabraid._provider: no linker needed. A v1-only library must keep working and must
refuse v2-only options loudly.
"""

from __future__ import annotations

import ctypes as ct
import unittest
from unittest.mock import Mock, patch

from seabraid import Runtime, SeabraidError, _abi
from seabraid._model import BatchInfo, Shape
from seabraid.constants import (
    BATCH_V2_VERSION, FORMAT_ROUTED_BF16, FORMAT_ROUTED_F16, FORMAT_ROUTED_F32, INVALID, OK, PROVIDER_ACCEPTS_BF16,
    PROVIDER_ACCEPTS_F16, PROVIDER_COMPACT_ROWS, PROVIDER_PARTIAL_ROUTES, PROVIDER_REFERENCE, PROVIDER_V3_VERSION,
)

from ._v2_support import BF16, HALVES, SHAPE, V1_SPEC, V3_SPEC, echo, half_provider


class MockedV3OpenTests(unittest.TestCase):
    """The native session against a library that exports seabraid_session_open, with no linker."""

    def setUp(self) -> None:
        self.library = Mock(spec=V3_SPEC)
        self.library.seabraid_abi_epoch.return_value = 1
        self.library.seabraid_session_open.side_effect = self._open
        self.library.seabraid_session_submit.side_effect = self._submit
        self.library.seabraid_session_status.side_effect = self._status
        self.library.seabraid_collect.side_effect = self._collect
        self.library.seabraid_close.return_value = 0
        self.opened: list[dict] = []
        self.batches: list[dict] = []
        self.execute = None
        self.keep: list = []
        loader = patch("seabraid._abi.ct.CDLL", return_value=self.library)
        loader.start()
        self.addCleanup(loader.stop)

    def _open(self, pointer: object, out: object) -> int:
        provider = ct.cast(pointer, ct.POINTER(_abi._ProviderV3)).contents
        self.opened.append(dict(
            version=provider.abi_version, size=provider.struct_size, operations=provider.operation_count,
            flags=provider.flags, config=tuple(provider.config.contents.__getattribute__(n) for n in
                                               ("abi_version", "hidden", "experts", "max_tokens", "top_k")),
            context=provider.context, has_execute=bool(provider.execute),
        ))
        self.execute = provider.execute
        self.keep.append(provider)
        ct.cast(out, ct.POINTER(ct.c_void_p))[0] = 41
        return 0

    def _submit(self, session: object, pointer: object, out: object) -> int:
        batch = ct.cast(pointer, ct.POINTER(_abi._BatchV2)).contents
        self.batches.append(dict(
            operation=batch.operation_id, epoch=batch.placement_epoch, format=batch.activation_format,
            tokens=batch.tokens, count=batch.activation_count, size=batch.struct_size, version=batch.abi_version,
            raw=ct.string_at(batch.activations, batch.activation_count * (4 if batch.activation_format == 1 else 2)),
        ))
        ct.cast(out, ct.POINTER(ct.c_uint64))[0] = 7
        return 0

    def _status(self, session: object, pointer: object) -> int:
        info = ct.cast(pointer, ct.POINTER(_abi._SessionStatusV1)).contents
        info.submitted, info.rows_submitted, info.rows_shipped = 3, 12, 5
        return 0

    def _collect(self, session: object, ticket: int, timeout: int, output: ct.Array, count: int) -> int:
        for i in range(count):
            output[i] = 0.5
        return 0

    def test_reference_open_waits_for_prepare_and_sets_the_reference_flag(self) -> None:
        with Runtime(__file__, reference=True) as runtime:
            self.library.seabraid_session_open.assert_not_called()  # the config is not known yet
            self.assertEqual(runtime.status().submitted, 0)  # no session, no native call, no error
            runtime.prepare(**SHAPE)
        (opened,) = self.opened
        self.assertEqual(
            (opened["version"], opened["size"], opened["operations"], opened["flags"], opened["config"]),
            (PROVIDER_V3_VERSION, 40, 1, PROVIDER_REFERENCE, (1, 2, 3, 4, 2)),
        )
        self.assertFalse(opened["has_execute"])
        self.library.seabraid_open_reference.assert_not_called()
        self.library.seabraid_prepare.assert_not_called()

    def test_reference_open_can_declare_partial_routes_and_compaction(self) -> None:
        with Runtime(__file__, reference=True, partial_routes=True, compact_rows=True) as runtime:
            runtime.prepare(**SHAPE)
        self.assertEqual(self.opened[0]["flags"], PROVIDER_REFERENCE | PROVIDER_PARTIAL_ROUTES | PROVIDER_COMPACT_ROWS)

    def test_provider_flags_operations_and_callback_ride_in_the_descriptor(self) -> None:
        with Runtime(__file__, provider=echo, operations=3, partial_routes=True, compact_rows=True,
                     accepts=(FORMAT_ROUTED_F16, FORMAT_ROUTED_BF16)) as runtime:
            runtime.prepare(**SHAPE)
        (opened,) = self.opened
        self.assertEqual(opened["operations"], 3)
        self.assertEqual(opened["flags"], PROVIDER_PARTIAL_ROUTES | PROVIDER_COMPACT_ROWS | PROVIDER_ACCEPTS_F16
                         | PROVIDER_ACCEPTS_BF16)
        self.assertTrue(opened["has_execute"])
        self.library.seabraid_open_multi.assert_not_called()

    def test_prepare_twice_is_a_state_error_and_a_failed_open_names_the_v3_call(self) -> None:
        with Runtime(__file__, reference=True) as runtime:
            runtime.prepare(**SHAPE)
            with self.assertRaises(RuntimeError):
                runtime.prepare(**SHAPE)
        self.library.seabraid_session_open.side_effect = lambda *_: 1
        with Runtime(__file__, reference=True) as runtime:
            with self.assertRaisesRegex(SeabraidError, "seabraid_session_open: invalid argument"):
                runtime.prepare(**SHAPE)

    def test_epoch_format_and_raw_halves_reach_the_c_batch(self) -> None:
        with Runtime(__file__, provider=half_provider([]), accepts=(FORMAT_ROUTED_F16,)) as runtime:
            prepared = runtime.prepare(**SHAPE)
            prepared.weave(HALVES, [0, 1, 2, 0], [0.5, 0.5, 0.25, 0.75], placement_epoch=(1 << 40) + 3,
                           activation_format=FORMAT_ROUTED_F16)
            prepared.weave([1.0, 2.0], [0, 1], [0.5, 0.5])
        first, second = self.batches
        self.assertEqual((first["version"], first["size"], first["epoch"], first["format"], first["tokens"]),
                         (BATCH_V2_VERSION, 80, (1 << 40) + 3, FORMAT_ROUTED_F16, 2))
        self.assertEqual((first["count"], first["raw"]), (4, HALVES))  # elements, not bytes
        self.assertEqual((second["epoch"], second["format"]), (0, FORMAT_ROUTED_F32))

    def test_row_counters_come_from_the_native_status(self) -> None:
        with Runtime(__file__, reference=True) as runtime:
            runtime.prepare(**SHAPE)
            status = runtime.status()
        self.assertEqual((status.submitted, status.rows_submitted, status.rows_shipped), (3, 12, 5))

    def test_the_v3_trampoline_hands_a_python_provider_exact_bytes_epoch_and_the_batch_operation(self) -> None:
        seen: list = []

        def serve(operation: int, config: Shape, acts: object, ids: object, weights: object, out: object,
                  info: BatchInfo) -> None:
            seen.append((operation, config, bytes(acts), list(ids), list(weights), len(out), info))
            out[0] = 9.0

        with Runtime(__file__, provider=serve, operations=2, accepts=(FORMAT_ROUTED_BF16,)) as runtime:
            runtime.prepare(**SHAPE)
            config = _abi._ConfigV1(1, 2, 3, 4, 2)
            payload = (ct.c_uint16 * 4).from_buffer_copy(BF16)
            ids = (ct.c_int32 * 4)(0, 1, 2, 0)
            weights = (ct.c_float * 4)(0.5, 0.5, 0.25, 0.75)
            batch = _abi._BatchV2(2, 80, 1, 0, 11, 2, FORMAT_ROUTED_BF16, ct.cast(payload, ct.c_void_p), 4, ids, 4,
                                  weights, 4)
            out = (ct.c_float * 4)()
            status = self.execute(None, ct.byref(config), ct.byref(batch), out, 4)
            self.assertEqual(status, OK)
            self.assertEqual(out[0], 9.0)
            batch.operation_id = 2  # past operation_count: refused before the callable runs
            self.assertEqual(self.execute(None, ct.byref(config), ct.byref(batch), out, 4), INVALID)
            batch.operation_id, batch.activation_format = 1, 77  # a format this binding does not know
            self.assertEqual(self.execute(None, ct.byref(config), ct.byref(batch), out, 4), INVALID)
        (record,) = seen
        self.assertEqual(record, (1, Shape(2, 3, 4, 2), BF16, [0, 1, 2, 0], [0.5, 0.5, 0.25, 0.75], 4,
                                  BatchInfo(FORMAT_ROUTED_BF16, 11)))

    def test_a_provider_without_halves_keeps_the_six_argument_call_on_the_v3_trampoline(self) -> None:
        with Runtime(__file__, provider=echo) as runtime:
            runtime.prepare(**SHAPE)
            config = _abi._ConfigV1(1, 2, 3, 4, 2)
            acts = (ct.c_float * 4)(1.0, 2.0, 3.0, 4.0)
            ids = (ct.c_int32 * 4)(0, 1, 2, 0)
            weights = (ct.c_float * 4)(0.5, 0.5, 0.25, 0.75)
            batch = _abi._BatchV2(2, 80, 0, 0, 0, 2, FORMAT_ROUTED_F32, ct.cast(acts, ct.c_void_p), 4, ids, 4,
                                  weights, 4)
            out = (ct.c_float * 4)()
            self.assertEqual(self.execute(None, ct.byref(config), ct.byref(batch), out, 4), OK)
            self.assertEqual(list(out), [2.0, 3.0, 4.0, 5.0])


class OlderLibraryTests(unittest.TestCase):
    """A library with the v1 entry points only keeps working and says so loudly for v2-only features."""

    def setUp(self) -> None:
        self.library = Mock(spec=V1_SPEC)
        self.library.seabraid_abi_epoch.return_value = 1
        loader = patch("seabraid._abi.ct.CDLL", return_value=self.library)
        loader.start()
        self.addCleanup(loader.stop)

    def test_v2_only_options_fail_loudly_instead_of_being_dropped(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "seabraid_session_open"):
            Runtime(__file__, provider=echo, partial_routes=True, compact_rows=True)
        with self.assertRaisesRegex(RuntimeError, "seabraid_session_open"):
            Runtime(__file__, provider=half_provider([]), accepts=(FORMAT_ROUTED_F16,))
        with self.assertRaisesRegex(RuntimeError, "seabraid_session_open"):
            Runtime(__file__, reference=True, partial_routes=True)

    def test_epoch_and_formats_need_the_v2_submit(self) -> None:
        self.library.seabraid_open_reference.side_effect = lambda out: (
            ct.cast(out, ct.POINTER(ct.c_void_p)).__setitem__(0, 41), 0)[1]
        self.library.seabraid_prepare.return_value = 0
        self.library.seabraid_close.return_value = 0
        with Runtime(__file__, reference=True) as runtime:
            prepared = runtime.prepare(**SHAPE)
            with self.assertRaisesRegex(RuntimeError, "seabraid_session_submit"):
                prepared.weave([1.0, 2.0], [0, 1], [0.5, 0.5], placement_epoch=2)
        self.library.seabraid_submit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
