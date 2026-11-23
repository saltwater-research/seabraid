# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""COMPACT_ROWS on the Python package: the row rule, bit-identical output, the provider's view, the row counters.

Checker for seabraid._rows and for ReferenceSession's compaction; the native cases (SEABRAID_LIBRARY) run the same
scripts on the engine and compare both outputs and counters with the reference session.
"""

from __future__ import annotations

import struct
import unittest

from seabraid import Runtime, SeabraidError, _rows
from seabraid._model import BatchInfo, Shape
from seabraid.constants import FORMAT_ROUTED_F16, INVALID

from ._v2_support import LIBRARY, reference_runtime

SHAPE = dict(hidden=2, experts=4, max_tokens=5, top_k=2)
# Five rows: 0 and 3 are skipped entirely, row 2 keeps one real pick beside a skip, rows 1 and 4 are full.
ACTS = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
IDS = [-1, -1, 0, 3, -1, 2, -1, -1, 1, 2]
WEIGHTS = [0.0, 0.0, 0.5, 0.25, 0.0, 1.0, 0.0, 0.0, 0.75, 0.125]
HALF_ROWS = struct.pack("<10e", *ACTS)
SKIPPED_ALL = ([1.0, 2.0, 3.0, 4.0], [-1, -1, -1, -1], [0.0, 0.0, 0.0, 0.0])


def shipping_provider(calls: list):
    """out = activations + 100 * the row's first real pick; records how many rows each call was given."""

    def serve(operation: int, config: Shape, acts: object, ids: object, weights: object, out: object) -> None:
        picks = list(ids)  # a provider view is read by index or copied: view[a:b] is refused (it would outlive the call)
        calls.append((len(acts) // config.hidden, picks))
        for row in range(len(acts) // config.hidden):
            pick = next((i for i in picks[row * config.top_k:(row + 1) * config.top_k] if i >= 0), 0)
            for col in range(config.hidden):
                out[row * config.hidden + col] = acts[row * config.hidden + col] + 100.0 * pick

    return serve


def run_pair(make: object) -> tuple[object, object]:
    """(uncompacted, compacted) results of one script; `make(compact_rows)` returns a Runtime."""
    results = []
    for compact in (False, True):
        with make(compact) as runtime:
            prepared = runtime.prepare(**SHAPE)
            results.append((prepared.weave(ACTS, IDS, WEIGHTS), runtime.status()))
    return tuple(results)


class RowRuleTests(unittest.TestCase):
    def test_a_row_is_kept_when_any_pick_is_real(self) -> None:
        self.assertEqual(_rows.kept_rows(IDS, 2), [1, 2, 4])
        self.assertEqual(_rows.kept_rows([-1, -1, -1, -1], 2), [])
        self.assertEqual(_rows.kept_rows([0, 1, 2, 3], 2), [0, 1])

    def test_gather_then_scatter_round_trips_into_zeros(self) -> None:
        rows = _rows.kept_rows(IDS, 2)
        self.assertEqual(_rows.gather(ACTS, rows, 2), [3.0, 4.0, 5.0, 6.0, 9.0, 10.0])
        self.assertEqual(_rows.scatter([7.0, 8.0, 9.0, 1.0, 2.0, 3.0], rows, 2, 5),
                         [0.0, 0.0, 7.0, 8.0, 9.0, 1.0, 0.0, 0.0, 2.0, 3.0])


class ReferenceCompactionTests(unittest.TestCase):
    def test_builtin_transform_output_is_bit_identical_with_and_without_compaction(self) -> None:
        plain, compact = run_pair(lambda c: reference_runtime(reference=True, partial_routes=True, compact_rows=c))
        self.assertEqual(plain[0], compact[0])
        self.assertEqual([struct.pack("<f", v) for v in plain[0]], [struct.pack("<f", v) for v in compact[0]])
        self.assertEqual(plain[0][:2], [0.0, 0.0])  # a skipped row is zeros either way
        self.assertEqual(plain[0][6:8], [0.0, 0.0])

    def test_python_provider_output_is_bit_identical_and_it_sees_only_the_kept_rows(self) -> None:
        calls: dict[bool, list] = {False: [], True: []}
        results = []
        for compact in (False, True):
            with reference_runtime(provider=shipping_provider(calls[compact]), partial_routes=True,
                                   compact_rows=compact) as runtime:
                prepared = runtime.prepare(**SHAPE)
                results.append(prepared.weave(ACTS, IDS, WEIGHTS))
        # The skipped rows differ: without compaction the callable writes them, with it they stay zero. Rows it was
        # given must agree to the bit, and the compacted call is shorter by exactly the skipped rows.
        kept = [1, 2, 4]
        for row in kept:
            self.assertEqual(results[0][row * 2:row * 2 + 2], results[1][row * 2:row * 2 + 2])
        self.assertEqual(results[1][0:2] + results[1][6:8], [0.0] * 4)
        self.assertEqual([c[0] for c in calls[False]], [5])
        self.assertEqual([c[0] for c in calls[True]], [3])
        self.assertEqual(calls[True][0][1], [0, 3, -1, 2, 1, 2])  # the kept rows' ids, in original order

    def test_an_all_skipped_batch_never_calls_the_provider_and_returns_zeros(self) -> None:
        calls: list = []
        with reference_runtime(provider=shipping_provider(calls), partial_routes=True, compact_rows=True) as runtime:
            prepared = runtime.prepare(**SHAPE)
            self.assertEqual(prepared.weave(*SKIPPED_ALL), [0.0] * 4)
            status = runtime.status()
        self.assertEqual(calls, [])
        self.assertEqual((status.submitted, status.collected, status.rows_submitted, status.rows_shipped),
                         (1, 1, 2, 0))

    def test_counters_show_what_compaction_saved(self) -> None:
        plain, compact = run_pair(lambda c: reference_runtime(reference=True, partial_routes=True, compact_rows=c))
        self.assertEqual((plain[1].rows_submitted, plain[1].rows_shipped), (5, 5))
        self.assertEqual((compact[1].rows_submitted, compact[1].rows_shipped), (5, 3))

    def test_a_rejected_batch_moves_no_counter(self) -> None:
        with reference_runtime(reference=True, partial_routes=True, compact_rows=True) as runtime:
            prepared = runtime.prepare(**SHAPE)
            with self.assertRaises(SeabraidError) as raised:
                prepared.weave([1.0, 2.0], [0, 9], [0.5, 0.5])
            self.assertEqual(raised.exception.status, INVALID)
            status = runtime.status()
        self.assertEqual((status.submitted, status.rows_submitted, status.rows_shipped), (0, 0, 0))

    def test_compaction_composes_with_f16_bytes(self) -> None:
        seen: list = []

        def serve(operation: int, config: Shape, acts: object, ids: object, weights: object, out: object,
                  info: BatchInfo) -> None:
            seen.append((bytes(acts), info.activation_format))
            for i, (value,) in enumerate(struct.iter_unpack("<e", bytes(acts))):
                out[i] = value

        with reference_runtime(provider=serve, partial_routes=True, compact_rows=True,
                               accepts=(FORMAT_ROUTED_F16,)) as runtime:
            result = runtime.prepare(**SHAPE).weave(HALF_ROWS, IDS, WEIGHTS, activation_format=FORMAT_ROUTED_F16)
        self.assertEqual(result, [0.0, 0.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0, 9.0, 10.0])
        self.assertEqual(seen, [(struct.pack("<6e", 3.0, 4.0, 5.0, 6.0, 9.0, 10.0), FORMAT_ROUTED_F16)])


@unittest.skipUnless(LIBRARY, "set SEABRAID_LIBRARY for native parity")
class NativeCompactionTests(unittest.TestCase):
    def test_builtin_transform_native_equals_reference_with_and_without_compaction(self) -> None:
        for compact in (False, True):
            with self.subTest(compact=compact):
                with reference_runtime(reference=True, partial_routes=True, compact_rows=compact) as ref, \
                        Runtime(LIBRARY, reference=True, partial_routes=True, compact_rows=compact) as native:
                    expected = ref.prepare(**SHAPE).weave(ACTS, IDS, WEIGHTS)
                    found = native.prepare(**SHAPE).weave(ACTS, IDS, WEIGHTS)
                    self.assertEqual([struct.pack("<f", v) for v in found], [struct.pack("<f", v) for v in expected])
                    self.assertEqual((ref.status().rows_submitted, ref.status().rows_shipped),
                                     (native.status().rows_submitted, native.status().rows_shipped))
                    self.assertEqual(native.status().rows_shipped, 3 if compact else 5)

    def test_native_compaction_is_bit_identical_to_uncompacted_for_the_builtin_transform(self) -> None:
        plain, compact = run_pair(lambda c: Runtime(LIBRARY, reference=True, partial_routes=True, compact_rows=c))
        self.assertEqual([struct.pack("<f", v) for v in plain[0]], [struct.pack("<f", v) for v in compact[0]])

    def test_python_provider_sees_only_kept_rows_on_the_worker_thread_and_matches_reference(self) -> None:
        native_calls: list = []
        ref_calls: list = []
        with Runtime(LIBRARY, provider=shipping_provider(native_calls), partial_routes=True, compact_rows=True) \
                as native, reference_runtime(provider=shipping_provider(ref_calls), partial_routes=True,
                                             compact_rows=True) as ref:
            found = native.prepare(**SHAPE).weave(ACTS, IDS, WEIGHTS)
            expected = ref.prepare(**SHAPE).weave(ACTS, IDS, WEIGHTS)
            self.assertEqual(found, expected)
            self.assertEqual(native_calls, ref_calls)
            self.assertEqual(native_calls[0][0], 3)
            status = native.status()
            self.assertEqual((status.rows_submitted, status.rows_shipped), (5, 3))

    def test_all_skipped_native_batch_returns_zeros_without_calling_the_provider(self) -> None:
        calls: list = []
        with Runtime(LIBRARY, provider=shipping_provider(calls), partial_routes=True, compact_rows=True) as native:
            self.assertEqual(native.prepare(**SHAPE).weave(*SKIPPED_ALL), [0.0] * 4)
            status = native.status()
        self.assertEqual(calls, [])
        self.assertEqual((status.rows_submitted, status.rows_shipped), (2, 0))

    def test_compact_rows_with_f16_bytes_matches_reference(self) -> None:
        def serve_into(seen: list):
            def serve(operation: int, config: Shape, acts: object, ids: object, weights: object, out: object,
                      info: BatchInfo) -> None:
                seen.append(bytes(acts))
                for i, (value,) in enumerate(struct.iter_unpack("<e", bytes(acts))):
                    out[i] = value
            return serve

        native_seen: list = []
        ref_seen: list = []
        kwargs = dict(partial_routes=True, compact_rows=True, accepts=(FORMAT_ROUTED_F16,))
        with Runtime(LIBRARY, provider=serve_into(native_seen), **kwargs) as native, \
                reference_runtime(provider=serve_into(ref_seen), **kwargs) as ref:
            found = native.prepare(**SHAPE).weave(HALF_ROWS, IDS, WEIGHTS, activation_format=FORMAT_ROUTED_F16)
            expected = ref.prepare(**SHAPE).weave(HALF_ROWS, IDS, WEIGHTS, activation_format=FORMAT_ROUTED_F16)
        self.assertEqual(found, expected)
        self.assertEqual(native_seen, ref_seen)


if __name__ == "__main__":
    unittest.main()
