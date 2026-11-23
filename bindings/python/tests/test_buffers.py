# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Buffer inputs and the out= path: validation messages, sequence/buffer parity, zero-copy packing.

Checker for seabraid._buffers and the buffer handling in seabraid.runtime. numpy cases are skipped when numpy is absent.
"""

from __future__ import annotations

import ctypes as ct
import gc
import struct
import unittest
from array import array

from seabraid import SeabraidError
from seabraid._buffers import FLOAT32, INT32, Operand
from tests.test_reference import reference_runtime

try:
    import numpy
except ImportError:  # numpy is optional for the package and for these tests
    numpy = None

ACTS = [1.0, 2.0, 3.0, -4.0, 0.5, 8.0]
IDS = [0, 2, 1, 0]  # no expert id repeats within a row (the runtime refuses that)
WTS = [0.25, 0.75, 0.5, 0.5]
SHAPE = dict(hidden=3, experts=3, max_tokens=2, top_k=2)


def weave(acts: object, ids: object, wts: object, **kwargs: object) -> object:
    with reference_runtime() as runtime:
        return runtime.prepare(**SHAPE).weave(acts, ids, wts, **kwargs)


class ParityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.expected = weave(ACTS, IDS, WTS)

    def test_array_module_buffers_match_sequences(self) -> None:
        self.assertEqual(weave(array("f", ACTS), array("i", IDS), array("f", WTS)), self.expected)

    def test_memoryviews_read_only_and_writable_match(self) -> None:
        acts = memoryview(array("f", ACTS))
        ids = memoryview(array("i", IDS)).toreadonly()
        self.assertEqual(weave(acts, ids, memoryview(array("f", WTS))), self.expected)

    def test_bytes_like_float_and_int_data_match(self) -> None:
        packed = (struct.pack("6f", *ACTS), struct.pack("4i", *IDS), bytearray(struct.pack("4f", *WTS)))
        self.assertEqual(weave(*packed), self.expected)

    @unittest.skipIf(numpy is None, "numpy not installed")
    def test_numpy_arrays_including_two_dimensional_rows(self) -> None:
        acts = numpy.array(ACTS, dtype=numpy.float32).reshape(2, 3)
        ids = numpy.array(IDS, dtype=numpy.int32).reshape(2, 2)
        self.assertEqual(weave(acts, ids, numpy.array(WTS, dtype=numpy.float32)), self.expected)

    def test_writable_buffer_is_aliased_and_read_only_is_copied(self) -> None:
        data = array("f", ACTS)
        self.assertEqual(ct.addressof(Operand(data, FLOAT32, "a").pack()), data.buffer_info()[0])
        copied = Operand(bytes(data.tobytes()), FLOAT32, "a").pack()
        self.assertEqual(list(copied), ACTS)

    def test_sequence_path_still_range_checks_expert_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "expert id must be between"):
            Operand([1 << 40], INT32, "expert_ids").pack()


class ValidationTests(unittest.TestCase):
    def check(self, acts: object, ids: object, wts: object, pattern: str) -> None:
        with self.assertRaisesRegex(ValueError, pattern):
            weave(acts, ids, wts)

    def test_wrong_element_type_names_the_argument_and_format(self) -> None:
        self.check(array("d", ACTS), IDS, WTS, r"activations must hold float32 values .*got format 'd'")
        self.check(ACTS, array("h", IDS), WTS, r"expert_ids must hold int32 values .*got format 'h'")
        self.check(ACTS, array("f", [0.0] * 4), WTS, r"expert_ids must hold int32")

    def test_non_contiguous_buffer_is_rejected(self) -> None:
        stepped = memoryview(array("f", range(12)))[::2]
        self.check(stepped, IDS, WTS, "activations must be C-contiguous")

    def test_partial_bytes_are_rejected(self) -> None:
        self.check(b"\x00" * 5, IDS, WTS, "multiple of 4")

    @unittest.skipIf(numpy is None, "numpy not installed")
    def test_foreign_byte_order_is_rejected(self) -> None:
        swapped = numpy.array(ACTS, dtype=">f4" if numpy.little_endian else "<f4")
        self.check(swapped, IDS, WTS, "activations must hold float32")

    def test_validation_happens_before_any_ticket_exists(self) -> None:
        with reference_runtime() as runtime:
            prepared = runtime.prepare(**SHAPE)
            with self.assertRaises(ValueError):
                prepared.weave(array("d", ACTS), IDS, WTS)
            self.assertEqual(prepared.weave(ACTS, IDS, WTS), weave(ACTS, IDS, WTS))
            self.assertEqual(runtime.status().submitted, 1)


class OutTests(unittest.TestCase):
    def test_out_is_filled_and_returned(self) -> None:
        out = array("f", bytes(24))
        self.assertIs(weave(ACTS, IDS, WTS, out=out), out)
        self.assertEqual(list(out), weave(ACTS, IDS, WTS))

    def test_out_accepts_writable_memoryview_and_numpy(self) -> None:
        out = memoryview(array("f", bytes(24)))
        weave(ACTS, IDS, WTS, out=out)
        self.assertEqual(out.tolist(), weave(ACTS, IDS, WTS))
        if numpy is not None:
            result = numpy.zeros(6, dtype=numpy.float32)
            weave(ACTS, IDS, WTS, out=result)
            self.assertEqual(result.tolist(), weave(ACTS, IDS, WTS))

    def test_bad_out_is_rejected_before_submit(self) -> None:
        with reference_runtime() as runtime:
            prepared = runtime.prepare(**SHAPE)
            for bad, pattern in (
                (array("f", bytes(20)), "exactly 6 float32 values, got 5"),
                (bytes(24), "writable"),
                ([0.0] * 6, "writable float32 buffer"),
                (array("d", bytes(48)), "out must hold float32"),
            ):
                with self.subTest(pattern=pattern), self.assertRaisesRegex(ValueError, pattern):
                    prepared.weave(ACTS, IDS, WTS, out=bad)
            self.assertEqual(runtime.status().submitted, 0)
            self.assertEqual(len(prepared.weave(ACTS, IDS, WTS)), 6)


def misaligned(code: str, values: list) -> memoryview:
    """A writable, C-contiguous, correctly-typed buffer whose first element sits one byte past a 4-byte boundary."""
    data = array(code, values).tobytes()
    raw = bytearray(1 + len(data))
    raw[1:] = data
    return memoryview(raw)[1:].cast(code)


class MisalignedTests(unittest.TestCase):
    """R7 LOW 7: a misaligned buffer was aliased, so the native engine refused it while the reference accepted it."""

    def test_a_misaligned_input_is_copied_to_aligned_memory(self) -> None:
        for kind, code, values in ((FLOAT32, "f", ACTS), (INT32, "i", IDS)):
            with self.subTest(code):
                source = misaligned(code, values)
                packed = Operand(source, kind, "x").pack()
                self.assertEqual(ct.addressof(packed) % 4, 0)
                self.assertEqual(list(packed), values)
                source[0] = 99  # a copy: the packed array does not see later writes
                self.assertEqual(packed[0], values[0])

    def test_an_aligned_writable_input_is_still_zero_copy(self) -> None:
        source = array("f", ACTS)
        packed = Operand(source, FLOAT32, "x").pack()
        source[0] = 99.0
        self.assertEqual(packed[0], 99.0)

    def test_every_session_accepts_a_misaligned_input_and_matches_the_aligned_result(self) -> None:
        got = weave(misaligned("f", ACTS), misaligned("i", IDS), misaligned("f", WTS))
        self.assertEqual(got, weave(ACTS, IDS, WTS))

    def test_a_misaligned_out_is_refused_because_it_cannot_be_both_aliased_and_aligned(self) -> None:
        with reference_runtime() as runtime:
            prepared = runtime.prepare(**SHAPE)
            with self.assertRaisesRegex(ValueError, "out must be 4-byte aligned"):
                prepared.weave(ACTS, IDS, WTS, out=misaligned("f", [0.0] * 6))
            self.assertEqual(runtime.status().submitted, 0)


class ExportReleaseTests(unittest.TestCase):
    """R7 LOW 10: a refused weave left the caller's array export-locked for as long as the exception lived."""

    def setUp(self) -> None:
        gc.disable()  # the old behaviour only healed when the cyclic collector ran
        self.addCleanup(gc.enable)

    def check(self, refuse: object, arrays: tuple | None = None, runtime_factory: object = reference_runtime) -> None:
        acts, ids, wts = arrays or (array("f", [1.0, 2.0]), array("i", [0, 9]), array("f", [0.5, 0.5]))
        saved = None
        with runtime_factory() as runtime:
            prepared = runtime.prepare(hidden=2, experts=3, max_tokens=2, top_k=2)
            try:
                refuse(prepared, acts, ids, wts)
            except (SeabraidError, ValueError) as error:
                saved = error  # e.g. a thread pool, a logger or a retry loop keeps it
            self.assertIsNotNone(saved)
            for buffer in (acts, ids, wts):
                buffer.append(buffer[0])  # BufferError while an export is still held
        self.assertIsNotNone(saved)

    def test_an_engine_refusal(self) -> None:
        self.check(lambda m, a, i, w: m.weave(a, i, w))  # expert id 9 does not exist

    def test_a_python_precheck_refusal(self) -> None:
        three_tokens = (array("f", [1.0] * 6), array("i", [0, 1, 2]), array("f", [0.5] * 3))  # max_tokens is 2
        self.check(lambda m, a, i, w: m.weave(a, i, w), three_tokens)

    def test_a_later_operand_refused_after_an_earlier_one_was_wrapped(self) -> None:
        self.check(lambda m, a, i, w: m.weave(a, array("d", [0.0, 1.0]), w))

    def test_a_refused_out_buffer(self) -> None:
        self.check(lambda m, a, i, w: m.weave(a, i, w, out=array("f", [0.0])))


if __name__ == "__main__":
    unittest.main()
