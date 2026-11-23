# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Turn caller buffers or plain sequences into ctypes arrays without per-element Python loops.

OWNER   : the Python-side rules for what counts as a float32 / int32 input or output buffer, and how it becomes the
          ctypes array the native call (or reference provider) consumes.
CONTRACT: Operand(data, kind, name) -> .count (elements), .pack() -> ctypes array; RawHalfOperand(data, name) is the same
          for F16/BF16 activations: any C-contiguous buffer is taken as raw bytes (2 per element, nothing is converted, so
          numpy is not needed), .count = elements, .pack() -> c_uint16 array; output_target(out, count, name) ->
          writable ctypes float array aliasing `out`; FLOAT32 / INT32 kinds. A C-contiguous buffer-protocol object
          (array.array, memoryview, numpy array, bytes-like) is used zero-copy when writable AND 4-byte aligned, and
          copied once with from_buffer_copy when read-only or misaligned (the engine refuses a misaligned pointer);
          a misaligned `out` is a ValueError; any other object is a plain sequence and takes the slow per-element path.
          Imports: stdlib + siblings; numpy is never imported here.
SIZE    : under 170 lines.
FAILURE : wrong item format, itemsize, byte order, contiguity, writability or (for out) alignment raises ValueError naming the argument;
          a wrong output length raises ValueError; a non-numeric sequence item raises TypeError (as ctypes does).
CHECKER : tests/test_buffers.py.
FINISH  : that test passes, including sequence/buffer parity and the out= path.
"""

from __future__ import annotations

import ctypes as ct
import sys
from dataclasses import dataclass

from ._model import _INT32_MAX, _INT32_MIN, checked_integer

_ITEM_BYTES = 4  # both wire elements (float32, int32) are 32 bits
_HALF_BYTES = 2  # F16 and BF16 elements
_BYTE_FORMATS = frozenset("Bbc")
_NATIVE_PREFIXES = frozenset(("", "@", "=", "<" if sys.byteorder == "little" else ">"))


@dataclass(frozen=True)
class Kind:
    ctype: type
    code: str  # the memoryview/array format character
    accepted: frozenset  # format characters of the right type with itemsize 4 (numpy int32 reports "l" on Windows)
    noun: str


FLOAT32 = Kind(ct.c_float, "f", frozenset("f"), "float32")
INT32 = Kind(ct.c_int32, "i", frozenset("il"), "int32")


def _plain_format(fmt: str) -> str:
    """Drop a native-byte-order prefix; keep a foreign one so it fails the format check with its text."""
    return fmt[1:] if fmt[:1] in _NATIVE_PREFIXES and len(fmt) > 1 else fmt


def view_of(data: object, kind: Kind, name: str, *, writable: bool = False) -> memoryview | None:
    """Flat view of `data` as `kind` elements, or None when `data` is not a buffer (a plain sequence).

    A refusal releases the view it opened: the exception's traceback keeps this frame alive, and an open view would
    keep the caller's array export-locked (no append/resize) for as long as the exception lives (R7 LOW 10).
    """
    try:
        raw = memoryview(data)
    except TypeError:
        return None
    try:
        return _flat(raw, kind, name, writable)
    except BaseException:
        raw.release()
        raise


def _flat(raw: memoryview, kind: Kind, name: str, writable: bool) -> memoryview:
    if not raw.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    if writable and raw.readonly:
        raise ValueError(f"{name} must be a writable buffer")
    fmt = _plain_format(raw.format)
    if raw.itemsize == 1 and fmt in _BYTE_FORMATS:
        if raw.nbytes % _ITEM_BYTES:
            raise ValueError(f"{name} byte length must be a multiple of {_ITEM_BYTES}")
    elif raw.itemsize != _ITEM_BYTES or fmt not in kind.accepted:
        raise ValueError(f"{name} must hold {kind.noun} values (format {kind.code!r}), got format {raw.format!r}")
    return raw.cast("B").cast(kind.code)


def _array_from(view: memoryview, kind: Kind, alias_name: str | None = None) -> ct.Array:
    """The ctypes array for `view`: aliased when writable and aligned, else one copy.

    `alias_name` names a target that MUST alias (the out= buffer, whose writes have to reach the caller): when it is
    misaligned there is no honest copy to return, so it is refused. WHY copy a misaligned input: the engine refuses
    a pointer that is not aligned for its element, while the reference accepts the same buffer; copying makes every
    session see the same aligned bytes (R7 LOW 7).
    """
    raw = view.cast("B")
    array_type = kind.ctype * len(view)
    if raw.readonly:  # a read-only source cannot back a writable ctypes array, so it is copied once
        return array_type.from_buffer_copy(raw)
    aliased = array_type.from_buffer(raw)
    if ct.addressof(aliased) % _ITEM_BYTES == 0:
        return aliased
    if alias_name is not None:
        aliased = None  # drop the exports before the traceback can hold this frame
        raw.release()
        raise ValueError(f"{alias_name} must be {_ITEM_BYTES}-byte aligned (its address is not a multiple of {_ITEM_BYTES})")
    return array_type.from_buffer_copy(raw)


class Operand:
    """One input argument: validated at construction, packed only after the caller's shape checks pass."""

    def __init__(self, data: object, kind: Kind, name: str) -> None:
        self._data, self._kind = data, kind
        self._view = view_of(data, kind, name)
        self.count = len(data) if self._view is None else len(self._view)

    def pack(self) -> ct.Array:
        if self._view is not None:
            return _array_from(self._view, self._kind)
        # Slow path: a plain sequence is converted element by element (integers are range-checked first).
        array_type = self._kind.ctype * self.count
        if self._kind is INT32:
            return array_type(*[checked_integer(v, _INT32_MIN, _INT32_MAX, "expert id") for v in self._data])
        return array_type(*self._data)


class RawHalfOperand:
    """F16/BF16 activations as the raw bytes of their 16-bit elements; the caller already encoded them."""

    def __init__(self, data: object, name: str) -> None:
        try:
            view = memoryview(data)
        except TypeError:
            raise ValueError(f"{name} must be a bytes-like buffer of 16-bit elements for F16/BF16") from None
        if not view.c_contiguous or view.nbytes % _HALF_BYTES:
            problem = (f"{name} must be C-contiguous" if not view.c_contiguous
                       else f"{name} byte length must be a multiple of {_HALF_BYTES}")
            view.release()  # the traceback would otherwise keep the caller's buffer export-locked
            raise ValueError(problem)
        raw = view.cast("B")
        self._raw = raw
        self.count = raw.nbytes // _HALF_BYTES

    def pack(self) -> ct.Array:
        # Always one copy: ctypes needs a uint16 array, and a copy also pins 2-byte alignment for the engine.
        return (ct.c_uint16 * self.count).from_buffer_copy(self._raw)


def output_target(out: object, count: int, name: str = "out") -> ct.Array:
    """A writable float32 ctypes array aliasing `out`, which must hold exactly `count` elements."""
    view = view_of(out, FLOAT32, name, writable=True)
    if view is None:
        raise ValueError(f"{name} must be a writable float32 buffer")
    try:
        if len(view) != count:
            raise ValueError(f"{name} must hold exactly {count} float32 values, got {len(view)}")
        return _array_from(view, FLOAT32, name)
    except BaseException:
        view.release()  # a refusal must not leave the caller's buffer export-locked (see view_of)
        raise
