# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""GGUF header reader for the planner (split out of plan.py to stay under the size cap).  UNTESTED-ON-HARDWARE.

OWNER     : the only code that parses GGUF bytes in the installer; _plan_model.read_model_facts() is its one caller.
CONTRACT  : scan_gguf(path) -> (kv, [(tensor_name, size_bytes, n_elements)]); ModelError on anything unreadable.  stdlib only.
FAILURE   : fail-fast with ModelError naming the file; never returns partial data.  The file is untrusted input: every length
            is bounded by the bytes left in the file and by a fixed cap, nesting is depth-capped and iterative, and any
            parse failure (including MemoryError/RecursionError) becomes ModelError.
CHECKER   : tests/install/test_plan.py builds synthetic GGUFs and reads them back; tests/install/test_plan_gguf_hostile.py
            feeds hostile headers.
"""
from __future__ import annotations

import os
import struct
from typing import Any


class ModelError(Exception):
    pass


_FIXED = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
# WHY these caps: real files sit orders of magnitude below them (longest key/template string is KBs, the largest vocab array is
# ~260k entries, tensors have <=4 dims), so a hostile header hits the cap, never a legitimate model.
MAX_STRING = 1 << 20
MAX_ARRAY = 1 << 20
MAX_DEPTH = 8
MAX_DIMS = 8
_MIN_KV, _MIN_TENSOR = 12, 24         # smallest on-disk size of one kv / tensor record: a count above size/min cannot be true


class _Reader:
    """Sequential reader whose every read is checked against the bytes that remain, so no length read from the file can
    allocate more than the file holds."""

    def __init__(self, f: Any, size: int, path: str) -> None:
        self.f, self.size, self.path = f, size, path

    def left(self) -> int:
        return self.size - self.f.tell()

    def take(self, n: int) -> bytes:
        if n < 0 or n > self.left():
            raise ModelError(f"{self.path}: GGUF header claims {n} bytes but only {max(self.left(), 0)} remain (truncated or corrupt)")
        return self.f.read(n)

    def skip(self, n: int) -> None:
        if n < 0 or n > self.left():
            raise ModelError(f"{self.path}: GGUF header skips {n} bytes but only {max(self.left(), 0)} remain (truncated or corrupt)")
        self.f.seek(n, 1)

    def u(self, fmt: str) -> Any:
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def length(self, what: str, cap: int) -> int:
        n = self.u("<Q")
        if n > cap:
            raise ModelError(f"{self.path}: GGUF {what} of {n} exceeds the limit {cap}")
        return n

    def string(self) -> str:
        return self.take(self.length("string length", MAX_STRING)).decode("utf-8", "replace")

    def count(self, what: str, min_record: int) -> int:
        n = self.u("<Q")
        if n > self.left() // min_record:
            raise ModelError(f"{self.path}: GGUF {what} count {n} cannot fit in the remaining {self.left()} bytes")
        return n

    def scalar(self, t: int) -> Any:
        if t == 8:
            return self.string()
        if t in _FMT:
            return self.u(_FMT[t])
        raise ModelError(f"{self.path}: unknown GGUF value type {t}")

    def array_header(self, depth: int) -> tuple:
        if depth > MAX_DEPTH:
            raise ModelError(f"{self.path}: GGUF arrays nested deeper than {MAX_DEPTH}")
        et = self.u("<I")
        n = self.u("<Q")
        if et in _FIXED:
            self.skip(n * _FIXED[et])         # bounded by skip(): O(1), no element is visited
            return et, 0
        if n > MAX_ARRAY:
            raise ModelError(f"{self.path}: GGUF array of {n} elements exceeds the limit {MAX_ARRAY}")
        return et, n

    def skip_array(self) -> None:
        """Skip one array value (its type and count are already unread).  Iterative: a stack of remaining counts replaces
        recursion, so nesting costs a list slot, not a Python frame."""
        stack = [self.array_header(1)]
        while stack:
            et, left = stack[-1]
            if left == 0:
                stack.pop()
                continue
            stack[-1] = (et, left - 1)
            if et == 9:
                stack.append(self.array_header(len(stack) + 1))
            else:
                self.scalar(et)

    def value(self, t: int) -> Any:
        if t == 9:
            self.skip_array()
            return None
        return self.scalar(t)


def _read_header(r: _Reader) -> tuple:
    if r.take(4) != b"GGUF":
        raise ModelError(f"{r.path}: not a GGUF file")
    version = r.u("<I")
    if version < 2:
        raise ModelError(f"{r.path}: GGUF v{version} unsupported")
    n_tensors, n_kv = r.u("<Q"), r.u("<Q")
    if n_kv > r.left() // _MIN_KV or n_tensors > r.left() // _MIN_TENSOR:
        raise ModelError(f"{r.path}: GGUF claims {n_kv} keys and {n_tensors} tensors, more than the file can hold")
    kv = {}
    for _ in range(n_kv):
        k = r.string()
        kv[k] = r.value(r.u("<I"))
    infos = []
    for _ in range(n_tensors):
        name = r.string()
        n_dims = r.u("<I")
        if n_dims > MAX_DIMS:
            raise ModelError(f"{r.path}: tensor {name[:40]!r} has {n_dims} dimensions")
        n_el = 1
        for _ in range(n_dims):
            n_el *= r.u("<Q")
        r.u("<I")                               # ggml type
        infos.append((name, r.u("<Q"), n_el))
    return kv, infos


def scan_gguf(path: str) -> tuple:
    """-> (kv dict of scalar/str values, [(tensor_name, size_bytes, n_elements)]).  Tensor sizes come from offset deltas, so no
    ggml type table is needed (and none can drift).  Any malformed or hostile header raises ModelError."""
    try:
        end = os.path.getsize(path)
        with open(path, "rb") as f:
            kv, infos = _read_header(_Reader(f, end, path))
            align = int(kv.get("general.alignment") or 32)
            data_start = -(-f.tell() // align) * align
    except ModelError:
        raise
    except (OSError, struct.error, MemoryError, RecursionError, ArithmeticError, ValueError) as e:
        # WHY one broad handler: whatever the bytes did, the planner's contract is "ModelError naming the file".
        raise ModelError(f"{path}: cannot read GGUF header: {type(e).__name__}: {e}") from e
    order = sorted(infos, key=lambda t: t[1])
    sizes = []
    for i, (name, off, n_el) in enumerate(order):
        nxt = order[i + 1][1] if i + 1 < len(order) else end - data_start
        sizes.append((name, nxt - off, n_el))
    return kv, sizes
