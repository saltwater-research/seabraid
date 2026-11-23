# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""install/_plan_gguf.py reads an untrusted file: hostile headers must end in ModelError, never MemoryError/RecursionError/a hang."""
from __future__ import annotations

import struct

import pytest

from install import _plan_gguf as G

Q, I = "<Q", "<I"


def _s(x: str) -> bytes:
    b = x.encode()
    return struct.pack(Q, len(b)) + b


def _file(tmp_path, body: bytes, n_tensors=0, n_kv=1):
    p = tmp_path / "h.gguf"
    p.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, n_tensors, n_kv) + body)
    return str(p)


def _kv(key, vtype, payload):
    return _s(key) + struct.pack(I, vtype) + payload


def test_valid_minimal_file_still_parses(tmp_path):
    kv, sizes = G.scan_gguf(_file(tmp_path, _kv("k", 4, struct.pack(I, 7))))
    assert kv == {"k": 7} and sizes == []


def test_huge_string_length_is_refused_without_allocating(tmp_path):
    body = struct.pack(Q, 2**62) + b"x" * 8                      # key length claims 4 EiB
    with pytest.raises(G.ModelError, match="exceeds the limit"):
        G.scan_gguf(_file(tmp_path, body))


def test_string_longer_than_the_file_is_refused(tmp_path):
    body = struct.pack(Q, 5000) + b"abc" * 4                       # under the cap, but beyond the file
    with pytest.raises(G.ModelError, match="remain"):
        G.scan_gguf(_file(tmp_path, body))


def test_deeply_nested_arrays_are_depth_capped_not_recursed(tmp_path):
    depth = 5000                                                 # would blow the Python stack if recursed
    payload = struct.pack("<IQ", 9, 1) * depth + struct.pack("<IQ", 4, 0)
    with pytest.raises(G.ModelError, match="nested deeper"):
        G.scan_gguf(_file(tmp_path, _kv("a", 9, payload)))


def test_nesting_within_the_cap_is_skipped(tmp_path):
    payload = struct.pack("<IQ", 9, 1) * 3 + struct.pack("<IQ", 4, 2) + struct.pack("<II", 1, 2)
    kv, _ = G.scan_gguf(_file(tmp_path, _kv("a", 9, payload) + _kv("b", 4, struct.pack(I, 9)), n_kv=2))
    assert kv["b"] == 9


def test_array_element_count_over_the_cap_is_refused(tmp_path):
    with pytest.raises(G.ModelError, match="exceeds the limit"):
        G.scan_gguf(_file(tmp_path, _kv("a", 9, struct.pack("<IQ", 8, 2**40))))


def test_fixed_array_longer_than_the_file_is_refused(tmp_path):
    with pytest.raises(G.ModelError, match="remain"):
        G.scan_gguf(_file(tmp_path, _kv("a", 9, struct.pack("<IQ", 10, 2**40))))


def test_counts_that_cannot_fit_are_refused(tmp_path):
    with pytest.raises(G.ModelError, match="more than the file can hold"):
        G.scan_gguf(_file(tmp_path, b"", n_tensors=2**50, n_kv=2**50))


def test_too_many_dimensions_is_refused(tmp_path):
    body = _s("t") + struct.pack(I, 10**6) + b"\0" * 64
    with pytest.raises(G.ModelError, match="dimensions"):
        G.scan_gguf(_file(tmp_path, body, n_tensors=1, n_kv=0))


@pytest.mark.parametrize("cut", [3, 8, 15, 30, 40])
def test_truncated_file_is_a_model_error(tmp_path, cut):
    data = open(_file(tmp_path, _kv("key", 8, _s("value"))), "rb").read()[:cut]
    t = tmp_path / "t.gguf"
    t.write_bytes(data)
    with pytest.raises(G.ModelError):
        G.scan_gguf(str(t))


@pytest.mark.parametrize("exc", [MemoryError, RecursionError])
def test_memory_and_recursion_errors_are_converted(tmp_path, monkeypatch, exc):
    p = _file(tmp_path, _kv("k", 4, struct.pack(I, 1)))

    def boom(self, n):
        raise exc()
    monkeypatch.setattr(G._Reader, "take", boom)
    with pytest.raises(G.ModelError, match=exc.__name__):
        G.scan_gguf(p)
