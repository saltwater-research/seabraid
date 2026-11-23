# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for the batch v2 / provider v3 tests: library spec lists, sample 16-bit payloads, providers, the reference-runtime
factory and the outcome helper that lets two sessions be compared class for class."""

from __future__ import annotations

import logging
import os
import struct
import warnings
from unittest.mock import patch

from seabraid import Runtime, SeabraidError
from seabraid._model import BatchInfo, Shape
from seabraid.constants import FORMAT_ROUTED_BF16

V1_SPEC = ["seabraid_abi_epoch", "seabraid_open", "seabraid_open_reference", "seabraid_open_multi",
            "seabraid_prepare", "seabraid_submit", "seabraid_collect", "seabraid_close"]
V3_SPEC = V1_SPEC + ["seabraid_session_open", "seabraid_session_submit", "seabraid_session_status"]
LIBRARY = os.environ.get("SEABRAID_LIBRARY")
SHAPE = dict(hidden=2, experts=3, max_tokens=4, top_k=2)
HALVES = struct.pack("<4e", 1.5, -2.0, 0.25, 8.0)  # two F16 rows of hidden=2
BF16 = b"".join(struct.pack("<f", v)[2:] for v in (1.5, -2.0, 0.25, 8.0))  # bf16 = the high half of an f32


def decode_f16(raw: bytes) -> list[float]:
    return list(struct.unpack(f"<{len(raw) // 2}e", raw))


def decode_bf16(raw: bytes) -> list[float]:
    return [struct.unpack("<f", b"\0\0" + raw[i:i + 2])[0] for i in range(0, len(raw), 2)]


def half_provider(seen: list) -> object:
    """Doubles each F16/BF16 element; records (format, epoch, byte length, kind of view, bytes)."""

    def serve(operation: int, config: Shape, acts: object, ids: object, weights: object, out: object,
              info: BatchInfo) -> None:
        raw = bytes(acts)
        seen.append((info.activation_format, info.placement_epoch, len(raw), view_kind(acts), raw))
        decode = decode_bf16 if info.activation_format == FORMAT_ROUTED_BF16 else decode_f16
        for i, value in enumerate(decode(raw)):
            out[i] = value * 2

    return serve


def view_kind(view: object) -> str:
    """"raw" for a bytes view (memoryview "B" or numpy uint8), "float" for float32; numpy is optional."""
    if getattr(view, "format", None) == "B" or str(getattr(view, "dtype", "")) == "uint8":
        return "raw"
    return "float" if getattr(view, "format", None) == "f" or str(getattr(view, "dtype", "")) == "float32" else "?"


def echo(operation: int, config: Shape, acts: object, ids: object, weights: object, out: object) -> None:
    for i in range(len(out)):
        out[i] = acts[i] + 1.0


def reference_runtime(**kwargs: object) -> Runtime:
    """A Runtime on the pure-Python reference session even when SEABRAID_LIBRARY is set."""
    with patch.dict(os.environ):
        os.environ.pop("SEABRAID_LIBRARY", None)
        with warnings.catch_warnings(), patch.object(logging.getLogger("seabraid"), "disabled", True):
            warnings.simplefilter("ignore", RuntimeWarning)
            return Runtime(**kwargs)


def outcome(call: object) -> object:
    """The result of `call`, or (error class name, status) so two sessions can be compared."""
    try:
        return call()
    except (SeabraidError, ValueError) as error:
        return (type(error).__name__, getattr(error, "status", None))




class EpochScript:
    """The same epoch story on any runtime; returns what each step produced."""

    def run(self, runtime: Runtime) -> list[object]:
        prepared = runtime.prepare(**SHAPE)
        good = ([1.0, 2.0], [0, 1], [0.5, 0.5])
        bad = ([1.0, 2.0], [0, 9], [0.5, 0.5])  # expert 9 does not exist
        steps = [
            lambda: prepared.weave(*good),                       # 0 before any nonzero epoch is fine
            lambda: prepared.weave(*good, placement_epoch=5),
            lambda: prepared.weave(*good, placement_epoch=4),    # stale -> BUSY_OR_STATE
            lambda: prepared.weave(*good, placement_epoch=0),    # 0 after nonzero -> INVALID
            lambda: prepared.weave(*bad, placement_epoch=9),     # rejected for another reason: must not move 5
            lambda: prepared.weave(*good, placement_epoch=5),    # equal is accepted, so 9 never landed
            lambda: prepared.weave(*good, placement_epoch=7),
        ]
        return [outcome(step) for step in steps]
