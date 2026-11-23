# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Plain data shared by the native and reference providers and the public Runtime.

OWNER   : the Python-side value types (Capabilities, SessionStatus, BatchInfo, Shape) and the integer-bounds guard.
CONTRACT: frozen dataclasses plus checked_integer(value, low, high, field) -> int. Imports: .constants only.
SIZE    : under 90 lines.
FAILURE : checked_integer raises ValueError (or TypeError for a non-integer) before ctypes could wrap a value.
CHECKER : tests/test_mock_runtime.py (bounds), tests/test_reference.py (reference use).
FINISH  : both pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from operator import index

from .constants import FORMAT_ROUTED_BF16, FORMAT_ROUTED_F16, FORMAT_ROUTED_F32, SourceModality

_UINT32_MAX = (1 << 32) - 1
_INT32_MIN = -(1 << 31)
_INT32_MAX = (1 << 31) - 1
_UINT64_MAX = (1 << 64) - 1
# Bytes per activation element for each format ID; F32 is always accepted, the other two need a declaration.
ELEMENT_BYTES = {FORMAT_ROUTED_F32: 4, FORMAT_ROUTED_F16: 2, FORMAT_ROUTED_BF16: 2}


@dataclass(frozen=True)
class Capabilities:
    """Numeric ABI operation and buffer-format IDs are defined in seabraid.h."""

    abi_version: int
    operation: int
    input_format: int
    output_format: int
    source_modalities: SourceModality


@dataclass(frozen=True)
class SessionStatus:
    submitted: int
    collected: int
    provider_failures: int
    wait_timeouts: int
    last_source_modalities: SourceModality
    outstanding: bool
    # Row counters (batch v2 / provider v3): rows callers submitted vs rows handed to the provider; they differ only when
    # COMPACT_ROWS dropped skipped rows. Defaulted so older positional construction keeps working.
    rows_submitted: int = 0
    rows_shipped: int = 0


@dataclass(frozen=True)
class BatchInfo:
    """Per-batch facts a provider declared extra formats for: the activation format ID and the placement epoch."""

    activation_format: int
    placement_epoch: int


@dataclass(frozen=True)
class Shape:
    """The fixed geometry a session is prepared with (the ABI's seabraid_config_v1 minus the version)."""

    hidden: int
    experts: int
    max_tokens: int
    top_k: int


def checked_integer(value: int, low: int, high: int, field: str) -> int:
    """Reject values ctypes would truncate or wrap at the ABI boundary."""
    number = index(value)
    if not low <= number <= high:
        raise ValueError(f"{field} must be between {low} and {high}")
    return number
