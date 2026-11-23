# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Which rows a COMPACT_ROWS provider sees, for the pure-Python reference session.

OWNER   : the Python mirror of src/rows.rs (the Rust file owns the rule: a row belongs to the provider exactly when
          one of its top_k picks is not a skip). This module only restates it so the reference session can be the
          parity oracle for compaction; it adds no rule of its own.
CONTRACT: kept_rows(ids, top_k) -> ascending row numbers; gather(items, rows, width) -> the kept rows of a flat
          row-major sequence; scatter(compact, rows, width, total_rows) -> a zero-filled float list with the kept
          rows' results written back. Imports: .constants only.
SIZE    : under 60 lines.
FAILURE : none of its own; callers pass validated, whole-row inputs.
CHECKER : tests/test_compaction.py (reference output equals uncompacted output; native parity with SEABRAID_LIBRARY).
FINISH  : that test passes.
"""

from __future__ import annotations

from collections.abc import Sequence

from .constants import SKIP_EXPERT_ID


def kept_rows(ids: Sequence[int], top_k: int) -> list[int]:
    """Rows with at least one real pick, in original order (src/rows.rs RowMap::derive)."""
    return [row for row in range(len(ids) // top_k)
            if any(i != SKIP_EXPERT_ID for i in ids[row * top_k:(row + 1) * top_k])]


def gather(items: Sequence, rows: Sequence[int], width: int) -> list:
    """Copy the kept rows of a flat sequence holding `width` items per row (src/rows.rs RowMap::gather)."""
    return [item for row in rows for item in items[row * width:(row + 1) * width]]


def scatter(compact: Sequence[float], rows: Sequence[int], width: int, total_rows: int) -> list[float]:
    """Zero everywhere, the kept rows' results in their original rows (src/rows.rs RowMap::scatter)."""
    full = [0.0] * (total_rows * width)
    for position, row in enumerate(rows):
        full[row * width:(row + 1) * width] = compact[position * width:(position + 1) * width]
    return full
