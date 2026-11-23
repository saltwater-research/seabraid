#!/usr/bin/env python3
# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""The ABI may only grow by appending: today's layout table is checked against the frozen snapshot of its epoch.

OWNER: src/layout/abi_layout.epochN.txt is history (the table frozen for epoch N) and owns no current number;
src/layout/abi_layout.txt stays the one owner of today's layout.  This checker owns the rule that relates them.
CONTRACT: the snapshot used is the newest ``abi_layout.epochN.txt`` with N <= SEABRAID_ABI_EPOCH (include/seabraid/
seabraid.h), and it MUST be N == SEABRAID_ABI_EPOCH: bumping the epoch without freezing a snapshot for it FAILS
(WHY: a skip on a bump let the break go unrecorded and left every later row unguarded).  Every snapshot row must
survive.  A field offset keeps its value.  A struct size is kept, or may GROW, but only for a struct whose snapshot
rows include ``.struct_size`` (the field that lets an old caller say how big its struct is).  A field added to a
snapshot struct must sit at or past the snapshot size.  New structs are free.  Row format: see abi_layout.txt.
SIZE: <=150 lines.  FAILURE: fail-fast and loud; exit 1 and ``row: reason`` per violation.
CHECKER: tests/test_checkers.py (green on the tree) and tests/test_sabotage.py (a v1 offset moved, a v2 offset
moved, and an epoch bumped with no snapshot, each in a copy).
Run:  python3 checks/check_abi_append.py [--root DIR]
"""

from __future__ import annotations

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as common  # noqa: E402

CURRENT = "src/layout/abi_layout.txt"
SNAPSHOT_RE = re.compile(r"^abi_layout\.epoch(\d+)\.txt$")
EPOCH_RE = re.compile(r"#define\s+SEABRAID_ABI_EPOCH\s+(\d+)u?\b")
HEADER = "include/seabraid/seabraid.h"


def parse(text: str, rel: str) -> dict[str, int]:
    """Rows as {"struct": size, "struct.field": offset}; a malformed or duplicate row is a hard error."""
    rows: dict[str, int] = {}
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2 or not parts[1].isdigit() or parts[0] in rows:
            raise SystemExit(f"check_abi_append: {rel}:{number}: bad or duplicate row {line!r}")
        rows[parts[0]] = int(parts[1])
    return rows


def _row_problem(key: str, was: int, now: int | None, old: dict[str, int]) -> str | None:
    """The rule for one snapshot row: a field keeps its offset, a struct never shrinks and grows only with .struct_size."""
    if now is None:
        return f"{key}: row removed (snapshot row was {was})"
    if "." in key:
        return f"{key}: offset moved {was} -> {now}" if now != was else None
    if now < was:
        return f"{key}: size shrank {was} -> {now}"
    if now > was and f"{key}.struct_size" not in old:
        return f"{key}: size grew {was} -> {now} but the struct has no .struct_size field to tell an old caller apart"
    return None


def _new_field_problem(key: str, now: int, old: dict[str, int]) -> str | None:
    """A field added to a snapshot struct must sit at or past the snapshot size."""
    struct, dot, _ = key.partition(".")
    if dot and key not in old and struct in old and now < old[struct]:
        return f"{key}: new field at offset {now} is inside the snapshot struct (size {old[struct]}); append past it"
    return None


def compare(old: dict[str, int], new: dict[str, int]) -> list[str]:
    """Violations of the append-only rule between the epoch snapshot and the current table."""
    found = [_row_problem(key, was, new.get(key), old) for key, was in old.items()]
    found += [_new_field_problem(key, now, old) for key, now in new.items()]
    return [problem for problem in found if problem]


def epoch_of(root: str) -> int | None:
    path = os.path.join(root, HEADER)
    if not os.path.exists(path):
        return None
    m = EPOCH_RE.search(common.read_text(root, HEADER) or "")
    return int(m.group(1)) if m else None


def check(root: str) -> tuple[list[str], list[str]]:
    folder = os.path.join(root, "src", "layout")
    snapshots = sorted(int(m.group(1)) for f in os.listdir(folder) if (m := SNAPSHOT_RE.match(f)))
    epoch = epoch_of(root)
    if epoch is None:
        return [f"{HEADER}: no SEABRAID_ABI_EPOCH define found"], []
    usable = [s for s in snapshots if s <= epoch]
    if not usable or usable[-1] != epoch:
        have = ", ".join(f"epoch{s}" for s in snapshots) or "none"
        return [f"src/layout: no snapshot for epoch {epoch} (have: {have}); freeze abi_layout.epoch{epoch}.txt with the bump"], []
    rel = f"src/layout/abi_layout.epoch{epoch}.txt"
    old = parse(common.read_text(root, rel) or "", rel)
    new = parse(common.read_text(root, CURRENT) or "", CURRENT)
    bad = [f"{rel}: {line}" for line in compare(old, new)]
    return bad, [f"note: epoch{epoch}: {len(old)} rows checked against {CURRENT}"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--root", default=common.default_root())
    bad, notes = check(ap.parse_args(argv).root)
    return common.finish("check_abi_append", bad, notes)


if __name__ == "__main__":
    sys.exit(main())
