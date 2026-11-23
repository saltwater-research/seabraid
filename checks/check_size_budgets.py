#!/usr/bin/env python3
# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""A file's own header SIZE budget must be true (R8 L5): "SIZE: under 150 lines" with 169 lines is a false claim.

OWNER: this checker owns the rule; each file's header owns its own number.  checks/check_size.py owns the repository-wide
limit and is unrelated (a file can be under 500 and still over the budget it wrote down).
CONTRACT: in the first HEADER_LINES lines of a .py/.rs/.c/.h/.cpp/.hpp file, ``SIZE: <=N lines`` / ``under N`` / ``at most N`` /
``hard cap N`` / ``N lines`` is read as that file's budget; the file's line count must be <= N.  A budget that names several
files (header and source, each, plus) cannot be checked from one file and is listed as a note, never silently ignored.  A SIZE
line this module cannot read is a failure (the format is part of the rule).  Budgets that are false TODAY are rows of
checks/size_budget_allow.txt (``path<TAB>lines<TAB>why``): the file may not grow past its row, the row must be lowered when it
shrinks, and it is deleted when the budget is true, so the list only shrinks and its owners are named in the why.
FAILURE: fail-fast and loud; exit 1 and ``path: N lines, header budget M`` per violation.  SIZE: <=150 lines.
CHECKER: tests/test_size_budgets.py.   Run:  python3 checks/check_size_budgets.py [--root DIR]
"""

from __future__ import annotations

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as common  # noqa: E402

ALLOW = "checks/size_budget_allow.txt"
SUFFIXES = (".py", ".rs", ".c", ".h", ".cpp", ".hpp")
HEADER_LINES = 50
TAG = re.compile(r"\bSIZE(?: BUDGET)?\s*:\s*([^\n]*)")
NUMBER = (re.compile(r"hard cap (\d+)"), re.compile(r"(?:<=|under|at most|up to)\s*~?(\d+)"), re.compile(r"^\s*~?(\d+) lines"))
MULTI = re.compile(r"\b(header|source|each|plus|including the test)\b|[a-z_]+\.(?:cpp|hpp|rs|py|c|h) \d+", re.I)


def budget_of(text: str) -> tuple[str, int | None]:
    """-> ("none"|"multi"|"bad"|"ok", N).  Only the first SIZE tag within the header counts."""
    head = "\n".join(text.splitlines()[:HEADER_LINES])
    m = TAG.search(head)
    if not m:
        return "none", None
    rest = re.split(r"\.(?:\s|$)", m.group(1))[0]  # the first sentence: later ones (FAILURE, CHECKER) are not the budget
    if not NUMBER[0].search(rest) and MULTI.search(rest):
        return "multi", None
    for rx in NUMBER:
        hit = rx.search(rest)
        if hit:
            return "ok", int(hit.group(1))
    return "bad", None


def load_allow(root: str) -> tuple[dict[str, int], list[str]]:
    rows, errs = {}, []
    path = os.path.join(root, ALLOW)
    for n, raw in enumerate(open(path, encoding="utf-8").read().splitlines() if os.path.exists(path) else [], 1):
        if raw.strip() and not raw.lstrip().startswith("#"):
            parts = raw.split("\t")
            ok = len(parts) == 3 and parts[1].strip().isdigit() and len(parts[2].strip()) >= 20
            errs += [] if ok else [f"{ALLOW}:{n}: want path<TAB>lines<TAB>why (why >= 20 chars)"]
            rows.update({parts[0].strip(): int(parts[1])} if ok else {})
    return rows, errs


def verdict(rel: str, lines: int, budget: int, rows: dict[str, int]) -> str | None:
    if lines <= budget:
        return f"{ALLOW}: stale row for {rel} (now {lines} lines, budget {budget}); delete it" if rel in rows else None
    if rel not in rows:
        return f"{rel}: {lines} lines, header budget {budget}"
    if lines > rows[rel]:
        return f"{rel}: {lines} lines, past its row of {rows[rel]} (header budget {budget})"
    return f"{ALLOW}: lower the row for {rel} to {lines} (budget {budget})" if lines < rows[rel] else None


def check(root: str) -> tuple[list[str], list[str]]:
    rows, bad = load_allow(root)
    notes, seen = [], set()
    for rel in common.tracked(root):
        text = common.read_text(root, rel) if rel.endswith(SUFFIXES) else None
        kind, budget = budget_of(text) if text is not None else ("none", None)
        if kind == "bad":
            bad.append(f"{rel}: SIZE line in the header that this checker cannot read (use '<=N lines' or 'under N lines')")
        elif kind == "multi":
            notes.append(f"note: {rel}: SIZE budget names several files; not machine-checked")
        elif kind == "ok":
            seen.add(rel)
            problem = verdict(rel, len(text.splitlines()), budget, rows)
            bad += [problem] if problem else []
    bad += [f"{ALLOW}: row for {rel} matches no file with a readable SIZE budget" for rel in sorted(set(rows) - seen)]
    return bad, notes + [f"note: {len(seen)} header budget(s) checked, {len(rows)} known-false row(s)"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--root", default=common.default_root())
    bad, notes = check(ap.parse_args(argv).root)
    return common.finish("check_size_budgets", bad, notes)


if __name__ == "__main__":
    sys.exit(main())
