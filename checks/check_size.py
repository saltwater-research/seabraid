#!/usr/bin/env python3
# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""No source file may exceed 500 lines unless a written exception allows it, and never beyond 750; past 400 it must be ratcheted.

OWNER: this checker owns the size budget (LIMIT, SOFT, HARD_MAX), the exception list checks/size_allow.json and the ratchet
checks/size_ratchet.json.  CONTRACT: a "source file" is ``common.is_source`` plus .yml/.yaml/.toml/.json (R8 L6: the CI workflow,
configs and data are read as code and grew unwatched; *.patch, docs and fixtures/ are not).  SOFT (400 = 80% of LIMIT, the project's
"split at ~80%" rule) is a FAIL, not a note: a file over it must be listed in the ratchet with its current length, may not
grow past that, and the row must be lowered the moment the file shrinks (``--write-ratchet`` does it; it never adds a row or
raises one) and deleted at or under SOFT, so the list only shrinks.  A file in 401..LIMIT with no row is the new violation.  The
exception list maps path -> {"ceiling": N, "why": "..."}: LIMIT < N <= HARD_MAX and a reason of >= 20 characters.
A listed file may never exceed its ceiling; a listed file back at or under LIMIT is stale and fails, so the list
only shrinks.  Exceptions are written by hand, never generated: a reason is the point of the entry.
WHY an exception at all: some files are one concept that reads worse split (a flat table, a generated binding, a
protocol state machine); WHY a hard maximum: past 1.5x the budget a file is two concepts, whatever its reason says.
FAILURE: fail-fast and loud; exit 1 and ``path: N lines (limit L | ceiling C)`` per violation; every excepted file
is printed with its reason so the exception stays visible.  SIZE: <=170 lines.
CHECKER: tests/test_checkers.py and tests/test_sabotage.py (a planted 501-line file; an excepted file that grows;
a ceiling over HARD_MAX; a missing reason; a stale entry) and tests/test_size_ratchet.py.
Run:  python3 checks/check_size.py [--root DIR] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as common  # noqa: E402

LIMIT = 500
SOFT = 400
HARD_MAX = 750
MIN_WHY = 20
ALLOW = "checks/size_allow.json"
RATCHET = "checks/size_ratchet.json"
SIZED_SUFFIXES = common.SOURCE_SUFFIXES + (".yml", ".yaml", ".toml", ".json")


def is_sized(rel: str) -> bool:
    return rel.endswith(SIZED_SUFFIXES) and "/fixtures/" not in f"/{rel}"


def line_counts(root: str) -> dict[str, int]:
    counts = {}
    for rel in common.tracked(root):
        text = common.read_text(root, rel) if is_sized(rel) else None
        if text is not None:
            counts[rel] = len(text.splitlines())
    return counts


def load_allow(root: str, limit: int = LIMIT) -> tuple[dict[str, int], dict[str, str], list[str]]:
    """-> (ceilings, reasons, errors).  A malformed entry is an error, never silently skipped."""
    path = os.path.join(root, ALLOW)
    if not os.path.exists(path):
        return {}, {}, []
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    ceilings, reasons, errors = {}, {}, []
    for rel, entry in sorted(raw.items()):
        ceiling = entry.get("ceiling") if isinstance(entry, dict) else None
        why = (entry.get("why") or "").strip() if isinstance(entry, dict) else ""
        if not isinstance(ceiling, int) or not limit < ceiling <= HARD_MAX:
            errors.append(f"{ALLOW}: {rel}: ceiling must be an integer in {limit + 1}..{HARD_MAX}, got {ceiling!r}")
        elif len(why) < MIN_WHY:
            errors.append(f"{ALLOW}: {rel}: needs a reason of at least {MIN_WHY} characters")
        else:
            ceilings[rel], reasons[rel] = ceiling, why
    return ceilings, reasons, errors


def soft_of(limit: int) -> int:
    """The split point: 80% of the limit (SOFT for the default limit)."""
    return limit * 4 // 5


def load_ratchet(root: str) -> dict[str, int]:
    path = os.path.join(root, RATCHET)
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        return {rel: n for rel, n in json.load(fh).items() if isinstance(n, int)}


def _ratchet_verdict(rel: str, n: int, ratchet: dict[str, int], soft: int) -> str | None:
    """A file over the split point: listed at exactly its length, or it is a violation."""
    if rel not in ratchet:
        return f"{rel}: {n} lines, over the {soft}-line split point and not in {RATCHET}; split it (a new file may not be ratcheted in)"
    if n > ratchet[rel]:
        return f"{rel}: {n} lines, past its ratchet of {ratchet[rel]} (a ratcheted file may only shrink)"
    if n < ratchet[rel]:
        return f"{rel}: {n} lines but {RATCHET} says {ratchet[rel]}; lower it (python checks/check_size.py --write-ratchet)"
    return None


def _file_verdict(rel: str, n: int, base: dict[str, int], limit: int, ratchet: dict[str, int]) -> tuple[str | None, str | None]:
    """-> (violation, note) for one file."""
    allowed = base.get(rel, limit)
    if n > allowed:
        return f"{rel}: {n} lines ({'ceiling %d' % allowed if rel in base else 'limit %d' % limit})", None
    if n > limit:
        return None, f"note: excepted file {rel}: {n} lines (ceiling {allowed})"
    # project rule: split at ~80% of the budget; since R8 L6 a file past it fails unless ratcheted, instead of printing a note
    return (_ratchet_verdict(rel, n, ratchet, soft_of(limit)), None) if n > soft_of(limit) else (None, None)


def judge(counts: dict[str, int], base: dict[str, int], limit: int, ratchet: dict[str, int] | None = None) -> tuple[list[str], list[str]]:
    """-> (violations, notes).  The exception list may only hold files that are still over the limit; the ratchet only files over the split point."""
    ratchet, soft = ratchet or {}, soft_of(limit)
    verdicts = [_file_verdict(rel, n, base, limit, ratchet) for rel, n in sorted(counts.items())]
    bad = [v for v, _ in verdicts if v]
    notes = [note for _, note in verdicts if note]
    bad += [f"{rel}: exception is stale (now {counts.get(rel, 'absent')} lines); remove it from {ALLOW}" for rel in sorted(base) if counts.get(rel, 0) <= limit]
    bad += [f"{rel}: ratchet row is stale (now {counts.get(rel, 'absent')} lines, split point {soft}); delete it from {RATCHET}" for rel in sorted(ratchet) if counts.get(rel, 0) <= soft]
    return bad, notes


def write_ratchet(root: str, counts: dict[str, int], limit: int) -> list[str]:
    """Lower or drop existing rows to the current lengths; NEVER add a row or raise one.  -> the changes made."""
    old, soft = load_ratchet(root), soft_of(limit)
    new = {rel: min(counts[rel], n) for rel, n in old.items() if counts.get(rel, 0) > soft}
    with open(os.path.join(root, RATCHET), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(dict(sorted(new.items())), fh, indent=2)
        fh.write("\n")
    return [f"{rel}: {old[rel]} -> {new.get(rel, 'deleted')}" for rel in sorted(old) if new.get(rel) != old[rel]]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--root", default=common.default_root())
    ap.add_argument("--limit", type=int, default=LIMIT)
    ap.add_argument("--write-ratchet", action="store_true", help="lower or drop ratchet rows to the current lengths (never adds)")
    args = ap.parse_args(argv)
    counts = line_counts(args.root)
    if args.write_ratchet:
        print("\n".join(write_ratchet(args.root, counts, args.limit)) or "check_size: ratchet already tight")
    ceilings, reasons, errors = load_allow(args.root, args.limit)
    bad, notes = judge(counts, ceilings, args.limit, load_ratchet(args.root))
    notes += [f"note: exception {rel}: {reasons[rel]}" for rel in sorted(reasons)]
    summary = f"note: {len(counts)} source file(s) checked, limit {args.limit}, hard maximum {HARD_MAX}"
    return common.finish("check_size", errors + bad, notes + [summary])


if __name__ == "__main__":
    sys.exit(main())
