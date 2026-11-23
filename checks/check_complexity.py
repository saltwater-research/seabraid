#!/usr/bin/env python3
# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Cyclomatic complexity gate: CCN <= 10 per function, with a short, documented, ratcheting exception list.

OWNER: this checker owns the complexity rule and reads checks/complexity_allow.txt; CI calls it and never lizard.
CONTRACT: scope = SCOPES below (every top-level directory that ships code; lizard parses the code files it knows, so
*.sh is ignored), and ``unscoped`` FAILS the run when a tracked code file sits outside SCOPES, so "every shipped dir is in
scope" is derived from the tree, not claimed.  Exception rows are ``path<TAB>function<TAB>ceiling<TAB>why``; a row is valid
only if ceiling is 11..HARD_MAX, the why is >= MIN_WHY chars, the function exists, currently exceeds DEFAULT (else the row
is stale: the list can only shrink), and is <= its ceiling (it may not grow); duplicate rows fail.  A function above
HARD_MAX can never be excepted; the ones that exist today (code that predates a scope) sit in checks/complexity_ratchet.txt
instead: ceiling > HARD_MAX and EQUAL to the current CCN, so a new violation fails, a split must lower the row, and the
row says which track splits it.
FAILURE: fail-fast; exit 1 with ``path:line: function CCN n > limit m`` per violation; exit 2 if lizard is missing.
SIZE: <=170 lines.  CHECKER: tests/test_complexity.py (temp trees for each rule, plus the real tree).
Run:  python3 checks/check_complexity.py [--root DIR]
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import importlib.util
import io
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as common  # noqa: E402

# WHY 10: the project standard; one reader can hold ~10 paths through a function at 2 a.m.
DEFAULT = 10
# WHY 15, never higher: the owner's decision; an exception buys room for a flat validator or dispatch, not for an
# orchestrator that should be split.  Rows above this are rejected, so the escape hatch has a floor under it.
HARD_MAX = 15
MIN_WHY = 20
# WHY pinned: CCN differs between lizard releases (1.24 flags Rust functions 1.26.1 does not), so the gate's yardstick is the
# version CI installs; any other version still runs but says so, and the real-tree test skips rather than judge by it.
LIZARD_PIN = "1.26.1"
ALLOW = "checks/complexity_allow.txt"
RATCHET = "checks/complexity_ratchet.txt"
# WHY every top-level code directory (R8 M1): the earlier list named subdirectories and silently skipped the rest of the
# tree (compat/strata/strata-ext held a 253-line main at CCN 86).  A directory is a scope as a whole; ``unscoped`` proves
# that no tracked code file lives outside these.
SCOPES = ("src", "include", "adapters", "bindings", "tools", "compat", "native-tests", "checks", "scripts", "examples",
          "tests", "build.rs", "docs/paper", "docs/img")  # docs/paper, docs/img: the figure generators are shipped code
# The suffixes lizard parses that this repository ships (anything else under a scope is data or shell).
CODE_SUFFIXES = (".py", ".rs", ".c", ".h", ".cpp", ".hpp", ".cc", ".cu", ".cuh")


def measure(root: str, scopes: tuple[str, ...] = SCOPES) -> dict[tuple[str, str], tuple[int, int]]:
    """-> {(repo-relative path, function): (max CCN, line of that worst instance)}.

    WHY ``--modified`` (-m): a switch/match counts as one decision; an enum or error-code mapping would otherwise
    inflate the score with branches a reader scans in one glance rather than holds as separate paths.
    WHY the CLI + --csv, not the library: the CLI is the documented stable surface and is what CI pinned before.
    """
    dirs = [os.path.join(root, s) for s in scopes if os.path.exists(os.path.join(root, s))]
    if not dirs:
        return {}
    out = subprocess.run([sys.executable, "-m", "lizard", "--csv", "-m", *dirs], capture_output=True, text=True, check=True).stdout
    found: dict[tuple[str, str], tuple[int, int]] = {}
    for row in csv.reader(io.StringIO(out)):
        ccn, name, path, line = int(row[1]), row[7], row[6], int(row[9])
        key = (os.path.relpath(path, root).replace("\\", "/"), name)
        if key not in found or ccn > found[key][0]:
            found[key] = (ccn, line)
    return found


def unscoped(root: str) -> list[str]:
    """Tracked code files that no SCOPES entry covers: a new top-level directory must be added to SCOPES, not skipped."""
    inside = lambda rel: any(rel == s or rel.startswith(s + "/") for s in SCOPES)  # noqa: E731
    return [rel for rel in common.tracked(root) if rel.endswith(CODE_SUFFIXES) and not inside(rel)]


def _row_errors(where: str, func: str, ceiling: int, why: str, floor: int, ceiling_max: int | None) -> list[str]:
    errs = []
    if not floor < ceiling or (ceiling_max is not None and ceiling > ceiling_max):
        errs.append(f"{where}: ceiling {ceiling} for {func} must be {floor + 1}..{ceiling_max or 'any'}")
    if len(why) < MIN_WHY:
        errs.append(f"{where}: why for {func} must be at least {MIN_WHY} chars")
    return errs


def load_rows(root: str, rel: str, floor: int, ceiling_max: int | None) -> tuple[dict[tuple[str, str], int], list[str]]:
    """-> ({(path, function): ceiling}, row errors).  Every rule on the row itself is enforced here.

    Allow rows take ceilings DEFAULT+1..HARD_MAX; ratchet rows (floor HARD_MAX, no maximum) hold what is above that.
    """
    rows: dict[tuple[str, str], int] = {}
    errs: list[str] = []
    path = os.path.join(root, rel)
    if not os.path.exists(path):
        return rows, errs
    with open(path, encoding="utf-8") as fh:
        for n, raw in enumerate(fh, 1):
            if not raw.strip() or raw.lstrip().startswith("#"):
                continue
            parts = raw.rstrip("\r\n").split("\t")
            if len(parts) != 4 or not parts[2].strip().isdigit():
                errs.append(f"{rel}:{n}: want path<TAB>function<TAB>ceiling<TAB>why")
                continue
            key, ceiling = (parts[0].strip(), parts[1].strip()), int(parts[2])
            errs += _row_errors(f"{rel}:{n}", key[1], ceiling, parts[3].strip(), floor, ceiling_max)
            errs += [f"{rel}:{n}: duplicate row for {key[0]} {key[1]}"] if key in rows else []
            rows[key] = ceiling
    return rows, errs


def _verdict(key: tuple[str, str], ccn: int, line: int, rows: dict[tuple[str, str], int], ratchet: dict[tuple[str, str], int]) -> str | None:
    """What is wrong with one measured function, or None."""
    file, func = key
    limit = ratchet.get(key, rows.get(key, DEFAULT))
    if ccn > limit:
        return f"{file}:{line}: {func} CCN {ccn} > limit {limit}" + (" (grew past its row)" if key in rows or key in ratchet else "")
    if key in rows and ccn <= DEFAULT:
        return f"{ALLOW}: stale row for {file} {func} (CCN is now {ccn} <= {DEFAULT}); delete it"
    if key in ratchet and ccn < limit:
        fix = f"lower the row to {ccn}" if ccn > HARD_MAX else f"delete it (use the allow file if still > {DEFAULT})"
        return f"{RATCHET}: {file} {func} is now CCN {ccn}, row says {limit}; {fix}"
    return None


def judge(found: dict[tuple[str, str], tuple[int, int]], rows: dict[tuple[str, str], int],
          ratchet: dict[tuple[str, str], int] | None = None) -> list[str]:
    ratchet = ratchet or {}
    bad = [v for key, (ccn, line) in sorted(found.items()) if (v := _verdict(key, ccn, line, rows, ratchet))]
    for name, table in ((ALLOW, rows), (RATCHET, ratchet)):
        bad += [f"{name}: row for {file} {func} matches no function" for (file, func) in sorted(table) if (file, func) not in found]
    return bad




def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    args = ap.parse_args(argv)
    if importlib.util.find_spec("lizard") is None:
        print("check_complexity: lizard is not installed (pip install lizard==1.26.1)", file=sys.stderr)
        return 2
    have = importlib.metadata.version("lizard")
    if have != LIZARD_PIN:
        print(f"check_complexity: note: lizard {have} is not the pinned {LIZARD_PIN}; counts may differ from CI", file=sys.stderr)
    rows, bad = load_rows(args.root, ALLOW, DEFAULT, HARD_MAX)
    ratchet, more = load_rows(args.root, RATCHET, HARD_MAX, None)
    found = measure(args.root)
    bad += more + [f"{rel}: code file outside SCOPES (add its directory to SCOPES in checks/check_complexity.py)" for rel in unscoped(args.root)]
    bad += judge(found, rows, ratchet)
    for line in bad:
        print(line)
    print(f"check_complexity: {len(found)} function(s), {len(rows)} exception row(s), {len(ratchet)} ratchet row(s), {len(bad)} violation(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
