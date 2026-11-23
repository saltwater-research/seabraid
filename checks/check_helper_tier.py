#!/usr/bin/env python3
# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Every Strata call site that asks ``seabraid()`` (named-channel helper only) must say why it is not ``process_tier()``.

WHY this gate exists: Strata knows two helper processes, the named-channel helper (``seabraid()``) and the private
Seabraid helper (``seabraid_tier()``); ``process_tier()`` is either.  On 2026-11-17 the RAM-copy rule still asked
``seabraid()``, so the private helper's experts were left out of RAM and every long prompt re-read them from the
drive (half the prompt speed).  Nothing flagged it, because a ``seabraid()`` test compiles and runs either way.
OWNER: this checker owns the rule; the reasons live in checks/helper_tier_allow.txt, one site per row.
CONTRACT: the patch stack is the compat series (compat/strata/patches/strata/*.patch, in name order) and then
adapters/strata/strata-v7-seabraid.patch.  A call site is an added line that calls ``.seabraid()`` or
``->seabraid()`` and that no later patch removes.  Each must match exactly one allow row, ``<fragment> | <why>``,
by substring; a reason has at least MIN_WHY characters.  A row that matches no site, or several, fails, so the
list cannot rot.  FAILURE: fail-fast, exit 1, one ``patch:line: text`` per unexplained site.  SIZE: <=150 lines.
CHECKER: tests/test_sabotage.py (a new ``seabraid()`` site; a stale row).
Run:  python3 checks/check_helper_tier.py [--root DIR]
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

SERIES = "compat/strata/patches/strata/*.patch"
LAST = "adapters/strata/strata-v7-seabraid.patch"
ALLOW = "checks/helper_tier_allow.txt"
MIN_WHY = 20
CALL = re.compile(r"(\.|->)seabraid\(\)")


def patch_stack(root: str) -> list[str]:
    stack = sorted(os.path.relpath(p, root).replace(os.sep, "/") for p in glob.glob(os.path.join(root, SERIES)))
    if os.path.exists(os.path.join(root, LAST)):
        stack.append(LAST)
    return stack


def diff_lines(lines: list[str]) -> tuple[list[tuple[int, str]], set[str]]:
    """-> ([(line, text)] of added ``seabraid()`` calls, {text} of every removed line) for one patch."""
    removed = {ln[1:].strip() for ln in lines if ln.startswith("-") and not ln.startswith("---")}
    added = [(n, ln[1:].strip()) for n, ln in enumerate(lines, 1)
             if ln.startswith("+") and not ln.startswith("+++") and CALL.search(ln)]
    return added, removed


def call_sites(root: str) -> list[tuple[str, int, str]]:
    """-> (patch, line, text) of every surviving added ``seabraid()`` call, in stack order."""
    sites: list[tuple[str, int, str]] = []
    for rel in patch_stack(root):
        with open(os.path.join(root, rel), encoding="utf-8", errors="replace") as fh:
            added, removed = diff_lines(fh.read().splitlines())
        sites = [s for s in sites if s[2] not in removed] + [(rel, n, text) for n, text in added]
    return sites


def load_allow(root: str) -> tuple[list[tuple[int, str, str]], list[str]]:
    """-> ([(row line, fragment, why)], errors).  A malformed row is an error, never skipped."""
    path = os.path.join(root, ALLOW)
    if not os.path.exists(path):
        return [], []
    rows, errors = [], []
    with open(path, encoding="utf-8") as fh:
        for n, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            frag, sep, why = (part.strip() for part in line.partition(" | "))
            if not sep or not frag:
                errors.append(f"{ALLOW}:{n}: expected '<fragment> | <why>'")
            elif len(why) < MIN_WHY:
                errors.append(f"{ALLOW}:{n}: needs a reason of at least {MIN_WHY} characters")
            else:
                rows.append((n, frag, why))
    return rows, errors


def check(root: str) -> list[str]:
    sites = call_sites(root)
    rows, errors = load_allow(root)
    explained: set[int] = set()
    for n, frag, _why in rows:
        hits = [i for i, s in enumerate(sites) if frag in s[2]]
        if len(hits) != 1:
            errors.append(f"{ALLOW}:{n}: fragment matches {len(hits)} call sites, needs exactly 1: {frag!r}")
        explained.update(hits)
    for i, (rel, n, text) in enumerate(sites):
        if i not in explained:
            errors.append(f"{rel}:{n}: seabraid() asks for the named-channel helper only; use process_tier() or add a "
                          f"row to {ALLOW} saying why the private helper is excluded: {text}")
    return errors


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    args = ap.parse_args(argv)
    errors = check(args.root)
    for e in errors:
        print(e)
    if errors:
        print(f"check_helper_tier: {len(errors)} problem(s)")
        return 1
    for n, frag, why in load_allow(args.root)[0]:
        print(f"check_helper_tier: allowed {frag!r}: {why}")
    print(f"check_helper_tier: ok ({len(call_sites(args.root))} seabraid() call sites, all explained)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
