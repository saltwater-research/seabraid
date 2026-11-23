#!/usr/bin/env python3
# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Every source file carries the SPDX license expression its path requires, in its first 5 lines.

OWNER: this checker owns the path -> expression table (RULES); scripts/add_spdx.py imports ``expected`` from here,
so the inserter and the checker can never disagree.  CONTRACT: a "source file" is .rs .c .h .cpp .hpp .cu .py .sh
.cmake, CMakeLists.txt or .cmake.in.  include/seabraid/*.h is the public ABI header and is dual-licensed
``Apache-2.0 OR MIT``; every other file of ours is ``Apache-2.0``.  *.patch files are diffs of someone else's
code and are exempt (their license is stated in NOTICE); anything under a ``fixtures/`` directory is test data.
SIZE: <=150 lines.  FAILURE: fail-fast and loud; exit 1 and ``path:1: expected <expr>, found <expr|missing>`` per file.
An unreadable source file is reported, not skipped.  CHECKER: tests/test_checkers.py plants a missing, a wrong
and a late (line 6) identifier.
Run:  python3 checks/check_spdx.py [--root DIR]
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as common  # noqa: E402

# (glob, expression); first match wins; the last row is the default.  fnmatch's "*" crosses "/", so ONLY the header dir is listed.
RULES = [("include/seabraid/*.h", "Apache-2.0 OR MIT"), ("*", "Apache-2.0")]
HEAD_LINES = 5
FOUND = re.compile(r"SPDX-License-Identifier:\s*(.*?)\s*(?:\*/|-->)?\s*$")


def expected(rel: str) -> str:
    return next(expr for glob, expr in RULES if fnmatch.fnmatch(rel, glob))


def found_expression(text: str) -> str | None:
    for line in text.splitlines()[:HEAD_LINES]:
        m = FOUND.search(line) if "SPDX-License-Identifier:" in line else None
        if m:
            return m.group(1)
    return None


def scan(root: str) -> list[str]:
    hits = []
    for rel in common.tracked(root):
        if not common.is_source(rel):
            continue
        text = common.read_text(root, rel)
        want, got = expected(rel), None if text is None else found_expression(text)
        if text is None:
            hits.append(f"{rel}:1: unreadable source file")
        elif got != want:
            hits.append(f"{rel}:1: expected {want}, found {got or 'missing'}")
    return hits


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--root", default=common.default_root())
    root = ap.parse_args(argv).root
    hits, n = scan(root), sum(1 for p in common.tracked(root) if common.is_source(p))
    return common.finish("check_spdx", hits, [f"note: {n} source file(s) checked"])


if __name__ == "__main__":
    sys.exit(main())
