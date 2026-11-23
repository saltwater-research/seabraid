#!/usr/bin/env python3
# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Fail if any tracked file still holds a fill-in-later marker (the word REPLACE, a hyphen, the word WITH).

OWNER: this checker owns the marker spelling; no other file lists it.  CONTRACT: one regex over each line of every tracked
text file; a hit is ``path:line: unfilled placeholder``.  WHY a checker and not a release-only gate: the marker that
motivated it (a fake DOI in .zenodo.json) sat in the tree for several commits and would have been published as real
citation metadata; a marker is a promise to fill a value, and a tree that ships must have none outstanding.
The pattern is assembled from two halves so this file, and the test that plants one, never contain it themselves.
SIZE: <=100 lines.  FAILURE: fail-fast and loud, exit 1, one line per hit.  No allow list: a legitimate use has to be reworded.
CHECKER: tests/test_sabotage.py (case ``placeholder-marker``) and tests/test_no_placeholders.py.
Run:  python3 checks/check_no_placeholders.py [--root DIR]
"""

from __future__ import annotations

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as common  # noqa: E402

MARKER = re.compile("REPLACE" + "-WITH")


def scan(root: str) -> list[str]:
    hits = []
    for rel in common.tracked(root):
        if rel.lower().endswith(common.BINARY_EXT):
            continue
        text = common.read_text(root, rel)
        if text is None:
            continue  # binary content has no lines to scan
        hits += [f"{rel}:{n}: unfilled placeholder" for n, line in enumerate(text.splitlines(), start=1) if MARKER.search(line)]
    return hits


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--root", default=common.default_root())
    return common.finish("check_no_placeholders", scan(ap.parse_args(argv).root))


if __name__ == "__main__":
    sys.exit(main())
