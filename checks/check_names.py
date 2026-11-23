#!/usr/bin/env python3
# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Post-rename gate: a retired project name must not appear anywhere in the tracked tree.

OWNER: the token rule is ``_common.name_regex`` (the same one scripts/rename.py rewrites with), so a name this gate
reports is exactly a name the renamer would have rewritten.  The only exceptions are whole English words in
``_common.EXCLUDED_WORDS`` and the files in ``checks/names_allow.txt``.
CONTRACT: ``--forbid NAME`` (repeatable) is checked, case-insensitively with no word boundary (kName, NAMEX, name_x all
hit), in the CONTENT and in the PATH of EVERY tracked or untracked-not-ignored file: README, docs, CMake, Cargo, NOTICE
and the patches included.  WHY every file: the earlier gate scanned only the runtime scopes and let five files through
(review B8).  Exceptions live in ``checks/names_allow.txt`` as ``path-glob<TAB>why``; a row without a why, or one whose
glob matches no file, is an error.  With no --forbid it exits 2 (usage): a gate with nothing to forbid must not look
green.  FAILURE: fail-fast and loud; exit 1 and ``path:line:col: forbidden name 'x'`` per hit; a file that cannot be
read is reported, not skipped.  SIZE: <=150 lines.
CHECKER: tests/test_checkers.py (the regex, the allow-list rules) and tests/test_sabotage.py (name in a path, in a
comment, as kName, and in README).
Run:  python3 checks/check_names.py --forbid seabraid [--root DIR]
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as common  # noqa: E402

ALLOW_FILE = "checks/names_allow.txt"


def load_allow(root: str) -> tuple[list[str], list[str]]:
    """(globs, errors).  Each row is ``glob<TAB>why``; the why is mandatory so every exception is defended in writing."""
    path = os.path.join(root, ALLOW_FILE)
    if not os.path.exists(path):
        return [], []
    globs, errors = [], []
    for n, raw in enumerate(open(path, encoding="utf-8"), start=1):
        if not raw.strip() or raw.startswith("#"):
            continue
        glob, _, why = raw.strip().partition("\t")
        if len(why.strip()) < 10:
            errors.append(f"{ALLOW_FILE}:{n}: row for {glob!r} needs a why (at least 10 characters)")
        globs.append(glob.strip())
    return globs, errors


def path_hits(rel: str, name: str) -> list[str]:
    rx = common.name_regex(name)
    return [f"{rel}: forbidden name {m.group(0)!r} in the path" for m in rx.finditer(rel) if not common.is_excluded(name, rel, m.start(), m.end())]


def text_hits(rel: str, text: str, name: str) -> list[str]:
    rx, out = common.name_regex(name), []
    for n, line in enumerate(text.splitlines(), start=1):
        out += [
            f"{rel}:{n}:{m.start() + 1}: forbidden name {m.group(0)!r}"
            for m in rx.finditer(line)
            if not common.is_excluded(name, line, m.start(), m.end())
        ]
    return out


def binary_hits(root: str, rel: str, name: str) -> list[str]:
    """A binary cannot be tokenised, so any occurrence of the bytes counts; an unreadable one is a hit too."""
    try:
        with open(os.path.join(root, rel), "rb") as fh:
            raw = fh.read().lower()
    except OSError as exc:
        return [f"{rel}: unreadable ({exc}); cannot prove the name is absent"]
    return [f"{rel}: binary file contains the bytes of forbidden name {name!r}"] if name.lower().encode() in raw else []


def scan(root: str, names: list[str]) -> list[str]:
    allow, errors = load_allow(root)
    files = common.tracked(root)
    hits = list(errors) + [f"{ALLOW_FILE}: glob {g!r} matches no file" for g in allow if not any(fnmatch.fnmatch(f, g) for f in files)]
    for rel in files:
        if any(fnmatch.fnmatch(rel, g) for g in allow):
            continue
        text = None if rel.lower().endswith(common.BINARY_EXT) else common.read_text(root, rel)
        for name in names:
            hits += path_hits(rel, name)
            hits += text_hits(rel, text, name) if text is not None else binary_hits(root, rel, name)
    return hits


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=common.default_root())
    ap.add_argument("--forbid", action="append", required=True, metavar="NAME", help="a retired name (repeatable)")
    a = ap.parse_args(argv)
    hits = scan(a.root, a.forbid)
    return common.finish("check_names", hits, [f"note: forbidding {', '.join(a.forbid)} in every tracked file except {ALLOW_FILE}"])


if __name__ == "__main__":
    sys.exit(main())
