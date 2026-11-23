#!/usr/bin/env python3
# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Insert the copyright + SPDX header into every source file that lacks one.  Idempotent.

OWNER: the expression per path is owned by ``checks/check_spdx.py`` (imported, never copied); this script only owns
WHERE and HOW the two lines are written.  CONTRACT: a file whose first 5 lines already hold an SPDX identifier is
left untouched (a WRONG expression is reported, never silently rewritten).  The header goes after a shebang and,
for Python, after a PEP 263 encoding line, so both stay where the interpreter needs them.  Comment syntax:
``//`` for Rust/C/C++/CUDA, ``#`` for Python/shell/CMake.  A UTF-8 BOM and the file's own CRLF/LF are preserved.
Without --write nothing is changed: it lists what WOULD change (the safe default for a tree others are editing).
FAILURE: unreadable or undecodable files are reported and make the exit code 1; nothing is skipped silently.
Run:  python3 scripts/add_spdx.py --root DIR [--write]
"""

from __future__ import annotations

import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "checks"))
import _common as common  # noqa: E402
import check_spdx as spdx  # noqa: E402

COPYRIGHT = "Copyright 2026 Saltwater Research LLC"
SLASH = (".rs", ".c", ".h", ".cpp", ".hpp", ".cu")
ENCODING = re.compile(r"^[ \t\f]*#.*?coding[:=][ \t]*[-\w.]+")


def comment_prefix(rel: str) -> str:
    return "//" if rel.endswith(SLASH) else "#"


def insert_at(lines: list[str], rel: str) -> int:
    """Index of the first line the header may occupy: after a shebang, then (Python only) an encoding line."""
    # WHY "#![": a Rust inner attribute starts with "#!" too, but it is code, not an interpreter line.
    at = 1 if lines and lines[0].startswith("#!") and not lines[0].startswith("#![") else 0
    if rel.endswith(".py") and at < len(lines) and ENCODING.match(lines[at]):
        at += 1
    return at


def with_header(rel: str, text: str) -> str | None:
    """The text with the header inserted, or None when it already has an SPDX identifier (nothing to do)."""
    bom, body = ("﻿", text[1:]) if text.startswith("﻿") else ("", text)
    nl = "\r\n" if "\r\n" in body else "\n"
    lines = body.split(nl)
    if spdx.found_expression(body) is not None:
        return None
    c = comment_prefix(rel)
    at = insert_at(lines, rel)
    lines[at:at] = [f"{c} {COPYRIGHT}", f"{c} SPDX-License-Identifier: {spdx.expected(rel)}"]
    return bom + nl.join(lines)


def run(root: str, write: bool) -> tuple[list[str], list[str]]:
    """-> (paths changed or that would change, problems)."""
    changed, problems = [], []
    for rel in common.tracked(root):
        if not common.is_source(rel):
            continue
        path = os.path.join(root, rel)
        try:
            with open(path, "rb") as fh:
                text = fh.read().decode("utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            problems.append(f"{rel}: cannot read: {exc}")
            continue
        new = with_header(rel, text)
        got = spdx.found_expression(text)
        if got is not None and got != spdx.expected(rel):
            problems.append(f"{rel}: has SPDX {got!r}, expected {spdx.expected(rel)!r}; not rewritten")
        if new is not None:
            changed.append(rel)
            if write:
                with open(path, "wb") as fh:
                    fh.write(new.encode("utf-8"))
    return changed, problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--root", required=True, help="tree to edit (required: never defaults to the live repo)")
    ap.add_argument("--write", action="store_true", help="apply the change; default is a dry run")
    args = ap.parse_args(argv)
    changed, problems = run(os.path.abspath(args.root), args.write)
    for rel in changed:
        print(("updated: " if args.write else "would update: ") + rel)
    for p in problems:
        print("problem: " + p)
    print(f"add_spdx: {len(changed)} file(s) {'updated' if args.write else 'would change'}, {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
