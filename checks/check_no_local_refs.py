#!/usr/bin/env python3
# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Fail if any tracked file contains a machine-specific value: a home path, a private IP, a local mount, a time of day.

OWNER: this checker owns the repo-wide list of generic machine-specific SHAPES (regexes).  It holds no names.
CONTRACT: a shape is a regex over one line; a hit is reported as ``path:line:col: kind``.  The shapes are home
directories, private IPv4 ranges, local mounts, an ssh target with a numeric host name, a time of day attached to a
date, a run id that embeds HHMMSS, a GPU UUID (GPU-xxxxxxxx-xxxx-), a dated attribution to a person (a parenthesis, the word owner, a year), and the maintainers' machine
used as a place name (the word box before job or run; their desktop's source archive).
SIZE: <=150 lines.  FAILURE: fail-fast and loud; exit 1 with one line per hit.  An unreadable tracked file is skipped
only when it is binary; git itself reports a missing one.
Exceptions live in ``checks/local_refs_allow.txt`` as ``path-glob<TAB>kind``; an entry matching nothing is an error.
CHECKER: tests/test_no_local_refs.py (clean tree, a planted positive and a clean negative for every shape).
Run:  python3 checks/check_no_local_refs.py [--root DIR]
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as common  # noqa: E402

SHAPES = [
    ("home-path", re.compile(r"/home/(?!user\b|me\b|<)[A-Za-z][\w.-]*|/Users/[A-Za-z][\w.-]*|[A-Za-z]:\\+Users\\+[A-Za-z]")),
    ("data-mount", re.compile(r"/media/[A-Za-z][\w.-]*|/mnt/[A-Za-z][\w.-]*/")),
    ("ssh-alias", re.compile(r"\bssh(?:\s+-\S+)*\s+[a-z]+\d{3,}\b")),
    (
        "private-ip",
        re.compile(
            r"(?<![\d.])(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}"
            r"|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}|100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d{1,3}\.\d{1,3})(?![\d.]*\d)"
        ),
    ),
    # WHY: a date alone is fine, but "date + clock time" or "date + part of day" dates a work session.
    ("time-of-day", re.compile(r"20\d\d-\d\d-\d\d[ T]~?\d{1,2}:\d\d|\(20\d\d-\d\d-\d\d (?:morning|midday|afternoon|evening|night)\)")),
    ("run-id-time", re.compile(r"20\d{6}-\d{6}")),
    # WHY: a GPU UUID identifies one physical card; the repo names devices by index or a --var, never by serial.
    ("gpu-uuid", re.compile(r"GPU-[0-9a-f]{8}-[0-9a-f]{4}-")),
    # WHY these two and not "the box" / "owner" / "independent audit": those words are legitimate shipped text (OWNER: is the
    # design-gate header, "the owner table" is the placement vocabulary, "independent audit" is the audit's cited name), so a
    # bare pattern would flag them.  What is never legitimate is a dated attribution to a person and the maintainers' own
    # machine used as a place name for a run (the word box before job or run) or the archive they keep on one.
    ("dated-owner", re.compile(r"\(owner, ?20\d\d")),
    ("internal-vocab", re.compile(r"\bbox (?:job|run)\b|\bdesktop source archive\b")),
]
ALLOW_FILE = "checks/local_refs_allow.txt"


def line_hits(line: str) -> list[tuple[int, str]]:
    """(column, kind) for the first match of every shape on one line."""
    return [(m.start(), kind) for kind, rx in SHAPES for m in [rx.search(line)] if m]


def load_allow(root: str) -> list[tuple[str, str]]:
    path = os.path.join(root, ALLOW_FILE)
    if not os.path.exists(path):
        return []
    rows = [raw.strip().partition("\t") for raw in open(path, encoding="utf-8") if raw.strip() and not raw.startswith("#")]
    return [(g.strip(), k.strip()) for g, _, k in rows]


def scan(root: str) -> list[str]:
    allow, used, hits = load_allow(root), set(), []
    for rel in common.tracked(root):
        if rel.lower().endswith(common.BINARY_EXT):
            continue
        text = common.read_text(root, rel)
        if text is None:
            continue  # binary or unreadable content has no lines to scan; a named-but-unreadable file is caught by git itself
        for n, line in enumerate(text.splitlines(), start=1):
            for col, kind in line_hits(line):
                ok = [i for i, (g, k) in enumerate(allow) if fnmatch.fnmatch(rel, g) and k in ("", kind)]
                used.update(ok)
                if not ok:
                    hits.append(f"{rel}:{n}:{col + 1}: {kind}")
    return hits + [f"{ALLOW_FILE}: unused allowlist entry: {allow[i][0]} {allow[i][1]}" for i in range(len(allow)) if i not in used]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--root", default=common.default_root())
    return common.finish("check_no_local_refs", scan(ap.parse_args(argv).root))


if __name__ == "__main__":
    sys.exit(main())
