#!/usr/bin/env python3
# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""The dependency edge is one-way: the core and its hosts never import or path-reference compat/.

OWNER: this checker owns the list of core scopes and the shape of a compat reference; checks/compat_edges_allow.txt
owns the (few) documented exceptions.  The rule itself is MAP.md's "core never depends on the add-on".
CONTRACT: no file of a core scope (adapters/ bindings/ scripts/ src/ include/ examples/ tools/ cmake/ native-tests/)
may contain ``compat/`` as a path component, ``"compat"`` as a quoted path part, or ``import compat`` / ``from compat``.
Only code and build files are read (source suffixes, .json/.toml/.yml/.cmake, CMakeLists.txt): prose (.md, .txt) and
.patch files describe the add-on and may name it.  WHY scripts/ is a core scope: add-on qualification tooling that
imports compat tools lives under compat/strata/scripts/, not beside the core's own scripts.
Exceptions: ``checks/compat_edges_allow.txt`` rows ``path<TAB>why`` (why >= 10 characters).  A row whose file no longer
has a reference is an error, so an exception cannot outlive its cause.  A BUILD-time dependency (an adapter that needs
a compat-patched Strata tree via STRATA_SOURCE) is not a reference of this shape; if one ever names compat/ in a
CMake file, it gets an allow row saying so.
SIZE: <=150 lines.  FAILURE: fail-fast and loud; exit 1 and ``path:line:col: compat reference`` per hit.
CHECKER: tests/test_checkers.py (planted import, path join, allow-list rules) and tests/test_sabotage.py (an import
planted in src/).
Run:  python3 checks/check_compat_edges.py [--root DIR]
"""

from __future__ import annotations

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as common  # noqa: E402

CORE_SCOPES = ("adapters/", "bindings/", "scripts/", "src/", "include/", "examples/", "tools/", "cmake/", "native-tests/")
CONFIG_SUFFIXES = (".json", ".toml", ".yml", ".yaml", ".cmake", ".in")
REFERENCE = re.compile(r"(?<![\w-])compat/|[\"']compat[\"']|^\s*(?:from|import)\s+compat\b")
ALLOW_FILE = "checks/compat_edges_allow.txt"


def is_code(rel: str) -> bool:
    return (common.is_source(rel) or rel.endswith(CONFIG_SUFFIXES)) and not rel.endswith(".patch")


def load_allow(root: str) -> tuple[dict[str, str], list[str]]:
    path, rows, errors = os.path.join(root, ALLOW_FILE), {}, []
    if not os.path.exists(path):
        return rows, errors
    for n, raw in enumerate(open(path, encoding="utf-8"), start=1):
        if not raw.strip() or raw.startswith("#"):
            continue
        rel, _, why = raw.strip().partition("\t")
        if len(why.strip()) < 10:
            errors.append(f"{ALLOW_FILE}:{n}: row for {rel!r} needs a why (at least 10 characters)")
        rows[rel.strip()] = why.strip()
    return rows, errors


def scan(root: str) -> list[str]:
    allow, hits = load_allow(root)
    hits = list(hits)
    used: set[str] = set()
    for rel in common.tracked(root):
        if not (common.in_scope(rel, CORE_SCOPES) and is_code(rel)):
            continue
        text = common.read_text(root, rel)
        found = [] if text is None else [(n, m.start() + 1) for n, line in enumerate(text.splitlines(), 1) for m in [REFERENCE.search(line)] if m]
        if rel in allow:
            if found:
                used.add(rel)
            continue
        hits += [f"{rel}:{n}:{col}: compat reference from a core scope (move the code under compat/strata/ or drop the reference)" for n, col in found]
    return hits + [f"{ALLOW_FILE}: row for {rel!r} matches no compat reference; delete it" for rel in allow if rel not in used]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=common.default_root())
    return common.finish("check_compat_edges", scan(ap.parse_args(argv).root))


if __name__ == "__main__":
    sys.exit(main())
