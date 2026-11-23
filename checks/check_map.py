#!/usr/bin/env python3
# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""MAP.md must describe the tree that exists: its modules, its checkers, its script dependencies and where unsafe may live (R8 L12).

OWNER: this checker owns the rule; MAP.md owns the prose.  CONTRACT, five checks, each a failure with ``MAP.md: <what>``:
  1. every top-level directory, and every ``src/<name>.rs`` and ``src/<name>/``, is named in MAP.md (in backticks);
  2. every backticked path in MAP.md that starts with a shipped directory exists (a renamed module cannot leave a ghost row);
  3. the ``checks/`` row names every ``checks/check_*.py`` by its stem (the row once listed topics, and missed check_links);
  4. the ``scripts/`` row names every third-party module the scripts import (``checks/`` counts for the checker modules; a module in the importing file's own directory is internal);
  5. the ``allow(unsafe_code)`` ATTRIBUTE appears in a .rs file only where the paragraph "appears only in:" lists it (the attribute
     on ``mod ffi`` in src/lib.rs is what lets the whole ffi tree use unsafe), and every path it lists still holds one (so the claim "unsafe only in ffi" cannot go false again, and a path cleaned up must be deleted from the list).
WHY a checker and not review: the first MAP.md listed 6 of 13 src files and claimed unsafe lived only in ffi while src/diag.rs
held a C-callable sink; both were found by an auditor, not by anything that runs.
FAILURE: fail-fast and loud; exit 1 and one line per problem.  SIZE: <=150 lines.  CHECKER: tests/test_map_checker.py.
Run:  python3 checks/check_map.py [--root DIR]
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as common  # noqa: E402

MAP = "MAP.md"
TICKED = re.compile(r"`([^`\n]+)`")
ATTRIBUTE = re.compile(r"#!?\[allow\([^)]*unsafe_code")
UNSAFE_LIST = re.compile(r"appears only in:(.*?)\n\s*\n", re.S)


def _names(text: str) -> set[str]:
    return {m.group(1).strip() for m in TICKED.finditer(text)}


def _top_dirs(files: list[str]) -> set[str]:
    return {rel.split("/")[0] for rel in files if "/" in rel}


def _named(mapped: set[str], *prefixes: str) -> bool:
    return any(t.startswith(p) for t in mapped for p in prefixes)


def _exists(files: list[str], path: str) -> bool:
    base = path.rstrip("/")
    return any(f == base or f.startswith(base + "/") for f in files)


def tree_problems(files: list[str], mapped: set[str]) -> list[str]:
    src = {m.group(1) for f in files if (m := re.match(r"src/([^/.]+)", f))}
    bad = [f"{MAP}: top-level directory {d}/ is not described" for d in sorted(_top_dirs(files)) if not _named(mapped, d + "/")]
    return bad + [f"{MAP}: src module `src/{n}.rs` / `src/{n}/` is not described" for n in sorted(src) if not _named(mapped, f"src/{n}.rs", f"src/{n}/")]


def ghost_problems(files: list[str], mapped: set[str]) -> list[str]:
    tops = _top_dirs(files)
    paths = [t for t in sorted(mapped) if re.fullmatch(r"[\w./-]+", t) and "/" in t and t.split("/")[0] in tops]
    return [f"{MAP}: `{t}` is named but does not exist" for t in paths if not _exists(files, t)]


def _imported(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Import):
        return [a.name for a in node.names]
    if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
        return [node.module]
    return []


def _siblings(files: list[str], rel: str) -> set[str]:
    """Module names of the .py files in ``rel``'s own directory.  WHY: scripts/release/publish.py importing rel_core is the
    package's own code, not a dependency."""
    return {os.path.basename(f)[:-3] for f in files if f.endswith(".py") and os.path.dirname(f) == os.path.dirname(rel)}


def _third_party_imports(root: str, files: list[str]) -> set[str]:
    found: set[str] = set()
    for rel in (f for f in files if f.startswith("scripts/") and f.endswith(".py")):
        for node in ast.walk(ast.parse(common.read_text(root, rel) or "")):
            found |= {n.split(".")[0] for n in _imported(node)} - _siblings(files, rel)
    return found - set(sys.stdlib_module_names)


def _row(text: str, key: str) -> str:
    return next((line for line in text.splitlines() if line.startswith(f"| `{key}`")), "")


def row_problems(root: str, files: list[str], text: str) -> list[str]:
    stems = sorted(os.path.basename(f)[:-3] for f in files if re.fullmatch(r"checks/check_\w+\.py", f))
    bad = [f"{MAP}: the `checks/` row does not name {s}" for s in stems if s not in _row(text, "checks/")]
    # a module that lives in checks/ (add_spdx imports check_spdx) is a dependency on `checks/`, not on a package
    wanted = {"checks/" if os.path.exists(os.path.join(root, "checks", n + ".py")) else n for n in _third_party_imports(root, files)}
    return bad + [f"{MAP}: the `scripts/` row does not name its dependency {n}" for n in sorted(wanted) if n not in _row(text, "scripts/")]


def _has_attribute(text: str) -> bool:
    """A real ``#[allow(unsafe_code)]`` / ``#![allow(unsafe_code)]`` on a non-comment line (prose that names it does not count)."""
    return any(ATTRIBUTE.search(line) for line in text.splitlines() if not line.lstrip().startswith("//"))


def _covered(path: str, listed: str) -> bool:
    return path == listed or (listed.endswith("/") and path.startswith(listed))


def unsafe_problems(root: str, files: list[str], text: str) -> list[str]:
    m = UNSAFE_LIST.search(text)
    if not m:
        return [f"{MAP}: no paragraph 'allow(unsafe_code) appears only in: ...' (the list the unsafe check reads)"]
    allowed = sorted(t for t in _names(m.group(1)) if t.endswith((".rs", "/")))  # paths only; `mod ffi` and the like are prose
    holders = [f for f in files if f.endswith(".rs") and _has_attribute(common.read_text(root, f) or "")]
    bad = [f"{f}: allow(unsafe_code) outside the places {MAP} lists" for f in holders if not any(_covered(f, a) for a in allowed)]
    return bad + [f"{MAP}: `{a}` is listed for allow(unsafe_code) but holds none; delete it from the list" for a in allowed if not any(_covered(f, a) for f in holders)]


def check(root: str) -> list[str]:
    if not os.path.exists(os.path.join(root, MAP)):
        return [f"{MAP}: missing"]
    text = common.read_text(root, MAP) or ""
    files = [f for f in common.tracked(root) if f != MAP]
    mapped = _names(text)
    return tree_problems(files, mapped) + ghost_problems(files, mapped) + row_problems(root, files, text) + unsafe_problems(root, files, text)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--root", default=common.default_root())
    return common.finish("check_map", check(ap.parse_args(argv).root))


if __name__ == "__main__":
    sys.exit(main())
