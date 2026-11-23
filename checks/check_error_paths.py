#!/usr/bin/env python3
# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Find places that swallow an error without recording it.  Project rule: every error is logged, none vanish.

OWNER: this checker owns the definition of "swallowed".  CONTRACT, per language:
  C/C++  a ``catch`` block with no log/rethrow/what() in it; a ``(void)`` cast of a function call.
  Python a bare ``except:``; an ``except ...:`` whose whole body is pass/continue/a bare constant (AST based).
  Rust   ``map_err(|_| ..)``; ``let _ = <call>``.
A line carrying ``allow-error-path: <reason>`` is exempt AND listed (``allowed: path:line: reason``).
MODES: runtime scopes (ERROR_SCOPES: src/ include/ adapters/ bindings/python/seabraid/ examples/ tools/ scripts/) must be ZERO under --strict-runtime;
without it they are only counted (other tracks are fixing those sites).  compat/strata/ and native-tests/ are a RATCHET: per-file counts
in checks/error_paths_baseline.json may shrink, never grow (--write-baseline rewrites it from the current tree).
SIZE: <=150 lines.  FAILURE: fail-fast and loud; exit 1 and ``path:line: kind`` per violation.  An unparseable
Python file is reported as a hit, never skipped.  CHECKER: tests/test_checkers.py (planted violation per language).
Run:  python3 checks/check_error_paths.py [--root DIR] [--strict-runtime] [--runtime-scope PREFIX ...] [--write-baseline]
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as common  # noqa: E402

BASELINE = "checks/error_paths_baseline.json"
# WHY more than common.RUNTIME_SCOPES (R8 L7): tools/ and scripts/ are shipped, run code whose swallowed errors hid a build fallback;
# the shared list stays the egress and no-local-refs definition of "the runtime".
ERROR_SCOPES = common.RUNTIME_SCOPES + ("tools/", "scripts/")
# WHY a ratchet for native-tests/ (C fixtures): one known catch-all (provider_smoke.cpp, a test provider that must not throw across the C boundary).
RATCHET_SCOPES = (common.COMPAT_SCOPE, "native-tests/")
ALLOW_TAG = re.compile(r"allow-error-path:\s*(\S.*)")
C_EXT = (".c", ".h", ".cpp", ".hpp", ".cc", ".cu", ".cuh")
# WHY mask: braces and keywords inside strings/comments must not steer block matching; masking keeps offsets and newlines.
MASK = re.compile(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'', re.S)
CATCH = re.compile(r"\bcatch\s*\([^)]*\)\s*\{")
VOID_CALL = re.compile(r"\(void\)\s*[\w:]+(?:(?:\.|->|::)\w+)*\s*\(")
LOG = re.compile(r"\bfprintf\s*\(|\bcerr\b|\bperror\b|\bfputs\s*\(|\bprintf\s*\(|\blog\w*\s*\(|\bLOG\w*|\bsyslog\b|\bthrow\b|\bwhat\s*\(|\bterminate\b|\babort\s*\(")
RUST_LINE = [(re.compile(r"\bmap_err\(\s*\|\s*_\s*\|"), "map_err-discards-error"), (re.compile(r"\blet\s+_\s*=\s*[^;]*\("), "let-underscore-call")]


def _mask(text: str) -> str:
    return MASK.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), text)


def _block_end(src: str, open_at: int) -> int:
    depth = 0
    for i in range(open_at, len(src)):
        depth += (src[i] == "{") - (src[i] == "}")
        if depth == 0:
            return i
    return len(src)


def c_hits(text: str) -> list[tuple[int, str]]:
    src, out = _mask(text), []
    for m in CATCH.finditer(src):
        if not LOG.search(src[m.end() : _block_end(src, m.end() - 1)]):
            out.append((src.count("\n", 0, m.start()) + 1, "catch-without-log"))
    out += [(src.count("\n", 0, m.start()) + 1, "void-cast-of-call") for m in VOID_CALL.finditer(src)]
    return out


def _silent(body: list[ast.stmt]) -> bool:
    return all(isinstance(s, (ast.Pass, ast.Continue)) or (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)) for s in body)


def py_hits(text: str) -> list[tuple[int, str]]:
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return [(exc.lineno or 1, "unparseable-python")]
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            kind = "bare-except" if node.type is None else "silent-except" if _silent(node.body) else None
            if kind:
                out.append((node.lineno, kind))
    return out


def rust_hits(text: str) -> list[tuple[int, str]]:
    return [(n, kind) for n, line in enumerate(text.splitlines(), 1) if not line.lstrip().startswith("//") for rx, kind in RUST_LINE if rx.search(line)]


def file_hits(rel: str, text: str) -> list[tuple[int, str]]:
    if rel.endswith(C_EXT):
        return c_hits(text)
    return py_hits(text) if rel.endswith(".py") else rust_hits(text) if rel.endswith(".rs") else []


def scan(root: str, runtime_scopes=ERROR_SCOPES):
    """-> (violations by file, allowed lines, per-file counts for runtime, per-file counts for compat)."""
    viol, allowed, runtime, compat = {}, [], {}, {}
    for rel in common.tracked(root):
        bucket = runtime if common.in_scope(rel, runtime_scopes) else compat if rel.startswith(RATCHET_SCOPES) else None
        text = common.read_text(root, rel) if bucket is not None and not rel.endswith(".patch") else None
        if text is None:
            continue
        lines = text.splitlines()
        for n, kind in file_hits(rel, text):
            tag = ALLOW_TAG.search(lines[n - 1]) if n <= len(lines) else None
            if tag:
                allowed.append(f"allowed: {rel}:{n}: {tag.group(1).strip()}")
            else:
                viol.setdefault(rel, []).append(f"{rel}:{n}: {kind}")
                bucket[rel] = bucket.get(rel, 0) + 1
    return viol, allowed, runtime, compat


def load_baseline(path: str, compat: dict[str, int], write: bool) -> dict[str, int]:
    if write:
        with open(path, "w", encoding="utf-8", newline="\n") as fh:  # LF on every OS: .gitattributes pins eol=lf
            fh.write(json.dumps(compat, indent=1, sort_keys=True) + "\n")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def ratchet(compat: dict[str, int], base: dict[str, int]) -> tuple[list[str], list[str]]:
    """(failures, notes): a file's count may shrink, never grow; a new file starts at zero."""
    fails = [f"{f}: {c} hit(s), baseline {base.get(f, 0)} (ratchet: may shrink, never grow)" for f, c in sorted(compat.items()) if c > base.get(f, 0)]
    notes = [f"note: {f} improved to {compat.get(f, 0)} (was {b}); run --write-baseline" for f, b in sorted(base.items()) if compat.get(f, 0) < b]
    return fails, notes


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--root", default=common.default_root())
    ap.add_argument("--strict-runtime", action="store_true", help="runtime scopes must have zero hits")
    ap.add_argument("--runtime-scope", action="append", help="override the runtime scope prefixes (repeatable)")
    ap.add_argument("--write-baseline", action="store_true", help="rewrite the compat ratchet from the current tree")
    args = ap.parse_args(argv)
    viol, allowed, runtime, compat = scan(args.root, tuple(args.runtime_scope or ERROR_SCOPES))
    base = load_baseline(os.path.join(args.root, BASELINE), compat, args.write_baseline)
    fails, notes = ratchet(compat, base)
    runtime_hits = [h for f in sorted(runtime) for h in viol[f]]
    summary = f"note: runtime hits {sum(runtime.values())} in {len(runtime)} file(s); compat hits {sum(compat.values())} in {len(compat)} file(s)"
    fails, notes = (fails + runtime_hits, notes) if args.strict_runtime else (fails, notes + runtime_hits)
    return common.finish("check_error_paths", fails, allowed + [summary] + notes)


if __name__ == "__main__":
    sys.exit(main())
