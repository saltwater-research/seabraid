#!/usr/bin/env python3
# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Tripwire for network use in the tracked tree: a loud, cheap check that the runtime does not phone home.

OWNER: this checker owns the "no egress" rule; the pattern tables are in _egress_patterns.py, the per-language comment
stripping and patch extraction in _egress_lang.py.  It is a TRIPWIRE, not a proof: it reads source text, so it cannot see
obfuscation (names assembled at run time, encoded strings, exec of fetched code, dlopen of a computed path) or anything
reached only through a dependency; the link-level gate in tests/test_no_egress.py (undefined socket symbols of the
built library) is the second, independent net.
CONTRACT: every tracked file is scanned (``*.patch``: added lines only).  Runtime scopes (src/, include/, adapters/,
bindings/python/seabraid/, examples/) allow nothing but ALLOW rows; everywhere else a URL is also allowed when its HOST is
loopback or is this package's own repository address in a pyproject.toml/Cargo.toml metadata field (_egress_meta.py), and documentation files (*.md, *.cff, LICENSE*, NOTICE*) may hold URLs.  A binary or unreadable tracked file is
itself a hit unless a BINARY_OK row names it.  ALLOW rows match the hit text (not the whole line); every row carries a WHY;
an unused row is a violation, so the tables cannot rot.  Allowed hits are PRINTED for review in CI logs.
SIZE: <=150 lines.  FAILURE: fail-fast; exit 1, ``path:line: egress: text`` per hit.  CHECKER: tests/test_no_egress.py
(one planted evasion per pattern family) plus tests/test_checkers.py and tests/test_sabotage.py.
Run:  python3 checks/check_no_egress.py [--root DIR] [--runtime-scope PREFIX ...]
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as common  # noqa: E402
import _egress_lang as lang  # noqa: E402
import _egress_meta as meta  # noqa: E402
import _egress_patterns as pats  # noqa: E402

# (path glob, regex searched in the HIT TEXT, why).  Each row is one reviewed exception; unused rows fail the run.
ALLOW = [
    ("compat/strata/install/build.py", r"github\.com/Niko1221/Strata", "upstream engine source URL the installer clones (overridable for mirrors)"),
    ("compat/strata/install/services.py", r"://\{host\}", "health probe; host argument defaults to loopback"),
    ("compat/strata/install/services.py", r"^import urllib", "loopback-only HTTP client for the health probe and /v1 requests"),
    ("compat/strata/install/_bench_lead_run.py", r"^import urllib", "benchmark client; every URL in this file is loopback (checked by the URL rule)"),
    ("compat/strata/install/_http.py", r"^import urllib", "loopback opener with proxies disabled (its WHY: a proxy must never see loopback traffic)"),
    ("compat/strata/tools/_loopback.py", r"^import urllib", "loopback JSON client with proxies disabled"),
    ("compat/strata/tools/capture_expert_preflight.py", r"^from urllib", "preflight client; its only URL is loopback"),
    ("compat/strata/tools/arms.sh", r"^curl$", "A/B driver; $URL is fixed to loopback at line 39"),
    ("compat/strata/tools/kl_gate.sh", r"^curl$", "KL gate client; $URL is built from 127.0.0.1 only"),
    ("compat/strata/tests/install/test_bench_lead.py", r"import (?:socket|http\.server)", "test fixture server bound to 127.0.0.1"),
    ("compat/strata/tests/install/test_hardening.py", r"import http\.server", "test fixture server bound to 127.0.0.1"),
    ("tests/test_placement_qual.py", r"^import http\.server", "stub server for the placement runner's dry run, bound to 127.0.0.1"),
    ("compat/strata/tests/test_kl_gate_sh.py", r"import argparse, http\.server", "fixture server script bound to 127.0.0.1"),
    ("compat/strata/tests/test_kl_gate_sh.py", r"cur[l]", "skipif guard: the shell gate needs the curl binary to exist"),
    ("compat/strata/tests/channel_*.cpp", r"connect\s*\(", "shm Channel::connect, not POSIX connect"),
    ("adapters/strata/wire/strata/seabraid/channel.hpp", r"connect\s*\(", "declaration of the shm Channel::connect member, not POSIX connect"),
    ("adapters/strata/static_qwen*.*pp", r"bind\(", "StaticQwenProvider::bind factory (binds a provider to the engine), not POSIX bind"),
    ("deny.toml", r"cur[l]", "cargo-deny BAN list entry: the crate is forbidden, not used"),
    ("deny.toml", r"crates\.io-index", "the one cargo registry cargo-deny permits (build-time dependency source, not runtime)"),
    ("tests/test_ci_workflow.py", r"crates\.io-index", "asserts the same cargo-deny registry allow-list"),
    ("scripts/release/rel_crate_credential.py", r"(?:index\.crates\.io/|github\.com/rust-lang/crates\.io-index)$",
     "canonical Cargo registry identifiers compared by the stdio credential provider; it makes no connection"),
    ("tests/test_no_local_refs.py", r"ss[h]", "test data: an ssh-alias SHAPE the local-refs checker must flag"),
    ("docs/img/*", r"www\.w3\.org/2000/svg", "SVG XML namespace identifier: a name, never fetched (the figures are self-contained)"),
    ("tests/test_l1_figures.py", r"www\.w3\.org/2000/svg", "the self-containment test names the one namespace URL it exempts"),
    ("tests/test_sabotage.py", r"Tcp[S]tream|imp[o]rt urllib|example\.invalid", "planted violation strings of the sabotage proof (copied tree only)"),
]
# (path glob, why): tracked binary files that are known not to carry code we ship.
BINARY_OK: list[tuple[str, str]] = []


def _row(rel: str, text: str, used: set[int]) -> str | None:
    for i, (glob, rx, why) in enumerate(ALLOW):
        if fnmatch.fnmatch(rel, glob) and re.search(rx, text):
            used.add(i)
            return f"allow-list: {why}"
    return None


def judge(rel: str, text: str, url: re.Match | None, runtime: bool, used: set[int], line: str = "", name: str = "") -> str | None:
    """Why this hit may stay, or None.  WHY host-only: the word 'localhost' elsewhere on a line proves nothing."""
    if url and not runtime and pats.loopback_host(url.group("auth")):
        return "loopback"
    if url and not runtime and meta.allowed(rel, line, name):
        return "package-metadata repository address (_egress_meta.py)"
    return _row(rel, text, used)


def line_hits(family: str, line: str):
    """(text, url match or None) per network use on one stripped line."""
    for m in pats.URL.finditer(line):
        yield m.group(0), m
    for p in pats.FAMILY_PATTERNS[family]:
        for m in p.finditer(line):
            yield m.group(0).strip(), None


def scan_lines(rel, family, lines, runtime, used, bad, allowed, docs, name=""):
    """``lines`` = [(reported line number, stripped text)]."""
    for n, line in lines:
        for text, url in line_hits(family, line):
            if family == "doc" and url:
                docs.append(text)
            elif (why := judge(rel, text, url, runtime, used, line, name)) is None:
                bad.append(f"{rel}:{n}: egress: {text}")
            else:
                allowed.append(f"allowed: {rel}:{n}: {why}: {text}")


def scan_file(rel, text, runtime, used, bad, allowed, docs, name=""):
    if rel.endswith(".patch"):
        for target, added in lang.patch_chunks(text):
            fam = lang.family_of(target, "")
            body = lang.strip_comments(fam, "\n".join(t for _, t in added)).split("\n")
            scan_lines(rel, fam, [(n, b) for (n, _), b in zip(added, body)], runtime, used, bad, allowed, docs, name)
        return
    fam = lang.family_of(rel, text)
    scan_lines(rel, fam, enumerate(lang.strip_comments(fam, text).split("\n"), start=1), runtime, used, bad, allowed, docs, name)


def scan(root: str, scopes: tuple[str, ...] = common.RUNTIME_SCOPES) -> tuple[list[str], list[str]]:
    bad, allowed, docs, used, bin_used = [], [], [], set(), set()
    name = meta.package_name(root)
    for rel in common.tracked(root):
        try:
            text = common.read_text(root, rel)
        except OSError as exc:
            bad.append(f"{rel}:1: egress-unreadable: {exc}")
            continue
        if text is None:
            hit = next((i for i, (g, _) in enumerate(BINARY_OK) if fnmatch.fnmatch(rel, g)), None)
            bin_used.add(hit)
            if hit is None:
                bad.append(f"{rel}:1: egress-binary: unscannable tracked file (NUL in first 8 KiB); add a BINARY_OK row with a WHY")
            continue
        scan_file(rel, text, common.in_scope(rel, scopes), used, bad, allowed, docs, name)
    bad += [f"unused ALLOW row: {g} {rx}" for i, (g, rx, _) in enumerate(ALLOW) if i not in used]
    bad += [f"unused BINARY_OK row: {g}" for i, (g, _) in enumerate(BINARY_OK) if i not in bin_used]
    if docs:
        allowed.append(f"allowed: {len(docs)} URL(s) in documentation files (*.md, *.cff, LICENSE*, NOTICE*): links are not code")
    return bad, allowed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--root", default=common.default_root())
    ap.add_argument("--runtime-scope", action="append", help="override the runtime scope prefixes (repeatable)")
    args = ap.parse_args(argv)
    bad, allowed = scan(args.root, tuple(args.runtime_scope or common.RUNTIME_SCOPES))
    return common.finish("check_no_egress", bad, allowed)


if __name__ == "__main__":
    sys.exit(main())
