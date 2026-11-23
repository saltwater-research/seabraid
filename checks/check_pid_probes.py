#!/usr/bin/env python3
# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""No code may probe or signal a helper by its bare pid, name the pidfd syscall twice, or wait on the channel outside the one wrapper.

WHY this gate exists: the engine's D4 prefill once asked ``::kill(helper_pid, 0)`` whether a helper lived; a recycled pid answers
"yes" for a stranger, and a helper that died while the engine waited 30 s on it ended the whole engine (HL-FIX).  A pid is a
number the kernel reuses; the identity the engine trusts is (pid, starttime) behind a pidfd (wire peer_watch.hpp).  Nothing flagged
the bare probe, because it compiles and works until the pid is reused.
OWNER: this checker owns the three rules; the reasons live in checks/pid_probes_allow.txt, one site per row.
RULES (a site is a line that matches, ignoring comments and string literals):
  1. ``kill(`` / ``killpg(`` anywhere: in repo C++ and in every line the patch stack adds and no later patch removes.
  2. ``SYS_pidfd_open`` anywhere but wire/strata/seabraid/pidfd.hpp (the one owner of the raw syscall).
  3. ``Channel::wait(`` in a line the patch stack adds to src/core/remote_*.cpp (the engine's remote experts): each wait must be
     the single ``bl_wait`` wrapper, which also wakes when the helper is lost; a bare wait sleeps out its whole deadline.
CONTRACT: the patch stack is the compat series (compat/strata/patches/strata/*.patch, in name order) and then
adapters/strata/strata-v7-seabraid.patch, as in check_helper_tier.py.  Each site must match exactly one allow row,
``<fragment> | <why>``, by substring of ``<file>: <line text>``; a reason has at least MIN_WHY characters.  A row that matches no
site, or several, fails, so the list cannot rot (a patch that removes a site must delete its row in the same change).
FAILURE: fail-fast, exit 1, one ``file:line: rule: text`` per unexplained site.  SIZE: <=150 lines.
CHECKER: tests/test_sabotage.py (a re-added kill, a second pidfd_open, a bare wait, a stale row).
Run:  python3 checks/check_pid_probes.py [--root DIR]
"""

from __future__ import annotations

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as common  # noqa: E402
from check_helper_tier import patch_stack  # noqa: E402  (one definition of the patch stack)

ALLOW = "checks/pid_probes_allow.txt"
PIDFD_OWNER = "adapters/strata/wire/strata/seabraid/pidfd.hpp"
MIN_WHY = 20
CPP = (".cpp", ".hpp", ".h", ".cc", ".cu", ".cuh", ".c")
KILL = re.compile(r"(?<![\w])(?:kill|killpg)\s*\(")
PIDFD = re.compile(r"\bSYS_pidfd_open\b")
WAIT = re.compile(r"\bChannel::wait\s*\(")
REMOTE = re.compile(r"src/core/remote_\w+\.cpp$")
STRING = re.compile(r'"(?:\\.|[^"\\])*"')
Site = tuple[str, int, str, str]   # (file, line, rule, text)


def code_of(text: str) -> str:
    """The line without string literals and trailing comments; "" for a comment-only line."""
    plain = STRING.sub('""', text).split("//")[0].strip()
    return "" if plain.startswith(("*", "/*")) else plain


def rules_hit(code: str, rel: str, in_remote: bool) -> list[str]:
    hits = ["kill" for _ in KILL.finditer(code)][:1]
    if PIDFD.search(code) and rel != PIDFD_OWNER:
        hits.append("pidfd_open")
    if in_remote and WAIT.search(code):
        hits.append("bare-wait")
    return hits


def added_sites(rel: str, lines: list[str]) -> list[Site]:
    """The sites one patch adds.  The target file comes from the latest ``diff --git`` header above the line."""
    sites: list[Site] = []
    target = ""
    for n, ln in enumerate(lines, 1):
        if ln.startswith("diff --git"):
            target = ln.rsplit(" b/", 1)[-1]
        elif ln.startswith("+") and not ln.startswith("+++"):
            hits = rules_hit(code_of(ln[1:]), target, bool(REMOTE.search(target)))
            sites += [(rel, n, rule, ln[1:].strip()) for rule in hits]
    return sites


def patch_sites(root: str) -> list[Site]:
    """Surviving added lines of the patch stack (a later patch that removes a line removes its site), in stack order."""
    sites: list[Site] = []
    for rel in patch_stack(root):
        with open(os.path.join(root, rel), encoding="utf-8", errors="replace") as fh:
            lines = fh.read().splitlines()
        removed = {ln[1:].strip() for ln in lines if ln.startswith("-") and not ln.startswith("---")}
        sites = [s for s in sites if s[3] not in removed] + added_sites(rel, lines)
    return sites


def cpp_sites(root: str) -> list[Site]:
    sites: list[Site] = []
    for rel in common.tracked(root):
        if not rel.endswith(CPP):
            continue
        text = common.read_text(root, rel) or ""
        for n, ln in enumerate(text.splitlines(), 1):
            sites += [(rel, n, rule, ln.strip()) for rule in rules_hit(code_of(ln), rel, False) if rule != "bare-wait"]
    return sites


def load_allow(root: str) -> tuple[list[tuple[int, str]], list[str]]:
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
                rows.append((n, frag))
    return rows, errors


def check(root: str) -> list[str]:
    sites = patch_sites(root) + cpp_sites(root)
    rows, errors = load_allow(root)
    explained: set[int] = set()
    for n, frag in rows:
        hits = [i for i, s in enumerate(sites) if frag in f"{s[0]}: {s[3]}"]
        if len(hits) != 1:
            errors.append(f"{ALLOW}:{n}: fragment matches {len(hits)} sites, needs exactly 1: {frag!r}")
        explained.update(hits)
    for i, (rel, n, rule, text) in enumerate(sites):
        if i not in explained:
            errors.append(f"{rel}:{n}: {rule}: use the PeerWatch pidfd identity / bl_wait / wire pidfd.hpp, or add a row to {ALLOW} saying why: {text}")
    return errors


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=common.default_root())
    args = ap.parse_args(argv)
    errors = check(args.root)
    for e in errors:
        print(e)
    if errors:
        print(f"check_pid_probes: {len(errors)} problem(s)")
        return 1
    print(f"check_pid_probes: ok ({len(load_allow(args.root)[0])} explained sites, no unexplained pid probe)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
