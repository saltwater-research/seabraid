#!/usr/bin/env python3
# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Rename the project: ``rename.py --from OLD --to NEW [--dry-run]``.

OWNER: the token rule (what counts as "the name") is owned by ``checks/_common.name_regex`` and shared with
``checks/check_names.py``, so the tool that renames and the gate that proves it agree by construction.
CONTRACT: file CONTENTS and file/directory NAMES are rewritten wherever the bare name occurs, with no word boundary
(kOld and OLDX included): old -> new, OLD -> NEW, any other casing -> New (Capitalised).  That covers identifiers,
headers, CMake package and targets (Old::x), env vars (OLD_*), library names (libold, old.dll, -lold), Python imports
and package dirs.  Only whole words listed in ``_common.EXCLUDED_WORDS`` (English words that contain the name) are
never touched.  Binary files are renamed by path only; a binary that CONTAINS the
name is reported, not rewritten.  .patch files change only when they contain the name (so upstream-only patches stay
byte-identical) and every hunk header's line counts are re-verified afterwards; one bad patch aborts the whole run.
ETYMOLOGY: a line carrying ``<!-- name-etymology -->`` explains why the OLD name was chosen, so no rewrite can keep it true;
the tool deletes that line and reports it, and the maintainers write the new name's story.
SAFETY: two phases.  Phase 1 plans everything in memory and refuses on any collision or bad patch; phase 2 writes.
Nothing is written under --dry-run.  Re-running after a rename finds nothing and is a no-op.  Run it on a clean git
tree and commit (or `git checkout .`) around it: it moves files with os.replace, not `git mv`.
FAILURE: fail-fast and loud; exit 1 with the reason, before any write.  Prints every replacement count by file.
Run:  python3 scripts/rename.py --from OLD --to NEW [--root DIR] [--dry-run]
"""

from __future__ import annotations

import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "checks"))
import _common as common  # noqa: E402

NEW_NAME = re.compile(r"^[a-z][a-z0-9]{2,20}$")
# WHY exempt: these files NAME the old project as their input (examples, fixtures); rewriting them would break the next run.
SELF_EXEMPT = frozenset({"scripts/rename.py", "tests/test_rename.py"})
ETYMOLOGY = "<!-- name-etymology -->"
HUNK = re.compile(r"^@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@")


class RenameError(Exception):
    """A refusal the user can act on; raised before anything is written."""


def casings(old: str, new: str) -> dict[str, str]:
    return {old: new, old.capitalize(): new.capitalize(), old.upper(): new.upper()}


def substitute(text: str, old: str, new: str) -> tuple[str, int]:
    """Replace name tokens in ``text``; returns (new text, replacement count)."""
    rx, table, n = common.name_regex(old), casings(old, new), 0

    def repl(m: re.Match) -> str:
        nonlocal n
        if common.is_excluded(old, m.string, m.start(), m.end()):
            return m.group(0)
        n += 1
        return table.get(m.group(0)) or (new.capitalize() if m.group(0)[:1].isupper() else new)

    return rx.sub(repl, text), n


def hunk_problems(text: str) -> list[str]:
    """Hunk headers whose declared line counts disagree with the body.  A patch with no hunks has none."""
    problems, lines, i = [], text.split("\n"), 0
    while i < len(lines):
        m = HUNK.match(lines[i])
        i += 1
        if not m:
            continue
        want = (int(m.group(1) or 1), int(m.group(2) or 1))
        old = new = 0
        while i < len(lines) and (old < want[0] or new < want[1]):
            tag = lines[i][:1]
            old += tag in (" ", "-", "")  # "" is a context line whose single space an editor stripped
            new += tag in (" ", "+", "")
            i += 1
        if (old, new) != want:
            problems.append(f"hunk wants -{want[0]} +{want[1]}, body has -{old} +{new}")
    return problems


def drop_etymology(text: str) -> tuple[str, int]:
    """Delete every line that carries the etymology marker (and the blank line after it, so no double gap is left)."""
    if ETYMOLOGY not in text:
        return text, 0
    out, skip_blank, n = [], False, 0
    for line in text.split("\n"):
        if ETYMOLOGY in line:
            n, skip_blank = n + 1, True
            continue
        if skip_blank and not line.strip():
            skip_blank = False
            continue
        skip_blank = False
        out.append(line)
    return "\n".join(out), n


def plan_content(root: str, rel: str, old: str, new: str) -> tuple[bytes | None, int, str | None]:
    """-> (new bytes or None when unchanged, replacement count, warning)."""
    with open(os.path.join(root, rel), "rb") as fh:
        raw = fh.read()
    if b"\0" in raw[:8192]:
        return None, 0, (f"{rel}: binary file contains the name; not rewritten" if old.encode() in raw.lower() else None)
    text = raw.decode("utf-8", "surrogateescape")  # WHY: byte-exact round trip, CRLF and odd bytes included
    out, n = substitute(text, old, new)
    out, dropped = drop_etymology(out)
    if dropped:
        warning = f"{rel}: removed the name-etymology line (it explained {old!r}); write one for {new!r}"
        return out.encode("utf-8", "surrogateescape"), n + dropped, warning
    if n == 0:
        return None, 0, None
    if rel.endswith(".patch"):
        # WHY compare with the original: a patch that was already inconsistent is not this tool's fault; only a NEW problem refuses.
        problems = hunk_problems(out)
        if len(problems) > len(hunk_problems(text)):
            raise RenameError(f"{rel}: patch hunks inconsistent after rename ({problems[0]}); refusing")
    return out.encode("utf-8", "surrogateescape"), n, None


def plan_path(rel: str, old: str, new: str) -> str:
    return "/".join(substitute(part, old, new)[0] for part in rel.split("/"))


def build_plan(root: str, old: str, new: str) -> dict:
    files, targets, warnings = [], {}, []
    for rel in common.tracked(root):
        data, n, warn = (None, 0, None) if rel in SELF_EXEMPT else plan_content(root, rel, old, new)
        dest = plan_path(rel, old, new)
        if dest in targets:
            raise RenameError(f"collision: {rel} and {targets[dest]} both become {dest}")
        targets[dest] = rel
        warnings += [warn] if warn else []
        if data is not None or dest != rel:
            files.append((rel, dest, data, n))
    refuse_overwrites(root, files)
    return {"files": files, "warnings": warnings}


def refuse_overwrites(root: str, files: list) -> None:
    """A destination that exists on disk (even untracked) and is not itself being moved away must not be clobbered."""
    sources = {r for r, _, _, _ in files}
    for rel, dest, _, _ in files:
        if dest != rel and dest not in sources and os.path.exists(os.path.join(root, dest)):
            raise RenameError(f"collision: {rel} -> {dest} would overwrite an existing file")


def apply_plan(root: str, plan: dict) -> None:
    for rel, dest, data, _ in plan["files"]:
        src, dst = os.path.join(root, rel), os.path.join(root, dest)
        if data is not None:
            with open(src, "wb") as fh:
                fh.write(data)
        if dest != rel:
            os.makedirs(os.path.dirname(dst) or root, exist_ok=True)
            os.replace(src, dst)
            parent = os.path.dirname(src)
            while parent != root and not os.listdir(parent):
                os.rmdir(parent)
                parent = os.path.dirname(parent)


def report(plan: dict, dry: bool) -> None:
    total = sum(n for *_, n in plan["files"])
    for rel, dest, _, n in sorted(plan["files"]):
        print(f"{n:5d} replacement(s)  {rel}" + (f"  ->  {dest}" if dest != rel else ""))
    for w in plan["warnings"]:
        print("warning: " + w)
    moved = sum(1 for rel, dest, _, _ in plan["files"] if dest != rel)
    print(f"rename: {'DRY RUN, nothing written: ' if dry else ''}{total} replacement(s) in {len(plan['files'])} file(s), {moved} path(s) renamed")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--from", dest="old", required=True)
    ap.add_argument("--to", dest="new", required=True)
    ap.add_argument("--root", default=common.default_root())
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    try:
        if not NEW_NAME.match(a.new) or not NEW_NAME.match(a.old) or a.old in a.new or a.new in a.old:
            raise RenameError(f"names must match [a-z][a-z0-9]{{2,20}} and not contain each other: {a.old!r} -> {a.new!r}")
        plan = build_plan(os.path.abspath(a.root), a.old, a.new)
        if not a.dry_run:
            apply_plan(os.path.abspath(a.root), plan)
    except RenameError as exc:
        print(f"rename: REFUSED: {exc}", file=sys.stderr)
        return 1
    report(plan, a.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
