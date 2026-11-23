# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""The Strata pin has one owner, patches/strata/STRATA_BASE (read by install/build.py): prose refers to it, never copies it.

WHY a scan: a SHA copied into a README or a patch message is right on the day it is written and silently wrong after the
next rebase.  This fails on any copy (7 or more leading hex digits of the pin) in the add-on or the Strata adapter, apart
from the owner itself and the dated history (docs/DESIGN-LOG.md, docs/research/), which records what was true then.
"""
from __future__ import annotations

import re
from pathlib import Path

ADDON = Path(__file__).resolve().parents[2]          # compat/strata
REPO = ADDON.parents[1]
OWNER = ADDON / "patches" / "strata" / "STRATA_BASE"
HISTORY = (ADDON / "docs" / "DESIGN-LOG.md", ADDON / "docs" / "research")
TEXT_SUFFIXES = {".md", ".py", ".patch", ".txt", ".cmake", ".cpp", ".hpp", ".sh", ".json", ".toml", ".yml", ".yaml"}


def _pin() -> str:
    return OWNER.read_text(encoding="utf-8").split()[0].lower()


def _is_scanned(p: Path) -> bool:
    """One file's verdict; split out of the walk so each stays well under the complexity limit."""
    if not p.is_file() or p == OWNER or "__pycache__" in p.parts:
        return False
    if p.suffix not in TEXT_SUFFIXES and p.name != "CMakeLists.txt":
        return False
    return not any(p == h or h in p.parents for h in HISTORY)


def _scanned():
    for root in (ADDON, REPO / "adapters" / "strata"):
        yield from (p for p in root.rglob("*") if _is_scanned(p))


def test_the_pin_is_a_full_sha():
    assert re.fullmatch(r"[0-9a-f]{40}", _pin())


def test_no_file_copies_the_pin():
    copy = re.compile(r"(?<![0-9a-f])" + _pin()[:7] + r"[0-9a-f]{0,33}(?![0-9a-f])", re.I)
    hits = [f"{p.relative_to(REPO)}:{n}" for p in _scanned()
            for n, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1) if copy.search(line)]
    assert not hits, "refer to patches/strata/STRATA_BASE instead of copying the pin: " + ", ".join(hits)
