# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""The security rule of the helper's file reader, as a checker (DESIGN 6): the serve path never opens a file.

OWNER: this file is the enforcing check for "no file is ever opened after the first round is served, and no path comes from the
wire".  Rules, each failing on its own:
  1. in strata-ext/src/file_fetch.cpp every open()/openat()/fopen()/creat() call sits between `// ---- LAUNCH ONLY` and
     `// ---- END LAUNCH ONLY` (the section FileTable::open_all reaches); the read path after it has none;
  2. FileTable::open_all is called only from startup code (helper.cpp, seabraid_sums.cpp), never from the serve loop or the round;
  3. no path-shaped member exists in the wire's FileRef / FileExtent (integers only), so no path can arrive from the wire.
"""
from __future__ import annotations

import re
from pathlib import Path

ADDON = Path(__file__).resolve().parents[1]
REPO = ADDON.parents[1]
SRC = ADDON / "strata-ext" / "src"
OPEN_CALL = re.compile(r"(?<![A-Za-z0-9_])(?:::)?(open|openat|open64|fopen|freopen|creat)\s*\(")
BEGIN, END = "// ---- LAUNCH ONLY", "// ---- END LAUNCH ONLY"


def _code(text: str) -> list[tuple[int, str]]:
    """(line number, line with // comments and string contents removed): so prose naming open() is not a call."""
    out = []
    for n, line in enumerate(text.splitlines(), 1):
        line = re.sub(r'"(?:\\.|[^"\\])*"', '""', line)
        out.append((n, line.split("//")[0]))
    return out


def _section(text: str) -> tuple[int, int] | None:
    """(first, last) 0-based lines of the markers, or None when they are not exactly one ordered pair."""
    lines = text.splitlines()
    begins = [i for i, l in enumerate(lines) if l.strip().startswith(BEGIN)]
    ends = [i for i, l in enumerate(lines) if l.strip().startswith(END)]
    return (begins[0], ends[0]) if len(begins) == 1 and len(ends) == 1 and begins[0] < ends[0] else None


def open_violations(text: str) -> list[str]:
    """Calls to open-like functions outside the LAUNCH ONLY section; a missing or unbalanced marker pair is itself a violation."""
    span = _section(text)
    if span is None:
        return ["the LAUNCH ONLY section markers are missing or not exactly one pair"]
    hits = [(n, OPEN_CALL.search(code)) for n, code in _code(text) if not span[0] < n - 1 < span[1]]
    return [f"line {n}: {m.group(1)}( outside the LAUNCH ONLY section" for n, m in hits if m]


def test_file_fetch_opens_only_in_the_launch_section():
    text = (SRC / "file_fetch.cpp").read_text(encoding="utf-8")
    assert open_violations(text) == []
    # the section does open files (otherwise the rule would pass on an empty section)
    assert any(OPEN_CALL.search(c) for _, c in _code(text)), "file_fetch.cpp opens nothing at all"


def test_the_rule_catches_an_open_on_the_serve_path():
    text = (SRC / "file_fetch.cpp").read_text(encoding="utf-8")
    serve = text.index("bool FileTable::read_buffered")
    sabotaged = text[:serve] + "static int leak(const char* p) { return ::open(p, O_RDONLY); }\n" + text[serve:]
    assert any("outside the LAUNCH ONLY" in v for v in open_violations(sabotaged))
    assert open_violations(text.replace(END, "// moved")) != []   # a vanished marker is not a pass


def test_open_all_is_called_only_at_startup():
    allowed = {"helper.cpp", "seabraid_sums.cpp", "file_fetch.cpp", "file_fetch.hpp"}
    callers = {p.name for p in SRC.glob("*") if p.suffix in {".cpp", ".hpp"} and re.search(r"\bopen_all\s*\(", p.read_text(encoding="utf-8"))}
    assert callers <= allowed, callers - allowed
    round_hpp = (ADDON / "strata-ext" / "include" / "strata" / "seabraid" / "tier_round.hpp").read_text(encoding="utf-8")
    code = "\n".join(c for _, c in _code(round_hpp))
    assert "open_all" not in code and "FileTable" not in code   # the round sees only the FileFetch concept, never the table


def test_a_file_ref_carries_integers_only():
    layout = (REPO / "adapters" / "strata" / "wire" / "strata" / "seabraid" / "channel_layout.hpp").read_text(encoding="utf-8")
    block = layout[layout.index("struct FileExtent"):layout.index("static_assert(sizeof(FileRef)")]
    assert not re.search(r"\b(char|string|path|std::)\b", re.sub(r"//.*", "", block)), "a FileRef member could carry a path"
