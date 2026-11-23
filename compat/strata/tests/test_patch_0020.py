# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Patch 0020 (a dead helper at start opens closed whatever state its last engine left): what it must contain, where it sits.

OWNER: this file pins 0020's position in the series and its load-bearing lines, like test_patch_0019.py does for 0019.  Whether it
APPLIES and COMPILES is proved on a Strata tree (`git am` of 0001..0020, then the v7 mbox, on STRATA_BASE).  Its behaviour is
proved by the two Strata tests it adds to seabraid_helper_loss_test (L11: a kShutdown channel with a dead peer opens closed within
1 s with one WARN; L12: a live helper in kShutdown is waited for, not read as dead).
"""
from __future__ import annotations

from pathlib import Path

ADDON = Path(__file__).resolve().parents[1]
REPO = ADDON.parents[1]
SERIES = sorted((ADDON / "patches" / "strata").glob("0*.patch"))
V7 = REPO / "adapters" / "strata" / "strata-v7-seabraid.patch"


def _p20() -> Path:
    found = [p for p in SERIES if p.name.startswith("0020-")]
    assert len(found) == 1, "exactly one 0020 patch"
    return found[0]


def _added(text: str, only: str) -> list[str]:
    """The added lines (without the '+') of the file section whose path is `only`."""
    out, cur = [], None
    for ln in text.splitlines():
        if ln.startswith("diff --git"):
            cur = ln.split(" b/")[-1]
        elif ln.startswith("+") and not ln.startswith("+++") and cur == only:
            out.append(ln[1:])
    return out


def _new_side(text: str, only: str) -> list[str]:
    """The lines of the file section whose path is `only` as they read after the patch (context and added lines, without the marker)."""
    out, cur = [], None
    for ln in text.splitlines():
        if ln.startswith("diff --git"):
            cur = ln.split(" b/")[-1]
        elif cur == only and ln[:1] in (" ", "+") and not ln.startswith("+++"):
            out.append(ln[1:])
    return out


def test_0020_is_the_last_of_the_compat_series_and_before_the_v7_patch():
    assert SERIES[-1] == _p20() and SERIES[-2].name.startswith("0019-")
    assert V7.exists()   # the v7 patch is applied after the whole compat series; it does not touch seabraid_open's open path


def test_0020_uses_the_series_header_style():
    text = _p20().read_text(encoding="utf-8")
    head = text.splitlines()[:5]
    assert head[0].startswith("From 0000000000000000000000000000000000000000 ")
    assert head[1] == "From: Seabraid <seabraid@example.invalid>" and head[2] == "Date: Sat, 1 Jan 2000 00:00:00 +0000"
    assert head[3].startswith("Subject: [PATCH 20/20] ")
    assert "Co-Authored-By" not in text


def test_0020_touches_exactly_seabraid_open_and_its_test():
    text = _p20().read_text(encoding="utf-8")
    files = [ln.split(" b/")[-1] for ln in text.splitlines() if ln.startswith("diff --git")]
    assert sorted(files) == ["src/core/remote_experts.cpp", "tests/core/seabraid_helper_loss_test.cpp"]


def _first(lines: list, needle: str) -> int:
    return next(i for i, ln in enumerate(lines) if needle in ln)


def test_0020_binds_the_identity_before_it_reads_the_state_word():
    text = _p20().read_text(encoding="utf-8")
    after = _new_side(text, "src/core/remote_experts.cpp")   # git shows the unmoved watch block as context, so read the new side, not only '+' lines
    watch = _first(after, "watch_ = std::move(w);")   # the end of the watch block (its `w->start(...)` line is outside the diff's context)
    first_state = _first(after, "uint32_t st = hd->state.load(")
    refusal = _first(after, "is not ready for a plan (state ")
    removed = [ln[1:] for ln in text.splitlines() if ln.startswith("-") and not ln.startswith("---")]
    assert _first(removed, "is not ready for a plan (state ") >= 0   # the old refusal, which sat BEFORE the watch block, is gone from there
    assert watch < first_state < refusal   # the order IS the fix: a state-first refusal is what ended the engine in live run S9a


def test_0020_waits_for_a_live_helper_and_opens_a_dead_one_closed():
    text = _p20().read_text(encoding="utf-8")
    body = "\n".join(_added(text, "src/core/remote_experts.cpp"))
    assert "seabraid::kShutdown" in body and "bl_wait(" in body and "seabraid_open_closed(" in body
    assert "Channel::wait(" not in body   # the one wait wrapper stays bl_wait (checks/check_pid_probes.py)


def test_0020_carries_both_tests_and_adds_no_pid_probe():
    text = _p20().read_text(encoding="utf-8")
    tests = "\n".join(_added(text, "tests/core/seabraid_helper_loss_test.cpp"))
    assert 'want("L11")' in tests and 'want("L12")' in tests
    assert "kill(" not in "\n".join(ln for ln in text.splitlines() if ln.startswith("+"))
