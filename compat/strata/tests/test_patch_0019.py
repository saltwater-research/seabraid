# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Patch 0019 (a lost helper closes its card, not the engine): what it must contain, where it sits, what it must not add.

OWNER: this file pins 0019's position in the series and its load-bearing lines, like test_patch_0018.py does for 0018.  Whether
it APPLIES and COMPILES is proved on a Strata tree (`git am` of 0001..0019, then the v7 mbox, on STRATA_BASE; the box and CI's
strata job).  Its behaviour is proved by the Strata tests it carries (seabraid_helper_loss_test L1-L9, lost_rows_test L10).
"""
from __future__ import annotations

import re
from pathlib import Path

ADDON = Path(__file__).resolve().parents[1]
REPO = ADDON.parents[1]
SERIES = sorted((ADDON / "patches" / "strata").glob("0*.patch"))
V7 = REPO / "adapters" / "strata" / "strata-v7-seabraid.patch"


def _p19() -> Path:
    found = [p for p in SERIES if p.name.startswith("0019-")]
    assert len(found) == 1, "exactly one 0019 patch"
    return found[0]


def _added(text: str, only: str | None = None) -> list[str]:
    """The added lines (without the '+'), optionally only those of the file section whose path is `only`."""
    out, cur = [], None
    for ln in text.splitlines():
        if ln.startswith("diff --git"):
            cur = ln.split(" b/")[-1]
        elif ln.startswith("+") and not ln.startswith("+++") and (only is None or cur == only):
            out.append(ln[1:])
    return out


def test_0019_follows_0018_and_precedes_0020_and_the_v7_patch():
    # 0019 was the last of the series until 0020 (the graceful-stop stale channel, live run S9a) was added after it
    assert SERIES[-2] == _p19() and SERIES[-3].name.startswith("0018-") and SERIES[-1].name.startswith("0020-")
    assert V7.exists()   # the v7 patch is applied after the whole compat series (it was regenerated to apply after 0019)


def test_0019_uses_the_series_header_style():
    text = _p19().read_text(encoding="utf-8")
    head = text.splitlines()[:5]
    assert head[0].startswith("From 0000000000000000000000000000000000000000 ")
    assert head[1] == "From: Seabraid <seabraid@example.invalid>" and head[2] == "Date: Sat, 1 Jan 2000 00:00:00 +0000"
    assert head[3].startswith("Subject: [PATCH 19/19] ")
    assert "Co-Authored-By" not in text


def test_0019_touches_exactly_the_loss_files():
    text = _p19().read_text(encoding="utf-8")
    files = [ln.split(" b/")[-1] for ln in text.splitlines() if ln.startswith("diff --git")]
    assert sorted(files) == sorted({"cmake/strata_gpu_engine.cmake", "include/strata/core/lost_rows.hpp",
                                    "include/strata/core/remote_experts.hpp", "src/core/expert_source.cpp",
                                    "src/core/remote_experts.cpp", "src/core/remote_prefill.cpp",
                                    "tests/core/seabraid_helper_loss_test.cpp", "tests/core/lost_rows_test.cpp",
                                    "tests/core/remote_experts_seabraid_opt_test.cpp", "tests/core/remote_prefill_test.cpp"})


def test_0019_carries_the_loss_needles():
    text = _p19().read_text(encoding="utf-8")
    for needle in ("bl_wait(", "seabraid_lose(", "bool rows_lost() const", "announce_identity", "seabraid::PeerWatch",
                   "plan_lost_rows(", "redo_lost_rows_single(", "redo_lost_rows_multi(", "quantize_window_act(",
                   "set_seabraid_deadline_for_test", "seabraid_open_closed(", "seabraid_closed() == 3", "std::this_thread::get_id() == bl_thread_"):
        assert needle in text, needle
    # both dispatch sites hand a lost helper's rows to the CPU: one call per dispatch, nowhere else
    assert len(re.findall(r"->rows_lost\(\)", "\n".join(_added(text, "src/core/expert_source.cpp")))) == 2
    # the prompt path reads liveness from the one predicate, not from a number
    assert "seabraid_closed() != 0" in "\n".join(_added(text, "src/core/remote_prefill.cpp"))


def test_0019_adds_no_pid_probe_and_no_stray_wait():
    added = "\n".join(_added(_p19().read_text(encoding="utf-8")))
    # a signal-0 probe of a number a recycled pid can answer is what this patch removes (D4's two sites); tests use raise()/_exit
    assert not re.search(r"(?<![A-Za-z_])kill\s*\(", added.replace("raise(SIGKILL)", ""))
    assert "SYS_pidfd_open" not in added   # the raw syscall has one owner, wire/strata/seabraid/pidfd.hpp
    # the single wait wrapper: no seabraid wait is added to a .cpp except through bl_wait
    for path in ("src/core/remote_experts.cpp", "src/core/remote_prefill.cpp"):
        assert "Channel::wait(" not in "\n".join(_added(_p19().read_text(encoding="utf-8"), path)), path
    hpp = "\n".join(_added(_p19().read_text(encoding="utf-8"), "include/strata/core/remote_experts.hpp"))
    assert hpp.count("Channel::wait(") == 1   # inside bl_wait


def test_0019_removes_the_old_prefill_probes():
    text = _p19().read_text(encoding="utf-8")
    removed = [ln[1:] for ln in text.splitlines() if ln.startswith("-") and not ln.startswith("---")]
    assert sum("::kill(pid, 0)" in ln for ln in removed) == 2
