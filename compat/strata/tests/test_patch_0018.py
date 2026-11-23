# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Patch 0018 (file moves): where an expert's bytes live, and the engine's side of one kTierSwapFile.

OWNER: this file pins what 0018 must contain and where it sits in the series; whether it APPLIES and COMPILES is proved on a Strata
tree (`git am` of the series on STRATA_BASE, then the v7 patch; CI's strata job and the Linux test-machine run).
"""
from __future__ import annotations

from pathlib import Path

ADDON = Path(__file__).resolve().parents[1]
REPO = ADDON.parents[1]
SERIES = sorted((ADDON / "patches" / "strata").glob("0*.patch"))
V7 = REPO / "adapters" / "strata" / "strata-v7-seabraid.patch"


def _p18() -> Path:
    found = [p for p in SERIES if p.name.startswith("0018-")]
    assert len(found) == 1, "exactly one 0018 patch"
    return found[0]


def test_0018_follows_0017_and_precedes_0019_and_the_v7_patch():
    # 0018 was the last of the series until 0019 (a lost helper closes its card) and then 0020 were added after it; their tests pin that
    assert SERIES[-3] == _p18() and SERIES[-4].name.startswith("0017-") and SERIES[-2].name.startswith("0019-")
    assert V7.exists()   # the v7 patch is applied after the whole compat series


def test_0018_uses_the_series_header_style():
    text = _p18().read_text(encoding="utf-8")
    head = text.splitlines()[:5]
    assert head[0].startswith("From 0000000000000000000000000000000000000000 ")
    assert head[1] == "From: Seabraid <seabraid@example.invalid>" and head[2] == "Date: Sat, 1 Jan 2000 00:00:00 +0000"
    assert head[3].startswith("Subject: [PATCH 18/18] ")
    assert "Co-Authored-By" not in text


def test_0018_carries_the_addressing_the_retier_call_and_the_test():
    text = _p18().read_text(encoding="utf-8")
    for needle in ("const std::vector<std::string>& file_paths() const",
                   "bool file_extents(int64_t layer, int64_t expert, FileExtent out[3], int& n) const",
                   "bool FileExpertSource::file_extents(",
                   "bool RemoteExperts::seabraid_retier_file(",
                   "seabraid::file_ref_valid(ref, file_size, file_count, need",
                   "seabraid::kTierSwapFile",
                   "test_file_extents_canonical"):
        assert needle in text, needle
    # one owner of addressing: file_extents reads spans(), the function copy_from_files and the page-cache probes already use
    assert "spans(layer, expert, sp)" in text and "size_t file; };" in text


def test_0018_stays_away_from_what_the_v7_patch_edits():
    # the v7 patch adds the engine's rung code; 0018 must not touch its regions (FR-E applies on top of it)
    text = _p18().read_text(encoding="utf-8")
    files = [ln.split(" b/")[-1] for ln in text.splitlines() if ln.startswith("diff --git")]
    assert sorted(files) == sorted({"include/strata/core/expert_source.hpp", "include/strata/core/remote_experts.hpp",
                                    "src/core/expert_source.cpp", "src/core/remote_experts.cpp", "tests/core/file_expert_source_test.cpp"})
