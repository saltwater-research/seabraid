# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""v7 patch commit 3 ("file moves"): what the engine side of --retier-file-moves must contain, and what it must not touch.

OWNER: this file pins the commit's file list and its load-bearing lines, like test_patch_0018.py does for 0018.  Whether the stack
APPLIES and COMPILES is proved on a Strata tree (`git am` of 0001..0018 then the v7 mbox on STRATA_BASE; the box and CI's strata job).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ADDON = Path(__file__).resolve().parents[1]
REPO = ADDON.parents[1]
V7 = REPO / "adapters" / "strata" / "strata-v7-seabraid.patch"


def _commits() -> list[str]:
    text = V7.read_text(encoding="utf-8")
    parts = re.split(r"(?m)^(?=From 0{40} Mon Sep 17 00:00:00 2001$)", text)
    return [p for p in parts if p.strip()]


def _files(commit: str) -> list[str]:
    return [ln.split(" b/")[-1] for ln in commit.splitlines() if ln.startswith("diff --git")]


def _added(commit: str) -> str:
    return "\n".join(ln[1:] for ln in commit.splitlines() if ln.startswith("+") and not ln.startswith("+++"))


def test_the_mbox_is_three_commits_and_the_third_is_file_moves():
    commits = _commits()
    assert len(commits) == 3
    for i, c in enumerate(commits, 1):
        assert f"Subject: [PATCH {i}/3] " in c.splitlines()[3] or f"[PATCH {i}/3]" in c[:400]
    third = commits[2]
    head = third.splitlines()[:5]
    assert head[1] == "From: Seabraid <seabraid@example.invalid>" and head[2] == "Date: Sat, 1 Jan 2000 00:00:00 +0000"
    assert "file-reference expert moves" in third[:400] and "Co-Authored-By" not in third


def test_commit_3_touches_exactly_the_engine_files_it_owns():
    # NOT expert_source.* nor src/core/remote_experts.cpp: those are patch 0018's regions (file_extents, seabraid_retier_file).
    assert sorted(_files(_commits()[2])) == sorted({
        "include/strata/core/ownership_map.hpp", "include/strata/core/ownership_rungs.hpp", "include/strata/core/remote_experts.hpp",
        "src/core/ownership_rungs.cpp", "src/core/remote_seabraid.cpp", "src/program/generate.cpp"})


def test_commit_3_carries_the_rung_contract_the_checks_and_the_wiring():
    added = _added(_commits()[2])
    for needle in (
            "virtual IoStatus replace_from_fallback(", "virtual bool reads_files() const { return false; }",   # RungIO defaults: REFUSED, no files
            "class FileMover {", "class ReadCheck {",
            "IoStatus RemoteRung::replace_from_fallback(", "bool reads_files() const override { return mover_ != nullptr && mover_->active(); }",
            "check_->read_ok(layer, expert, out.data(), out.size(), err)",                                     # main verifies its own reads
            "ram_install_misses_", "FileFacts seabraid_file_facts() const",
            "bool compact_rows = false", "pack, native, 0, 60000, 1800000, compact_rows",                       # the launch struct's switch
            "o.seabraid.compact_rows", "parse_engine_flag(a, next, o.seabraid, seabraid_err)",
            "struct EngineFileBackend final : seabraid::strata_adapter::FileBackend",
            "re.seabraid_retier_file(op, ref, sizes, count, res, err)",
            "PackSums::load(PackSums::pack_file(o.pack), facts)",
            "rung->set_file_moves(placement_file_movers.back().get(), placement_sums.get())",
            "fallback_rung->set_read_check(placement_sums.get())",
            "placement_cfg.fallback_counters = [fb = fallback_rung.get()]",                                     # TRACE handover
            "tier->set_blob_bytes(slot_bytes)",                                                                 # REVFIX handover
            "void after_round() noexcept override { seabraid::strata_adapter::log_file_io(*file_io); }"):
        assert needle in added, needle


def test_the_engine_counter_line_is_the_one_placement_qual_parses():
    # adapters/strata/file_moves.cpp writes the line; arms.json's io_patterns read it (the format is documented in ONE place: here)
    source = (REPO / "adapters" / "strata" / "file_moves.cpp").read_text(encoding="utf-8")
    fmt = "helper file io: direct_bytes=%llu buffered_bytes=%llu reads=%llu bad_sum=%llu"
    assert fmt in source
    line = "seabraid: INFO: " + fmt.replace("%llu", "{}").format(1100, 30, 7, 1)
    patterns = json.loads((ADDON / "scripts" / "placement-qual" / "arms.json").read_text(encoding="utf-8"))["io_patterns"]
    got = {k: re.search(patterns[k], line) for k in ("file_direct_bytes", "file_buffered_bytes", "file_reads", "bad_sum")}
    assert {k: int(m.group(1)) for k, m in got.items() if m} == {"file_direct_bytes": 1100, "file_buffered_bytes": 30, "file_reads": 7, "bad_sum": 1}


def test_commit_3_adds_no_helper_tier_call_site_and_no_path_in_a_log_line():
    added = _added(_commits()[2])
    assert ".seabraid()" not in added                       # checks/check_helper_tier.py's rule: a new call site needs a reviewed row
    assert "o.pack" not in re.sub(r"PackSums::pack_file\(o\.pack\)", "", added)   # the pack path goes to the reader and nowhere else
