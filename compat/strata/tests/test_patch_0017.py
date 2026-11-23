# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Patch 0017 (pyramid, track C): one channel per helper process, and the blob swap with a read-back.

OWNER: this file pins what 0017 must contain and where it sits in the series; whether it APPLIES is proved on a Strata tree
(`git am` of the series on STRATA_BASE, then the v7 patch; CI's strata job and the Linux test-machine run).  An optional local proof:
set STRATA_TREE to a checkout at STRATA_BASE and the series is applied in a scratch clone.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

ADDON = Path(__file__).resolve().parents[1]
REPO = ADDON.parents[1]
SERIES = sorted((ADDON / "patches" / "strata").glob("0*.patch"))
V7 = REPO / "adapters" / "strata" / "strata-v7-seabraid.patch"


def _p17() -> Path:
    found = [p for p in SERIES if p.name.startswith("0017-")]
    assert len(found) == 1, "exactly one 0017 patch"
    return found[0]


def test_0017_follows_0016_and_precedes_0018_and_the_v7_patch():
    assert SERIES[-4] == _p17() and SERIES[-5].name.startswith("0016-") and SERIES[-3].name.startswith("0018-")   # 0019 and 0020 follow 0018
    assert V7.exists()   # the v7 patch is applied after the whole compat series (ci.yml)


def test_0017_uses_the_series_header_style():
    head = _p17().read_text(encoding="utf-8").splitlines()[:5]
    assert head[0].startswith("From 0000000000000000000000000000000000000000 ")
    assert head[1] == "From: Seabraid <seabraid@example.invalid>" and head[2] == "Date: Sat, 1 Jan 2000 00:00:00 +0000"
    assert head[3].startswith("Subject: [PATCH 17/17] ")
    assert "Co-Authored-By" not in _p17().read_text(encoding="utf-8")


def test_0017_carries_the_three_pieces_and_the_list_reader():
    text = _p17().read_text(encoding="utf-8")
    for needle in ("bool channel_name(int card, std::string& name, std::string& err)",
                   "bool RemoteExperts::seabraid_retier_blob(const seabraid::TierWireOp& op, const uint8_t* bytes, size_t n, uint8_t& res, std::string& err)",
                   "bool RemoteExperts::tier_verify(int layer, int expert, const uint8_t* blob, std::string& err)",
                   "seabraid::kTierSwapBlob", "channel_name(device - seabraid::kDeviceBase"):
        assert needle in text, needle
    # the single-name reads are gone: SEABRAID_STRATA_SHM is read in exactly one place
    added = [ln for ln in text.splitlines() if ln.startswith("+") and 'getenv("SEABRAID_STRATA_SHM")' in ln]
    removed = [ln for ln in text.splitlines() if ln.startswith("-") and 'getenv("SEABRAID_STRATA_SHM")' in ln]
    assert len(added) == 1 and len(removed) == 2


def test_0017_adds_no_named_channel_only_call_site():
    # checks/check_helper_tier.py's rule, asserted here too so the failure names this patch
    assert not re.search(r"^\+.*(\.|->)seabraid\(\)", _p17().read_text(encoding="utf-8"), re.M)


def test_0017_does_not_open_the_v7_patch_context():
    # the v7 patch inserts above `namespace strata::core {` in remote_experts.hpp; a hunk of ours there would conflict
    text = _p17().read_text(encoding="utf-8")
    hpp = text.split("diff --git a/include/strata/core/remote_experts.hpp", 1)[1].split("diff --git", 1)[0]
    first_line = min(int(m) for m in re.findall(r"^@@ -(\d+)", hpp, re.M))
    assert first_line > 100, "no hunk near the include block the v7 patch edits"
    assert "+namespace seabraid {" in hpp


@pytest.mark.skipif(not os.environ.get("STRATA_TREE") or shutil.which("git") is None, reason="STRATA_TREE (a Strata checkout at STRATA_BASE) not set")
def test_series_then_v7_applies_on_strata_base():
    base = (ADDON / "patches" / "strata" / "STRATA_BASE").read_text().strip()
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "strata"
        subprocess.run(["git", "clone", "-q", "--local", os.environ["STRATA_TREE"], str(work)], check=True)
        subprocess.run(["git", "-C", str(work), "checkout", "-q", base], check=True)
        cfg = ["-c", "user.name=t", "-c", "user.email=t@invalid"]
        subprocess.run(["git", "-C", str(work), *cfg, "am", "-q", *map(str, SERIES), str(V7)], check=True)
