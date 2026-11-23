# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Checker for scripts/overflow/ov_cap.py: the capacity arithmetic and the rule that a cap must actually overflow.  Pure, CPU only."""
import copy
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ovf_synth as S  # noqa: E402
import ov_cap  # noqa: E402
import qual_cfg  # noqa: E402


def cfg_with(**cap):
    cfg = copy.deepcopy(S.load_cfg())
    cfg["capacity"].update(cap)
    return cfg


def test_o1_arithmetic_matches_the_hand_calculation():
    p = S.load_cfg()["plans"]["O1"]
    blob = 47.0 / (48 * 512)
    assert p.ram_blobs == int(6.0 / blob) == 3137
    assert p.card_blobs == 3000 and p.file_blobs == 24576 - 3000 - 3137 == 18439
    assert p.file_share == pytest.approx(0.7503, abs=1e-4)
    assert p.engine_cap_bytes == 16 * 2**30 and p.helper_cap_bytes == 6 * 2**30


def test_helpers_add_card_slots_and_shrink_the_ssd_share():
    plans = S.load_cfg()["plans"]
    assert [plans[a].card_blobs for a in ("O1", "O2", "O3")] == [3000, 6000, 9000]
    assert plans["O1"].file_share > plans["O2"].file_share > plans["O3"].file_share >= 0.40
    assert plans["O3"].helper_slots == (3000, 3000) and plans["R"].main_slots is None


def test_every_capped_arm_prints_its_arithmetic_with_the_margin():
    text = "\n".join(S.load_cfg()["plans"]["O2"].lines)
    assert "SSD = 15439 blobs" in text and "rule: >= 40%" in text and "margin 22.8 points" in text
    assert "ESTIMATE" in text and "swap 0" in text


def test_a_cap_that_does_not_overflow_is_refused_at_plan_time():
    with pytest.raises(qual_cfg.ConfigError, match="arm O1: only 20.8% .* does not overflow"):
        ov_cap.build(cfg_with(main_slots=9000, ram_gib=20.0))


def test_slots_larger_than_the_card_are_refused():
    with pytest.raises(qual_cfg.ConfigError, match="main card holds"):
        ov_cap.build(cfg_with(main_slots=14000))
    with pytest.raises(qual_cfg.ConfigError, match="helper card holds"):
        ov_cap.build(cfg_with(helper_slots=[14000, 100]))


def test_more_helpers_than_configured_slots_is_refused():
    with pytest.raises(qual_cfg.ConfigError, match="helper_slots"):
        ov_cap.build(cfg_with(helper_slots=[3000]))


def test_an_arm_without_a_tiers_block_is_refused():
    cfg = copy.deepcopy(S.load_cfg())
    del cfg["arms"]["O1"]["tiers"]
    with pytest.raises(qual_cfg.ConfigError, match="tiers"):
        ov_cap.build(cfg)


def test_bad_capacity_values_are_named():
    for key, bad in (("experts_gib", 0), ("min_file_share", 1.5), ("helper_slots", [0]), ("layers", -1)):
        with pytest.raises(qual_cfg.ConfigError, match=key):
            ov_cap.build(cfg_with(**{key: bad}))
