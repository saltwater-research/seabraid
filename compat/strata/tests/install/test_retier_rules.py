# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""The installer's mirror of the engine's --retier* rules (install/_retier_rules.py), held to the engine by ONE shared case file.

tests/data/retier_cases.tsv is also run through the engine's own retier_parse_flag / retier_check / retier_check_helpers by
tests/retier_rules_case_test.cpp (CTest seabraid_retier_rules_case_test).  Same rows, same expected text, so a rule edited on one side
only fails the other.  The second half drives the same findings through config.make_config, the place an operator meets them."""
from __future__ import annotations

import pathlib

import pytest
from test_ladder import make

from install import _ladder
from install import _retier_rules as R
from install import config as C

CASES = pathlib.Path(__file__).resolve().parent.parent / "data" / "retier_cases.tsv"


def rows():
    out = []
    for line in CASES.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            name, helpers, args, err = line.split("\t")
            out.append(pytest.param(int(helpers), [] if args == "-" else ["" if a == "''" else a for a in args.split(" ")], None if err == "-" else err, id=name))
    return out


def test_the_case_file_is_not_empty_and_covers_both_outcomes():
    cases = rows()
    assert len(cases) >= 60
    assert any(c.values[2] is None for c in cases) and any(c.values[2] is not None for c in cases)


@pytest.mark.parametrize("helpers,args,error", rows())
def test_the_mirror_gives_the_engines_verdict_and_text_on_every_shared_case(helpers, args, error):
    assert R.check(args, helpers) == error


# ---- the findings, through make_config ------------------------------------------------------------------------------------

@pytest.mark.parametrize("override", [("--retier-moves", "1"), ("--retier-moves=8",), ("--retier-every", "0"), ("--retier-margin", "0.5"),
                                      ("--retier-cooldown", "-1"), ("--retier-ceiling", "2"), ("--retier-fallback", "2")])
def test_ladder_overrides_the_engine_would_reject_are_refused_at_plan_time(override):
    with pytest.raises(C.ConfigRefused):
        make(2, C.ConfigOptions(ladder=True, extra_args=override))


def test_ladder_moves_override_is_refused_with_the_engines_own_wording():
    with pytest.raises(C.ConfigRefused, match=r"--retier-hops 3 needs --retier-moves >= 6 .* got --retier-moves 1"):
        make(2, C.ConfigOptions(ladder=True, extra_args=("--retier-moves", "1")))
    make(2, C.ConfigOptions(ladder=True, extra_args=("--retier-moves", "12", "--retier-every", "16")))   # a valid override still wins


def test_the_ladder_with_file_moves_and_a_low_moves_override_is_refused():
    with pytest.raises(C.ConfigRefused, match="--retier-moves >= 6"):
        make(2, C.ConfigOptions(ladder=True, file_moves=True, extra_args=("--retier-moves", "1")))


def test_the_equals_promote_spelling_is_refused_the_engine_does_not_know_it():
    with pytest.raises(C.ConfigRefused, match="--retier-promote=seabraid"):
        make(2, C.ConfigOptions(ladder=True, extra_args=("--retier-promote=seabraid",)))
    with pytest.raises(C.ConfigRefused, match="unknown argument: --retier-promote=seabraid"):
        make(2, C.ConfigOptions(extra_args=("--retier-promote=seabraid", "--adapt-every", "64", "--adapt-swaps", "8")))


@pytest.mark.parametrize("extra", [("--retier", "on"), ("--retier-promote", "seabraid"), ("--retier", "on", "--retier-promote", "seabraid")])
def test_the_ladder_owns_its_retier_flags_a_second_copy_is_refused(extra):
    with pytest.raises(C.ConfigRefused, match="already"):
        make(2, C.ConfigOptions(ladder=True, extra_args=extra))
    args = make(2, C.ConfigOptions(ladder=True)).cfg_json["args"]
    assert args.count("--retier") == 1 and args.count("--retier-promote") == 1   # and the unadorned ladder renders each once


@pytest.mark.parametrize("extra", [("--retier", "on", "--retier-promote", "legacy"), ("--retier", "on", "--retier-promote", "on"), ("--retier", "on"),
                                   ("--retier", "zero-op")])
def test_legacy_aliases_and_modes_without_seabraid_are_refused_with_k_helpers(extra):
    adapt = ("--adapt-every", "64", "--adapt-swaps", "8")
    with pytest.raises(C.ConfigRefused, match="serves ONE helper"):
        make(2, C.ConfigOptions(extra_args=extra + adapt))
    make(1, C.ConfigOptions(extra_args=extra + adapt))


def test_retier_without_adapt_flags_takes_the_engines_defaults_and_an_explicit_zero_is_refused():
    make(2, C.ConfigOptions(extra_args=("--retier", "on", "--retier-promote", "seabraid")))   # engine: adapt_every 4, adapt_swaps 96
    with pytest.raises(C.ConfigRefused, match="routing counts"):
        make(2, C.ConfigOptions(extra_args=("--retier", "on", "--retier-promote", "seabraid", "--adapt-every", "0")))


def test_every_ladder_arg_set_the_installer_renders_passes_the_engines_rules():
    for k in (1, 2, 3):
        for fb in (True, False):
            assert R.check(_ladder.ladder_args(k, fb), k) is None


# ---- the facts the C++ fixture copies from generate.cpp, pinned to the patches that carry them -------------------------------
# The fixture re-implements generate.cpp's parse loop, so a change to the engine there would not fail it.  What the patches show
# of generate.cpp is pinned here; `adapt_every = 4` and the "unknown argument" text are NOT in any patch context (verified), so
# those two stay an unpinned copy and the case-file header says so.
ROOT = pathlib.Path(__file__).resolve().parents[4]
PATCHES = [ROOT / "adapters" / "strata" / "strata-v7-seabraid.patch", *sorted((ROOT / "compat" / "strata" / "patches" / "strata").glob("0004-*.patch"))]


def test_the_engines_adapt_swaps_default_in_the_patch_context_is_the_one_the_mirror_names():
    seen = [p for p in PATCHES if "     int adapt_swaps = 96;" in p.read_text(encoding="utf-8", errors="replace")]
    assert len(seen) == len(PATCHES), "a patch no longer shows `int adapt_swaps = 96;`: re-read generate.cpp and update ENGINE_ADAPT_DEFAULTS"
    assert R.ENGINE_ADAPT_DEFAULTS["--adapt-swaps"] == 96


def test_the_patch_still_hands_retier_flags_to_retier_parse_flag_and_leaves_the_rest_to_the_unknown_flag_branch():
    text = (ROOT / "compat" / "strata" / "patches" / "strata" / "0004-D7-call-site-retier-schedules-tier-policy-rounds-off.patch").read_text(
        encoding="utf-8", errors="replace")
    assert "seabraid::retier_parse_flag(a, next, o.retier, retier_err); rr != 0" in text
    assert "An unknown flag is an ERROR and not a warning" in text
