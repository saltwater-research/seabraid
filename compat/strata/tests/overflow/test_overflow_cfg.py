# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Checker for scripts/overflow/ov_cfg.py and the dry run of overflow.py: the commands, the cgroup prefix, the hooks, the plan order, and
that a dry run starts and creates nothing.  CPU only; no engine, no systemd."""
import os
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ovf_synth as S  # noqa: E402
import ov_cfg  # noqa: E402
import overflow as OV  # noqa: E402
import qual_cfg  # noqa: E402

VARS = ["--var", "gpu=0", "--var", "rocm=/rocm", "--var", "pack=/pack", "--var", "native=/model.gguf"]


def argv_of(cfg, arm, workload="speed"):
    return ov_cfg.engine_command(cfg, qual_cfg.Run(workload, arm, "base", 0), "/run")[0]


def test_capped_engine_starts_in_a_scope_with_the_exact_byte_limit_and_no_swap():
    cfg = S.load_cfg()
    argv = argv_of(cfg, "O1")
    assert argv[:4] == ["systemd-run", "--user", "--scope", "--quiet"]
    assert "MemoryMax=17179869184" in argv and "MemorySwapMax=0" in argv
    assert argv[argv.index("--") + 1] == "bash"
    assert argv[-4:] == ["--expert-cache", "3000", "--resident-budget-gib", "6"]


def test_the_uncapped_reference_has_no_scope_no_budget_and_no_file_tier_flag():
    argv = argv_of(S.load_cfg(), "R")
    assert argv[0] == "bash" and "systemd-run" not in argv and "--resident-budget-gib" not in argv
    assert argv[-4:] == ["--expert-cache", "auto", "--expert-cache-device1", "auto"]


def test_helper_is_in_its_own_limited_scope_and_o1_has_none():
    cfg = S.load_cfg()
    h = ov_cfg.helper_specs(cfg, qual_cfg.Run("speed", "O2", "base", 0))
    assert len(h) == 1 and h[0][0][:2] == ["systemd-run", "--user"] and "MemoryMax=6442450944" in h[0][0]
    assert h[0][2] == "/dev/shm/seabraid-ovf1"
    assert ov_cfg.helper_specs(cfg, qual_cfg.Run("speed", "O1", "base", 0)) == []
    r = ov_cfg.helper_specs(cfg, qual_cfg.Run("speed", "R", "base", 0))
    assert r[0][0][0] != "systemd-run"  # the reference helper is not limited


def test_o3_has_two_helpers_on_two_channels_and_the_engines_comma_list():
    cfg = S.load_cfg(gpu=0)
    h = ov_cfg.helper_specs(cfg, qual_cfg.Run("speed", "O3", "base", 0))
    assert [x[2] for x in h] == ["/dev/shm/seabraid-ovf1", "/dev/shm/seabraid-ovf2"]
    run = qual_cfg.Run("speed", "O3", "base", 0)
    argv, env = ov_cfg.engine_command(cfg, run, "/run")
    assert env["SEABRAID_STRATA_SHM"] == "/seabraid-ovf1,/seabraid-ovf2"
    assert argv[-4:] == ["--expert-cache-device1", "3000", "--expert-cache-device2", "3000"] and "--retier" not in argv
    assert not hasattr(ov_cfg, "blocked")  # O3 is runnable now: nothing refuses it before the engine log shows the channels


def test_the_ladder_arm_takes_its_flags_and_env_from_the_installer_by_import_and_nothing_else_differs_from_o3():
    from install import _ladder
    cfg = S.load_cfg()
    o3, o3l = argv_of(cfg, "O3"), argv_of(cfg, "O3L")
    assert o3l[: len(o3)] == o3 and o3l[len(o3):] == _ladder.ladder_args(2, True)
    env3 = ov_cfg.engine_command(cfg, qual_cfg.Run("speed", "O3", "base", 0), "/run")[1]
    env3l = ov_cfg.engine_command(cfg, qual_cfg.Run("speed", "O3L", "base", 0), "/run")[1]
    assert {k: v for k, v in env3l.items() if k not in env3} == dict(_ladder.LADDER_ENGINE_ENV) and not set(_ladder.LADDER_ENGINE_ENV) & set(env3)
    assert not any(a.startswith("--retier") for arm in ("O1", "O2", "O3", "R") for a in argv_of(cfg, arm))
    assert ov_cfg.ladder_flags(2) == (_ladder.ladder_args(2, True), dict(_ladder.LADDER_ENGINE_ENV))


def test_ladder_extra_args_reach_only_the_ladder_arm_and_unbuffered_reaches_every_capped_arm():
    cfg = S.load_cfg(ladder_extra_args="--adapt-every 64 --adapt-swaps 8", unbuffered="1")
    assert argv_of(cfg, "O3L")[-4:] == ["--adapt-every", "64", "--adapt-swaps", "8"]
    assert "--adapt-every" not in argv_of(cfg, "O3") and "--adapt-every" not in argv_of(cfg, "O2")
    envs = {a: ov_cfg.engine_command(cfg, qual_cfg.Run("speed", a, "base", 0), "/run")[1] for a in ("R", "O1", "O3", "O3L")}
    assert [envs[a].get("STRATA_UNBUFFERED_LOAD") for a in ("R", "O1", "O3", "O3L")] == [None, "1", "1", "1"]


def test_the_installers_refusals_apply_to_a_conflicting_ladder_arm():
    with pytest.raises(qual_cfg.ConfigError, match="arm O3L: .*--remote-expert-opt"):
        S.load_cfg(ladder_extra_args="--remote-expert-opt 1")
    with pytest.raises(qual_cfg.ConfigError, match="arm O3L: .*legacy"):
        S.load_cfg(ladder_extra_args="--retier-promote legacy")


def test_plan_order_is_reference_first_then_capacity_order_then_noise_and_reference_speed():
    cfg = S.load_cfg()
    tags = [r.tag for r in ov_cfg.plan(cfg, ["R", "O1", "O2"])]
    assert tags == ["w7-R-s0", "speed-O1-s0", "w7-O1-s0", "speed-O2-s0", "w7-O2-s0", "w7-R-s1", "speed-R-s0"]
    assert [r.tag for r in ov_cfg.plan(cfg)][-4:-2] == ["speed-O3-s0", "w7-O3-s0"] and not any("O3L" in r.tag for r in ov_cfg.plan(cfg))  # the default sweep runs O3 (static), not O3L
    assert [r.tag for r in ov_cfg.plan(cfg, ["R", "O1"], kl=False)] == ["speed-O1-s0", "speed-R-s0"]


def test_estimate_is_labelled_and_scales_with_the_starts():
    cfg = S.load_cfg()
    one, three = ov_cfg.estimate(cfg, ov_cfg.plan(cfg, ["R", "O1"])), ov_cfg.estimate(cfg, ov_cfg.plan(cfg, ["R", "O1", "O2"]))
    assert 0 < one[0] < one[1] < three[1] and "ESTIMATE" in cfg["estimate"]["note"]


def test_the_capture_command_uses_the_arms_own_engine_config_inside_the_scope():
    cfg = S.load_cfg()
    cap = ov_cfg.capture_command(cfg, qual_cfg.Run("w7", "O2", "base", 0), "/run")
    assert cap[:4] == ["systemd-run", "--user", "--scope", "--quiet"] and "<forced_cfg_O2>" in cap and cap[-2:] == ["--path", "decode"]
    assert ov_cfg.capture_command(cfg, qual_cfg.Run("w7", "R", "base", 0), "/run")[0] == "python3"


def test_a_live_load_without_a_machine_var_fails_before_anything_starts():
    with pytest.raises(qual_cfg.ConfigError, match="gpu"):
        ov_cfg.engine_command(ov_cfg.load(os.path.join(S.SCRIPTS, "arms.json"), {}, False), qual_cfg.Run("speed", "O1", "base", 0), "/run")


def test_dry_run_prints_the_plan_the_arithmetic_and_the_mechanism_and_creates_nothing(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    before = sorted(os.listdir(tmp_path))
    rc = OV.main(["--out", str(tmp_path / "out"), "--dry-run", "--arms", "R,O1,O2,O3,O3L", *VARS])
    text = capsys.readouterr().out
    assert rc == 0 and sorted(os.listdir(tmp_path)) == before  # no out directory, no file
    for needle in ("CAPACITY", "SSD = 18439 blobs", "MemoryMax=17179869184", "MEMORY LIMIT", "PAGE CACHE", "DATA GATE", "LADDER (arm O3L", "--retier-promote seabraid",
                   "STRATA_RAM_ADAPT", "zero stale-placement records", "ESTIMATE (not measured"):
        assert needle in text


def test_bad_arguments_exit_2():
    assert OV.main(["--dry-run"]) == 2
    assert OV.main(["--out", "x", "--dry-run", "--arms", "R,ZZ"]) == 2
