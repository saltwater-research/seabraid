# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Checker for scripts/payoff/pay_cfg.py and the dry-run CLI: the three arms differ in exactly the flags the design names,
the plan rotates the arm order, a bad sweep or a missing machine var fails before anything starts, a dry run creates nothing.
CPU only; no engine is started (the dry run is a subprocess that prints)."""
import json
import os
import pathlib
import subprocess
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import payoff_synth as S  # noqa: E402
import pay_cfg  # noqa: E402
import qual_cfg  # noqa: E402
import qual_engine  # noqa: E402

CONFIG = os.path.join(S.SCRIPTS, "arms.json")
PAYOFF = os.path.join(S.SCRIPTS, "payoff.py")


@pytest.fixture(scope="module")
def cfg():
    return pay_cfg.load(CONFIG, {}, dry=True)


def run_of(cfg, workload, arm):
    return next(r for r in pay_cfg.plan(cfg) if r.workload == workload and r.arm == arm)


def argv_env(cfg, workload, arm):
    return qual_engine.build_command(cfg, run_of(cfg, workload, arm), "/r")


def after(argv, flag):
    return argv[argv.index(flag) + 1]


def test_arm_flags_are_exactly_the_design(cfg):
    a, _ = argv_env(cfg, "cap8000", "A")
    b, _ = argv_env(cfg, "cap8000", "B")
    c, _ = argv_env(cfg, "cap8000", "C")
    assert (after(a, "--pcie-frac"), after(b, "--pcie-frac"), after(c, "--pcie-frac")) == ("0", "1", "0")
    assert "--expert-cache-device1" in c and "--expert-cache-device1" not in a + b
    assert after(c, "--expert-cache-device1") == "auto"


def test_every_arm_gets_the_same_cap_ram_budget_and_context(cfg):
    for wl in ("cap11000", "cap2500"):
        rows = []
        for arm in "ABC":
            argv, _ = argv_env(cfg, wl, arm)
            rows.append((after(argv, "--expert-cache"), after(argv, "--resident-budget-gib"), after(argv, "--max-context"), "--stats" in argv))
        assert len(set(rows)) == 1 and rows[0][0] == wl[3:] and rows[0][3]


def test_helper_env_is_only_arm_c_and_a_b_drop_it(cfg):
    _, env_a = argv_env(cfg, "cap8000", "A")
    _, env_c = argv_env(cfg, "cap8000", "C")
    assert env_a["NOHELPER"] == "1" and "SEABRAID_STRATA_SHM" not in env_a and "SEABRAID_QWEN_HELPER" not in env_a
    assert env_c["SEABRAID_STRATA_SHM"] == "/seabraid-payoff" and "NOHELPER" not in env_c
    assert qual_engine.build_helper(cfg, run_of(cfg, "cap8000", "A"), "/r") is None
    assert "--shm" in qual_engine.build_helper(cfg, run_of(cfg, "cap8000", "C"), "/r")[0]


def test_quality_starts_add_short_read_and_logpos_only_there(cfg):
    speed, env_s = argv_env(cfg, "cap8000", "A")
    kl, env_k = argv_env(cfg, "cap8000kl", "A")
    assert "--short-read" not in speed and after(kl, "--short-read") == "1000"
    assert "STRATA_LOGPOS" not in env_s and env_k["STRATA_LOGPOS"] == "/r/rows.tsv" and env_k["STRATA_LOGPOS_TOPK"] == "32"
    assert env_s["STRATA_DECODE_TIMING"] == "1"


def test_plan_covers_every_arm_and_cap_and_rotates_the_order(cfg):
    speed = [r for r in pay_cfg.plan(cfg) if not r.workload.endswith("kl")]
    assert len(speed) == 15 and len({(r.workload, r.arm) for r in speed}) == 15
    first_arm_per_cap = [next(r.arm for r in speed if r.workload == f"cap{c}") for c in (11000, 8000, 6000, 4000, 2500)]
    assert first_arm_per_cap == ["A", "B", "C", "A", "B"]


def test_baseline_repeats_its_quality_start_for_the_noise_floor(cfg):
    kl = [r for r in pay_cfg.plan(cfg) if r.workload.endswith("kl")]
    assert len(kl) == 15 + 1 and kl[-1] == qual_cfg.Run("cap11000kl", "A", "base", 1)


def test_starts_option_repeats_each_arm_per_point(cfg):
    runs = pay_cfg.plan(cfg, starts=2)
    assert sum(1 for r in runs if r.workload == "cap8000" and r.arm == "C") == 2


def test_no_kl_removes_the_quality_starts():
    c = pay_cfg.load(CONFIG, {}, dry=True, sweep={"kl": False})
    assert not [r for r in pay_cfg.plan(c) if r.workload.endswith("kl")]


@pytest.mark.parametrize("sweep, word", [({"caps": [11000, 4000]}, "3 distinct"), ({"caps": [8000, 8000, 4000]}, "3 distinct"),
                                         ({"n_requests": 5}, "n_requests"), ({"arms": ["A", "B"]}, "second_gpu_arm"),
                                         ({"baseline_arm": "Z"}, "baseline_arm")])
def test_bad_sweeps_fail_before_anything_starts(sweep, word):
    with pytest.raises(qual_cfg.ConfigError, match=word):
        pay_cfg.load(CONFIG, {}, dry=True, sweep=sweep)


def test_live_load_refuses_an_unset_machine_var(cfg):
    live = pay_cfg.load(CONFIG, {}, dry=False)
    with pytest.raises(qual_cfg.ConfigError, match="no value"):
        pay_cfg.describe(live, pay_cfg.plan(live)[0])


def test_estimate_is_a_range_and_grows_with_starts(cfg):
    runs = pay_cfg.plan(cfg)
    lo, hi = pay_cfg.estimate(cfg, runs)
    assert 0 < lo < hi
    assert pay_cfg.estimate(cfg, runs[:3])[1] < hi


def test_dry_run_prints_every_command_and_creates_nothing(tmp_path):
    out = tmp_path / "never"
    done = subprocess.run([sys.executable, "-B", PAYOFF, "--out", str(out), "--dry-run"], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert not out.exists()
    starts = [ln for ln in done.stdout.splitlines() if ln.startswith("START ")]
    assert len(starts) == 31 and done.stdout.count("  engine: ") == 31
    assert "--pcie-frac 1" in done.stdout and "SEABRAID_STRATA_SHM=/seabraid-payoff" in done.stdout and "ESTIMATE" in done.stdout
    assert "helper: bin/helper-cuda" in done.stdout


def test_dry_run_with_overrides_prints_the_given_values(tmp_path):
    done = subprocess.run([sys.executable, "-B", PAYOFF, "--out", str(tmp_path / "x"), "--dry-run", "--caps", "9000,5000,3000", "--arms", "A,C",
                           "--kl", "off", "--var", "gpu=2", "--var", "port=9001"], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert "PLAN 6 engine starts" in done.stdout and "CUDA_VISIBLE_DEVICES=2" in done.stdout and "QUAL_PORT=9001" in done.stdout


def test_live_run_without_machine_vars_exits_2_and_starts_nothing(tmp_path):
    out = tmp_path / "live"
    done = subprocess.run([sys.executable, "-B", PAYOFF, "--out", str(out)], capture_output=True, text=True, timeout=60)
    assert done.returncode == 2 and "config error" in done.stderr and not out.exists()


def test_config_file_is_valid_json_with_the_documented_flags_in_the_note():
    doc = json.load(open(CONFIG, encoding="utf-8"))["_doc"]
    for flag in ("--expert-cache", "--pcie-frac", "--expert-cache-device1", "--resident-budget-gib", "--max-context", "--short-read", "--stats"):
        assert flag in doc
