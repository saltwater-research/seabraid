# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""The expert pyramid, installer side (PY DESIGN 5.8 / 7): ONE owner of the server's SEABRAID_STRATA_SHM list, k <= 3 helpers, rung
order == card id order, the opt-in ladder flag set, and the refusals that keep one publisher per rung.  No GPU, no host."""
from __future__ import annotations

import dataclasses
from dataclasses import replace

import pytest
from test_config import host_paths, host_report
from test_plan import FS, KITS, R9700, RTX3060, G, Gpu, Host, model
from test_services import HELPERS, PY, STRATA, Cfg, mkplan, render

from install import config as C
from install import plan as P
from install import services as S

RTX3090 = Gpu(0, "nvidia", "RTX 3090", "sm_86", G(24), G(23.5), pcie_gen_max=3)
C3060 = replace(RTX3060, index=1)
EXTRA = (Gpu(2, "nvidia", "RTX 3060 B", "sm_86", G(12), G(11.5)), Gpu(3, "nvidia", "RTX 3060 C", "sm_86", G(12), G(11.5)))
CONF = "/data/cfg/seabraid.json"


def make(n_help, opts=None, plan=None):
    """config.make_config on a real services-side Plan with n_help helpers (the path setup takes)."""
    return C.make_config(host_report(), plan or mkplan(n_help), host_paths(), opts or C.ConfigOptions(), isdir=lambda p: False)


def unit(n_help, opts=None):
    """The server unit text rendered from config's real output."""
    plan = mkplan(n_help)
    cfg = replace(make(n_help, opts, plan), cfg_json_path=CONF)
    return render(plan, cfg, STRATA, HELPERS, python=PY)[S.SERVER_UNIT]


def env_line(text, key):
    return next(line for line in text.splitlines() if line.startswith(f"Environment={key}="))


def flag(args, name):
    return args[args.index(name) + 1]


def real_plan(*cards, **kw):
    return P.make_plan(Host(gpus=cards, toolkits=KITS, filesystems=FS), model(expert=33, layers=7, experts=13), "/models/m.gguf", **kw)


# ---- the regression: config overwrote services' SEABRAID_STRATA_SHM with the first channel only --------------------------------

@pytest.mark.parametrize("n", [1, 2, 3])
def test_the_server_unit_carries_every_helper_channel_in_helper_index_order(n):
    want = ",".join(f"/seabraid{i}" for i in range(n))
    assert make(n).server_env["SEABRAID_STRATA_SHM"] == want                       # config is the owner ...
    assert env_line(unit(n), "SEABRAID_STRATA_SHM") == f"Environment=SEABRAID_STRATA_SHM={want}"   # ... and the unit shows exactly its value


def test_the_list_is_in_helper_index_order_whatever_order_the_roles_come_in():
    p = mkplan(3)
    shuffled = replace(p, roles=(p.roles[0], p.roles[3], p.roles[1], p.roles[2]))
    assert make(3, plan=shuffled).server_env["SEABRAID_STRATA_SHM"] == "/seabraid0,/seabraid1,/seabraid2"


def test_config_server_env_is_the_complete_truth_for_the_fixed_switches():
    env = make(2).server_env
    assert all(env[k] == v for k, v in C.SERVER_ENV.items())
    srv = unit(2)
    assert all(f"Environment={k}={v}" in srv for k, v in C.SERVER_ENV.items())


def test_services_only_renders_it_it_adds_neither_the_list_nor_the_fixed_switches():
    cfg = Cfg(server_env={"SEABRAID_STRATA_SHM": "/seabraid0,/seabraid1"})
    srv = render(mkplan(2), cfg, STRATA, HELPERS, python=PY)[S.SERVER_UNIT]
    assert "Environment=SEABRAID_STRATA_SHM=/seabraid0,/seabraid1" in srv
    assert "SEABRAID_STRATA_EXCLUDE_HELPER_RAM" not in srv and "STRATA_DISJOINT_ADAPT" not in srv
    assert not hasattr(S, "SERVER_FIXED_ENV")


def test_a_config_without_the_list_is_refused_for_a_plan_with_helpers():
    with pytest.raises(S.ServiceError, match="SEABRAID_STRATA_SHM"):
        render(mkplan(2), Cfg(server_env={}), STRATA, HELPERS, python=PY)


# ---- k = 2 and 3 plans, k = 4 refused; ranking -> rung order ------------------------------------------------------------

@pytest.mark.parametrize("cards,k", [((R9700, RTX3090, C3060), 2), ((R9700, RTX3090, C3060, EXTRA[0]), 3)])
def test_k_helper_plans_get_one_cache_flag_one_channel_and_one_shm_each(cards, k):
    pl = real_plan(*cards)
    assert pl.ready and [h.helper_index for h in pl.helpers()] == list(range(k))
    cfg = C.make_config(host_report(), pl, host_paths(), isdir=lambda p: False)
    args = cfg.cfg_json["args"]
    assert [a for a in args if a.startswith("--expert-cache-device")] == [f"--expert-cache-device{n}" for n in range(1, k + 1)]
    assert [flag(args, f"--expert-cache-device{n}") for n in range(1, k + 1)] == ["auto"] * k
    assert cfg.server_env["SEABRAID_STRATA_SHM"] == ",".join(f"/seabraid{i}" for i in range(k))
    assert [flag(cfg.helper_args[i], "--shm") for i in range(k)] == [f"/seabraid{i}" for i in range(k)]


def test_four_helpers_are_refused_with_the_reason_and_config_will_not_start():
    pl = real_plan(R9700, RTX3090, C3060, *EXTRA)
    assert len(pl.helpers()) == 4 and not pl.ready and pl.mode == "infeasible"
    assert any("three remote tiers" in r and "4 helpers" in r for r in pl.refusals)
    assert "REFUSED: " in pl.explain()
    with pytest.raises(C.ConfigRefused, match="three remote tiers"):
        C.make_config(host_report(), pl, host_paths(), isdir=lambda p: False)


def test_three_helpers_is_the_cap_not_a_refusal():
    assert real_plan(R9700, RTX3090, C3060, EXTRA[0]).ready


def test_ranking_order_is_the_rung_order_and_the_plan_prints_every_name_of_it():
    lead = {("amd", 0): 24.0, ("nvidia", 0): 30.0, ("nvidia", 1): 40.0}
    rows = {("amd", 0): 100.0, ("nvidia", 0): 150.0, ("nvidia", 1): 200.0}
    cards = (R9700, RTX3090, C3060)
    pl = real_plan(*cards, bench_lead=lead, bench=rows)
    assert [(h.name, h.helper_index) for h in pl.helpers()] == [("RTX 3060", 0), ("RTX 3090", 1)]     # faster helper = rung 1
    text = pl.explain()
    assert "HELPER0 nvidia gpu1 RTX 3060" in text and "HELPER1 nvidia gpu0 RTX 3090" in text
    assert "/seabraid0 = --expert-cache-device1 = rung 1" in text and "/seabraid1 = --expert-cache-device2 = rung 2" in text


# ---- the ladder flag set: opt-in, one flag set, never with --remote-expert-opt ------------------------------------------

def test_moves_are_opt_in_by_default():
    assert C.ConfigOptions().ladder is False
    for k in (1, 2, 3):
        cfg = make(k)
        assert not any(a.startswith("--retier") for a in cfg.cfg_json["args"])
        assert "STRATA_RAM_ADAPT" not in cfg.cfg_json["env"] and "STRATA_RAM_ADAPT2" not in cfg.cfg_json["env"]


@pytest.mark.parametrize("k", [1, 2, 3])
def test_the_ladder_adds_the_seabraid_flag_set_sized_by_k(k):
    cfg = make(k, C.ConfigOptions(ladder=True))
    args = cfg.cfg_json["args"]
    assert flag(args, "--retier") == "on" and flag(args, "--retier-promote") == "seabraid" and flag(args, "--retier-fallback") == "1"
    assert flag(args, "--retier-hops") == str(k + 1) and flag(args, "--retier-moves") == str(2 * (k + 1))
    assert cfg.cfg_json["env"]["STRATA_RAM_ADAPT"] == "4" and cfg.cfg_json["env"]["STRATA_RAM_ADAPT2"] == "1"
    assert "--remote-expert-opt" not in args


def counts_ok(args):
    """Python mirror of the engine's check_counts (retier_loop.cpp): the LAST --adapt-every / --adapt-swaps must both be > 0."""
    def last(name):
        v = [args[i + 1] for i, a in enumerate(args[:-1]) if a == name]
        return int(v[-1]) if v else 0
    return last("--adapt-every") > 0 and last("--adapt-swaps") > 0


@pytest.mark.parametrize("k", [1, 2, 3])
def test_the_ladder_argv_passes_the_engines_adapt_count_rule(k):
    """The engine exits 2 at startup unless both adapt flags are > 0; the old ladder set carried neither."""
    assert counts_ok(make(k, C.ConfigOptions(ladder=True)).cfg_json["args"])
    assert not counts_ok(make(k).cfg_json["args"])                                       # no ladder: not the ladder's business


def test_a_user_adapt_value_wins_and_a_non_positive_one_is_refused():
    args = make(2, C.ConfigOptions(ladder=True, extra_args=("--adapt-every", "32", "--adapt-swaps=4"))).cfg_json["args"]
    assert counts_ok(args) and args[-3:] == ["--adapt-every", "32", "--adapt-swaps=4"]      # theirs come last: the engine takes the last
    for bad in (("--adapt-every", "0"), ("--adapt-swaps=-1",), ("--adapt-every", "x")):
        with pytest.raises(C.ConfigRefused, match="adapt"):
            make(2, C.ConfigOptions(ladder=True, extra_args=bad))


def test_the_fallback_boundary_has_its_own_switch():
    args = make(2, C.ConfigOptions(ladder=True, ladder_fallback=False)).cfg_json["args"]
    assert flag(args, "--retier-fallback") == "0" and flag(args, "--retier-promote") == "seabraid"


def test_the_ladder_never_runs_with_remote_expert_opt():
    with pytest.raises(C.ConfigRefused, match="remote-expert-opt"):
        make(2, C.ConfigOptions(ladder=True, extra_args=("--remote-expert-opt",)))
    assert "--remote-expert-opt" in make(2, C.ConfigOptions(extra_args=("--remote-expert-opt",))).cfg_json["args"]   # not the ladder's rule


def test_the_ladder_needs_a_helper_and_will_not_run_beside_a_legacy_promote():
    with pytest.raises(C.ConfigRefused, match="at least one helper"):
        make(0, C.ConfigOptions(ladder=True))
    with pytest.raises(C.ConfigRefused, match="legacy"):
        make(2, C.ConfigOptions(ladder=True, extra_args=("--retier-promote", "legacy")))


# ---- legacy retier and D4 prefill remote compute are one-helper only --------------------------------------------------

ADAPT = ("--adapt-every", "64", "--adapt-swaps", "8")   # the engine refuses --retier without routing counts


@pytest.mark.parametrize("extra", [("--retier", "on"), ("--retier", "on", "--retier-promote", "legacy")])
def test_legacy_retier_is_refused_with_more_than_one_helper_and_names_the_way_out(extra):
    with pytest.raises(C.ConfigRefused, match="--retier-promote seabraid"):
        make(2, C.ConfigOptions(extra_args=extra + ADAPT))
    make(1, C.ConfigOptions(extra_args=extra + ADAPT))                                         # one helper: the legacy loop is fine


def test_the_seabraid_promote_and_retier_off_are_allowed_with_k_helpers():
    make(3, C.ConfigOptions(extra_args=("--retier", "on", "--retier-promote", "seabraid") + ADAPT))
    make(3, C.ConfigOptions(extra_args=("--retier", "off")))


def test_d4_prefill_remote_compute_is_refused_with_more_than_one_helper():
    on = C.ConfigOptions(extra_env={"STRATA_PREFILL_REMOTE_COMPUTE": "1"})
    with pytest.raises(C.ConfigRefused, match="STRATA_PREFILL_REMOTE_COMPUTE"):
        make(2, on)
    make(1, on)
    make(3, C.ConfigOptions(extra_env={"STRATA_PREFILL_REMOTE_COMPUTE": "0"}))


def test_options_are_frozen_so_the_flag_cannot_be_flipped_after_a_plan_is_judged():
    with pytest.raises(dataclasses.FrozenInstanceError):
        C.ConfigOptions().ladder = True
