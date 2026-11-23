# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""PLAN ranking: who is MAIN.  Capacity first, bandwidth second, architecture last; a measurement replaces the proxy order
but never the 'can it hold main' gate.  Self-contained fakes (no shared fixtures)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from install import plan as P

G = lambda x: int(x * (1 << 30))  # noqa: E731


@dataclass(frozen=True)
class Gpu:
    index: int
    vendor: str
    name: str
    arch: Optional[str]
    vram_total_bytes: Optional[int]
    vram_free_bytes: Optional[int]
    gtt_total_bytes: Optional[int] = None
    pcie_gen_max: Optional[int] = 4
    pcie_width_max: Optional[int] = 16
    uuid: Optional[str] = None
    holders: tuple = ()


@dataclass(frozen=True)
class Toolkit:
    vendor: str
    kind: str
    root: str
    version: str
    compiler: str
    host_compiler: Optional[str] = None
    host_compiler_tried: tuple = ()
    lib_dirs: tuple = ()


@dataclass(frozen=True)
class Filesystem:
    mount: str
    free_bytes: Optional[int]
    rotational: Optional[bool]
    fstype: str = "ext4"


@dataclass(frozen=True)
class Host:
    gpus: tuple
    toolkits: tuple = (Toolkit("amd", "rocm", "/opt/r", "7", "/opt/r/clang++", lib_dirs=("/opt/r/lib",)),
                       Toolkit("nvidia", "cuda", "/opt/c", "12.4", "/opt/c/nvcc", host_compiler="/usr/bin/gcc-13"))
    filesystems: tuple = (Filesystem("/", G(500), False), Filesystem("/dev/shm", G(15), None, "tmpfs"))
    ram_total_bytes: Optional[int] = G(64)
    ram_available_bytes: Optional[int] = G(50)
    ttm_pages_limit_bytes: Optional[int] = G(30)
    memlock_bytes: Optional[int] = 8 << 20
    blockers: tuple = ()


R9700 = Gpu(0, "amd", "AMD Radeon AI PRO R9700", "gfx1201", G(32), G(31.5), gtt_total_bytes=G(30))
RTX3090 = Gpu(0, "nvidia", "NVIDIA GeForce RTX 3090", "sm_86", G(24), G(23.5))
RTX4060 = Gpu(1, "nvidia", "NVIDIA GeForce RTX 4060", "sm_89", G(8), G(7.5))


def model(expert=5.0, dense=3.0) -> P.ModelFacts:
    return P.ModelFacts(48, 512, G(expert), G(dense), 2560, 1510400)


def plan(*gpus, bench=None, bench_lead=None, why="", opts=None, **kw) -> P.Plan:
    return P.make_plan(Host(gpus=gpus), model(**kw), "/m.gguf", bench=bench, bench_why=why, bench_lead=bench_lead,
                       **({"opts": opts} if opts else {}))


def test_a_4060_never_outranks_a_3090_for_main_on_architecture_alone():
    pl = plan(RTX4060, RTX3090)
    assert pl.main().name.endswith("3090") and pl.ranking == "heuristic"


def test_this_boxs_r9700_32g_is_main_over_the_3090_24g():
    pl = plan(RTX3090, R9700)
    assert pl.main().name.endswith("R9700")


def test_a_card_too_small_for_mains_resident_set_is_never_main_even_if_it_has_more_total_vram():
    # index 1: two cards never share (vendor, index), and rank() now refuses duplicate keys
    held = Gpu(1, "nvidia", "NVIDIA GeForce RTX 5090", "sm_120", G(32), G(2.0))   # 32 GiB card, 2 GiB free: something holds it
    pl = plan(held, RTX3090)
    # Not merely "3090 is main": without the gate the held card is tried as main, rejected, and DROPPED.  With it, it is
    # demoted and still serves as a helper.  Asserting the helper role is what proves the gate ran.
    assert pl.main().name.endswith("3090") and [h.name for h in pl.helpers()] == [held.name]


def test_a_too_small_card_stays_off_main_even_when_it_measures_fastest():
    held = Gpu(1, "nvidia", "NVIDIA GeForce RTX 5090", "sm_120", G(32), G(2.0))
    pl = plan(RTX3090, held, bench_lead={("nvidia", 0): 100.0, ("nvidia", 1): 1.0})   # the held card has the lowest lead time
    assert pl.ranking == "bench" and pl.main().gpu_index == 0 and [h.gpu_index for h in pl.helpers()] == [1]


def test_equal_vram_falls_to_bandwidth_not_generation():
    # 24 GiB each: the 4090 reads 1008 GB/s, the 3090 936; the SKU table decides, and the note says why.
    rtx4090 = Gpu(1, "nvidia", "NVIDIA GeForce RTX 4090", "sm_89", G(24), G(23.5))
    pl = plan(RTX3090, rtx4090)
    assert pl.main().name.endswith("4090") and "equal VRAM, tie broken by memory bandwidth" in pl.explain()


def test_measurement_overrides_the_heuristic_order_when_they_disagree():
    heur = plan(R9700, RTX3090)
    assert heur.main().name.endswith("R9700")
    meas = plan(R9700, RTX3090, bench_lead={("amd", 0): 26.7, ("nvidia", 0): 24.0})
    assert meas.ranking == "bench" and meas.main().name.endswith("3090")


def test_a_missing_measurement_for_one_card_means_heuristic_for_all():
    pl = plan(R9700, RTX3090, bench_lead={("amd", 0): 1.0, ("nvidia", 0): None}, why="x")
    assert pl.ranking == "heuristic" and pl.main().name.endswith("R9700")


def test_explain_states_the_measured_reason():
    pl = plan(R9700, RTX3090, bench_lead={("amd", 0): 23.0, ("nvidia", 0): 24.0})
    assert ("  main: AMD Radeon AI PRO R9700 (measured lead job, 1 stream: 23.0 ms/window, 32.0 GiB) over "
            "NVIDIA GeForce RTX 3090 (24.0 ms/window, 24.0 GiB)") in pl.explain()


def test_explain_states_the_heuristic_reason_and_why_not_measured():
    pl = plan(R9700, RTX3090, why="no prebuilt seabraid-helper for amd")
    assert ("  main: AMD Radeon AI PRO R9700 by heuristic (32.0 GiB vs 24.0 GiB, "
            "not measured: no prebuilt seabraid-helper for amd)") in pl.explain()


def test_explain_names_a_runner_up_that_is_too_small():
    held = Gpu(1, "nvidia", "NVIDIA GeForce RTX 5090", "sm_120", G(32), G(2.0))
    assert "is too small to hold main's resident set" in plan(RTX3090, held, why="x").explain()


def test_candidates_are_exactly_the_cards_make_plan_ranks():
    bad = Gpu(0, "intel", "Arc", "xe", G(16), G(16))
    c = P.candidates(Host(gpus=(R9700, RTX3090, bad)))
    assert {(x.vendor, x.index) for x in c} == {("amd", 0), ("nvidia", 0)}


def test_bandwidth_table_rows_each_carry_a_citation():
    import re
    from seabraid import ranking as R
    rows = [l for l in open(R.__file__, encoding="utf-8").read().splitlines() if re.match(r'\s+\("[^"]+", [\d.]+\),', l)]
    assert len(rows) == len(R._BANDWIDTH_GBS) and all("#" in l and "Gbps" in l for l in rows)


# ---------------------------------------------------------------------------------------------- R1-R3 (N-card ladder)
from seabraid import ranking as R  # noqa: E402

RTX3060 = Gpu(1, "nvidia", "NVIDIA GeForce RTX 3060", "sm_86", G(12), G(11.5))
ROWS = {("amd", 0): 270000.0, ("nvidia", 0): 226000.0}                  # DESIGN-LOG 2026-11-09: rows/s favours the R9700
LEAD1 = {("amd", 0): 26.7, ("nvidia", 0): 24.0}                          # one stream: the 3090 is the faster card 1
LEAD4 = {("amd", 0): 24.0, ("nvidia", 0): 26.0}                          # four streams: the R9700 is
S4 = P.PlanOptions(streams=4)


def test_r1_lead_job_picks_card_1_even_when_expert_rows_per_s_says_the_other_card():
    pl = plan(R9700, RTX3090, bench=ROWS, bench_lead=LEAD1)
    assert pl.main().name.endswith("3090") and pl.ranking == "bench" and pl.rest_ranking == "bench"
    assert "measured lead job, 1 stream: 24.0 ms/window" in pl.explain()


def test_r1_expert_rows_alone_never_decide_card_1_and_the_plan_says_the_lead_bench_was_not_run():
    pl = plan(R9700, RTX3090, bench=ROWS)
    assert pl.main().name.endswith("R9700") and pl.ranking == "heuristic" and pl.rest_ranking == "bench"
    out = pl.explain()
    assert "not measured: the lead-job bench was not run" in out and "measured lead job" not in out


def test_r1_the_regime_follows_the_configured_streams_and_never_blends_them():
    one = plan(R9700, RTX3090, bench_lead=LEAD1, opts=P.PlanOptions(streams=1))
    four = plan(R9700, RTX3090, bench_lead=LEAD4, opts=S4)
    assert one.main().name.endswith("3090") and four.main().name.endswith("R9700")
    assert "4 streams: 24.0 ms/window" in four.explain()
    two = plan(R9700, RTX3090, bench_lead=LEAD1, opts=P.PlanOptions(streams=2))      # no regime for 2: proxy, said plainly
    assert two.ranking == "heuristic" and "no lead-job regime for 2 streams" in two.explain()


def test_r1_a_nan_or_zero_lead_figure_drops_every_card_to_the_proxy():
    for bad in (float("nan"), 0.0, -1.0):
        pl = plan(R9700, RTX3090, bench_lead={("amd", 0): 26.7, ("nvidia", 0): bad})
        assert pl.ranking == "heuristic" and pl.main().name.endswith("R9700")


def test_r1_the_user_can_always_choose_card_1_and_the_plan_says_so():
    pl = plan(R9700, RTX3090, bench_lead=LEAD1, opts=P.PlanOptions(main_card=("amd", 0)))     # the bench says 3090
    assert pl.main().name.endswith("R9700") and pl.ranking == "user" and "chosen by the user" in pl.explain()
    gone = plan(R9700, RTX3090, bench_lead=LEAD1, opts=P.PlanOptions(main_card=("nvidia", 7)))
    assert gone.main().name.endswith("3090") and "--main-card nvidia:7 is not among the usable cards" in gone.explain()


def test_r2_cards_after_card_1_are_ordered_by_expert_rows_per_s_then_free_vram():
    lead = {("nvidia", 0): 24.0, ("amd", 0): 30.0, ("nvidia", 1): 40.0}
    rows = {("nvidia", 0): 1.0, ("amd", 0): 100.0, ("nvidia", 1): 200.0}
    pl = plan(RTX3090, R9700, RTX3060, bench_lead=lead, bench=rows, expert=40.0)
    assert [r.name[-4:] for r in pl.roles] == ["3090", "3060", "9700"]                       # rows/s: 3060 200 > R9700 100
    heur = plan(RTX3090, R9700, RTX3060, bench_lead=lead, expert=40.0)
    assert [r.name[-4:] for r in heur.roles] == ["3090", "9700", "3060"] and heur.rest_ranking == "heuristic"


def test_r3a_a_slow_card_is_excluded_on_an_all_vram_plan_and_admitted_on_an_overflow_plan():
    on = P.PlanOptions(admission=True)
    small = plan(R9700, RTX3060, expert=5.0, opts=on)                                         # 5 GiB fits in the R9700 alone
    assert [r.name[-4:] for r in small.roles] == ["R9700"[-4:]] and small.helpers() == ()
    assert "R3a: 0 bytes would leave VRAM" in small.explain() and "excluded nvidia gpu1" in small.explain()
    big = plan(R9700, RTX3060, expert=40.0, opts=on)                                          # 40 GiB overflows the R9700
    assert [h.name[-4:] for h in big.helpers()] == ["3060"] and "R3a ok: would hold" in big.explain()
    off = plan(R9700, RTX3060, expert=5.0)                                                    # admission is opt-in
    assert [h.name[-4:] for h in off.helpers()] == ["3060"]


def test_r3_dropping_a_middle_card_keeps_the_helper_channel_numbers_contiguous():
    bad = R.Terms(host_us=60, hop_us=75, t1_us=40, tk_us=80, pull_us=20)
    pl = plan(RTX3090, R9700, RTX3060, bench_lead={("nvidia", 0): 24.0, ("amd", 0): 30.0, ("nvidia", 1): 40.0},
              bench={("nvidia", 0): 1.0, ("amd", 0): 200.0, ("nvidia", 1): 100.0}, expert=40.0,
              opts=P.PlanOptions(admission=True, floors={("amd", 0): bad}))
    assert [(r.name[-4:], r.helper_index) for r in pl.roles] == [("3090", None), ("3060", 0)]     # R9700 (channel 0) dropped
    assert "R3b: layer cost with the card 215 us > without 60 us" in pl.explain()


def _slots(*budgets):
    return [R.Slot(("v", i), f"card{i}", int(b * (1 << 30))) for i, b in enumerate(budgets)]


FLOORS_OK = R.Terms(host_us=60, hop_us=75, t1_us=40, tk_us=30, pull_us=500)
FLOORS_BAD = R.Terms(host_us=60, hop_us=75, t1_us=40, tk_us=80, pull_us=20)


def test_r3b_layer_cost_is_pure_and_prints_every_term():
    ok = R.layer_cost(FLOORS_OK)
    assert ok.with_card_us == 60 + max(40, 75 + 30) and ok.without_card_us == 540 and ok.ok
    for term in ("host 60", "t1 40", "hop 75", "tk 30", "pull 500"):
        assert term in ok.terms
    assert not R.layer_cost(FLOORS_BAD).ok and not R.layer_cost(R.Terms(float("nan"), 1, 1, 1, 1)).ok


def test_r3b_a_card_that_makes_the_layer_slower_is_excluded_with_the_floors_named():
    adm, ex = R.admit(_slots(20, 10), 25 << 30, {("v", 1): FLOORS_BAD})
    assert [a.slot.name for a in adm] == ["card0"] and ex[0][0] == ("v", 1)
    assert ex[0][1].startswith("R3b: layer cost with the card 215 us > without 60 us") and "hop 75" in ex[0][1] and "5.0 GiB" in ex[0][1]
    adm, ex = R.admit(_slots(20, 10), 25 << 30, {("v", 1): FLOORS_OK})
    assert [a.slot.name for a in adm] == ["card0", "card1"] and not ex and "R3b ok" in adm[1].note


def test_r3_without_floors_a_card_is_admitted_on_r3a_alone_and_r3b_is_reported_not_evaluated():
    adm, ex = R.admit(_slots(20, 10), 25 << 30)
    assert len(adm) == 2 and not ex and "R3b not evaluated" in adm[1].note


def test_r3a_counts_every_card_before_it_not_only_card_1():
    adm, ex = R.admit(_slots(20, 10, 10), 28 << 30)                                           # card1 covers the last 8 GiB
    assert [a.slot.name for a in adm] == ["card0", "card1"] and ex[0][0] == ("v", 2) and "R3a" in ex[0][1]
