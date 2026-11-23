# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Checker for scripts/payoff/pay_analyze.py and pay_render.py on synthetic starts whose crossover is known by construction:
the second GPU ties at the large caps and pays below a boundary, so a wrong bound, a wrong gate or an extrapolating statement
changes a sentence the test knows.  The confidence bound is qual_stats.welch_bounds (placement-qual owns it).  CPU only."""
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import payoff_synth as S  # noqa: E402
import pay_analyze as PA  # noqa: E402
import pay_cfg  # noqa: E402
import pay_render as PR  # noqa: E402

CFG = pay_cfg.load(S.SCRIPTS + "/arms.json", {}, dry=True)
CAPS = [11000, 8000, 6000, 4000, 2500]
# decode tok/s (mean) per arm per cap: C ties A at the two large caps and pays below; B is the slowest everywhere but the first cap
DECODE = {"A": [100, 80, 55, 40, 30], "B": [95, 60, 50, 45, 42], "C": [100.5, 80.5, 75, 70, 66]}
PREFILL = {"A": [900, 800, 700, 600, 500], "B": [950, 900, 880, 860, 840], "C": [905, 805, 705, 605, 505]}


def jitter(k):
    return ((k * 7) % 5 - 2) * 0.2  # deterministic +-0.4 around the mean


def requests(arm, dec, pre, n=6):
    out = []
    for k in range(n + 1):  # request 0 is the warm-up
        counted = k >= 1
        out.append({"phase": "decode", "k": k, "counted": counted, "ok": True, "usable": counted, "decode_tps": dec + jitter(k), "decode_ms": 6000.0,
                    "pcie_share_of_misses": {"A": 0.0, "B": 0.98, "C": 0.0}[arm], "has_remote": arm == "C", "helper_requests": 900 if arm == "C" else None,
                    "staging_ms": 120.0 if arm == "B" else 210.0, "merge_wait_ms": 940.0 if arm == "C" else None, "local_compute_ms": 600.0,
                    "gpu_reach_wait_ms": 1800.0, "io_wait_ms": 7.5})
        out.append({"phase": "prefill", "k": k, "counted": counted, "ok": True, "usable": counted, "prefill_tps": pre + 10 * jitter(k)})
    return out


def start(arm, ci, slots=None, mode="speed", **kw):
    cap = CAPS[ci]
    return {"arm": arm, "cap": cap, "mode": mode, "idx": 0, "tag": f"cap{cap}-{arm}", "aborted": False, "error": None, "kl_files": {},
            "startup": {"cap_slots": slots or cap, "helper_up": arm == "C"}, "requests": requests(arm, DECODE[arm][ci], PREFILL[arm][ci]), **kw}


def results(**kw):
    return [start(a, ci, **kw) for ci in range(5) for a in "ABC"]


def test_crossover_is_located_between_the_adjacent_measured_caps():
    an = PA.analyze(results(), CFG, [])
    status = {p.cap: p.status["decode"] for p in an.points}
    assert status == {11000: "tie", 8000: "tie", 6000: "pays", 4000: "pays", 2500: "pays"}
    assert "boundary lies between cap 8000 (tie) and cap 6000 (pays)" in an.crossover["decode"]
    assert "measured caps 2500..11000" in an.crossover["decode"]


def test_prefill_is_judged_separately_from_decode():
    an = PA.analyze(results(), CFG, [])
    # B's prefill beats C everywhere from the second cap on, so C never pays in prefill and loses to B at the small caps
    assert {p.cap: p.status["prefill"] for p in an.points}[2500] == "loses"
    assert "no crossover inside the measured range" in an.crossover["prefill"]


def test_bounds_come_from_welch_and_carry_the_margin_rule():
    an = PA.analyze(results(), CFG, [])
    low = next(p for p in an.points if p.cap == 6000)
    lower, upper, diff = low.compare["decode"]["A"]
    assert diff == pytest.approx(100 * (75 - 55) / 55) and lower < diff < upper and lower > CFG["gates"]["margin_pct"]


def test_unmatched_cap_is_undecided_and_never_interpolated():
    r = results()
    for s in r:
        if s["cap"] == 6000 and s["arm"] == "B":
            s["startup"]["cap_slots"] = 5200  # the engine shrank B's cache: the budgets are no longer matched
    an = PA.analyze(r, CFG, [])
    assert {p.cap: p.status["decode"] for p in an.points}[6000] == "undecided"
    assert "Undecided (not interpolated): cap 6000" in an.crossover["decode"]
    assert "boundary lies between cap 8000 (tie) and cap 4000 (pays)" in an.crossover["decode"]  # the gap is named, not filled in


def test_a_failed_quality_gate_removes_the_arm_speed_and_the_point_becomes_undecided():
    rows = [{"cap": 6000, "arm": "C", "idx": 0, "text": "t", "kind": "gate", "rc": 1, "line": "KL-GATE FAIL"}]
    an = PA.analyze(results(), CFG, rows)
    pt = next(p for p in an.points if p.cap == 6000)
    assert pt.arms["C"].valid_starts == 0 and pt.arms["C"].samples["decode"] == [] and pt.status["decode"] == "undecided"
    assert any("quality gate FAIL" in r for r in pt.arms["C"].reasons)


def test_all_points_paying_says_no_crossover_inside_the_range_not_where_it_is():
    r = results()
    for s in r:
        if s["arm"] == "C":
            s["requests"] = requests("C", 200, 900)
    s = PA.analyze(r, CFG, []).crossover["decode"]
    assert "at every decided cap" in s and "no crossover inside the measured range" in s


def test_statement_for_a_non_monotone_pattern_refuses_a_single_boundary():
    s = PA.crossover_statement("decode", [(11000, "pays"), (8000, "tie"), (6000, "pays")])
    assert "no single crossover" in s and "pays at cap 11000, 6000" in s


def test_statement_with_nothing_decided():
    assert "no point could be decided" in PA.crossover_statement("decode", [(8000, "undecided")])


def test_fewer_than_two_samples_is_undecided():
    r = results()
    for s in r:
        if s["arm"] == "A" and s["cap"] == 4000:
            for q in s["requests"]:
                q["usable"] = q["usable"] and q["k"] == 1
    # only one usable request per phase fails the n >= 6 start gate: the arm has no valid start at that point
    pt = next(p for p in PA.analyze(r, CFG, []).points if p.cap == 4000)
    assert pt.arms["A"].valid_starts == 0 and pt.status["decode"] == "undecided"


def test_render_shows_tables_bounds_and_the_limit():
    text = PR.render(PA.analyze(results(), CFG, []), CFG)
    assert "=== cap 6000: logged expert-cache slots A 6000, B 6000, C 6000; matched ===" in text
    assert "second GPU decode: PAYS" in text and "vs A: +36.4%" in text
    assert "LIMIT: samples are requests inside ONE engine start" in text and "CROSSOVER (measured range only):" in text


def test_render_shows_the_merge_wait_for_arm_c_only():
    lines = PR.render(PA.analyze(results(), CFG, []), CFG).splitlines()
    c_row = next(ln for ln in lines if ln.startswith("C ") and "940" in ln)  # the merge wait column
    a_row = next(ln for ln in lines if ln.startswith("A ") and "600" in ln and " - " in ln)  # arm A: local compute 600, merge wait '-'
    assert c_row and a_row


def test_verdict_rows_for_the_html_report():
    v = PR.verdicts(PA.analyze(results(), CFG, []))
    assert [x["id"] for x in v][:5] == ["cap11000", "cap8000", "cap6000", "cap4000", "cap2500"]
    assert all(x["status"] == "WARN" for x in v[:5])  # UNGATED (no kl rows): a warning, never a silent pass
    assert {x["id"] for x in v[5:]} == {"X-decode", "X-prefill"}
