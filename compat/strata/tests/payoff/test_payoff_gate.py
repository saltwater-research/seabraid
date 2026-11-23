# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Checker for scripts/payoff/pay_gate.py: a start that is not the arm it claims is invalid, a quality gate that fails or
cannot run keeps the speed out, and the real tools/kl.py decides the KL verdicts (synthetic LOGPOS rows, real exit codes).
CPU only."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import payoff_synth as S  # noqa: E402
import pay_cfg  # noqa: E402
import pay_gate as G  # noqa: E402

CFG = pay_cfg.load(S.SCRIPTS + "/arms.json", {}, dry=True)
GATES = CFG["gates"]
TEXTS = CFG["sweep"]["kl_texts"]


def req(**kw):
    base = {"phase": "decode", "ok": True, "usable": True, "counted": True, "pcie_share_of_misses": None, "has_remote": False, "helper_requests": None}
    return {**base, **kw}


def start(arm, requests, slots=8000, helper=False, **kw):
    return {"arm": arm, "cap": 8000, "mode": "speed", "idx": 0, "startup": {"cap_slots": slots, "helper_up": helper}, "requests": requests,
            "kl_files": {}, "aborted": False, "error": None, **kw}


def test_arm_a_must_not_read_experts_over_pcie_or_have_a_helper():
    assert G.config_problems("A", start("A", [req(pcie_share_of_misses=0.0)]), GATES) == ([], [])
    bad, _ = G.config_problems("A", start("A", [req(pcie_share_of_misses=0.6)]), GATES)
    assert "--pcie-frac 0 did not hold" in bad[0]
    bad, _ = G.config_problems("A", start("A", [req(has_remote=True)], helper=True), GATES)
    assert "not a one-GPU arm" in bad[0]


def test_arm_b_must_move_its_misses_to_pcie():
    assert G.config_problems("B", start("B", [req(pcie_share_of_misses=0.97)]), GATES) == ([], [])
    bad, _ = G.config_problems("B", start("B", [req(pcie_share_of_misses=0.55)]), GATES)
    assert "--pcie-frac 1 did not hold" in bad[0]
    bad, notes = G.config_problems("B", start("B", [req()]), GATES)
    assert bad == [] and "same run" in notes[0]  # no misses at all: not invalid, but A and B are indistinguishable here


def test_arm_c_needs_the_helper_channel_and_notes_an_idle_helper():
    bad, _ = G.config_problems("C", start("C", [req(helper_requests=10)], helper=False), GATES)
    assert "never joined" in bad[0]
    assert G.config_problems("C", start("C", [req(helper_requests=10)], helper=True), GATES) == ([], [])
    bad, notes = G.config_problems("C", start("C", [req(helper_requests=0)], helper=True), GATES)
    assert bad == [] and "idle second GPU" in notes[0]


def test_an_unconfirmed_cap_invalidates_any_arm():
    bad, _ = G.config_problems("A", start("A", [req()], slots=None), GATES)
    assert "cap was not confirmed" in bad[0]


def full(n=6):
    return [req(phase=p, k=k) for k in range(n) for p in ("decode", "prefill")]


def test_start_verdict_requires_n_usable_requests_per_phase_and_a_passing_gate():
    ok = G.start_verdict(start("A", full()), "PASS", CFG)
    assert ok["valid"] and ok["reasons"] == []
    short = G.start_verdict(start("A", full(5)), "PASS", CFG)
    assert not short["valid"] and "only 5 of 6 usable decode" in short["reasons"][0]
    for kl in ("FAIL", "INVALID"):
        v = G.start_verdict(start("A", full()), kl, CFG)
        assert not v["valid"] and "quality gate " + kl in v["reasons"][0]
    assert "UNGATED" in G.start_verdict(start("A", full()), "NONE", CFG)["notes"][0]
    assert not G.start_verdict(start("A", full(), aborted=True), "PASS", CFG)["valid"]
    assert not G.start_verdict(start("A", [], error="the engine did not start"), "PASS", CFG)["valid"]


def kl_start(tmp_path, arm, cap, idx, noise, seed=1):
    files = {}
    for k, t in enumerate(TEXTS):
        p = S.write_logpos(tmp_path / f"{arm}-{cap}-{idx}-{t}.tsv", seed=seed + k, noise=noise)
        files[t] = {"path": p, "rows": 60, "ok": True}
    return {"arm": arm, "cap": cap, "mode": "kl", "idx": idx, "kl_files": files, "requests": [], "startup": {}}


def test_kl_gates_use_the_real_gate_baseline_is_arm_a_at_the_largest_cap(tmp_path):
    results = [kl_start(tmp_path, "A", 8000, 0, 0.0), kl_start(tmp_path, "A", 8000, 1, 0.0), kl_start(tmp_path, "B", 8000, 0, 0.0),
               kl_start(tmp_path, "C", 8000, 0, 6.0), kl_start(tmp_path, "A", 4000, 0, 0.0)]
    rows = G.kl_gates(results, CFG)
    by = {(r["arm"], r["cap"], r["idx"], r["text"]): r for r in rows}
    assert by[("A", 8000, 0, TEXTS[0])]["kind"] == "baseline"
    assert by[("A", 8000, 1, TEXTS[0])]["kind"] == "noise" and by[("A", 8000, 1, TEXTS[0])]["rc"] == 0
    assert all(by[("B", 8000, 0, t)]["rc"] == 0 and "KL-GATE PASS" in by[("B", 8000, 0, t)]["line"] for t in TEXTS)
    assert all(by[("C", 8000, 0, t)]["rc"] == 1 and "KL-GATE FAIL" in by[("C", 8000, 0, t)]["line"] for t in TEXTS)
    status = G.kl_status(rows)
    assert status[(8000, "B")] == "PASS" and status[(8000, "C")] == "FAIL" and status[(8000, "A")] == "PASS"


def test_a_missing_baseline_makes_every_gate_invalid_never_a_pass(tmp_path):
    rows = G.kl_gates([kl_start(tmp_path, "B", 8000, 0, 0.0), kl_start(tmp_path, "C", 8000, 0, 0.0)], CFG)
    assert rows and all(r["rc"] == 2 for r in rows)
    assert set(G.kl_status(rows).values()) == {"INVALID"}


def test_an_empty_rows_file_is_invalid_not_pass(tmp_path):
    base, arm = kl_start(tmp_path, "A", 8000, 0, 0.0), kl_start(tmp_path, "B", 8000, 0, 0.0)
    arm["kl_files"][TEXTS[0]]["rows"] = 0
    rows = G.kl_gates([base, arm], CFG)
    assert next(r for r in rows if r["arm"] == "B" and r["text"] == TEXTS[0])["rc"] == 2


def test_kl_status_worst_verdict_wins():
    rows = [{"cap": 1, "arm": "B", "kind": "gate", "rc": rc} for rc in (0, 2, 1)]
    assert G.kl_status(rows)[(1, "B")] == "FAIL"
    assert G.kl_status(rows[:2])[(1, "B")] == "INVALID"
    assert G.kl_status([{"cap": 1, "arm": "B", "kind": "noise", "rc": 1}]) == {}


def test_kl_thresholds_are_passed_only_when_the_config_sets_them(tmp_path):
    seen = []
    fake = lambda a: (seen.append(a), (0, "x"))[1]  # noqa: E731
    pair = [kl_start(tmp_path, "A", 8000, 0, 0.0), kl_start(tmp_path, "B", 8000, 0, 0.0)]
    G.kl_gates(pair, {**CFG, "thresholds": {"kl_max": 0.5}}, run=fake)
    assert seen[0][-2:] == ["--max-kl", "0.5"] and "--min-top1" not in seen[0]
    seen.clear()
    G.kl_gates(pair, CFG, run=fake)
    assert "--max-kl" not in seen[0]  # kl.py's own defaults stay the single source
