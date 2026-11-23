# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Checker for the tools/kl.py target pairing (F5), the four separate result verdicts (stall-inclusive decode, per-text
quality, convergence, helper-loss recovery), the host-RAM-headroom arm dimension, and that live promotion is opt-in in
every arm but the explicit move arms.  Pure Python, stdlib + pytest."""
import placement_qual_fixtures  # noqa: F401  (puts scripts/placement-qual on sys.path)
import qual_cfg  # noqa: E402
import qual_engine  # noqa: E402
from placement_qual_fixtures import (  # noqa: E402
    CFG, QUAL, SETTLED, JIT, TEXTS, GOOD_KL, _rounds, _run, _req, _w1, _w2, _w3, _w6, good_results, kl_rec, verdicts, status,
)
from test_kl_gate import gate, logits, write  # noqa: E402


# ---- F5: a KL pair must force the same targets at the same positions ----
def test_kl_gate_rejects_equal_length_pair_with_different_targets(tmp_path):
    rows = logits(n=60)
    write(tmp_path / "base.tsv", rows)
    lines = write(tmp_path / "arm.tsv", rows)
    other = ["\t".join([f.split("\t")[0], str(int(f.split("\t")[1]) + 1)] + f.split("\t")[2:]) for f in lines]
    write(tmp_path / "arm.tsv", rows, lines=other)
    rc, out, _ = gate(tmp_path / "base.tsv", tmp_path / "arm.tsv")
    assert rc == 2 and "INVALID" in out and "target ids differ" in out, out
    assert "PASS" not in out


def test_kl_gate_still_passes_the_same_targets(tmp_path):
    rows = logits(n=60)
    write(tmp_path / "base.tsv", rows)
    write(tmp_path / "arm.tsv", rows)
    assert gate(tmp_path / "base.tsv", tmp_path / "arm.tsv")[0] == 0


def test_kl_gate_target_swap_between_two_positions_is_rejected(tmp_path):
    rows = logits(n=60)
    lines = write(tmp_path / "base.tsv", rows)
    f = [x.split("\t") for x in lines]
    f[3][1], f[4][1] = f[4][1], f[3][1]
    write(tmp_path / "arm.tsv", rows, lines=["\t".join(x) for x in f])
    assert gate(tmp_path / "base.tsv", tmp_path / "arm.tsv")[0] == 2


# ---- separate verdicts (audit decision 1): D1 stall-inclusive decode, Q1 per-text quality, C4 convergence, C9 recovery ----
def _with(res, key, idx, new):
    return [new if (r["workload"], r["key"], r["idx"]) == ("W1", key, idx) else r for r in res]


def test_d1_long_pause_in_a_live_arm_fails_while_c1_still_passes():
    res = _with(good_results(), "P2", 0, _w1("P2", 0, 100.0, first_extra=12.0))  # 12 s pause in the first request: skipped by the tok/s mean, counted by the stall count
    vs = verdicts(res)
    assert status(vs, "C1") == ["PASS", "PASS"] and status(vs, "D1") == ["FAIL"]
    assert "P2 long 1/9 warn 0" in [v for v in vs if v["id"] == "D1"][0]["detail"]


def test_d1_short_warnings_are_counted_separately_and_do_not_fail():
    d1 = [v for v in verdicts(_with(good_results(), "P2", 0, _w1("P2", 0, 100.0, first_extra=3.0))) if v["id"] == "D1"][0]
    assert d1["status"] == "PASS" and "P2 long 0/9 warn 1" in d1["detail"]


def test_d1_pauses_that_cut_the_stall_inclusive_rate_fail_non_inferiority():
    res = [_w1("P2", i, 100.0) for i in range(3)]
    for r in res:
        r["phases"]["decode"][1]["wall_s"] *= 1.5  # pauses inside the counted requests: engine tok/s untouched, wall tok/s falls 33%
    res = [r for r in good_results() if not (r["workload"] == "W1" and r["key"] == "P2")] + res
    vs = verdicts(res)
    assert status(vs, "C1") == ["PASS", "PASS"] and status(vs, "D1") == ["FAIL"]


def _texts_fail(text, starts):
    """Pooled passes in both starts; ``text`` fails (rc 1) in the given starts, every other per-text record passes."""
    return [kl_rec(0, 0), kl_rec(1, 0)] + [kl_rec(i, 1 if (t == text and i in starts) else 0, text=t) for i in (0, 1) for t in TEXTS]


def test_q1_text_failing_in_every_live_start_fails_even_if_pooled_passes():
    vs = verdicts(good_results(), _texts_fail("kl_t2.txt", (0, 1)))
    assert status(vs, "C8") == ["PASS"] and status(vs, "Q1") == ["FAIL"]


def test_q1_text_failing_in_one_start_only_is_not_decidable():
    assert status(verdicts(good_results(), _texts_fail("kl_t2.txt", (0,))), "Q1") == ["WARN"]


def test_q1_undocumented_exit_code_in_every_start_fails():
    kl = [kl_rec(0, 0), kl_rec(1, 0)] + [kl_rec(i, -9 if t == "kl_t3.txt" else 0, text=t) for i in (0, 1) for t in TEXTS]
    assert status(verdicts(good_results(), kl), "Q1") == ["FAIL"]


def test_c9_recovery_time_is_required_and_bounded():
    for bad in (None, 500.0):
        res = [{**r, "fault": {**r["fault"], "recovery_s": bad}} if r["workload"] == "W5" else r for r in good_results()]
        assert status(verdicts(res), "C9") == ["FAIL"]


def test_h1_moves_off_vs_on_at_each_fixed_budget():
    res = [r for r in good_results() if not (r["workload"] == "W6" and r["key"] == "P2/ram_lo")]
    res += [_run("W6", "P2/ram_lo", i, phases={"decode": [{"decode_tps": 80.0 + JIT[i]}] * 3}) for i in range(3)]
    h = [v for v in verdicts(res) if v["id"] == "H1"][0]
    assert h["status"] == "FAIL" and "P2 vs P0 at ram_lo" in h["detail"] and "budget varied, arm fixed" in h["detail"]


def test_aggregate_lists_all_five_and_never_merges_them():
    a = [v for v in verdicts(good_results()) if v["id"] == "AGG"][0]
    assert a["status"] == "PASS" and all(f"{c}: PASS" in a["detail"] for c in ("D1", "Q1", "C4", "C9", "C10"))
    bad = verdicts(_with(good_results(), "P2", 0, _w1("P2", 0, 100.0, first_extra=12.0)))
    a = [v for v in bad if v["id"] == "AGG"][0]
    assert a["status"] == "FAIL" and "D1: FAIL" in a["detail"] and "C9: PASS" in a["detail"]
    assert {c: status(bad, c)[0] for c in ("D1", "Q1", "C4", "C9")} == {"D1": "FAIL", "Q1": "PASS", "C4": "PASS", "C9": "PASS"}


# ---- (b) RAM headroom is a config dimension; (c) live promotion is opt-in ----
OVERRIDES = {k: "x" for k in CFG["vars"] if str(CFG["vars"][k]).startswith("UNSET")}
MOVE_ARMS = {"P2", "P2z", "N1", "N2", "F0", "F1"}  # the explicit move arms: the only arms that carry --retier / --retier-promote (F0/F1: the file-moves persona)


def _all_run_commands():
    cfg = qual_cfg.load(str(QUAL / "arms.json"), OVERRIDES)
    return cfg, [(r, qual_engine.build_command(cfg, r, "rd")[0]) for r in qual_cfg.plan(cfg, None)]


def test_live_promotion_is_opt_in_in_every_arm_but_the_move_arms():
    cfg, cmds = _all_run_commands()
    shared = [cfg["engine"]["argv"]] + [w.get("extra_args", []) for w in cfg["workloads"].values()] + [v.get("args", []) for v in cfg["variants"].values()]
    for part in shared:
        assert not {"--retier", "--retier-promote"} & set(part), part  # no shared layer may switch moves on for every arm (--retier-margin etc. only tune an arm that is on)
    for r, argv in cmds:
        assert ("--retier-promote" in argv) == (r.arm in MOVE_ARMS), (r.key, argv)
        assert ("--retier" in argv) == (r.arm in MOVE_ARMS), (r.key, argv)
    assert {a for a, spec in cfg["arms"].items() if "policy" in spec} == MOVE_ARMS


def _without_budget(argv):
    i = argv.index("--resident-budget-gib")
    return argv[:i] + argv[i + 2:]


def test_ram_budget_variants_vary_one_flag_with_the_arm_fixed():
    _, cmds = _all_run_commands()
    w6 = {r.key: argv for r, argv in cmds if r.workload == "W6"}
    assert set(w6) == {f"{a}/ram_{b}" for a in ("P0", "P2") for b in ("hi", "mid", "lo")}
    for a in ("P0", "P2"):
        cells = [w6[f"{a}/ram_{b}"] for b in ("hi", "mid", "lo")]
        assert len({c[c.index("--resident-budget-gib") + 1] for c in cells}) == 3
        assert _without_budget(cells[0]) == _without_budget(cells[1]) == _without_budget(cells[2])  # nothing but the budget differs
    for b in ("hi", "mid", "lo"):  # moves on vs off at a fixed budget: the budget is identical, only the move arm differs
        on, off = w6[f"P2/ram_{b}"], w6[f"P0/ram_{b}"]
        assert on[on.index("--resident-budget-gib") + 1] == off[off.index("--resident-budget-gib") + 1] and "--retier-promote" not in off
