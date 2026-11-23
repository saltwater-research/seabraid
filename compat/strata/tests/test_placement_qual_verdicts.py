# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Checker for the placement qualification verdicts (scripts/placement-qual: qual_judge, qual_stats, qual_kl): an
interrupted KL checker never passes (F2), the 95% bound uses the real Student-t quantile (F3), convergence needs a
reference and a settled commit stream (F4), and every criterion declines when its evidence is removed (the module
contract "never passes on absence of evidence").  Pure Python, stdlib + pytest, no engine, no network."""
import math
import subprocess

import pytest

import placement_qual_fixtures  # noqa: F401  (puts scripts/placement-qual on sys.path)
import qual_judge as qj  # noqa: E402
import qual_kl  # noqa: E402
import qual_report  # noqa: E402
import qual_separate as qs  # noqa: E402
import qual_stats as st  # noqa: E402
from placement_qual_fixtures import (  # noqa: E402
    CFG, QUAL, SETTLED, JIT, TEXTS, GOOD_KL, _rounds, _run, _req, _w1, _w2, _w3, _w6, good_results, kl_rec, verdicts, status,
)


def test_fixture_passes_everything():
    # F1 (file-moves persona) is WARN here: this fixture has no file-moves starts, and absence never passes (test_placement_qual_file_moves.py owns its evidence).
    want = {c: "PASS" for c in ("W0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "D1", "Q1", "H1", "AGG")}
    assert {v["id"]: v["status"] for v in verdicts(good_results())} == {**want, "F1": "WARN"}


# ---- the contract: one test per criterion that removes its evidence and asserts no PASS ----
def _drop(*pairs):
    return lambda res: [r for r in res if (r["workload"], r["key"]) not in pairs]


def _no_rounds(res):
    return [{**r, "rounds": []} for r in res]


ABSENCE = {
    "W0": _drop(("W0", "P2")),
    "C1": _drop(*[("W1", k) for k in ("P0", "N1")]),
    "C2": _drop(("W1", "P2z")),
    "C3": _no_rounds,
    "C4": _drop(("W2", "N1")),
    "C5": _no_rounds,
    "C6": _drop(*[("W3", k) for k in ("N1/off", "N2/off", "P0/off", "P2/off")]),
    "C7": _drop(("W3", "N1/on"), ("W3", "N2/on")),
    "C8": lambda res: [r for r in res if r["workload"] != "W4"],
    "C9": _drop(("W5", "P2")),
    "C10": lambda res: [r for r in res if r["workload"] != "W7"],
    "D1": _drop(("W1", "P0"), ("W1", "N1")),
    "Q1": lambda res: [r for r in res if r["workload"] != "W4"],
    "H1": _drop(("W6", "P0/ram_lo")),
    "AGG": _drop(("W5", "P2")),
}


@pytest.mark.parametrize("cid", sorted(ABSENCE))
def test_removing_the_evidence_never_passes(cid):
    assert set(status(verdicts(good_results()), cid)) == {"PASS"}, "vacuous: the baseline must pass the criterion first"
    assert "PASS" not in status(verdicts(ABSENCE[cid](good_results())), cid)


def test_no_results_at_all_passes_nothing():
    assert {v["status"] for v in qs.judge_all(CFG, [], [])} == {"WARN"}


# ---- F2: an interrupted or crashed KL checker never yields PASS ----
@pytest.mark.parametrize("rc,want", [(0, "PASS"), (1, "FAIL"), (2, "WARN"), (-9, "FAIL"), (-15, "FAIL"), (137, "FAIL"), (3, "FAIL"), (None, "FAIL"), (True, "FAIL")])
def test_c8_exit_codes(rc, want):
    assert status(verdicts(good_results(), [kl_rec(0, 0), kl_rec(1, rc)]), "C8") == [want]


def test_c8_ignores_per_text_records_for_the_decision():
    assert status(verdicts(good_results(), [kl_rec(0, 0), kl_rec(1, 0), kl_rec(0, 1, text="kl_t1.txt")]), "C8") == ["PASS"]


def test_c8_exit_zero_without_the_gates_own_pass_line_is_not_a_pass():
    assert status(verdicts(good_results(), [kl_rec(0, 0), kl_rec(1, 0, line="")]), "C8") == ["WARN"]


def test_c8_missing_and_partial_pooled_records_are_not_decidable():
    res = good_results()
    assert status(verdicts(res, []), "C8") == ["WARN"]
    assert status(verdicts(res, [kl_rec(0, 0)]), "C8") == ["WARN"], "start 1 has no record"
    per_text = [kl_rec(i, 0, text=t) for i in (0, 1) for t in ("kl_t1.txt", "kl_t2.txt")]
    assert status(verdicts(res, per_text), "C8") == ["WARN"], "per-text records never stand in for the pooled gate"


def test_c8_fewer_starts_than_configured_is_not_decidable():
    res = [r for r in good_results() if not (r["workload"] == "W4" and r["arm"] == "P2" and r["idx"] == 1)]
    assert status(verdicts(res, [kl_rec(0, 0)]), "C8") == ["WARN"]


def test_c8_one_failing_record_beats_a_pass_for_the_same_start():
    assert status(verdicts(good_results(), [kl_rec(0, 0), kl_rec(1, 0), kl_rec(1, -9)]), "C8") == ["FAIL"]


class _Rec:
    """Recorder double: runs fn, records FAIL on an exception, like qual_report.Recorder.step."""
    def __init__(self, tmp):
        self.r = qual_report.Recorder(str(tmp))

    def step(self, name, fn):
        return self.r.step(name, fn)


@pytest.mark.parametrize("rc,step", [(0, "PASS"), (2, "PASS"), (1, "FAIL"), (-9, "FAIL"), (137, "FAIL"), (None, "FAIL")])
def test_qual_kl_gate_step_records_the_raw_code(monkeypatch, tmp_path, rc, step):
    monkeypatch.setattr(qual_kl, "_run", lambda args: (rc, "KL-GATE x"))
    rec = _Rec(tmp_path)
    got = qual_kl._gate_step(rec, "kl-x", [])
    assert got == (rc, "KL-GATE x") and rec.r.rows[-1]["status"] == step


def test_qual_kl_run_maps_a_timeout_to_no_exit_code(monkeypatch):
    def boom(*a, **k):
        raise subprocess.TimeoutExpired("kl.py", 600)
    monkeypatch.setattr(qual_kl.subprocess, "run", boom)
    assert qual_kl._run([])[0] is None


# ---- F3: the one-sided 95% bound uses the Student-t quantile ----
def _tail(t, df, n=40000):
    """P(T > t) by Simpson integration of the t density (independent of the incomplete-beta inversion)."""
    c = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(df * math.pi)
    f = lambda x: c * (1 + x * x / df) ** (-(df + 1) / 2)  # noqa: E731
    h = t / n
    s = f(0) + f(t) + sum((4 if i % 2 else 2) * f(i * h) for i in range(1, n))
    return 0.5 - s * h / 3


@pytest.mark.parametrize("df,want", [(1, 6.3138), (2, 2.9200), (3, 2.3534), (4, 2.1318), (5, 2.0150), (10, 1.8125), (30, 1.6973), (120, 1.6577)])
def test_t95_known_quantiles(df, want):
    assert st.t95(df) == pytest.approx(want, abs=6e-5)


@pytest.mark.parametrize("df", [1.0, 1.005, 1.5, 2.7, 7.3, 55.0])
def test_t95_inverts_the_density_integral(df):
    assert _tail(st.t95(df), df) == pytest.approx(0.05, abs=2e-7)


def test_t95_rejects_df_below_one():
    for bad in (0.99, 0.0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            st.t95(bad)


def test_two_starts_against_a_constant_reference_use_t1():
    lo, hi, diff = st.welch_bounds([99.4, 100.6], [100.0, 100.0])
    assert lo == pytest.approx(-3.788, abs=1e-3) and hi == pytest.approx(3.788, abs=1e-3) and diff == 0.0


def test_that_pair_does_not_pass_a_3_percent_limit():
    res = [_run("W1", k, i, phases={"decode": [{"decode_tps": v}] * 2}) for k, vs in (("P2", (99.4, 100.6)), ("P0", (100.0, 100.0))) for i, v in enumerate(vs)]
    assert status(qj.c1(CFG, res), "C1")[0] == "FAIL"


def test_welch_effective_df_near_one():
    arm, ref = [99.95, 100.05], [99.0, 101.0]  # va = 0.0025, vr = 1: df = 1.00500
    lo, _, diff = st.welch_bounds(arm, ref)
    assert lo == pytest.approx(diff - st.t95(1.005) * math.sqrt(0.0025 + 1.0), abs=1e-3)  # percent of mean(ref) = 100


# ---- F4: convergence needs a reference, and churn is not settling ----
def _c4(w2):
    return qj.c4(CFG, [r for r in good_results() if r["workload"] != "W2"] + w2)


def _w2_all(p2, n1, n2, share=0.3):
    return ([_w2("P2", i, p2) for i in range(2)] + [_w2("N1", i, n1, share) for i in range(2)] + [_w2("N2", i, n2) for i in range(2)]
            + [_w2(k, i, SETTLED) for k in ("P0", "N0") for i in range(2)])


def test_c4_genuine_settling_passes():
    assert _c4(_w2_all(SETTLED, SETTLED, SETTLED))["status"] == "PASS"


def test_c4_no_reference_is_indeterminate_not_a_pass():
    """N1 moved nothing: its median is None, the old limit was inf and everything passed."""
    v = _c4(_w2_all(SETTLED, [0] * 14, SETTLED))
    assert v["status"] == "WARN" and "no reference" in v["detail"]


def test_c4_constant_churn_is_not_convergence():
    churn = [2] * 40
    assert _c4(_w2_all(churn, churn, churn))["status"] != "PASS"
    assert _c4(_w2_all(churn, SETTLED, SETTLED))["status"] == "FAIL"


def test_c4_churn_in_the_tail_of_the_horizon_fails():
    assert _c4(_w2_all([2, 2, 1, 1] + [0] * 6 + [1, 1], SETTLED, SETTLED))["status"] == "FAIL"


def test_c4_missing_post_shift_comparison_is_indeterminate():
    res = [r for r in good_results() if not (r["workload"] == "W2" and r["key"] == "N0")]
    assert qj.c4(CFG, res)["status"] == "WARN"


def test_c4_without_the_legacy_share_is_indeterminate():
    res = [r for r in good_results() if r["workload"] != "W2"] + _w2_all(SETTLED, SETTLED, SETTLED, share=None)
    v = qj.c4(CFG, res)
    assert v["status"] == "WARN" and "legacy-publisher helper_share" in v["detail"]


def test_c4_too_few_rounds_for_the_horizon_is_indeterminate():
    assert _c4(_w2_all([2, 1, 0], SETTLED, SETTLED))["status"] == "WARN"


def test_settle_statuses():
    assert st.settle(SETTLED, 12, 4) == ("settled", 4, pytest.approx(8 / 12))
    assert st.settle([1] * 20, 12, 4)[0] == "churn"
    assert st.settle([0] * 20, 12, 4)[0] == "none"
    assert st.settle([1, 0], 12, 4)[0] == "short"
