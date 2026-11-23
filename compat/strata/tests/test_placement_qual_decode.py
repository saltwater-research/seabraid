# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Checker for the decode-path quality gate of the placement qualification (W7 capture, C10 verdict).

The W4 KL gate scores PREFILL only, where the helper tier never runs (live run G1, 2026-11-19: tier-on and tier-off gave
byte-identical W4 rows), so C8 cannot see a decode-path change.  W7 teacher-forces a fixed continuation through the real
decode windows with the tier engaged and C10 judges it against the main-card-alone reference.  These tests prove: C10 never
passes on absent or partial evidence, a crashed checker is a FAIL, a pair that did not force the same text is not a pass,
C8 says it is prefill-only, and the capture/pooling/pairing code does what the verdict assumes.  No engine, no network."""
import json
import sys

import pytest

import placement_qual_fixtures  # noqa: F401  (puts scripts/placement-qual on sys.path)
import qual_cfg  # noqa: E402
import qual_report  # noqa: E402
from placement_qual_fixtures import CFG, GOOD_DECODE_KL, GOOD_KL, NOISE, QUAL, _run, dec_rec, good_results, kl_rec, verdicts, status  # noqa: E402

W4_KL = [k for k in GOOD_KL if k not in GOOD_DECODE_KL]


def _kl(*decode):
    return W4_KL + list(decode)


def _c10(res=None, kl=None):
    return [v for v in verdicts(good_results() if res is None else res, GOOD_KL if kl is None else kl) if v["id"] == "C10"][0]


# ---- absence: baseline PASS first, then each piece of evidence removed ----
def test_baseline_passes_and_reports_the_noise_floor_next_to_it():
    v = _c10()
    assert v["status"] == "PASS" and "noise floor" in v["detail"] and "KL(top32) mean 0.00102" in v["detail"]


def _no_w7(res):
    return [r for r in res if r["workload"] != "W7"]


def _no_reference(res):
    return [r for r in res if not (r["workload"] == "W7" and r["arm"] == "M")]


def _no_arm_start(res):
    return [r for r in res if not (r["workload"] == "W7" and r["arm"] == "P2" and r["idx"] == 1)]


@pytest.mark.parametrize("cut", [_no_w7, _no_reference, _no_arm_start], ids=["no-W7-starts", "no-M-reference", "one-P2-start-missing"])
def test_removing_a_start_is_not_decidable(cut):
    assert _c10()["status"] == "PASS", "vacuous: the baseline must pass first"
    assert _c10(res=cut(good_results()))["status"] == "WARN"


@pytest.mark.parametrize("drop", [lambda k: False, lambda k: True, lambda k: k["pair"] == "D-P2-s1-vs-M-s0", lambda k: k["pair"].startswith("D-P0-")],
                         ids=["none", "all-decode-records", "one-P2-start", "whole-P0-arm"])
def test_removing_decode_records_is_not_decidable(drop):
    kl = [k for k in GOOD_KL if not (k["text"] == "decode-pooled" and drop(k))]
    assert _c10(kl=kl)["status"] == ("PASS" if len(kl) == len(GOOD_KL) else "WARN")


def test_prefill_records_never_stand_in_for_decode_records():
    assert _c10(kl=W4_KL + [NOISE])["status"] == "WARN"


def test_a_per_text_style_record_is_not_the_pooled_decision():
    kl = [k for k in GOOD_KL if k["pair"] != "D-P2-s1-vs-M-s0"] + [dec_rec("P2", 1, 0, text="decode-kl_t1")]
    assert _c10(kl=kl)["status"] == "WARN"


def test_fewer_starts_than_configured_is_not_decidable():
    cfg = {**CFG, "workloads": {**CFG["workloads"], "W7": {**CFG["workloads"]["W7"], "n": 3}}}
    vs = verdicts(good_results(), GOOD_KL, cfg=cfg)
    assert status(vs, "C10") == ["WARN"]


# ---- failure contract: a checker that crashed or was killed is a FAIL, a gate FAIL is a FAIL, INVALID is not a pass ----
@pytest.mark.parametrize("rc,want", [(0, "PASS"), (1, "FAIL"), (2, "WARN"), (-9, "FAIL"), (137, "FAIL"), (3, "FAIL"), (None, "FAIL"), (True, "FAIL")])
def test_c10_exit_codes(rc, want):
    kl = [k for k in GOOD_KL if k["pair"] != "D-P2-s1-vs-M-s0"] + [dec_rec("P2", 1, rc)]
    assert _c10(kl=kl)["status"] == want


def test_exit_zero_without_the_gates_own_pass_line_is_not_a_pass():
    kl = [k for k in GOOD_KL if k["pair"] != "D-P0-s0-vs-M-s0"] + [dec_rec("P0", 0, 0, line="")]
    assert _c10(kl=kl)["status"] == "WARN"


def test_one_failing_record_beats_a_pass_for_the_same_start():
    assert _c10(kl=GOOD_KL + [dec_rec("P2", 0, -9)])["status"] == "FAIL"


def test_a_failing_decode_path_does_not_hide_behind_a_passing_prefill_gate():
    kl = [k for k in GOOD_KL if k["pair"] != "D-P2-s0-vs-M-s0"] + [dec_rec("P2", 0, 1, line="KL-GATE FAIL: mean KL 0.2")]
    vs = verdicts(good_results(), kl)
    assert status(vs, "C8") == ["PASS"] and status(vs, "C10") == ["FAIL"] and status(vs, "AGG") == ["FAIL"]


# ---- C8 states what it covers; AGG lists C10 ----
def test_c8_says_it_is_prefill_only_and_does_not_cover_the_helper_tier():
    for kl in (GOOD_KL, []):
        d = [v for v in verdicts(good_results(), kl) if v["id"] == "C8"][0]["detail"].lower()
        assert "prefill only" in d and "helper tier" in d and "c10" in d


def test_agg_lists_c10_and_is_not_pass_while_c10_is_not():
    agg = lambda kl: [v for v in verdicts(good_results(), kl) if v["id"] == "AGG"][0]  # noqa: E731
    assert "decode-path quality C10: PASS" in agg(GOOD_KL)["detail"] and agg(GOOD_KL)["status"] == "PASS"
    assert agg(W4_KL)["status"] == "WARN" and "C10: WARN" in agg(W4_KL)["detail"]


# ---- the target-mismatch refusal of tools/kl.py reaches C10 as INVALID, never PASS (real kl.py) ----
def _rows(path, targets, lp=-0.5):
    with open(path, "w", encoding="utf-8") as fh:
        for pos, tgt in enumerate(targets):
            fh.write(f"{pos}\t{tgt}\t{lp}\t{tgt}\t{lp}\t1\t-9.0\t{lp}\t{tgt}:{lp}\t99:-3.0\n")


def _res(tmp, arm, idx, targets):
    p = tmp / f"{arm}{idx}.tsv"
    _rows(p, targets)
    return _run("W7", arm, idx, kl_files={"forced": {"path": str(p), "rows": len(targets), "ok": True}})


def _compare(tmp, results):
    import qual_decode
    return qual_decode.compare(CFG, results, qual_report.Recorder(str(tmp / "out")))


def test_mismatched_targets_are_invalid_not_a_pass(tmp_path):
    ref, same, other = _res(tmp_path, "M", 0, [5, 6, 7]), _res(tmp_path, "P2", 0, [5, 6, 7]), _res(tmp_path, "P2", 1, [5, 6, 8])
    recs = {k["pair"]: k for k in _compare(tmp_path, [ref, same, other])}
    assert recs["D-P2-s0-vs-M-s0"]["rc"] == 0 and "KL-GATE PASS" in recs["D-P2-s0-vs-M-s0"]["line"]
    bad = recs["D-P2-s1-vs-M-s0"]
    assert bad["rc"] == 2 and "target ids differ" in bad["line"] and bad["text"] == "decode-pooled"
    assert _c10(kl=[k for k in GOOD_KL if k["pair"] != "D-P2-s1-vs-M-s0"] + [bad])["status"] == "WARN"


def test_compare_pairs_each_live_start_with_the_lowest_m_start_and_records_the_noise_floor(tmp_path):
    rs = [_res(tmp_path, a, i, [5, 6, 7]) for a, i in (("M", 1), ("M", 0), ("P0", 0), ("P2", 1))]
    recs = _compare(tmp_path, rs)
    assert sorted(k["pair"] for k in recs) == ["D-P0-s0-vs-M-s0", "D-P2-s1-vs-M-s0", "D-noise-M-s1-vs-M-s0"]
    assert [k["text"] for k in recs if "noise" in k["pair"]] == ["decode-noise"]
    assert all(k["rc"] == 0 for k in recs)


def test_compare_without_a_reference_or_workload_makes_no_record(tmp_path):
    assert _compare(tmp_path, [_res(tmp_path, "P2", 0, [5])]) == []
    import qual_decode
    cfg = {**CFG, "workloads": {k: v for k, v in CFG["workloads"].items() if k != "W7"}}
    assert qual_decode.compare(cfg, [], qual_report.Recorder(str(tmp_path / "o2"))) == []


def test_unusable_rows_file_is_invalid(tmp_path):
    ref = _res(tmp_path, "M", 0, [5, 6])
    arm = _run("W7", "P2", 0, kl_files={"forced": {"path": "missing", "rows": 0, "ok": True}})
    assert [k["rc"] for k in _compare(tmp_path, [ref, arm])] == [2]


# ---- capture and pooling ----
FAKE = '''import json, sys
out = sys.argv[sys.argv.index("--out") + 1]
rows = [[0, 5], [1, 6], [1, 6], [2, 7]]  # position 1 appears twice: a rejected draft window rewrote the row
with open(out, "w") as fh:
    for pos, tgt in rows[:3]: fh.write(f"{pos}\\t{tgt}\\t-0.5\\t{tgt}\\t-0.5\\t1\\t-9\\t-0.5\\t{tgt}:-0.5\\n")
    for pos, tgt in rows[3:]: fh.write(f"{pos}\\t{tgt}\\t-0.5\\t{tgt}\\t-0.5\\t1\\t-9\\t-0.5\\t{tgt}:-0.5\\n")
json.dump({"items": [{"first_line": 0, "end_line": 3}, {"first_line": 3, "end_line": 4}]}, open(out + ".meta.json", "w"))
sys.exit(int(sys.argv[sys.argv.index("--rc") + 1]))
'''


def _ctx(tmp_path, rc, dry=False):
    import qual_work
    script = tmp_path / "fake_force_check.py"
    script.write_text(FAKE, encoding="utf-8")
    cfg = json.loads(json.dumps(CFG))
    for name in ("force_check", "forced_set", "forced_cfg_M", "forced_cfg_P0", "forced_cfg_P2"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
        cfg["vars"][name] = str(tmp_path / name)
    cfg["workloads"]["W7"]["capture"]["argv"] = [sys.executable, "-B", str(script), "--out", "{rundir}/forced.tsv", "--rc", str(rc)]
    return qual_work.Ctx(cfg, "http://127.0.0.1:1", qual_report.Recorder(str(tmp_path / "out")), dry, str(tmp_path / "out"))


def test_capture_pools_last_row_per_position_with_per_item_offsets(tmp_path):
    import qual_decode
    got, note = qual_decode.capture(_ctx(tmp_path, 0), qual_cfg.Run("W7", "P2", "base", 0))
    rows = open(got["kl_files"]["forced"]["path"], encoding="utf-8").read().splitlines()
    assert [r.split("\t")[0] for r in rows] == ["0", "1", "1000002"] and got["ok"] and got["kl_files"]["forced"]["rows"] == 3
    assert got["workload"] == "W7" and got["arm"] == "P2" and got["key"] == "P2" and got["idx"] == 0 and "3 pooled rows" in note


def test_capture_that_exits_nonzero_is_a_failed_step_with_no_result(tmp_path):
    import qual_decode
    ctx = _ctx(tmp_path, 3)
    with pytest.raises(RuntimeError, match="exit 3"):
        qual_decode.capture(ctx, qual_cfg.Run("W7", "M", "base", 0))


def test_capture_with_a_missing_tool_fails_before_running_anything(tmp_path):
    import qual_decode
    ctx = _ctx(tmp_path, 0)
    ctx.cfg["vars"]["forced_set"] = str(tmp_path / "nope.json")
    with pytest.raises(RuntimeError, match="forced_set"):
        qual_decode.capture(ctx, qual_cfg.Run("W7", "M", "base", 0))


def test_dry_capture_runs_nothing_and_yields_a_result_with_no_rows(tmp_path):
    import qual_decode
    got, note = qual_decode.capture(_ctx(tmp_path, 3, dry=True), qual_cfg.Run("W7", "M", "base", 0))
    assert "DRY-RUN" in note and got["kl_files"] == {} and not (tmp_path / "out" / "runs").exists()


# ---- config: W7 is a validated workload; its capture program is never a service manager ----
def test_w7_is_planned_with_every_arm_and_validated():
    cfg = qual_cfg.load(str(QUAL / "arms.json"), {k: "x" for k in CFG["vars"] if str(CFG["vars"][k]).startswith("UNSET")})
    runs = [r for r in qual_cfg.plan(cfg, None) if r.workload == "W7"]
    assert sorted({r.arm for r in runs}) == ["M", "P0", "P2"] and len(runs) == 3 * cfg["workloads"]["W7"]["n"]


@pytest.mark.parametrize("edit,msg", [(lambda w: w["capture"].update(argv=["systemctl", "stop", "x"]), "not a capture tool"),
                                      (lambda w: w.pop("capture"), "capture.argv"),
                                      (lambda w: w["arms"].update(N1=["base"]), "forced_cfg")], ids=["forbidden-program", "no-capture", "arm-without-forced-cfg"])
def test_bad_w7_config_is_refused_before_anything_starts(tmp_path, edit, msg):
    cfg = json.loads(json.dumps(CFG))
    edit(cfg["workloads"]["W7"])
    path = tmp_path / "arms.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(qual_cfg.ConfigError, match=msg):
        qual_cfg.load(str(path), {k: "x" for k in cfg["vars"] if str(cfg["vars"][k]).startswith("UNSET")})
