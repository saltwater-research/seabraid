# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Checker for the `file-moves` persona of the placement qualification (FR DESIGN 9: arms FR-A1..A3 = W8..W10) and its F1 verdict.

The persona is opt-in (never part of the default plan), runs the blob arm F0 against the file-reference arm F1 with ONE variable
(--retier-file-moves 0|1), and F1 is PASS only with the engine's file counters present, file bytes moved, zero bad sums, and no
file read in the blob arm.  Absence of any counter is WARN, never PASS (the module contract of qual_judge).  Pure Python."""
import copy
import json

import pytest

import placement_qual_fixtures  # noqa: F401  (puts scripts/placement-qual on sys.path)
import qual_cfg  # noqa: E402
import qual_judge_file as qf  # noqa: E402
import qual_separate as qs  # noqa: E402
import qual_stats as st  # noqa: E402
from placement_qual_fixtures import CFG, QUAL, _run, good_results  # noqa: E402

PERSONA_WLS = ("W8", "W9", "W10")
UNSET = {k: "x" for k, v in CFG["vars"].items() if str(v).startswith("UNSET")}


def test_the_persona_workloads_exist_and_carry_the_persona_tag():
    assert {k for k, v in CFG["workloads"].items() if v.get("persona") == "file-moves"} == set(PERSONA_WLS)
    for k in PERSONA_WLS:
        assert set(CFG["workloads"][k]["arms"]) == {"F0", "F1"}


def test_the_two_arms_differ_in_exactly_the_file_moves_flag():
    a0, a1 = CFG["arms"]["F0"], CFG["arms"]["F1"]
    assert a0["args"][a0["args"].index("--retier-file-moves") + 1] == "0"
    assert a1["args"][a1["args"].index("--retier-file-moves") + 1] == "1"
    strip = lambda a: [x for i, x in enumerate(a["args"]) if x != "--retier-file-moves" and a["args"][i - 1] != "--retier-file-moves"]  # noqa: E731
    assert strip(a0) == strip(a1) and a0["env"] == a1["env"] and a0["helper_start"] == a1["helper_start"]


def test_the_config_validates_and_the_persona_is_not_in_the_default_plan():
    cfg = qual_cfg.load(str(QUAL / "arms.json"), UNSET)
    assert all(r.workload not in PERSONA_WLS for r in qual_cfg.plan({**cfg, "workloads": {k: v for k, v in cfg["workloads"].items() if not v.get("persona")}}, None))


def test_run_py_selects_the_persona_only_when_asked(tmp_path):
    import run
    args = run.parse_args(["--out", str(tmp_path), "--persona", "file-moves"])
    var = [f"{k}={v}" for k, v in UNSET.items()]
    args.var = var
    cfg, full, runs = run.select(args)
    assert set(cfg["workloads"]) == set(PERSONA_WLS) and {r.arm for r in runs} == {"F0", "F1"}
    plain = run.parse_args(["--out", str(tmp_path)])
    plain.var = var
    assert not (set(run.select(plain)[0]["workloads"]) & set(PERSONA_WLS))
    bad = run.parse_args(["--out", str(tmp_path), "--persona", "nope"])
    bad.var = var
    with pytest.raises(qual_cfg.ConfigError, match="persona"):
        run.select(bad)


# ---- the F1 verdict ----------------------------------------------------------------------------------------------------

def _counters(reads=40.0, direct=60e6, buffered=0.0, bad=0.0):
    return {"file_reads": [reads], "file_direct_bytes": [direct], "file_buffered_bytes": [buffered], "bad_sum": [bad]}


def _file_set(f1_io=None, f0_io=None):
    out = []
    for wl in PERSONA_WLS:
        for i in range(CFG["workloads"][wl]["n"]):
            for arm, io in (("F0", f0_io if f0_io is not None else _counters(0.0, 0.0)), ("F1", f1_io if f1_io is not None else _counters())):
                for v in CFG["workloads"][wl]["arms"][arm]:
                    out.append(_run(wl, arm if v == "base" else f"{arm}/{v}", i, io=io))
    return out


def f1(res, cfg=CFG):
    return [v for v in qs.judge_all(cfg, good_results() + res, []) if v["id"] == "F1"]


def test_f1_passes_with_counters_bytes_moved_and_no_bad_sum():
    assert [v["status"] for v in f1(_file_set())] == ["PASS"]


@pytest.mark.parametrize("io,why", [({}, "no counters"), ({"bad_sum": [0.0]}, "only bad_sum"),
                                     (_counters(reads=0.0, direct=0.0), "nothing moved by file")], ids=lambda x: x if isinstance(x, str) else "")
def test_f1_declines_without_the_counters_or_without_file_bytes(io, why):
    assert [v["status"] for v in f1(_file_set(f1_io=io))] == ["WARN"], why


def test_f1_fails_on_any_bad_sum_even_with_everything_else_good():
    assert [v["status"] for v in f1(_file_set(f1_io=_counters(bad=1.0)))] == ["FAIL"]


def test_f1_fails_when_the_blob_arm_read_a_file():
    assert [v["status"] for v in f1(_file_set(f0_io=_counters()))] == ["FAIL"]


def test_f1_declines_when_a_required_start_is_missing():
    res = [r for r in _file_set() if not (r["workload"] == "W8" and r["arm"] == "F1" and r["idx"] == 0)]
    assert [v["status"] for v in f1(res)] == ["WARN"]


def test_f1_declines_with_no_file_moves_starts_at_all():
    assert [v["status"] for v in f1([])] == ["WARN"]


def test_f1_is_absent_when_the_persona_is_not_selected_and_never_enters_agg():
    cfg = copy.deepcopy(CFG)
    cfg["workloads"] = {k: v for k, v in cfg["workloads"].items() if not v.get("persona")}
    assert f1([], cfg) == []
    agg = [v for v in qs.judge_all(CFG, good_results() + _file_set(f1_io=_counters(bad=1.0)), []) if v["id"] == "AGG"][0]
    assert "F1" not in agg["detail"]


def test_the_counter_patterns_parse_the_one_provisional_log_line():
    line = "helper file io: direct_bytes=1500000 buffered_bytes=0 reads=3 bad_sum=0\n"
    got = st.parse_io(line, CFG["io_patterns"])
    assert got["file_direct_bytes"] == [1500000.0] and got["file_buffered_bytes"] == [0.0] and got["file_reads"] == [3.0] and got["bad_sum"] == [0.0]
    assert "PROVISIONAL" in CFG["io_patterns_file_note"]
    json.dumps(CFG)  # the note is data, not a stray comment key the loader would reject
