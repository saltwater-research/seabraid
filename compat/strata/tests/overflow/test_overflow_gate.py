# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Checker for ov_gate.py and ov_parse.py: a run that really read the SSD passes; every way of fooling the test (page cache, no limit, shrunk
cap, wrong arithmetic, cold counters) is INVALID.  Each sabotage changes ONE thing of a start that otherwise passes."""
import copy
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ovf_synth as S  # noqa: E402
import ov_gate  # noqa: E402
import ov_parse  # noqa: E402

CFG = S.load_cfg()


def verdict(start, arm="O2", cfg=CFG):
    return ov_gate.speed_verdict(start, cfg["plans"][arm], cfg)


def sabotaged(edit, arm="O2"):
    s = copy.deepcopy(S.good_start(CFG, arm))
    edit(s)
    return verdict(s, arm)


@pytest.mark.parametrize("arm", ["O1", "O2", "O3"])
def test_a_start_that_really_overflowed_to_the_ssd_passes(arm):
    v = verdict(S.good_start(CFG, arm), arm)
    assert v["valid"], v["reasons"]


def test_SABOTAGE_a_page_cache_served_run_is_refused():
    """The engine says it read 400 MB from the file tier per request, but the disk and the process counters barely moved: RAM served it."""
    def edit(s):
        for r in s["requests"]:
            r["dev_read_mb"], r["proc_read_mb"] = 4.0, 3.0
    v = sabotaged(edit)
    assert not v["valid"]
    assert any("page cache" in r and "NVMe proof failed on 3 of 3" in r for r in v["reasons"])


def test_SABOTAGE_only_the_device_counter_is_cold_is_still_refused():
    v = sabotaged(lambda s: [r.update(dev_read_mb=10.0) for r in s["requests"]])
    assert not v["valid"] and any("disk counter" in r for r in v["reasons"])
    v = sabotaged(lambda s: [r.update(proc_read_mb=10.0) for r in s["requests"]])
    assert not v["valid"] and any("process counter" in r for r in v["reasons"])


def test_SABOTAGE_a_file_tier_that_read_nothing_means_nothing_overflowed():
    v = sabotaged(lambda s: [r.update(file_mb=0.0) for r in s["requests"]])
    assert not v["valid"] and any("no expert came from the files" in r for r in v["reasons"])
    v = sabotaged(lambda s: [r.update(file_mb=None) for r in s["requests"]])
    assert not v["valid"]


def test_SABOTAGE_a_missing_counter_is_never_read_as_zero_io():
    v = sabotaged(lambda s: [r.update(dev_read_mb=None) for r in s["requests"]])
    assert not v["valid"] and any("no disk read counter" in r for r in v["reasons"])


def test_SABOTAGE_the_cgroup_limit_not_in_force_is_refused():
    def edit(s):
        for row in s["cgroup"]:
            row["memory_max"] = "max"
    v = sabotaged(edit)
    assert not v["valid"] and any("no memory limit" in r for r in v["reasons"])


def test_SABOTAGE_a_limit_that_differs_from_the_plan_or_has_swap_is_refused():
    plan = CFG["plans"]["O2"]
    v = sabotaged(lambda s: s["cgroup"][0].update(memory_max=plan.engine_cap_bytes * 2))
    assert not v["valid"] and any("the plan is" in r for r in v["reasons"])
    v = sabotaged(lambda s: s["cgroup"][0].update(swap_max="max"))
    assert not v["valid"]


def test_SABOTAGE_an_uncapped_helper_next_to_a_capped_engine_is_refused():
    v = sabotaged(lambda s: s["cgroup"][-1].update(memory_max="max"))
    assert not v["valid"] and any("helper-cuda" in r for r in v["reasons"])


def test_SABOTAGE_no_cgroup_readback_at_all_is_refused():
    v = sabotaged(lambda s: s.update(cgroup=[]))
    assert not v["valid"] and any("not proven in force" in r for r in v["reasons"])
    v = sabotaged(lambda s: s.update(cgroup=[x for x in s["cgroup"] if x["comm"] != "strata"]))
    assert not v["valid"] and any("engine's own cgroup was not read" in r for r in v["reasons"])


def test_SABOTAGE_an_oom_kill_inside_the_limit_is_refused_with_the_remedy():
    v = sabotaged(lambda s: s["memory"].update(oom_kill=1))
    assert not v["valid"] and any("engine_overhead_gib" in r for r in v["reasons"])


def test_SABOTAGE_model_files_still_cached_at_start_are_refused_but_an_unmeasurable_cache_is_only_a_note():
    v = sabotaged(lambda s: s["preflight"].update(resident_mb=3273.0))
    assert not v["valid"] and any("still in the page cache" in r for r in v["reasons"])
    v = sabotaged(lambda s: s["preflight"].update(resident_mb=None, errors=["fincore could not measure"]))
    assert v["valid"] and any("fincore" in n for n in v["notes"])
    assert not sabotaged(lambda s: s.update(preflight=None))["valid"]


def test_SABOTAGE_a_shrunk_cap_or_changed_ram_budget_means_the_arms_are_not_matched():
    assert any("the cap was changed" in r for r in sabotaged(lambda s: s["startup"].update(cap_slots=2000))["reasons"])
    assert any("RAM budget" in r for r in sabotaged(lambda s: s["startup"].update(ram_budget_gib=3.0))["reasons"])
    assert any("no 'expert cache N slots'" in r for r in sabotaged(lambda s: s["startup"].update(cap_slots=None))["reasons"])


def test_SABOTAGE_wrong_arithmetic_is_caught_against_the_engines_own_startup_lines():
    v = sabotaged(lambda s: s["startup"].update(complement_gib=20.0))
    assert not v["valid"] and any("experts_gib or the slot counts are wrong" in r for r in v["reasons"])
    v = sabotaged(lambda s: s["startup"].update(ram_held_blobs=15000))
    assert not v["valid"] and any("on the files" in r for r in v["reasons"])
    assert any("RAM budget' startup line" in r for r in sabotaged(lambda s: s["startup"].update(ram_budget_gib=None))["reasons"])


def test_SABOTAGE_a_helper_channel_that_does_not_match_the_arm_is_refused():
    assert any("helper channels" in r for r in sabotaged(lambda s: s["startup"].update(channels=()))["reasons"])
    assert any("helper channels" in r for r in sabotaged(lambda s: s["startup"].update(channels=("/seabraid-ovf1", "/seabraid-ovf2")), "O2")["reasons"])
    assert any("helper channels" in r for r in sabotaged(lambda s: s["startup"].update(channels=("/seabraid-ovf1",)), "O1")["reasons"])
    assert any("helper channels" in r for r in sabotaged(lambda s: s["startup"].update(channels=("/other",)), "O2")["reasons"])


def test_SABOTAGE_o3_needs_BOTH_channels_up_in_the_engine_log():
    """The coordinator's rule: the run is refused only if the engine log does not show both helper channels up."""
    assert any("helper channels ['/seabraid-ovf1'] up" in r for r in sabotaged(lambda s: s["startup"].update(channels=("/seabraid-ovf1",)), "O3")["reasons"])
    assert not sabotaged(lambda s: None, "O3")["reasons"]


def test_a_ladder_arm_that_really_moved_every_rung_passes():
    v = verdict(S.good_start(CFG, "O3L"), "O3L")
    assert v["valid"], v["reasons"]
    assert [(m["lower"], m["upper"], m["src"]) for m in S.good_start(CFG, "O3L")["moves"]] == [(3, 2, "fallback"), (2, 1, "card"), (1, 0, "card")]


def test_SABOTAGE_a_ladder_that_never_moved_is_refused_even_though_its_flags_were_accepted():
    v = sabotaged(lambda s: s.update(moves=[]), "O3L")
    assert not v["valid"] and any("never committed a hop" in r for r in v["reasons"])


def test_SABOTAGE_moves_that_never_came_from_the_fallback_rung_are_refused():
    def edit(s):
        s["moves"] = [m for m in s["moves"] if m["src"] != "fallback"]
    v = sabotaged(edit, "O3L")
    assert not v["valid"] and any("no move from the fallback rung" in r for r in v["reasons"])


@pytest.mark.parametrize("missing", [1, 2])
def test_SABOTAGE_a_ladder_that_never_reached_one_helper_rung_is_refused(missing):
    v = sabotaged(lambda s: s.update(moves=[m for m in s["moves"] if m["upper"] != missing]), "O3L")
    assert not v["valid"] and any(f"no hop onto helper rung {missing}" in r for r in v["reasons"])


def test_SABOTAGE_a_stale_epoch_refusal_in_a_ladder_run_is_refused():
    v = sabotaged(lambda s: s.update(stale=1), "O3L")
    assert not v["valid"] and any("stale-placement record" in r for r in v["reasons"])
    assert verdict(S.good_start(CFG, "O3L"), "O3L")["valid"]


def test_a_static_arm_is_not_required_to_move_but_stray_moves_are_noted():
    assert verdict(S.good_start(CFG, "O3"), "O3")["valid"]
    v = sabotaged(lambda s: s.update(moves=S.ladder_moves(2)), "O3")
    assert v["valid"] and any("no ladder flags" in n for n in v["notes"])


def test_the_move_lines_and_stale_records_are_parsed_from_the_adapters_formats():
    text = S.move_lines(2) + "Seabraid Qwen layer 3 refused: stale placement epoch 9; work stays local\n"
    assert ov_parse.moves(text) == S.ladder_moves(2) and ov_parse.stale_count(text) == 1
    assert ov_parse.stale_count("seabraid retier: helper residency is STALE after an applied round (x)") == 1
    assert ov_parse.moves("placement round=3 committed=2") == [] and ov_parse.stale_count("placement round=3") == 0
    assert ov_parse.startup(S.budget_text(CFG["plans"]["O3"])).channels == ("/seabraid-ovf1", "/seabraid-ovf2")


def test_SABOTAGE_a_model_on_a_spinning_disk_or_an_unresolved_disk_is_refused_unless_allowed():
    v = sabotaged(lambda s: s["disk"].update(rotational=True))
    assert not v["valid"] and any("rotational" in r for r in v["reasons"])
    cfg = S.load_cfg(allow_rotational=1)
    s = S.good_start(cfg)
    s["disk"]["rotational"] = True
    assert ov_gate.speed_verdict(s, cfg["plans"]["O2"], cfg)["valid"]
    assert not sabotaged(lambda s: s.update(disk=None))["valid"]


def test_SABOTAGE_too_few_counted_requests_or_an_aborted_start_is_refused():
    assert any("usable decode requests" in r for r in sabotaged(lambda s: s["requests"][2].update(usable=False))["reasons"])
    assert any("aborted" in r for r in sabotaged(lambda s: s.update(aborted=True))["reasons"])
    assert sabotaged(lambda s: s.update(error="engine died"))["reasons"] == ["engine died"]


def test_every_reason_is_listed_not_only_the_first():
    def edit(s):
        s["startup"]["cap_slots"] = 1
        s["memory"]["oom_kill"] = 2
        s["cgroup"] = []
    assert len(sabotaged(edit)["reasons"]) >= 3


def test_the_reference_is_judged_without_the_overflow_checks_but_notes_a_file_tier_that_was_used():
    s = S.good_start(CFG, "R")
    assert verdict(s, "R")["valid"]
    s["requests"] = [S.request(p, k, k >= 1, file_mb=900.0) for k in range(4) for p in ("decode", "prefill")]
    v = verdict(s, "R")
    assert v["valid"] and any("not an all-VRAM reference" in n for n in v["notes"])


def test_arm_status_never_passes_invalid_data_and_distinguishes_ungated_from_failed():
    ok, bad = {"valid": True}, {"valid": False}
    assert ov_gate.arm_status([bad, ok], "PASS", False, True) == "INVALID"
    assert ov_gate.arm_status([], "PASS", False, True) == "INVALID"
    assert ov_gate.arm_status([ok], "FAIL", False, True) == "FAIL"
    assert ov_gate.arm_status([ok], "NONE", False, True) == "INVALID"  # a wanted quality gate with no result is not a pass
    assert ov_gate.arm_status([ok], "NONE", False, False) == "UNGATED"
    assert ov_gate.arm_status([ok], "PASS", False, True) == "PASS" and ov_gate.arm_status([ok], "NONE", True, True) == "PASS"


def w7(arm, idx, rows=60, mb=5000.0, **kw):
    rec = {"workload": "w7", "arm": arm, "idx": idx, "mode": "forced", "tag": f"w7-{arm}-s{idx}", "error": None, "cgroup": S.good_start(CFG, arm)["cgroup"],
           "preflight": S.good_start(CFG, arm)["preflight"], "disk": S.good_start(CFG, arm)["disk"], "dev_read_mb": mb, "memory": {"oom_kill": 0},
           "kl_files": {"forced": {"path": f"/p/{arm}{idx}.tsv", "rows": rows, "ok": True}}}
    rec.update(kw)
    return rec


def test_w7_capture_needs_rows_a_device_counter_that_rose_and_the_limit_in_force():
    plan = CFG["plans"]["O2"]
    assert ov_gate.w7_verdict(w7("O2", 0), plan, CFG)["valid"]
    assert any("no rows" in r for r in ov_gate.w7_verdict(w7("O2", 0, rows=0), plan, CFG)["reasons"])
    assert any("did not overflow" in r for r in ov_gate.w7_verdict(w7("O2", 0, mb=3.0), plan, CFG)["reasons"])
    assert any("did not overflow" in r for r in ov_gate.w7_verdict(w7("O2", 0, mb=None), plan, CFG)["reasons"])
    bad = w7("O2", 0)
    bad["cgroup"] = [dict(r, memory_max="max") for r in bad["cgroup"]]
    assert not ov_gate.w7_verdict(bad, plan, CFG)["valid"]
    assert not ov_gate.w7_verdict(w7("O2", 0, error="capture exit 1"), plan, CFG)["valid"]


def test_kl_rows_gate_the_arm_against_the_reference_and_report_the_noise_floor():
    calls = []

    def fake(args):
        calls.append(args)
        return (0, "PASS: mean KL 0.001")

    rows = ov_gate.kl_rows([w7("R", 0), w7("O1", 0), w7("R", 1)], CFG, fake)
    assert [(r["arm"], r["kind"], r["rc"]) for r in rows] == [("O1", "gate", 0), ("R", "noise", 0)]
    assert calls[0] == ["--baseline", "/p/R0.tsv", "--arm", "/p/O10.tsv", "--max-kl", "0.03", "--min-top1", "0.95"]
    assert calls[1] == ["/p/R0.tsv", "/p/R1.tsv"]
    assert ov_gate.kl_status(rows, "O1") == "PASS" and ov_gate.kl_status(rows, "O2") == "NONE"


def test_a_lost_reference_makes_every_gate_row_invalid_never_a_pass():
    rows = ov_gate.kl_rows([w7("O1", 0)], CFG, lambda a: (0, "PASS"))
    assert [r["rc"] for r in rows] == [2] and ov_gate.kl_status(rows, "O1") == "INVALID"
    rows = ov_gate.kl_rows([w7("R", 0), w7("O1", 0, rows=0)], CFG, lambda a: (0, "PASS"))
    assert ov_gate.kl_status(rows, "O1") == "INVALID"


def test_kl_status_fail_beats_invalid_beats_pass():
    mk = lambda rc: {"arm": "O1", "kind": "gate", "rc": rc, "idx": 0, "line": ""}  # noqa: E731
    assert ov_gate.kl_status([mk(0), mk(2)], "O1") == "INVALID" and ov_gate.kl_status([mk(0), mk(1), mk(2)], "O1") == "FAIL"


def test_the_startup_lines_are_parsed_from_the_engines_own_formats():
    plan = CFG["plans"]["O2"]
    s = ov_parse.startup(S.budget_text(plan))
    assert s.cap_slots == 3000 and s.ram_budget_gib == 6.0 and s.ram_held_blobs == 3137 and s.helper_up and s.file_mode == "through the file cache"
    assert s.complement_gib == pytest.approx(plan.complement_gib, abs=0.01) and s.slot_gib == pytest.approx(47.0 / 24576, rel=1e-3)
    empty = ov_parse.startup("")
    assert empty.cap_slots is None and empty.ram_budget_gib is None and empty.slot_gib is None
