# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""End-to-end checker for scripts/overflow/overflow.py with a synthetic engine: no process is started and no HTTP is sent.

The fake engine appends engine-format log lines for every request and bumps a PLANTED procfs/cgroup tree like the real engine would; the
real runner, the real ov_proc readers, the real data gate, the real payoff request loop and the real tools/kl.py run on top.  The
sabotage runs prove the harness refuses a page-cache-served run, an unlimited cgroup, a failed quality gate and a lost capture.  CPU only."""
import json
import os
import pathlib
import re
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ovf_synth as S  # noqa: E402
import ov_cache  # noqa: E402
import ov_capture  # noqa: E402
import ov_cfg  # noqa: E402
import overflow as OV  # noqa: E402
import qual_work  # noqa: E402

def status(summary: str, arm: str) -> str:
    """The roll-up status of an arm, read from its ``=== arm: label -> STATUS ===`` header in SUMMARY.txt."""
    return re.search(rf"=== {arm}: .* -> (\w+) ===", summary).group(1)


CURRENT: list = []
SCEN: dict = {}


class FakeEngine:
    """Quacks like OvEngine for what the runner uses; its log is the synthetic engine log, its I/O lands in the planted tree."""

    def __init__(self, cfg, run, rundir, url, dry):
        self.cfg, self.run, self.rundir, self.plan = cfg, run, rundir, cfg["plans"][run.arm]
        self.text, self.cum = "", 0.0

    def start(self):
        os.makedirs(self.rundir, exist_ok=True)
        if self.run.arm in SCEN.get("dead", ()):
            raise RuntimeError("synthetic start failure")
        self.text = S.budget_text(self.plan) + (S.move_lines(self.plan.helpers) if self.plan.ladder and self.run.arm not in SCEN.get("nomoves", ()) else "") if self.plan.capped else "strata generate: expert cache 12000 slots, 22.90 GiB of VRAM; policy is\nseabraid: helper 3090 on channel /x, 22 GiB free\n"
        CURRENT[:] = [self]
        return "fake started"

    def wait_ready(self):
        return "fake ready"

    def log_text(self):
        return self.text

    def pids(self):
        return SCEN["tree"].pids(with_helper=self.plan.helpers > 0)

    def stop(self):
        return 0


def fake_send(ctx, prompt, max_tokens):
    eng, tree = CURRENT[0], SCEN["tree"]
    phase = "decode" if max_tokens > 1 else "prefill"
    mb = 0.0 if not eng.plan.capped else (400.0 if phase == "decode" else 900.0)
    eng.cum += mb
    cold = eng.run.arm in SCEN.get("pagecache", ())
    tree.bump(int(mb * 1e6 * (0.01 if cold else 1.2)), int(mb * 1e6 * (0.01 if cold else 1.1)))
    eng.text += S.decode_log(eng.cum, mb * 1.05) if phase == "decode" else S.prefill_log(eng.cum)
    return {"ok": True, "wall_s": 0.1, "sha": "abc", "text": "x", "completion_n": 200, "decode_tps": 8.0, "approx": False}


def fake_run(argv, env, capped, timeout, logpath):
    """ov_capture._run on the planted tree: writes what force_check would, returns what the poll loop would have seen."""
    rundir = os.path.dirname(logpath)
    arm = os.path.basename(rundir).split("-")[1]
    S.write_forced(rundir, seed=1 if arm == "R" else 1, noise=SCEN.get("kl_noise", {}).get((arm, rundir.endswith("s1")), 0.0))
    tree = SCEN["tree"]
    err = "capture exit 1" if arm in SCEN.get("w7_dead", ()) else None
    rows = tree.host().cgroup_report(tree.pids(with_helper=False)) if capped else []
    mb = 5000.0 if capped and arm not in SCEN.get("pagecache", ()) else 2.0
    return {"rc": 1 if err else 0, "seen": {"cg": S.CG, "cgroup": rows, "oom_kill": 0, "max_hits": 5, "peak": 12 << 30}, "error": err, "dev_read_mb": mb, "kill_note": None}


def setup(tmp_path, monkeypatch):
    SCEN.clear()
    plans = S.load_cfg()["plans"]
    SCEN["tree"] = S.Tree(tmp_path / "tree", plans["O2"])
    SCEN["tree"].set_events(S.CG, oom=0, hits=4)
    cfg = json.load(open(os.path.join(S.SCRIPTS, "arms.json"), encoding="utf-8"))
    cfg["gates"]["settle_s"], cfg["gates"]["settle_timeout_s"] = 0, 0.2
    conf = tmp_path / "arms.json"
    conf.write_text(json.dumps(cfg), encoding="utf-8")
    (tmp_path / "pack").mkdir()
    (tmp_path / "model.gguf").write_bytes(b"x")
    forced = []
    for arm, p in plans.items():
        f = tmp_path / f"forced-{arm}.json"
        args = ["--expert-cache", str(p.main_slots), "--resident-budget-gib", str(p.ram_gib)] if p.capped else ["--expert-cache", "auto"]
        args += [x for i, n in enumerate(p.helper_slots, 1) for x in (f"--expert-cache-device{i}", str(n))]
        lenv = {}
        if p.ladder:
            flags, lenv = ov_cfg.ladder_flags(p.helpers)
            args += flags
        f.write_text(json.dumps({"args": args, "env": lenv}), encoding="utf-8")
        forced += ["--var", f"forced_cfg_{arm}={f}"]
    monkeypatch.setattr(qual_work, "send", fake_send)
    monkeypatch.setattr(ov_capture, "_run", fake_run)
    monkeypatch.setattr(ov_cache, "preflight", lambda files, mx, **kw: {"files": len(files), "resident_mb": 12.0, "max_resident_mb": mx, "errors": [], "ok": True})
    return ["--config", str(conf), "--var", "gpu=0", "--var", "rocm=/rocm", "--var", f"pack={tmp_path / 'pack'}", "--var", f"native={tmp_path / 'model.gguf'}",
            "--var", "disk=" + S.DISK, "--var", "force_check=/fc.py", "--var", "forced_set=/fs.json", *forced]


def run_main(tmp_path, monkeypatch, extra=(), arms="R,O1,O2"):
    args = setup(tmp_path, monkeypatch)
    out = tmp_path / "out"
    return out, OV.main(["--out", str(out), "--arms", arms, *args, *extra], engine_factory=FakeEngine, host=SCEN["tree"].host())


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("chain")
    with pytest.MonkeyPatch.context() as mp:
        out, rc = run_main(tmp, mp)
    return out, rc


def test_a_run_that_really_used_the_ssd_exits_0_with_every_arm_passing(chain):
    out, rc = chain
    summary = (out / "SUMMARY.txt").read_text(encoding="utf-8")
    assert rc == 0, summary
    assert [status(summary, a) for a in ("R", "O1", "O2")] == ["PASS"] * 3
    assert "INVALID" not in summary and "WHAT THIS RUN CANNOT SHOW" in summary and "no ranking between arms" in summary


def test_the_summary_shows_the_witnesses_and_the_cgroup_readback(chain):
    out, _ = chain
    summary = (out / "SUMMARY.txt").read_text(encoding="utf-8")
    assert "engine file tier MB 400.0" in summary and "disk MB 480.0" in summary and "process MB 440.0" in summary
    assert "peak 9.00 GiB, memory.max hit 4 times" in summary and "OOM kills 0" in summary and "SSD = 15439 blobs" in summary


def test_the_chain_files_record_every_step_in_the_planned_order(chain):
    out, _ = chain
    for name in ("chain.tsv", "chain_report.html", "results.json", "analysis.json"):
        assert (out / name).exists()
    rows = (out / "chain.tsv").read_text(encoding="utf-8")
    order = [rows.index(n) for n in ("w7-R-s0-capture", "speed-O1-s0-start", "w7-O1-s0-capture", "speed-O2-s0-measure", "w7-R-s1-capture", "speed-R-s0-stop", "kl-gate-O2-s0")]
    assert order == sorted(order) and "kl-noise-R-s1" in rows and "verdict-O2" in rows


def test_every_request_record_carries_the_three_disk_witnesses(chain):
    out, _ = chain
    res = json.loads((out / "results.json").read_text(encoding="utf-8"))
    o1 = next(r for r in res if r["arm"] == "O1" and r["mode"] == "speed")
    dec = [q for q in o1["requests"] if q["phase"] == "decode" and q["counted"]]
    assert len(dec) == 3
    assert [(q["file_mb"], q["dev_read_mb"] > 400, q["proc_read_mb"] > 400) for q in dec] == [(400.0, True, True)] * 3
    assert o1["cgroup"][0]["memory_max"] == 16 * 2**30 and o1["startup"]["ram_budget_gib"] == 6.0
    assert o1["disk"]["rotational"] is False


def test_SABOTAGE_a_page_cache_served_arm_exits_1_and_is_named(tmp_path, monkeypatch):
    args = setup(tmp_path, monkeypatch)
    SCEN["pagecache"] = {"O1"}
    out = tmp_path / "out"
    rc = OV.main(["--out", str(out), "--arms", "R,O1,O2", *args], engine_factory=FakeEngine, host=SCEN["tree"].host())
    summary = (out / "SUMMARY.txt").read_text(encoding="utf-8")
    assert rc == 1
    assert status(summary, "O1") == "INVALID" and "served from the page cache" in summary and "did not overflow to the SSD" in summary
    assert status(summary, "O2") == "PASS" and status(summary, "R") == "PASS"  # only the cold arm is refused
    assert "FAIL" in (out / "chain.tsv").read_text(encoding="utf-8")


def test_SABOTAGE_an_unlimited_cgroup_exits_1(tmp_path, monkeypatch):
    args = setup(tmp_path, monkeypatch)
    SCEN["tree"].set_limit(S.CG, "max", 0)
    out = tmp_path / "out"
    rc = OV.main(["--out", str(out), "--arms", "R,O1", *args], engine_factory=FakeEngine, host=SCEN["tree"].host())
    assert rc == 1 and "no memory limit" in (out / "SUMMARY.txt").read_text(encoding="utf-8")


def test_SABOTAGE_a_failing_decode_path_kl_gate_exits_1_and_names_the_arm(tmp_path, monkeypatch):
    args = setup(tmp_path, monkeypatch)
    SCEN["kl_noise"] = {("O2", False): 6.0}
    out = tmp_path / "out"
    rc = OV.main(["--out", str(out), "--arms", "R,O1,O2", *args], engine_factory=FakeEngine, host=SCEN["tree"].host())
    summary = (out / "SUMMARY.txt").read_text(encoding="utf-8")
    assert rc == 1 and status(summary, "O2") == "FAIL" and status(summary, "O1") == "PASS"


def test_SABOTAGE_a_dead_engine_or_a_lost_capture_makes_the_arm_invalid_and_the_run_goes_on(tmp_path, monkeypatch):
    args = setup(tmp_path, monkeypatch)
    SCEN["dead"], SCEN["w7_dead"] = {"O1"}, {"O2"}
    out = tmp_path / "out"
    rc = OV.main(["--out", str(out), "--arms", "R,O1,O2", *args], engine_factory=FakeEngine, host=SCEN["tree"].host())
    summary = (out / "SUMMARY.txt").read_text(encoding="utf-8")
    assert rc == 1 and [status(summary, a) for a in ("O1", "O2", "R")] == ["INVALID", "INVALID", "PASS"]


def test_with_the_quality_gate_off_the_run_passes_as_ungated_never_as_pass(tmp_path, monkeypatch):
    out, rc = run_main(tmp_path, monkeypatch, extra=["--kl", "off"])
    summary = (out / "SUMMARY.txt").read_text(encoding="utf-8")
    assert rc == 0 and status(summary, "O1") == "UNGATED" and "OFF (UNGATED)" in summary and "w7-" not in (out / "chain.tsv").read_text(encoding="utf-8")


def test_a_forced_engine_config_that_disagrees_with_the_plan_is_refused_before_the_window_is_spent(tmp_path, monkeypatch):
    args = setup(tmp_path, monkeypatch)
    (tmp_path / "forced-O1.json").write_text(json.dumps({"args": ["--expert-cache", "5000", "--resident-budget-gib", "6"]}), encoding="utf-8")
    out = tmp_path / "out"
    rc = OV.main(["--out", str(out), "--arms", "R,O1", *args], engine_factory=FakeEngine, host=SCEN["tree"].host())
    chain_tsv = (out / "chain.tsv").read_text(encoding="utf-8")
    assert rc == 1 and "w7-O1-s0-capture" in chain_tsv and "the plan needs 3000" in (out / "SUMMARY.txt").read_text(encoding="utf-8") + chain_tsv + "".join(
        p.read_text(encoding="utf-8") for p in (out / "steps").iterdir())


def test_forced_cfg_check_unit():
    plan = S.load_cfg()["plans"]["O2"]
    assert ov_capture.forced_cfg_problems("/nope.json", plan)[0].startswith("cannot read")


def test_the_ladder_arm_passes_with_its_moves_and_o3_with_two_channels(tmp_path, monkeypatch):
    out, rc = run_main(tmp_path, monkeypatch, arms="R,O3,O3L")
    summary = (out / "SUMMARY.txt").read_text(encoding="utf-8")
    assert rc == 0, summary
    assert [status(summary, a) for a in ("O3", "O3L")] == ["PASS", "PASS"]
    assert "placement moves (rung 0 = main card): {'1->0 card': 1, '2->1 card': 1, '3->2 fallback': 1}; stale-placement records 0" in summary
    assert "ladder: live moves ON" in summary and summary.count("placement moves (rung 0") == 1  # only the ladder arm reports moves


def test_SABOTAGE_a_ladder_run_with_no_placement_move_lines_is_invalid_while_static_o3_still_passes(tmp_path, monkeypatch):
    args = setup(tmp_path, monkeypatch)
    SCEN["nomoves"] = {"O3L"}
    out = tmp_path / "out"
    rc = OV.main(["--out", str(out), "--arms", "R,O3,O3L", *args], engine_factory=FakeEngine, host=SCEN["tree"].host())
    summary = (out / "SUMMARY.txt").read_text(encoding="utf-8")
    assert rc == 1 and status(summary, "O3L") == "INVALID" and status(summary, "O3") == "PASS"
    assert "the ladder never committed a hop" in summary


def test_a_ladder_arms_engine_config_without_the_ladder_flags_is_refused_before_the_window_is_spent(tmp_path, monkeypatch):
    setup(tmp_path, monkeypatch)
    (tmp_path / "forced-O3L.json").write_text(json.dumps({"args": ["--expert-cache", "3000", "--resident-budget-gib", "6", "--expert-cache-device1", "3000",
                                                                      "--expert-cache-device2", "3000"], "env": {}}), encoding="utf-8")
    (tmp_path / "forced-O3.json").write_text(json.dumps({"args": ["--expert-cache", "3000", "--resident-budget-gib", "6", "--expert-cache-device1", "3000",
                                                                    "--expert-cache-device2", "3000", "--retier", "on"], "env": {}}), encoding="utf-8")
    plans = S.load_cfg()["plans"]
    assert any("--retier" in r for r in ov_capture.forced_cfg_problems(str(tmp_path / "forced-O3L.json"), plans["O3L"]))
    assert any("STRATA_RAM_ADAPT" in r for r in ov_capture.forced_cfg_problems(str(tmp_path / "forced-O3L.json"), plans["O3L"]))
    assert any("no ladder" in r for r in ov_capture.forced_cfg_problems(str(tmp_path / "forced-O3.json"), plans["O3"]))
