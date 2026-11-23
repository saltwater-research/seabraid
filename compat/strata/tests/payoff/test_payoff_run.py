# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""End-to-end checker for scripts/payoff/payoff.py with a synthetic engine: no process is started and no HTTP is sent.

The fake engine appends engine-format log lines (payoff_synth) for every request, ``qual_work.send`` is monkeypatched to feed
it, and the real runner, the real pay_gate (real tools/kl.py on synthetic rows) and the real analysis run on top.  The test
asserts what a reader gets: the crossover sentence in SUMMARY.txt, the chain files, and the exit codes of a failed quality gate
and of a dead start.  CPU only."""
import json
import os
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import payoff_synth as S  # noqa: E402
import payoff as PO  # noqa: E402
import pay_cfg  # noqa: E402
import pay_measure as PM  # noqa: E402
import qual_cfg  # noqa: E402
import qual_report  # noqa: E402
import qual_work  # noqa: E402

CAPS = "9000,6000,3000"
DECODE = {"A": {9000: 100, 6000: 55, 3000: 30}, "B": {9000: 95, 6000: 50, 3000: 42}, "C": {9000: 100.5, 6000: 75, 3000: 66}}
VARS = ["--var", "gpu=0", "--var", "rocm=/rocm", "--var", "pack=/pack", "--var", "native=/model.gguf"]
CURRENT = []


class FakeEngine:
    """Quacks like qual_engine.Engine for what the runner uses; its log is the synthetic engine log."""

    dead_arm = None
    kl_noise = {}

    def __init__(self, cfg, run, rundir, url, dry):
        self.cfg, self.run, self.rundir = cfg, run, rundir
        self.cap, self.kl = cfg["workloads"][run.workload]["cap"], cfg["workloads"][run.workload]["mode"] == "kl"
        self.text, self.sent, self.texts_sent = "", 0, 0

    def start(self):
        os.makedirs(self.rundir, exist_ok=True)
        if self.run.arm == type(self).dead_arm:
            raise RuntimeError("synthetic start failure")
        self.text = S.startup_text(self.cap, helper=self.run.arm == "C")
        CURRENT[:] = [self]
        return "fake started"

    def wait_ready(self):
        return "fake ready"

    def log_text(self):
        return self.text

    def stop(self):
        return 0


def fake_send(ctx, prompt, max_tokens):
    eng = CURRENT[0]
    arm, cap = eng.run.arm, eng.cap
    if eng.kl:
        noise = type(eng).kl_noise.get((arm, cap), 0.0)
        S.write_logpos(os.path.join(eng.rundir, "rows.tsv"), seed=10 + eng.texts_sent, noise=noise)
        eng.texts_sent += 1
        return {"ok": True, "wall_s": 0.1, "sha": "kl", "text": "", "completion_n": 1, "decode_tps": None, "approx": True}
    phase = "decode" if max_tokens > 1 else "prefill"
    eng.sent += 1
    tps = DECODE[arm][cap] + ((eng.sent * 7) % 5 - 2) * 0.2
    eng.text += S.request_text(arm, phase, tps, 500.0 + 10 * eng.sent)
    return {"ok": True, "wall_s": 0.1, "sha": "abc", "text": "x", "completion_n": 400, "decode_tps": tps, "approx": False}


def config_file(tmp_path):
    cfg = json.load(open(os.path.join(S.SCRIPTS, "arms.json"), encoding="utf-8"))
    cfg["gates"]["settle_s"], cfg["gates"]["settle_timeout_s"] = 0, 0.2
    p = tmp_path / "arms.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    return str(p)


def run_main(tmp_path, monkeypatch, engine=FakeEngine, extra=()):
    """Run the whole CLI on ``engine``; returns (out dir, exit code)."""
    monkeypatch.setattr(qual_work, "send", fake_send)
    out = tmp_path / "out"
    return out, PO.main(["--out", str(out), "--config", config_file(tmp_path), "--caps", CAPS, *VARS, *extra], engine_factory=engine)


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    """One full run of the CLI on the synthetic engine; the tests below read its outputs."""
    tmp = tmp_path_factory.mktemp("chain")
    with pytest.MonkeyPatch.context() as mp:
        out, rc = run_main(tmp, mp)
    return out, rc


def test_full_chain_locates_the_crossover_in_the_summary(chain):
    out, rc = chain
    assert rc == 0
    summary = (out / "SUMMARY.txt").read_text(encoding="utf-8")
    assert "boundary lies between cap 9000 (tie) and cap 6000 (pays)" in summary
    assert "=== cap 3000: logged expert-cache slots A 3000, B 3000, C 3000; matched ===" in summary
    assert "second GPU decode: PAYS" in summary and "UNGATED" not in summary  # the quality gate ran for every arm


def test_full_chain_writes_the_recorded_chain_files(chain):
    out, _ = chain
    for name in ("chain.tsv", "chain_report.html", "results.json", "analysis.json"):
        assert (out / name).exists()
    rows = (out / "chain.tsv").read_text(encoding="utf-8")
    assert "cap6000-C-s0-start" in rows and "cap6000kl-C-s0-measure" in rows and "kl-noise-cap9000-A-s1" in rows


def chain_requests(chain, phase):
    out, _ = chain
    res = json.loads((out / "results.json").read_text(encoding="utf-8"))
    start = next(r for r in res if r["arm"] == "C" and r["cap"] == 6000 and r["mode"] == "speed")
    return res, [q for q in start["requests"] if q["phase"] == phase]


def test_full_chain_records_every_start_and_the_counted_requests(chain):
    res, dec = chain_requests(chain, "decode")
    assert len(res) == 3 * 3 * 2 + 1 and {r["mode"] for r in res} == {"speed", "kl"}
    assert len(dec) == 7 and sum(q["counted"] for q in dec) == 6


def test_full_chain_records_components_and_tier_deltas_per_request(chain):
    _, dec = chain_requests(chain, "decode")
    assert dec[3]["merge_wait_ms"] == 940.0 and dec[3]["helper_requests"] == 1200
    assert dec[3]["ram_blobs"] == 0  # cumulative tiers differenced per request


def test_full_chain_prefill_requests_did_not_reuse_the_prompt_cache(chain):
    _, pre = chain_requests(chain, "prefill")
    assert pre and max(q["reused"] for q in pre) <= 5


def test_a_failing_quality_gate_exits_1_and_removes_that_arm_from_the_point(tmp_path, monkeypatch):
    out, rc = run_main(tmp_path, monkeypatch, engine=_noisy_engine({("C", 6000): 6.0}))
    assert rc == 1
    summary = (out / "SUMMARY.txt").read_text(encoding="utf-8")
    assert "quality gate FAIL" in summary and "second GPU decode: UNDECIDED" in summary
    assert "Undecided (not interpolated): cap 6000" in summary


def _noisy_engine(noise):
    class Noisy(FakeEngine):
        kl_noise = noise

    return Noisy


def test_a_dead_start_is_recorded_and_its_point_is_undecided(tmp_path, monkeypatch):
    class Dead(FakeEngine):
        dead_arm = "B"

    out, rc = run_main(tmp_path, monkeypatch, engine=Dead, extra=("--kl", "off"))
    assert rc == 1  # a failed step is a failed run
    summary = (out / "SUMMARY.txt").read_text(encoding="utf-8")
    assert "B INVALID: the engine did not start or become ready" in summary and "UNDECIDED" in summary


def ctx_for(tmp_path):
    cfg = pay_cfg.load(os.path.join(S.SCRIPTS, "arms.json"), {}, dry=True, sweep={"kl": False})
    cfg["gates"].update(settle_s=0, settle_timeout_s=0.1)
    return qual_work.Ctx(cfg, "http://127.0.0.1:1", qual_report.Recorder(str(tmp_path / "rec")), False, str(tmp_path)), cfg


def test_two_consecutive_failures_abort_the_start(tmp_path, monkeypatch):
    ctx, cfg = ctx_for(tmp_path)
    monkeypatch.setattr(qual_work, "send", lambda c, p, m: {"ok": False, "error": "OSError: hung", "wall_s": 1.0})
    eng = FakeEngine(cfg, qual_cfg.Run("cap11000", "A", "base", 0), str(tmp_path / "e"), "", False)
    eng.start()
    rec = PM.measure_speed(ctx, eng, qual_cfg.Run("cap11000", "A", "base", 0), cfg)
    assert rec["aborted"] and len(rec["requests"]) == 2 and not any(r["ok"] for r in rec["requests"])
    assert sum(1 for r in ctx.rec.rows if r["status"] == "FAIL") == 2


def test_a_prefill_that_reused_the_prompt_cache_is_not_usable(tmp_path, monkeypatch):
    ctx, cfg = ctx_for(tmp_path)
    run = qual_cfg.Run("cap11000", "A", "base", 0)
    eng = FakeEngine(cfg, run, str(tmp_path / "e"), "", False)
    eng.start()
    CURRENT[:] = [eng]

    def send(c, prompt, mt):
        text = S.request_text("A", "decode" if mt > 1 else "prefill", 40.0, 400.0, reused=3000 if mt == 1 else 5)
        eng.text += text
        return {"ok": True, "wall_s": 0.1, "sha": "x"}

    monkeypatch.setattr(qual_work, "send", send)
    rec = PM.measure_speed(ctx, eng, run, cfg)
    pre = [r for r in rec["requests"] if r["phase"] == "prefill"]
    assert pre and not any(r["usable"] for r in pre) and "prompt cache" in "".join(pre[0].get("problems", []) or [pre[0].get("error", "")])
    assert any(r["usable"] for r in rec["requests"] if r["phase"] == "decode")


def test_a_request_without_an_engine_prompt_line_is_refused_not_guessed(tmp_path, monkeypatch):
    ctx, cfg = ctx_for(tmp_path)
    run = qual_cfg.Run("cap11000", "A", "base", 0)
    eng = FakeEngine(cfg, run, str(tmp_path / "e"), "", False)
    eng.start()
    monkeypatch.setattr(qual_work, "send", lambda c, p, m: {"ok": True, "wall_s": 0.1, "sha": "x"})
    rec = PM.measure_speed(ctx, eng, run, cfg)
    assert rec["aborted"] and all(not r["usable"] and "prompt" in r["error"] for r in rec["requests"])
