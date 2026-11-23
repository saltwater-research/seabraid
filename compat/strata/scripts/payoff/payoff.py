#!/usr/bin/env python3
# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Second-GPU payoff runner: three arms (local RAM experts / host weights transferred / second GPU) over a sweep of expert caps.

OWNER: this file is only the CLI and the per-start loop.  Config and plan are pay_cfg.py, what one start measures
pay_measure.py, log parsing pay_parse.py, the data gate pay_gate.py, statistics and the crossover pay_analyze.py, text
pay_render.py; engine lifetime, the recorded chain and the HTTP client are placement-qual's (qual_engine, qual_report, qual_work).
CONTRACT: ``payoff.py --out DIR [--dry-run] [--caps N,N,..] [--arms A,B,C] [--starts N] [--kl on|off] [--server URL]
[--config FILE] [--var k=v]``.  Live: writes DIR/chain.tsv, steps/, SUMMARY.txt, chain_report.html, results.json,
analysis.json and runs/<tag>/ (engine.log, rows).  Exit 0 = every step ran and no quality gate FAILed, 1 = a step failed or a
gate FAILed, 2 = bad config or arguments (nothing was started).
--dry-run starts NOTHING and sends nothing: it prints every engine and helper command with its env, the request plan and
the GPU-window ESTIMATE, and creates no files.  A live run needs an exclusive GPU window: it starts and stops its OWN
engine processes by exact tag and pid (qual_engine), never a service manager, and it does not touch the GPU lock file
(the maintainers' window decision is a precondition, README section "How to run it in a GPU window").
FAILURE: fail-open-loud on a step (FAIL row, stderr line, next start); fail-fast on config.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time

import pay_analyze
import pay_cfg
import pay_gate
import pay_measure
import pay_paths  # noqa: F401  (puts placement-qual on sys.path)
import pay_parse
import pay_render
import qual_cfg
import qual_engine
import qual_report
import qual_work


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Measure when a second GPU pays off: three arms over a sweep of expert caps.")
    ap.add_argument("--out", required=True, help="output directory (created by a live run; a dry run creates nothing)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, every command with its env and the window ESTIMATE; start and send nothing")
    ap.add_argument("--caps", default="", help="comma list of expert caps in slots, replacing sweep.caps (needs >= 3)")
    ap.add_argument("--arms", default="", help="comma list of arms, replacing sweep.arms")
    ap.add_argument("--starts", type=int, default=1, help="engine starts per arm per cap point (default 1; 2+ shows start-to-start spread)")
    ap.add_argument("--kl", choices=("on", "off"), default="", help="the teacher-forced quality starts (default: sweep.kl); off = UNGATED speeds")
    ap.add_argument("--server", default="", help="engine server base URL (default http://127.0.0.1:<port var>)")
    ap.add_argument("--config", default=os.path.join(pay_paths.HERE, "arms.json"))
    ap.add_argument("--var", action="append", default=[], metavar="K=V", help="override a config var (gpu, rocm, pack, native, launcher, helper, port, shm, ram_gib, max_context)")
    return ap.parse_args(argv)


def _csv(text: str) -> list[str]:
    return [x for x in text.split(",") if x]


def _sweep_overrides(args: argparse.Namespace) -> dict:
    over: dict = {}
    if args.caps:
        over["caps"] = [int(c) for c in _csv(args.caps)]
    if args.arms:
        over["arms"] = _csv(args.arms)
    if args.kl:
        over["kl"] = args.kl == "on"
    return over


def select(args: argparse.Namespace) -> tuple[dict, list[qual_cfg.Run]]:
    """(cfg, ordered starts) or ConfigError.  Every command is built up front so a missing var fails before anything starts."""
    if args.starts < 1:
        raise qual_cfg.ConfigError("--starts must be >= 1")
    over = dict(kv.split("=", 1) for kv in args.var)
    cfg = pay_cfg.load(args.config, over, args.dry_run, _sweep_overrides(args))
    runs = pay_cfg.plan(cfg, args.starts)
    for run in runs:
        pay_cfg.describe(cfg, run)
    return cfg, runs


def print_plan(cfg: dict, runs: list[qual_cfg.Run]) -> None:
    sw = cfg["sweep"]
    print(f"PLAN {len(runs)} engine starts: caps {sorted(sw['caps'], reverse=True)} x arms {sw['arms']}, order rotated per cap point")
    print(f"PLAN each speed start: {pay_cfg.request_count(cfg)} rounds of one decode ({sw['decode_tokens']} tokens, greedy) and one prefill "
          f"(~{sw['prefill_tokens']} tokens, distinct first line, max_tokens 1) request; the first {sw['warmup']} round(s) are not counted")
    print(f"PLAN each quality start: {len(sw['kl_texts'])} teacher-forced texts {sw['kl_texts']}" if sw["kl"] else "PLAN quality starts: OFF (speeds will be UNGATED)")
    for i, run in enumerate(runs, 1):
        print(f"START {i:3d} {run.tag} [{cfg['workloads'][run.workload]['mode']}] arm {run.arm}: {cfg['arms'][run.arm].get('label', '')}")
        print("\n".join(pay_cfg.describe(cfg, run)))
    lo, hi = pay_cfg.estimate(cfg, runs)
    print(f"ESTIMATE (not measured; config estimate block): GPU window {lo / 3600:.1f} h to {hi / 3600:.1f} h for {len(runs)} starts")


def _stop(eng) -> str:
    left = eng.stop()
    if left:
        raise qual_engine.EngineError(f"{left} process(es) of the engine group survived the stop")
    return "engine group gone"


def _failed_start(run: qual_cfg.Run, cfg: dict, eng, err: str) -> dict:
    rec = pay_measure.start_record(run, cfg, pay_parse.parse_startup(eng.log_text()))
    rec["error"] = err
    return rec


def _measure(ctx: qual_work.Ctx, eng, run: qual_cfg.Run) -> tuple[dict, str]:
    fn = pay_measure.measure_kl if ctx.cfg["workloads"][run.workload]["mode"] == "kl" else pay_measure.measure_speed
    rec = fn(ctx, eng, run, ctx.cfg)
    return rec, f"{len(rec['requests'])} requests, {len(rec['kl_files'])} kl texts, aborted={rec['aborted']}"


def one_start(ctx: qual_work.Ctx, run: qual_cfg.Run, results: list[dict], engine_factory) -> None:
    """Start, wait ready, measure, stop: each a recorded step.  A start that fails is kept as an errored record (its point is then undecided)."""
    eng = engine_factory(ctx.cfg, run, os.path.join(ctx.outdir, "runs", run.tag), ctx.url, False)
    started = ctx.rec.step(f"{run.tag}-start", lambda: (None, eng.start()))
    ready = started.ok and ctx.rec.step(f"{run.tag}-ready", lambda: (None, eng.wait_ready())).ok
    if ready:
        res = ctx.rec.step(f"{run.tag}-measure", lambda: _measure(ctx, eng, run))
        results.append(res.value if res.ok else _failed_start(run, ctx.cfg, eng, res.err))
    else:
        results.append(_failed_start(run, ctx.cfg, eng, "the engine did not start or become ready"))
    ctx.rec.step(f"{run.tag}-stop", lambda: (None, _stop(eng)))


def _write_outputs(cfg: dict, args: argparse.Namespace, rec: qual_report.Recorder, results: list[dict], kl_rows: list[dict],
                   an: pay_analyze.Analysis, verdicts: list[dict]) -> None:
    header = f"Second-GPU payoff, LIVE, {time.strftime('%Y-%m-%d %H:%M:%S')}; {len(results)} starts recorded"
    summary = qual_report.write_summary(args.out, header, verdicts, pay_render.render(an, cfg))
    for name, obj in (("results.json", results), ("analysis.json", {"kl": kl_rows, "analysis": dataclasses.asdict(an)})):
        with open(os.path.join(args.out, name), "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=1, default=str)
    with open(summary, encoding="utf-8") as fh:
        qual_report.write_html(args.out, "Second-GPU payoff chain", verdicts, rec.rows, fh.read())


def finish(cfg: dict, args: argparse.Namespace, rec: qual_report.Recorder, results: list[dict]) -> int:
    kl_rows = pay_gate.kl_gates(results, cfg)
    for r in kl_rows:
        status = {0: "PASS", 1: "FAIL"}.get(r["rc"], "WARN")
        rec.mark(f"kl-{r['kind']}-cap{r['cap']}-{r['arm']}-s{r['idx']}-{r['text']}", status, r["line"])
    an = pay_analyze.analyze(results, cfg, kl_rows)
    verdicts = pay_render.verdicts(an)
    for v in verdicts:
        rec.mark(f"verdict-{v['id']}", v["status"], f"{v['title']}: {v['detail']}")
    _write_outputs(cfg, args, rec, results, kl_rows, an, verdicts)
    bad_steps = [r for r in rec.rows if r["status"] == "FAIL" and not r["name"].startswith(("verdict-", "kl-"))]
    bad_kl = [r for r in kl_rows if r["rc"] == 1]
    print(f"steps {len(rec.rows)}, failed steps {len(bad_steps)}, failed quality gates {len(bad_kl)}; report {os.path.join(args.out, 'chain_report.html')}")
    return 1 if bad_steps or bad_kl else 0


def main(argv: list[str], engine_factory=qual_engine.Engine) -> int:
    args = parse_args(argv)
    try:
        cfg, runs = select(args)
    except (qual_cfg.ConfigError, ValueError) as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    print_plan(cfg, runs)
    if args.dry_run:
        return 0
    server = args.server or f"http://127.0.0.1:{cfg['vars']['port']}"
    url = server if "/v1/" in server else server.rstrip("/") + "/v1/chat/completions"
    os.makedirs(args.out, exist_ok=True)
    ctx = qual_work.Ctx(cfg, url, qual_report.Recorder(args.out), False, args.out)
    results: list[dict] = []
    for run in runs:
        one_start(ctx, run, results, engine_factory)
    return finish(cfg, args, ctx.rec, results)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
