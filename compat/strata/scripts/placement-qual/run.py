#!/usr/bin/env python3
# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Live-qualification runner (S6) for Seabraid placement on Strata: runs DESIGN.md section 5 and records the chain.

OWNER: this file is only the CLI and the per-start loop; config and plan are qual_cfg.py, engine lifetime qual_engine.py,
workloads qual_work.py, statistics qual_stats.py, criteria qual_judge.py, KL pairing qual_kl.py, report qual_report.py.
CONTRACT: ``run.py --arms P0,P2 --out DIR [--dry-run] [--server URL] [--config FILE] [--workloads W1,W2] [--var k=v]``.
Writes DIR/chain.tsv (name, start epoch, seconds, rc, status, note), SUMMARY.txt, chain_report.html, results.json,
steps/NNNN-*.txt and one engine.log per start.  Exit 0 = every step ran and every decidable criterion passed
(dry run: every step ran), 1 = a step or criterion failed, 2 = bad config or arguments (nothing was started).
Needs an exclusive window: it starts and stops its OWN engine processes by exact pgid (never a service manager).
--dry-run starts no engine: it prints the planned step list, sends a capped number of starts (--dry-n, default 1) to
--server (a stub), and still writes the timings file and the report; the criteria are advisory there.
FAILURE: fail-open-loud on a step (FAIL row, stderr line, next run), fail-fast on config.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qual_cfg  # noqa: E402
import qual_decode  # noqa: E402
import qual_engine  # noqa: E402
import qual_judge  # noqa: E402
import qual_separate  # noqa: E402
import qual_kl  # noqa: E402
import qual_report  # noqa: E402
import qual_stats as st  # noqa: E402
import qual_work  # noqa: E402

DEFAULT_SERVER = "http://127.0.0.1:8794"


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run the section 5 placement qualification and record the chain.")
    ap.add_argument("--arms", default="", help="comma list of arms (default: all arms of the selected workloads)")
    ap.add_argument("--workloads", default="", help="comma list, e.g. W1,W2 (default: all)")
    ap.add_argument("--out", required=True, help="output directory (created)")
    ap.add_argument("--dry-run", action="store_true", help="plan only: no engine is started; requests go to --server")
    ap.add_argument("--persona", default="", help="select the workloads of an opt-in persona (e.g. file-moves: FR-A1..A3 = W8..W10); never part of the default plan")
    ap.add_argument("--n", default="", help="per-workload start counts, e.g. W1=4,W3=2 (the protocol's full n is the default)")
    ap.add_argument("--rejudge", default="", help="comma list of finished run dirs: judge their results.json together into --out (starts nothing)")
    ap.add_argument("--dry-n", type=int, default=1, help="starts per workload and arm sent in a dry run (default 1)")
    ap.add_argument("--server", default=DEFAULT_SERVER, help="engine server base URL (loopback or the given LAN host)")
    ap.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "arms.json"))
    ap.add_argument("--var", action="append", default=[], metavar="K=V", help="override a config var (engine, helper, shm, port)")
    return ap.parse_args(argv)


def chat_url(server: str) -> str:
    return server if "/v1/" in server else server.rstrip("/") + "/v1/chat/completions"


def step_names(cfg: dict, run: qual_cfg.Run) -> list[str]:
    wl = cfg["workloads"][run.workload]
    if wl["kind"] == "forced":
        return [f"{run.tag}-capture"]  # one external capture command: it starts and stops its own engine
    names = [f"{run.tag}-start", f"{run.tag}-ready"]
    names += [f"{run.tag}-{ph['name']}-r{k + 1}" for ph in wl["phases"] for k in range(ph["count"])]
    names += [f"{run.tag}-kl-{t}" for t in wl.get("texts", [])]
    return names + [f"{run.tag}-stop"]


def print_plan(cfg: dict, runs: list[qual_cfg.Run]) -> int:
    total = 0
    for i, run in enumerate(runs, 1):
        names = step_names(cfg, run)
        total += len(names)
        print(f"PLAN {i:3d} {run.workload} {run.key:<8} start {run.idx}: {len(names)} steps: {names[0]} .. {names[-1]}")
    print(f"PLAN total: {len(runs)} engine starts, {total} recorded steps (+ kl gates and criteria rows)")
    return total


def _bring_up(ctx: qual_work.Ctx, run: qual_cfg.Run, attempt: int) -> tuple[qual_engine.Engine, str, bool]:
    """One start attempt: (engine, step tag, ready).  Attempt 2 gets its own run dir and step names, so the failed first one stays in the record."""
    tag = run.tag if attempt == 1 else f"{run.tag}-a{attempt}"
    eng = qual_engine.Engine(ctx.cfg, run, os.path.join(ctx.outdir, "runs", tag), ctx.url, ctx.dry)
    started = ctx.rec.step(f"{tag}-start", lambda: (None, eng.start()))
    return eng, tag, started.ok and ctx.rec.step(f"{tag}-ready", lambda: (None, eng.wait_ready())).ok


def one_run(ctx: qual_work.Ctx, run: qual_cfg.Run, results: list[dict]) -> None:
    """Start, wait ready, workload, stop: each a recorded step.  A start that fails is stopped and tried ONCE more (a GPU
    hiccup such as the #267 verify timeout must not starve an arm; the failed attempt stays in the chain as a FAIL row).
    A forced-decode start (W7) is one capture step and is not retried: it is long, and a failed one is a FAIL row and no evidence."""
    if ctx.cfg["workloads"][run.workload]["kind"] == "forced":
        done = ctx.rec.step(f"{run.tag}-capture", lambda: qual_decode.capture(ctx, run))
        if done.ok:
            results.append(done.value)
        return
    for attempt in (1, 2):
        eng, tag, ready = _bring_up(ctx, run, attempt)
        if ready:
            wres = ctx.rec.step(f"{tag}-workload", lambda: _workload(ctx, eng, run))
            if wres.ok:
                results.append(wres.value)
        ctx.rec.step(f"{tag}-stop", lambda: (None, _stop(eng)))
        if ready:
            return


def _workload(ctx, eng, run) -> tuple[dict, str]:
    res = qual_work.run_workload(ctx, eng, run)
    return res, f"{len(res['rounds'])} round lines, ok={res['ok']}"


def _stop(eng) -> str:
    left = eng.stop()
    if left:
        raise qual_engine.EngineError(f"{left} process(es) of the engine group survived the stop")
    return "engine group gone"


def _w1_lines(results: list[dict]) -> list[str]:
    out = ["W1 decode tok/s (mean of requests 2-3 per start), per arm: n, mean, min, max"]
    for key in sorted({r["key"] for r in results if r["workload"] == "W1"}):
        xs = [x for x in (qual_judge._decode(r) for r in qual_judge._runs(results, "W1", key)) if x is not None]
        out.append(f"  {key:<8} n={len(xs)} mean={st.mean(xs) or 0:.2f} min={min(xs, default=0):.2f} max={max(xs, default=0):.2f}")
    return out


def _latency_lines(results: list[dict]) -> list[str]:
    out = ["Swap latency over all round lines (us): arm field p50 p95 max n"]
    fields = ("plan_us", "fence_us", "stage_us", "replace_us", "publish_us", "follow_us")
    for key in sorted({r["key"] for r in results if r.get("rounds")}):
        table = st.latency_table([rd for r in results if r["key"] == key for rd in r["rounds"]], fields)
        out += [f"  {key:<8} {f:<11} {v['p50']:.0f} {v['p95']:.0f} {v['max']:.0f} n={v['n']}" for f, v in table.items() if v["n"]]
    return out


def tables(results: list[dict]) -> str:
    """Plain-text numbers for SUMMARY.txt: W1 decode per arm and swap latency per arm."""
    return "\n".join(_w1_lines(results) + _latency_lines(results))


def finish(cfg: dict, args: argparse.Namespace, rec, results: list[dict], planned: int) -> int:
    cfg["_failed_starts"] = [r["name"] for r in rec.rows if r["status"] == "FAIL" and r["name"].endswith(("-start", "-ready"))]
    kl = qual_kl.compare(cfg, results, rec) + qual_decode.compare(cfg, results, rec)
    verdicts = qual_separate.judge_all(cfg, results, kl)
    for v in verdicts:
        rec.mark(f"criterion-{v['id']}", v["status"], f"{v['title']}: {v['detail']}")
    mode = "DRY RUN (criteria advisory)" if args.dry_run else "LIVE"
    header = f"Seabraid placement qualification, {mode}, {time.strftime('%Y-%m-%d %H:%M:%S')}\nserver {args.server}; {len(results)} starts with results; {planned} steps planned"
    summary = qual_report.write_summary(args.out, header, verdicts, tables(results))
    with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1, default=str)
    with open(summary, encoding="utf-8") as fh:
        qual_report.write_html(args.out, "Placement qualification chain", verdicts, rec.rows, fh.read())
    return _report_and_exit(rec, verdicts, args)


def _report_and_exit(rec, verdicts: list[dict], args: argparse.Namespace) -> int:
    bad_steps = [r for r in rec.rows if r["status"] == "FAIL" and not r["name"].startswith("criterion-")]
    bad_crit = [v for v in verdicts if v["status"] == "FAIL"]
    print(f"steps {len(rec.rows)}, failed steps {len(bad_steps)}, failed criteria {len(bad_crit)}; report {os.path.join(args.out, 'chain_report.html')}")
    return 1 if bad_steps or (bad_crit and not args.dry_run) else 0


def _validate(cfg: dict, arms: list[str] | None, wls: list[str]) -> None:
    unknown = [a for a in (arms or []) if a not in cfg["arms"]] + [w for w in wls if w not in cfg["workloads"]]
    if unknown:
        raise qual_cfg.ConfigError(f"unknown arm or workload: {unknown}")


def _csv(text: str) -> list[str]:
    return [x for x in text.split(",") if x]


def _need_runs(runs: list) -> None:
    if not runs:
        raise qual_cfg.ConfigError("the selection leaves nothing to run")


def _selected(workloads: dict, wls: list[str], persona: str) -> dict:
    """The workloads to run.  An explicit --workloads list wins; otherwise --persona selects that persona's own workloads, and the
    default plan leaves every persona out (opt-in evidence must not lengthen an ordinary window)."""
    personas = {v["persona"] for v in workloads.values() if v.get("persona")}
    if persona and persona not in personas:
        raise qual_cfg.ConfigError(f"unknown persona {persona!r}: have {sorted(personas)}")
    return {k: v for k, v in workloads.items() if (k in wls if wls else v.get("persona", "") == persona)}


def select(args: argparse.Namespace) -> tuple[dict, list[qual_cfg.Run], list[qual_cfg.Run]]:
    """(cfg, full plan, runs to execute) or ConfigError: the whole selection is validated before anything starts."""
    cfg = qual_cfg.load(args.config, dict(kv.split("=", 1) for kv in args.var))
    wls, arms = _csv(args.workloads), _csv(args.arms) or None
    _validate(cfg, arms, wls)
    cfg["workloads"] = _selected(cfg["workloads"], wls, args.persona)
    for kv in _csv(args.n):
        name, _, n = kv.partition("=")
        if name not in cfg["workloads"] or not n.isdigit() or int(n) < 1:
            raise qual_cfg.ConfigError(f"--n {kv!r}: expected WORKLOAD=N for a selected workload")
        cfg["workloads"][name]["n"] = int(n)
    runs = qual_cfg.plan(cfg, arms, args.dry_n if args.dry_run else None)
    _need_runs(runs)
    return cfg, qual_cfg.plan(cfg, arms), runs


def rejudge(args: argparse.Namespace) -> int:
    """Combine finished invocations: their results.json and chain rows become one verdict set and one report in --out."""
    cfg = qual_cfg.load(args.config, dict(kv.split("=", 1) for kv in args.var))
    os.makedirs(args.out, exist_ok=True)
    rec, results = qual_report.Recorder(args.out), []
    for d in _csv(args.rejudge):
        with open(os.path.join(d, "results.json"), encoding="utf-8") as fh:
            results += json.load(fh)
        rec.import_rows(os.path.join(d, "chain.tsv"), os.path.basename(d.rstrip("/")))
    have = {r["workload"] for r in results}   # an opt-in persona that no combined run executed is not judged (F1 would WARN on every ordinary rejudge)
    cfg["workloads"] = {k: v for k, v in cfg["workloads"].items() if not v.get("persona") or k in have}
    return finish(cfg, args, rec, results, 0)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.rejudge:
        return rejudge(args)
    try:
        cfg, full, runs = select(args)
    except (qual_cfg.ConfigError, ValueError) as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    planned = print_plan(cfg, full)
    os.makedirs(args.out, exist_ok=True)
    ctx = qual_work.Ctx(cfg, chat_url(args.server), qual_report.Recorder(args.out), args.dry_run, args.out)
    results: list[dict] = []
    for run in runs:
        one_run(ctx, run, results)
    return finish(cfg, args, ctx.rec, results, planned)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
