#!/usr/bin/env python3
# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Capped-overflow test: run a model whose experts do not fit the cards plus a small RAM budget, so the rest genuinely live on the SSD.

OWNER: this file is only the CLI and the per-start loop.  Capacity arithmetic ov_cap.py, config and plan ov_cfg.py, process launch
ov_engine.py, measurement ov_measure.py / ov_capture.py, witnesses ov_proc.py, page cache ov_cache.py, the data gate ov_gate.py, text
ov_report.py; engine readiness, the recorded chain and the HTTP client are placement-qual's, the request loop is payoff's.
CONTRACT: ``overflow.py --out DIR [--dry-run] [--arms R,O1,O2] [--kl on|off] [--ref-repeats N] [--server URL] [--config FILE]
[--var k=v]`` and ``overflow.py --probe-cgroup``.  Live: DIR/chain.tsv, steps/, SUMMARY.txt, chain_report.html, results.json,
analysis.json, runs/<tag>/.  Exit 0 = every arm PASSed (or is UNGATED with --kl off) and no step failed; 1 = an arm is INVALID or FAIL, or
a step failed; 2 = bad config or arguments (nothing was started).
--dry-run starts NOTHING, sends nothing, reads no model file and creates no file: it prints the capacity arithmetic, the memory-limit
mechanism, every command with its env, the data gate and the GPU-window ESTIMATE.
--probe-cgroup starts one transient scope with a 64 MiB limit for one `sh` and reads memory.max and memory.swap.max back: the check that
the box really delegates the memory controller to a user scope.  It touches no GPU, no engine and no file.
A live run needs an exclusive GPU window (README section "How to run it in a GPU window"); it starts and stops only its OWN processes.
FAILURE: fail-open-loud on a step (FAIL row, stderr line, next start); fail-fast on config.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time

import ov_capture
import ov_cfg
import ov_engine
import ov_gate
import ov_measure
import ov_parse
import ov_paths
import ov_report
import qual_cfg
import qual_engine
import qual_report
import qual_work

MECHANISM = [
    "MEMORY LIMIT: every capped engine, helper and W7 capture starts under `systemd-run --user --scope -p MemoryMax=<bytes> -p MemorySwapMax=0`.",
    "  A cgroup v2 limit charges page cache too, so the OS cannot serve more than (limit - the engine's own memory) of 'SSD' reads from RAM.",
    "  It is proven, not assumed: memory.max and memory.swap.max are read back from the cgroup of the engine's own pid and compared with the plan.",
    "PAGE CACHE: before every capped start the model files are evicted (posix_fadvise DONTNEED, no privilege) and `fincore` must show them nearly absent.",
    "  WHY: a cache page is charged to whoever first read it; pages an earlier start left behind are free to this one and would serve its reads.",
    "DATA GATE (INVALID never passes): memory.max readback equal to the plan | the engine's file-tier bytes > 0 on every counted decode request |",
    "  the model's disk counter AND the engine's own read_bytes both rise by at least min_device_frac x those bytes | the engine's slot count, RAM",
    "  budget and experts-left-for-files match the plan | no OOM kill | W7 decode-path KL against the uncapped reference.",
]


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Capped-overflow test: experts that fit nowhere else genuinely live on the SSD.")
    ap.add_argument("--out", default="", help="output directory (created by a live run; a dry run creates nothing)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, the arithmetic, every command and the window ESTIMATE; start and send nothing")
    ap.add_argument("--probe-cgroup", action="store_true", help="check that a user scope with a memory limit works on this machine, then exit")
    ap.add_argument("--arms", default="", help="comma list of arms, replacing sweep.arms (R is the reference; O3L is the live-ladder arm, off by default)")
    ap.add_argument("--kl", choices=("on", "off"), default="", help="the W7 decode-path quality captures (default: sweep.kl); off = UNGATED")
    ap.add_argument("--ref-repeats", type=int, default=0, help="W7 captures of the reference (2 = the second one is the noise floor)")
    ap.add_argument("--server", default="", help="engine server base URL (default http://127.0.0.1:<port var>)")
    ap.add_argument("--config", default=os.path.join(ov_paths.HERE, "arms.json"))
    ap.add_argument("--var", action="append", default=[], metavar="K=V", help="override a config var (gpu, rocm, pack, native, launcher, helper, forced_cfg_<arm>, ladder_extra_args, ...)")
    return ap.parse_args(argv)


def sweep_overrides(args: argparse.Namespace) -> dict:
    out: dict = {}
    if args.arms:
        out["arms"] = [a for a in args.arms.split(",") if a]
    if args.kl:
        out["kl"] = args.kl == "on"
    if args.ref_repeats:
        out["ref_repeats"] = args.ref_repeats
    return out


def select(args: argparse.Namespace) -> tuple[dict, list[qual_cfg.Run]]:
    """(cfg, ordered starts) or ConfigError.  Every command is built up front so a missing var fails before anything starts."""
    cfg = ov_cfg.load(args.config, dict(kv.split("=", 1) for kv in args.var), args.dry_run, sweep_overrides(args))
    runs = ov_cfg.plan(cfg)
    for run in runs:
        ov_cfg.describe(cfg, run)
    return cfg, runs


def print_plan(cfg: dict, runs: list[qual_cfg.Run]) -> None:
    sw = cfg["sweep"]
    print(f"PLAN {len(runs)} starts, arms {sw['arms']} (reference {sw['reference_arm']}); each speed start: {sw['n_requests'] + sw['warmup']} rounds of one decode "
          f"({sw['decode_tokens']} tokens, greedy) and one prefill (~{sw['prefill_tokens']} tokens, distinct first line, max_tokens 1); the first {sw['warmup']} round(s) are not counted")
    print("CAPACITY (arithmetic; every input is a config ESTIMATE the engine's own startup lines are checked against):")
    for arm in sw["arms"]:
        print("  " + "\n  ".join(cfg["plans"][arm].lines))
    print("\n".join(MECHANISM))
    flags, lenv = ov_cfg.ladder_flags(2)
    print(f"LADDER (arm O3L, imported from install/_ladder.py): {' '.join(flags)} + env {lenv}; extra = {cfg['vars']['ladder_extra_args'] or '(none)'}; "
          f"unbuffered = {cfg['vars']['unbuffered'] or '(engine decides)'}")
    print("  O3L data gate also needs: a 'placement move' from the fallback rung onto a helper rung, a hop onto each helper rung, and zero stale-placement records.")
    for i, run in enumerate(runs, 1):
        print(f"START {i:2d} {run.tag} [{run.workload}] arm {run.arm}: {cfg['arms'][run.arm].get('label', '')}")
        print("\n".join(ov_cfg.describe(cfg, run)))
    lo, hi = ov_cfg.estimate(cfg, runs)
    print(f"ESTIMATE (not measured; config estimate block): GPU window {lo / 3600:.1f} h to {hi / 3600:.1f} h for {len(runs)} starts")


def probe_cgroup() -> int:
    """One scope, 64 MiB, one sh: memory.max and memory.swap.max read back from the scope's own cgroup.  0 = works, 1 = does not."""
    limit = 64 << 20
    script = 'p=/sys/fs/cgroup$(sed -n "s/^0:://p" /proc/self/cgroup); echo "$p"; cat "$p/memory.max" "$p/memory.swap.max"'
    env = {**os.environ, "XDG_RUNTIME_DIR": ov_cfg.default_xdg()}
    try:
        p = subprocess.run(ov_cfg.scope_prefix(limit) + ["sh", "-c", script], capture_output=True, text=True, timeout=60, env=env)
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"PROBE FAIL: cannot run systemd-run: {exc}")
        return 1
    lines = p.stdout.split()
    ok = p.returncode == 0 and lines[-2:] == [str(limit), "0"]
    print(f"PROBE {'OK' if ok else 'FAIL'}: scope {lines[0] if lines else '?'} memory.max/memory.swap.max read back {lines[-2:]} (wanted [{limit!r}, '0']); rc {p.returncode} {p.stderr.strip()[:200]}")
    return 0 if ok else 1


def _stop(eng) -> str:
    left = eng.stop()
    if left:
        raise qual_engine.EngineError(f"{left} process(es) of the engine group survived the stop")
    return "engine group gone"


def failed_record(run: qual_cfg.Run, cfg: dict, error: str, pre: dict | None = None) -> dict:
    """A start that did not produce a measurement, kept so its arm is INVALID and says why."""
    return {"workload": run.workload, "cap": 0, "mode": "speed" if run.workload == "speed" else "forced", "arm": run.arm, "idx": run.idx, "tag": run.tag,
            "startup": vars(ov_parse.startup("")), "requests": [], "kl_files": {}, "aborted": False, "error": error, "preflight": pre}


def _preflight(ctx: qual_work.Ctx, env, run: qual_cfg.Run) -> dict | None:
    plan = ctx.cfg["plans"][run.arm]
    res = ctx.rec.step(f"{run.tag}-preflight", lambda: _pre_step(ctx.cfg, env, plan))
    return res.value if res.ok else {"resident_mb": None, "max_resident_mb": ctx.cfg["gates"]["max_resident_mb"], "errors": [res.err], "ok": False, "files": 0}


def _pre_step(cfg: dict, env, plan) -> tuple[dict | None, str]:
    pre = ov_measure.prepare(cfg, env, plan)
    return pre, "uncapped reference: page cache left alone" if pre is None else f"{pre['files']} model files evicted; resident {pre['resident_mb']} MB; errors {pre['errors']}"


def speed_start(ctx: qual_work.Ctx, run: qual_cfg.Run, env, results: list[dict], factory) -> None:
    """Preflight, start, wait ready, measure, stop: each a recorded step.  A start that fails is kept as an errored record."""
    pre = _preflight(ctx, env, run)
    if pre and pre["resident_mb"] is not None and pre["resident_mb"] > ctx.cfg["gates"]["max_resident_mb"]:
        results.append(failed_record(run, ctx.cfg, f"{pre['resident_mb']:.0f} MB of the model files stayed in the page cache after eviction: not started", pre))
        return
    eng = factory(ctx.cfg, run, os.path.join(ctx.outdir, "runs", run.tag), ctx.url, False)
    started = ctx.rec.step(f"{run.tag}-start", lambda: (None, eng.start()))
    ready = started.ok and ctx.rec.step(f"{run.tag}-ready", lambda: (None, eng.wait_ready())).ok
    if ready:
        res = ctx.rec.step(f"{run.tag}-measure", lambda: _measured(ctx, eng, run, env, pre))
        results.append(res.value if res.ok else failed_record(run, ctx.cfg, res.err, pre))
    else:
        results.append(failed_record(run, ctx.cfg, "the engine did not start or become ready", pre))
    ctx.rec.step(f"{run.tag}-stop", lambda: (None, _stop(eng)))


def _measured(ctx, eng, run, env, pre) -> tuple[dict, str]:
    rec = ov_measure.measure_speed_start(ctx, eng, run, env, pre)
    return rec, f"{len(rec['requests'])} requests, aborted={rec['aborted']}, sampler: {rec['sampler_note'] or 'ok'}"


def w7_start(ctx: qual_work.Ctx, run: qual_cfg.Run, env, results: list[dict]) -> None:
    pre = _preflight(ctx, env, run)
    res = ctx.rec.step(f"{run.tag}-capture", lambda: _captured(ctx, run, env, pre))
    results.append(res.value if res.ok else failed_record(run, ctx.cfg, res.err, pre))
    if res.ok and res.value["error"]:  # the record is kept (cgroup readback, counters) and the failure is its own FAIL row
        ctx.rec.mark(f"{run.tag}-capture-error", "FAIL", res.value["error"])


def _captured(ctx, run, env, pre) -> tuple[dict, str]:
    rec = ov_capture.capture(ctx, run, env, pre)
    return rec, f"{rec['kl_files'].get('forced', {}).get('rows', 0)} pooled rows, disk {rec['dev_read_mb']} MB over the capture, error {rec['error']}"


def _write_outputs(args: argparse.Namespace, rec: qual_report.Recorder, results: list[dict], kl: list[dict], arms: list[ov_report.ArmReport], text: str, rows: list[dict]) -> None:
    header = f"Capped-overflow test, LIVE, {time.strftime('%Y-%m-%d %H:%M:%S')}; {len(results)} starts recorded"
    summary = qual_report.write_summary(args.out, header, rows, text)
    for name, obj in (("results.json", results), ("analysis.json", {"kl": kl, "arms": [{"arm": a.arm, "status": a.status, "kl": a.kl, "plan": dataclasses.asdict(a.plan)} for a in arms]})):
        with open(os.path.join(args.out, name), "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=1, default=str)
    with open(summary, encoding="utf-8") as fh:
        qual_report.write_html(args.out, "Capped-overflow chain", rows, rec.rows, fh.read())


def _kl_marks(rec: qual_report.Recorder, kl: list[dict]) -> None:
    for r in kl:
        rec.mark(f"kl-{r['kind']}-{r['arm']}-s{r['idx']}", {0: "PASS", 1: "FAIL"}.get(r["rc"], "WARN"), r["line"])


def finish(cfg: dict, args: argparse.Namespace, rec: qual_report.Recorder, results: list[dict]) -> int:
    kl = ov_gate.kl_rows(results, cfg) if cfg["sweep"]["kl"] else []
    _kl_marks(rec, kl)
    arms = ov_report.build(results, cfg, kl)
    rows = ov_report.verdicts(arms)
    for v in rows:
        rec.mark(f"verdict-{v['id']}", v["status"], f"{v['title']}: {v['detail']}")
    _write_outputs(args, rec, results, kl, arms, ov_report.render(arms, kl, cfg), rows)
    bad_steps = [r for r in rec.rows if r["status"] == "FAIL" and not r["name"].startswith(("verdict-", "kl-"))]
    bad_arms = [a.arm for a in arms if a.status in ("INVALID", "FAIL")]
    print(f"steps {len(rec.rows)}, failed steps {len(bad_steps)}, arms INVALID or FAIL {bad_arms}; report {os.path.join(args.out, 'chain_report.html')}")
    return 1 if bad_steps or bad_arms else 0


def main(argv: list[str], engine_factory=ov_engine.OvEngine, host=None) -> int:
    args = parse_args(argv)
    if args.probe_cgroup:
        return probe_cgroup()
    if not args.out:
        print("config error: --out is required", file=sys.stderr)
        return 2
    try:
        cfg, runs = select(args)
        env = None if args.dry_run else ov_measure.environment(cfg, host)
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
        if run.workload == "speed":
            speed_start(ctx, run, env, results, engine_factory)
        else:
            w7_start(ctx, run, env, results)
    return finish(cfg, args, ctx.rec, results)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
