# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""W7 decode-path capture of one arm, run inside the arm's memory-limited scope, with the disk witnessed over the whole capture.

OWNER: this module owns "run one force_check forced-decode capture under the overflow conditions".  The capture tool, the forced set
and the row pooling are placement-qual's (qual_decode.pool, the same ``score --path decode`` command); the KL arithmetic is kl.py's.
WHY decode path: a prefill teacher-forcing (W4) never runs the helper tier or the decode-time file tier (placement-qual's own finding,
qual_decode docstring); only forced decode windows exercise the path an overflowed model uses.
WHY not qual_decode.capture: it hides the capture's pid, and the cgroup readback and the counters need it; the command, the pooling
and the output files are the same.
CONTRACT: ``forced_cfg_problems(path, plan) -> [reason]`` (the arm's engine config must carry exactly the capacity flags the plan prints,
checked BEFORE the window is spent); ``capture(ctx, run, env, pre) -> record`` (kl_files, cgroup readback, memory counters, disk MB).
The record has no per-request engine log, so its proof the SSD was read is the device counter over the capture (ov_gate.w7_verdict).
SAFETY: a capped capture's stragglers are killed through ITS OWN scope's cgroup.kill (ov_proc.Host.cg_kill refuses any other cgroup);
the uncapped reference's capture group is killed by process group, like qual_decode.
FAILURE: raises on a refused config; any other failure comes back in ``record['error']`` so the arm is INVALID and the run goes on.
CHECKER: compat/strata/tests/overflow/test_overflow_run.py (forced-cfg check, fake capture).
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time

import ov_cfg
import ov_paths  # noqa: F401
import qual_decode

POLL_S = 2.0


def _num(x) -> object:
    try:
        return float(x)
    except (TypeError, ValueError):
        return x


def _flags(args: list) -> dict[str, str]:
    return {a: str(args[i + 1]) for i, a in enumerate(args[:-1]) if isinstance(a, str) and a.startswith("--")}


def _wanted(plan) -> dict:
    """The flags and values an arm's engine config must carry: the capacity flags, and for a ladder arm the installer's ladder flags."""
    want = {"--expert-cache": plan.main_slots, "--resident-budget-gib": plan.ram_gib}
    want.update({f"--expert-cache-device{i}": n for i, n in enumerate(plan.helper_slots, 1)})
    if plan.ladder:
        want.update(_flags(ov_cfg.ladder_flags(plan.helpers)[0]))
    return want


def _ladder_problems(got: dict, env: dict, plan) -> list[str]:
    """A ladder arm's config carries the ladder env too; every other arm carries no ladder flag or env (else the arms are not matched)."""
    flags, lenv = ov_cfg.ladder_flags(max(plan.helpers, 1))
    if plan.ladder:
        return [f"engine config env {k} is {env.get(k)!r}, the ladder needs {v!r}" for k, v in lenv.items() if str(env.get(k)) != v]
    stray = [f for f in got if f.startswith("--retier")] + [k for k in lenv if k in env]
    return [f"engine config has {x} but the arm has no ladder" for x in stray]


def forced_cfg_problems(path: str, plan) -> list[str]:
    """Compare the flags and env in the arm's engine config JSON with the plan: any difference makes the arms unmatched."""
    try:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        got, env = _flags(doc["args"]), doc.get("env", {})
    except (OSError, ValueError, KeyError) as exc:
        return [f"cannot read the engine config {path}: {exc}"]
    if not plan.capped:
        return ["the reference's engine config carries --resident-budget-gib: it must be uncapped"] if "--resident-budget-gib" in got else []
    want = _wanted(plan)
    out = [f"engine config {f} is {got.get(f)!r}, the plan needs {v!r}" for f, v in want.items() if _num(got.get(f)) != _num(v)]
    extra = [f for f in got if f.startswith("--expert-cache-device") and f not in want]
    return out + [f"engine config has {f} but the arm has {plan.helpers} helper(s)" for f in extra] + _ladder_problems(got, env, plan)


def _sample(host, pid: int, seen: dict) -> None:
    """One poll of a live capture: its cgroup, the processes in it, the limit readback and the running maxima of the counters."""
    cg = host.cgroup_of(pid)
    if not cg:
        return
    seen["cg"] = cg
    pids = host.cg_pids(cg)
    if pids:
        seen["cgroup"] = host.cgroup_report(pids)
    _majflt, oom, hits, peak = host.cg_counters({cg})
    seen["oom_kill"] = max(seen["oom_kill"], oom or 0)
    seen["max_hits"] = max(seen["max_hits"], hits or 0)
    seen["peak"] = max(seen["peak"] or 0, peak or 0) or None


def _stop(proc: subprocess.Popen, host, env, seen: dict, capped: bool) -> str | None:
    """Kill the capture's whole tree: through its scope when capped (an engine may have left the process group), else by process group."""
    if capped and seen.get("cg"):
        return host.cg_kill(seen["cg"], env.own_cg) if os.path.isdir(f"{host.cgroot}{seen['cg']}") else None
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGKILL)
    return None


def _poll(proc: subprocess.Popen, host, capped: bool, timeout: float, seen: dict) -> str | None:
    """Poll a live capture until it exits or the timeout passes; returns the timeout error text, or None."""
    t0 = time.time()
    while proc.poll() is None and time.time() - t0 < timeout:
        if capped:
            _sample(host, proc.pid, seen)
        time.sleep(POLL_S)
    return f"capture did not finish in {timeout:.0f} s: killed" if proc.poll() is None else None


def _run(argv: list[str], env, capped: bool, timeout: float, logpath: str) -> dict:
    seen: dict = {"cg": None, "cgroup": [], "oom_kill": 0, "max_hits": 0, "peak": None}
    host, disk = env.host, (env.disk.name if env.disk else None)
    d0 = host.disk_read_bytes(disk) if disk else None
    child_env = {**os.environ, "XDG_RUNTIME_DIR": os.environ.get("XDG_RUNTIME_DIR") or ov_cfg.default_xdg()}
    with open(logpath, "wb") as log:
        proc = subprocess.Popen(argv, env=child_env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        err = _poll(proc, host, capped, timeout, seen)
        kill_err = _stop(proc, host, env, seen, capped)
        proc.wait(timeout=60)
    d1 = host.disk_read_bytes(disk) if disk else None
    return {"rc": proc.returncode, "seen": seen, "error": err or (f"capture exit {proc.returncode}" if proc.returncode else None),
            "dev_read_mb": None if d0 is None or d1 is None else max(0, d1 - d0) / 1e6, "kill_note": kill_err}


def capture(ctx, run, env, pre: dict | None) -> dict:
    """One forced-decode capture.  The arm's engine config is checked first; the rest is recorded, not raised."""
    cfg, plan = ctx.cfg, ctx.cfg["plans"][run.arm]
    bad = forced_cfg_problems(cfg["vars"][f"forced_cfg_{run.arm}"], plan)
    if bad:
        raise RuntimeError("; ".join(bad))
    rundir = os.path.join(ctx.outdir, "runs", run.tag)
    os.makedirs(rundir, exist_ok=True)
    res = _run(ov_cfg.capture_command(cfg, run, rundir), env, plan.capped, cfg["w7"]["timeout_s"], os.path.join(rundir, "capture.log"))
    rec = {"workload": run.workload, "cap": plan.card_blobs or 0, "mode": "forced", "arm": run.arm, "idx": run.idx, "tag": run.tag, "startup": None,
           "requests": [], "aborted": False, "error": res["error"], "kl_files": {}, "preflight": pre, "disk": vars(env.disk) if env.disk else None,
           "cgroup": res["seen"]["cgroup"], "memory": {k: res["seen"][k] for k in ("oom_kill", "max_hits", "peak")}, "dev_read_mb": res["dev_read_mb"],
           "kill_note": res["kill_note"]}
    if not rec["error"]:
        out = os.path.join(rundir, "pooled.tsv")
        rows = qual_decode.pool(os.path.join(rundir, "forced.tsv"), out)
        rec["kl_files"] = {qual_decode.KEY: {"path": out, "rows": rows, "ok": True}}
    return rec

