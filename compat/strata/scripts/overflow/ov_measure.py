# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""What one overflow speed start measures: payoff's matched request stream, each request joined to the disk, process and cgroup counters.

OWNER: this module owns the per-start measurement record of the overflow test.  The request loop and the engine-log joins are
payoff's (pay_measure.measure_speed); the counter reads are ov_proc.py's; judging is ov_gate.py's.  This module only adds the
witnesses around the loop and the facts the gate needs (startup lines, cgroup readback, page-cache preflight, disk identity).
CONTRACT: ``environment(cfg) -> Env`` (live only: resolves the model's disk); ``prepare(cfg, env, plan) -> preflight dict | None``
(evict + measure the page cache before the engine starts; None for an uncapped arm); ``measure_speed_start(ctx, eng, run, env, pre)
-> start record``.  A request record gains ``dev_read_mb``, ``proc_read_mb`` and ``majflt``: the counter deltas across that request.
FAILURE: an unreadable counter is None and ov_gate turns it into a named INVALID reason; a store/request count mismatch (the pairing
is by order, see ov_engine.sampled_sends) sets every delta to None rather than guessing.
CHECKER: compat/strata/tests/overflow/test_overflow_run.py (fake engine, planted procfs).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import ov_cache
import ov_engine
import ov_paths  # noqa: F401
import ov_parse
import ov_proc
import pay_measure
import qual_cfg


@dataclass
class Env:
    host: ov_proc.Host
    disk: ov_proc.Disk | None
    files: list[str]
    own_cg: str | None


def environment(cfg: dict, host: ov_proc.Host | None = None) -> Env:
    """Resolve what a live run needs from the machine: the model files and the disk they live on (``--var disk`` overrides the disk).  ConfigError if the model is not there."""
    host = host or ov_proc.Host()
    v = cfg["vars"]
    try:
        st = os.stat(v["native"])
    except OSError as exc:
        raise qual_cfg.ConfigError(f"var native does not name a readable file ({v['native']!r}): {exc}") from exc
    disk = host.disk_by_name(v["disk"]) if v["disk"] != "auto" else host.disk_for_dev(os.major(st.st_dev), os.minor(st.st_dev))
    return Env(host, disk, ov_cache.model_files(v["native"], v["pack"]), host.cgroup_of(os.getpid()))


def prepare(cfg: dict, env: Env, plan) -> dict | None:
    """Before a capped start: evict the model files from the page cache and measure what is left (None for the uncapped reference)."""
    return ov_cache.preflight(env.files, cfg["gates"]["max_resident_mb"]) if plan.capped else None


def _mb(after: int | None, before: int | None) -> float | None:
    return None if after is None or before is None else max(0, after - before) / 1e6


def _deltas(b: ov_proc.Snap | None, a: ov_proc.Snap | None) -> dict:
    """The counter deltas of one request from its two snapshots; None where either snapshot or either reading is missing."""
    if a is None or b is None:
        return {"dev_read_mb": None, "proc_read_mb": None, "majflt": None}
    flt = None if a.majflt is None or b.majflt is None else a.majflt - b.majflt
    return {"dev_read_mb": _mb(a.dev_bytes, b.dev_bytes), "proc_read_mb": _mb(a.proc_bytes, b.proc_bytes), "majflt": flt}


def _attach(requests: list[dict], store: list) -> str | None:
    """Give every request its counter deltas, or None everywhere if the pairing by order does not hold (returns the note)."""
    ok = len(store) == len(requests)
    for i, r in enumerate(requests):
        r.update(_deltas(*store[i]) if ok else _deltas(None, None))
    return None if ok else f"counter snapshots ({len(store)}) and requests ({len(requests)}) differ in number: every delta is unknown"


def _memory(first: ov_proc.Snap, last: ov_proc.Snap) -> dict:
    return {"oom_kill": last.oom_kill, "max_hits": last.max_hits, "peak": last.peak,
            "dev_mb": _mb(last.dev_bytes, first.dev_bytes), "proc_mb": _mb(last.proc_bytes, first.proc_bytes),
            "majflt": (last.majflt - first.majflt) if last.majflt is not None and first.majflt is not None else None}


def measure_speed_start(ctx, eng, run, env: Env, pre: dict | None) -> dict:
    """The matched request stream of one speed start with the witnesses around every request.  The engine must be ready."""
    probe = ov_proc.Probe(env.host, env.disk.name if env.disk else "", eng.pids)
    store: list = []
    first = probe.snap()
    with ov_engine.sampled_sends(probe, store):
        rec = pay_measure.measure_speed(ctx, eng, run, ctx.cfg)
    last = probe.snap()
    plan = ctx.cfg["plans"][run.arm]
    text = eng.log_text()
    rec["startup"] = vars(ov_parse.startup(text))
    rec["moves"], rec["stale"] = ov_parse.moves(text), ov_parse.stale_count(text)
    rec["cap"] = plan.card_blobs or 0
    rec["sampler_note"] = _attach(rec["requests"], store)
    rec.update({"preflight": pre, "disk": vars(env.disk) if env.disk else None, "cgroup": env.host.cgroup_report(eng.pids()), "memory": _memory(first, last)})
    return rec
