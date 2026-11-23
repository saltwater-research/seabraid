# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""The data gate: did this start really run with its experts on the SSD, under the memory limit, as the arm it claims?  INVALID never passes.

OWNER: this module owns "may this start's numbers count".  The witnesses are measured elsewhere (ov_proc.py, ov_measure.py); the
KL arithmetic is tools/kl.py's (pay_gate.run_kl_py).  Thresholds come from the ``gates`` block of arms.json.
CONTRACT: ``speed_verdict(start, plan, cfg) -> {valid, reasons, notes}``; ``w7_verdict(rec, plan, cfg) -> same``;
``kl_rows(results, cfg, run) -> rows``; ``arm_status(verdicts, kl, reference, needs_kl) -> PASS | FAIL | INVALID | UNGATED``.
WHY three independent witnesses of "the SSD was read": the engine's own file-tier byte counter says what it asked the file tier for,
not where the bytes came from; the device counter (machine-wide) and the engine's own /proc read_bytes (attributable) say what the
storage layer transferred.  A page-cache-served run shows engine bytes > 0 with device and process bytes near 0: that is refused
(``min_device_frac``).  A run whose file tier reads nothing is refused as well: nothing overflowed.
WHY the cgroup is read back and not trusted: the limit was requested through a launcher argv; only ``memory.max`` of the engine
process's own cgroup, read from /proc/<pid>/cgroup and the cgroup tree, shows it is in force.  An engine, helper or launcher process
that sits in a cgroup with no limit makes the whole start INVALID.
FAILURE: pure (no I/O besides run_kl_py's one subprocess); an absent witness is a named reason, never a pass.
CHECKER: compat/strata/tests/overflow/test_overflow_gate.py (planted page-cache run, uncapped cgroup, shrunk cap, wrong arithmetic).
"""

from __future__ import annotations

import os

import ov_paths  # noqa: F401
import pay_gate


def _decode(reqs: list[dict]) -> list[dict]:
    return [r for r in reqs if r.get("phase") == "decode" and r.get("counted") and r.get("usable")]


def _mean(xs: list) -> float | None:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _one_request(r: dict, g: dict) -> str | None:
    """The first reason a counted decode request is not proof that the SSD was read, or None."""
    f, d, p = r.get("file_mb"), r.get("dev_read_mb"), r.get("proc_read_mb")
    if f is None or f < g["min_file_mb"]:
        return f"the engine counted {f} MB of file-tier reads (< {g['min_file_mb']}): no expert came from the files"
    for name, got in (("disk", d), ("process", p)):
        if got is None:
            return f"no {name} read counter for this request"
        if got < g["min_device_frac"] * f:
            return (f"the engine's file tier read {f:.1f} MB but the {name} counter rose only {got:.1f} MB (< {g['min_device_frac']} x): "
                    "the reads were served from the page cache, not the SSD")
    return None


def nvme_problems(start: dict, g: dict) -> list[str]:
    """Per counted decode request, engine bytes against the device and process counters; one summarised reason when any request fails."""
    reqs = _decode(start["requests"])
    bad = [(i, why) for i, why in ((i, _one_request(r, g)) for i, r in enumerate(reqs)) if why]
    if not bad:
        return []
    return [f"NVMe proof failed on {len(bad)} of {len(reqs)} counted decode requests; first (request {bad[0][0]}): {bad[0][1]}"]


def _limit_problem(r: dict, plan, helper_comm: str) -> str | None:
    """The reason one process's cgroup is not the planned one: unlimited, a different byte limit, or swap allowed."""
    if not isinstance(r["memory_max"], int):
        return f"process {r['pid']} ({r['comm']}) runs in a cgroup with no memory limit ({r['cgroup']}, memory.max={r['memory_max']})"
    want = plan.helper_cap_bytes if r["comm"] == helper_comm else plan.engine_cap_bytes
    if r["memory_max"] != want or r["swap_max"] != 0:
        return f"process {r['pid']} ({r['comm']}): memory.max {r['memory_max']} / swap.max {r['swap_max']} but the plan is {want} / 0"
    return None


def cgroup_problems(start: dict, plan, cfg: dict) -> list[str]:
    """The limit read back from the cgroup tree: engine and helpers carry exactly the planned memory.max and no swap; nothing runs uncapped."""
    rows = start.get("cgroup") or []
    if not rows:
        return ["no cgroup readback for this start: the memory limit is not proven in force"]
    helper = os.path.basename(cfg["vars"]["helper"])[:15]
    out = [p for p in (_limit_problem(r, plan, helper) for r in rows) if p]
    comm = cfg["engine"]["engine_comm"]
    if not any(r["comm"] == comm for r in rows):
        out.append(f"no process named {comm!r} in the start's process group: the engine's own cgroup was not read")
    return out


def _budget_problems(s: dict, plan, g: dict) -> list[str]:
    out = []
    if s["ram_budget_gib"] is None:
        return ["no 'RAM budget' startup line: the RAM budget / file tier was not engaged"]
    if abs(s["ram_budget_gib"] - plan.ram_gib) > 0.01:
        out.append(f"the engine's RAM budget is {s['ram_budget_gib']} GiB, the plan {plan.ram_gib}")
    comp = s["complement_gib"]
    if abs(comp - plan.complement_gib) > g["complement_tol_frac"] * plan.complement_gib:
        out.append(f"the engine's experts outside the cards are {comp} GiB, the plan {plan.complement_gib:.2f} (> {100 * g['complement_tol_frac']:.0f}% apart): "
                   "capacity.experts_gib or the slot counts are wrong, so the arithmetic is not the engine's")
    slot = (s["cap_gib"] / s["cap_slots"]) if s["cap_slots"] and s["cap_gib"] else plan.complement_gib / max(1, plan.file_blobs)
    on_file = comp - (s["ram_held_blobs"] or 0) * slot
    if on_file < g["min_file_plan_frac"] * plan.file_gib:
        out.append(f"the engine leaves {on_file:.1f} GiB of experts on the files, less than {g['min_file_plan_frac']} x the planned {plan.file_gib:.1f} GiB")
    return out


def _channel_problems(s: dict, plan, expected: list[str]) -> list[str]:
    """The helper channels the engine log shows up against the arm's: one per helper process, none for a one-card arm."""
    got = list(s.get("channels") or [])
    if sorted(got) == sorted(expected[: plan.helpers]):
        return []
    return [f"the engine log shows helper channels {got} up, the arm needs {expected[: plan.helpers]}: a helper process did not join (or one too many did)"]


def startup_problems(start: dict, plan, g: dict, expected_channels: list[str]) -> list[str]:
    """The engine's own startup lines against the plan: slot count, helper channels, RAM budget, experts left for the files."""
    s = start["startup"]
    if s["cap_slots"] is None:
        return ["no 'expert cache N slots' line in the engine log: the cap was not confirmed"]
    if not plan.capped:
        return []
    out = [] if s["cap_slots"] == plan.main_slots else [f"the engine logged {s['cap_slots']} main slots, the plan {plan.main_slots}: the cap was changed"]
    return out + _channel_problems(s, plan, expected_channels) + _budget_problems(s, plan, g)


def ladder_problems(start: dict, plan) -> list[str]:
    """A ladder arm must show the pyramid moving: committed hops from the fallback rung onto a helper rung, a hop onto EACH helper rung, no stale epoch."""
    if not plan.ladder:
        return []
    mv = start.get("moves") or []
    if not mv:
        return ["no 'placement move' line in the engine log: the ladder never committed a hop (flags accepted is not the same as moving)"]
    out = [] if any(m["src"] == "fallback" and 1 <= m["upper"] <= plan.helpers for m in mv) else ["no move from the fallback rung (RAM/SSD) onto a helper rung"]
    out += [f"no hop onto helper rung {r}" for r in range(1, plan.helpers + 1) if not any(m["upper"] == r for m in mv)]
    return out + _stale_problem(start)


def _stale_problem(start: dict) -> list[str]:
    n = start.get("stale")
    return [f"{n} stale-placement record(s) in the engine log: a routing decision ran against an older table"] if n else []


def cache_problems(start: dict, plan, g: dict) -> tuple[list[str], list[str]]:
    """(reasons, notes) of the page-cache preflight: model files still resident above the limit are refused; an unmeasurable cache is a note."""
    pre, notes = start.get("preflight"), []
    if not plan.capped:
        return [], notes
    if pre is None:
        return ["no page-cache preflight record for this start"], notes
    if pre["resident_mb"] is not None and pre["resident_mb"] > g["max_resident_mb"]:
        return [f"{pre['resident_mb']:.0f} MB of the model files were still in the page cache at start (> {g['max_resident_mb']}): early reads would come from RAM"], notes
    notes += [f"page cache: {e}" for e in pre["errors"]]
    return [], notes


def disk_problems(start: dict, cfg: dict) -> list[str]:
    d = start.get("disk")
    if d is None:
        return ["the model's disk was not resolved in /proc/diskstats: no device witness is possible (pass --var disk=<name>)"]
    if d["rotational"] and cfg["vars"]["allow_rotational"] != "1":
        return [f"the model lives on {d['parent']}, a rotational disk: this test is about SSD (--var allow_rotational=1 runs it anyway)"]
    return []


def _counts(start: dict, cfg: dict) -> list[str]:
    n = cfg["sweep"]["n_requests"]
    out = []
    for phase in ("decode", "prefill"):
        usable = [r for r in start["requests"] if r["phase"] == phase and r["counted"] and r.get("usable")]
        if len(usable) < n:
            out.append(f"only {len(usable)} of {n} usable {phase} requests")
    return out


def speed_verdict(start: dict, plan, cfg: dict) -> dict:
    """May this speed start count?  All problems are listed, not only the first."""
    g = cfg["gates"]
    if start.get("error"):
        return {"valid": False, "reasons": [start["error"]], "notes": []}
    reasons = _counts(start, cfg) + startup_problems(start, plan, g, [cfg["vars"]["shm"], cfg["vars"]["shm2"]]) + ladder_problems(start, plan)
    if start.get("aborted"):
        reasons.append(f"aborted after {g['max_consecutive_failures']} consecutive failed requests")
    notes: list[str] = []
    if plan.capped:
        c_reasons, notes = cache_problems(start, plan, g)
        reasons += c_reasons + cgroup_problems(start, plan, cfg) + disk_problems(start, cfg) + nvme_problems(start, g)
        mem = start.get("memory") or {}
        if mem.get("oom_kill"):
            reasons.append(f"the kernel OOM-killed {mem['oom_kill']} process(es) inside the limit: capacity.engine_overhead_gib is too small")
        notes += [f"memory.max was hit {mem['max_hits']} times and the peak was {mem['peak'] / 2**30:.2f} GiB" if mem.get("peak") else "no memory.peak readback"]
    else:
        notes += _reference_notes(start, g)
    if start.get("moves") and not plan.ladder:
        notes.append(f"{len(start['moves'])} placement move line(s) in an arm that has no ladder flags")
    return {"valid": not reasons, "reasons": reasons, "notes": notes}


def _reference_notes(start: dict, g: dict) -> list[str]:
    f = _mean([r.get("file_mb") for r in _decode(start["requests"])])
    return [f"reference read {f:.1f} MB per decode request from the files (> {g['reference_max_file_mb']}): not an all-VRAM reference"] if f and f > g["reference_max_file_mb"] else []


def w7_verdict(rec: dict, plan, cfg: dict) -> dict:
    """May this forced-decode capture count?  It has no per-request engine log, so its proof is the device counter over the whole capture."""
    g = cfg["gates"]
    if rec.get("error"):
        return {"valid": False, "reasons": [rec["error"]], "notes": []}
    reasons = [] if rec["kl_files"].get("forced", {}).get("rows", 0) > 0 else ["the capture produced no rows"]
    if plan.capped:
        reasons += cache_problems(rec, plan, g)[0] + cgroup_problems(rec, plan, cfg) + disk_problems(rec, cfg)
        mb = rec.get("dev_read_mb")
        if mb is None or mb < g["w7_min_device_mb"]:
            reasons.append(f"the disk counter rose {mb} MB over the capture (< {g['w7_min_device_mb']}): the experts did not overflow to the SSD")
        if (rec.get("memory") or {}).get("oom_kill"):
            reasons.append("the kernel OOM-killed a process inside the limit")
    return {"valid": not reasons, "reasons": reasons, "notes": []}


def _rows_path(rec: dict | None) -> str | None:
    """The pooled rows file of a W7 capture, or None when the capture is missing or scored no positions."""
    f = (rec or {}).get("kl_files", {}).get("forced") or {}
    return f.get("path") if f.get("rows", 0) > 0 and f.get("ok") else None


def _kl_one(base: dict | None, r: dict, ref: str, th: dict, run) -> dict:
    """One W7 capture against the reference's first: the decision gate for a capped arm, the report-only noise floor for the reference's repeat."""
    a, b, kind = _rows_path(base), _rows_path(r), ("noise" if r["arm"] == ref else "gate")
    row = {"arm": r["arm"], "idx": r["idx"], "kind": kind}
    if a is None or b is None:
        return {**row, "rc": 2, "line": "no usable rows (capture failed, empty, or no reference)"}
    args = [a, b] if kind == "noise" else ["--baseline", a, "--arm", b, "--max-kl", str(th["kl_max"]), "--min-top1", str(th["top1_min"])]
    rc, line = run(args)
    return {**row, "rc": rc, "line": line}


def kl_rows(results: list[dict], cfg: dict, run=pay_gate.run_kl_py) -> list[dict]:
    """Each capped W7 capture against the reference's first one (decision), the reference's second against its first (noise, report only)."""
    ref = cfg["sweep"]["reference_arm"]
    w7 = [r for r in results if r["mode"] == "forced"]
    base = next((r for r in w7 if r["arm"] == ref and r["idx"] == 0), None)
    return [_kl_one(base, r, ref, cfg["thresholds"], run) for r in w7 if r is not base]


def kl_status(rows: list[dict], arm: str) -> str:
    """Worst gate verdict of an arm: FAIL beats INVALID beats PASS; no gate row = NONE."""
    mine = [r["rc"] for r in rows if r["arm"] == arm and r["kind"] == "gate"]
    if not mine:
        return "NONE"
    return "FAIL" if 1 in mine else ("INVALID" if any(rc not in (0, 1) for rc in mine) else "PASS")


def arm_status(verdicts: list[dict], kl: str, reference: bool, needs_kl: bool) -> str:
    """INVALID if a required start is missing or failed the data gate, or a wanted quality gate has no result; FAIL on a failed quality gate;
    UNGATED if the data is valid and the quality gate was switched off; else PASS."""
    if not verdicts or not all(v["valid"] for v in verdicts):
        return "INVALID"
    if reference:
        return "PASS"
    if kl == "NONE":
        return "INVALID" if needs_kl else "UNGATED"
    return kl
