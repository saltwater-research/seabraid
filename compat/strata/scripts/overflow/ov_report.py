# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""From starts and gate verdicts to one status per arm, the SUMMARY.txt text, and the criteria rows of the HTML chain report.

OWNER: this module owns the layout and the per-arm roll-up; every number is a mean of values the starts already hold, and every
verdict is ov_gate's (presentation never judges).  Speeds are reported with n and the spread and with NO ranking between arms: three
counted requests per phase cannot support one, and saying so here is the point.
CONTRACT: ``build(results, cfg, kl) -> [ArmReport]``; ``render(arms, kl, cfg) -> str``; ``verdicts(arms) -> [qual_report verdict dicts]``.
FAILURE: pure; a missing value prints ``-``.  An arm with a required start missing is INVALID, never absent.
CHECKER: compat/strata/tests/overflow/test_overflow_run.py (the text a reader sees is asserted).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import ov_gate
import ov_paths  # noqa: F401
import qual_stats as st


@dataclass
class ArmReport:
    arm: str
    plan: object
    label: str
    speed: list = field(default_factory=list)  # [(start, verdict)]
    w7: list = field(default_factory=list)  # [(rec, verdict)]
    kl: str = "NONE"
    status: str = "INVALID"


def _required(rep: ArmReport, needs_w7: bool) -> list[dict]:
    """The verdicts that must ALL be valid for the arm to count; empty (so INVALID) when a required start is missing."""
    complete = bool(rep.speed) and (bool(rep.w7) or not needs_w7)
    return [v for _s, v in rep.speed + (rep.w7 if needs_w7 else [])] if complete else []


def build(results: list[dict], cfg: dict, kl: list[dict]) -> list[ArmReport]:
    ref, needs_w7, out = cfg["sweep"]["reference_arm"], cfg["sweep"]["kl"], []
    for arm in cfg["sweep"]["arms"]:
        plan = cfg["plans"][arm]
        rep = ArmReport(arm, plan, cfg["arms"][arm].get("label", ""))
        rep.speed = [(s, ov_gate.speed_verdict(s, plan, cfg)) for s in results if s["arm"] == arm and s["mode"] == "speed"]
        rep.w7 = [(s, ov_gate.w7_verdict(s, plan, cfg)) for s in results if s["arm"] == arm and s["mode"] == "forced"]
        rep.kl = ov_gate.kl_status(kl, arm)
        rep.status = ov_gate.arm_status(_required(rep, needs_w7), rep.kl, arm == ref, needs_w7)
        out.append(rep)
    return out


def _f(x, spec: str = ".1f") -> str:
    return "-" if x is None else format(x, spec)


def _sd(xs: list[float]) -> float | None:
    return None if len(xs) < 2 else (sum((x - st.mean(xs)) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def _col(reqs: list[dict], phase: str, key: str) -> list[float]:
    return [r[key] for r in reqs if r["phase"] == phase and r["counted"] and r.get("usable") and r.get(key) is not None]


def _speed_lines(starts: list[dict]) -> list[str]:
    reqs = [r for s in starts for r in s["requests"]]
    d, p = _col(reqs, "decode", "decode_tps"), _col(reqs, "prefill", "prefill_tps")
    lines = [f"  decode tok/s mean {_f(st.mean(d))} sd {_f(_sd(d))} n {len(d)};  prefill tok/s mean {_f(st.mean(p))} sd {_f(_sd(p))} n {len(p)}   (no ranking between arms: n is too small)"]
    cells = [("engine file tier MB", "file_mb"), ("disk MB", "dev_read_mb"), ("process MB", "proc_read_mb"), ("engine OS-read MB", "os_read_mb"),
             ("major faults", "majflt"), ("decode waited ms", "io_wait_ms")]
    lines.append("  per counted decode request, mean: " + "; ".join(f"{name} {_f(st.mean(_col(reqs, 'decode', k)))}" for name, k in cells))
    return lines


def _mem_lines(start: dict) -> list[str]:
    m, pre = start.get("memory") or {}, start.get("preflight")
    peak = None if not m.get("peak") else m["peak"] / 2**30
    lines = [f"  cgroup: peak {_f(peak, '.2f')} GiB, memory.max hit {m.get('max_hits')} times, OOM kills {m.get('oom_kill')}; disk over the start {_f(m.get('dev_mb'))} MB, process {_f(m.get('proc_mb'))} MB"]
    if pre:
        lines.append(f"  page cache before the start: {_f(pre['resident_mb'], '.0f')} MB of the model files resident after eviction (limit {pre['max_resident_mb']:.0f} MB)")
    return lines


def _move_line(start: dict) -> str:
    """Committed hops by (lower->upper, src) as the placement driver logged them, and the stale-record count."""
    hops: dict[str, int] = {}
    for m in start.get("moves") or []:
        key = f"{m['lower']}->{m['upper']} {m['src']}"
        hops[key] = hops.get(key, 0) + 1
    return f"  placement moves (rung 0 = main card): {dict(sorted(hops.items())) or 'none'}; stale-placement records {start.get('stale')}"


def _w7_lines(rep: ArmReport) -> list[str]:
    lines: list[str] = []
    for s, v in rep.w7:
        lines.append(f"  W7 capture {s['tag']}: {s['kl_files'].get('forced', {}).get('rows', 0)} pooled rows, disk {_f(s.get('dev_read_mb'))} MB over the capture")
        lines += [f"  INVALID W7 start {s['tag']}: {r}" for r in v["reasons"]]
    return lines


def _arm_lines(rep: ArmReport, kl: list[dict]) -> list[str]:
    lines = [f"=== {rep.arm}: {rep.label} -> {rep.status} ==="] + [f"  {ln.strip()}" for ln in rep.plan.lines]
    for s, v in rep.speed:
        lines += _speed_lines([s]) + _mem_lines(s) + ([_move_line(s)] if rep.plan.ladder else [])
        lines += [f"  INVALID speed start {s['tag']}: {r}" for r in v["reasons"]] + [f"  note {s['tag']}: {n}" for n in v["notes"]]
    lines += _w7_lines(rep)
    lines += [f"  KL {r['kind']} s{r['idx']}: rc {r['rc']} {r['line']}" for r in kl if r["arm"] == rep.arm]
    if not rep.speed:
        lines.append("  INVALID: no speed start recorded for this arm")
    return lines


LIMITS = ["WHAT THIS RUN CANNOT SHOW:",
          "  - It does not rank arms by speed: 3 counted requests per phase give a spread, not a significance claim.",
          "  - The disk counter is machine-wide; it is trusted only together with the engine's own /proc read_bytes and the engine's file-tier bytes.",
          "  - It characterises the one drive the model sits on (named in the plan output); another drive may behave differently.",
          "  - Only the ladder arm (O3L) exercises live moves; the other arms are the static fill order (cards, RAM, SSD).",
          "  - Both helper processes of O3 and O3L share one card unless --var gpu2 names another, so their slots share that card's VRAM."]


def render(arms: list[ArmReport], kl: list[dict], cfg: dict) -> str:
    sw = cfg["sweep"]
    lines = [f"Capped-overflow test, arms {','.join(a.arm for a in arms)}; reference {sw['reference_arm']}; n = {sw['n_requests']} counted rounds after {sw['warmup']} warm-up; "
             f"quality: {'W7 decode-path KL against the uncapped reference' if sw['kl'] else 'OFF (UNGATED)'}.", ""]
    for rep in arms:
        lines += _arm_lines(rep, kl) + [""]
    return "\n".join(lines + LIMITS)


def verdicts(arms: list[ArmReport]) -> list[dict]:
    """One criteria row per arm: PASS only when the data gate and the quality gate both passed; UNGATED is a WARN; INVALID and FAIL are FAIL."""
    status = {"PASS": "PASS", "UNGATED": "WARN", "INVALID": "FAIL", "FAIL": "FAIL"}
    out = []
    for a in arms:
        reasons = [r for _s, v in a.speed + a.w7 for r in v["reasons"]]
        detail = f"{a.status}; kl {a.kl}" + (f"; {reasons[0]}" if reasons else "")
        out.append({"id": a.arm, "title": f"arm {a.arm}: {a.label}", "status": status[a.status], "detail": detail})
    return out
