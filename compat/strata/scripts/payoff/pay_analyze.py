# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""From valid starts to a table per cap point and a crossover statement limited to the measured range (pure, no I/O).

OWNER: this module owns "how a speed number and a payoff verdict are computed".  The confidence bound is
qual_stats.welch_bounds (placement-qual owns the t quantile); validity is pay_gate; the parsing is pay_parse.
CONTRACT: ``analyze(results, cfg, kl_rows) -> Analysis``.  Per cap point and metric (decode tok/s, prefill tok/s) the
second-GPU arm is compared with each rival arm: welch_bounds(second, rival) gives (lower, upper, diff) as percent of the
rival's mean, each a one-sided 95% bound.  Status per point: ``pays`` = lower bound above +margin against EVERY rival;
``loses`` = upper bound below -margin against at least one; ``tie`` = neither; ``undecided`` = an arm has no valid start,
fewer than 2 samples, or the arms did not get the same cap (the engine's logged slot count differs).
The crossover statement names the boundary between two ADJACENT measured caps, or says the measured range holds none.
It never interpolates and never extrapolates: below the smallest or above the largest measured cap it says "not measured".
LIMIT (stated in every report): samples are requests inside one engine start per arm per point, so the bound covers
request-to-request noise only, not start-to-start variance; use ``--starts 2`` or more to see the latter.
FAILURE: pure; a missing measurement is None / undecided, never a number.
CHECKER: compat/strata/tests/payoff/test_payoff_analyze.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pay_gate
import qual_stats as st

METRICS = {"decode": "decode_tps", "prefill": "prefill_tps"}
COMPONENTS = ("staging_ms", "merge_wait_ms", "local_compute_ms", "gpu_reach_wait_ms", "io_wait_ms", "decode_ms")


@dataclass
class ArmPoint:
    arm: str
    starts: int = 0
    valid_starts: int = 0
    slots: list[int] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    samples: dict[str, list[float]] = field(default_factory=lambda: {m: [] for m in METRICS})
    comps: dict[str, float | None] = field(default_factory=dict)
    kl: str = "NONE"


@dataclass
class Point:
    cap: int
    arms: dict[str, ArmPoint]
    matched: bool = False
    compare: dict[str, dict[str, tuple | None]] = field(default_factory=dict)  # metric -> rival -> (lower, upper, diff)
    status: dict[str, str] = field(default_factory=dict)  # metric -> pays / loses / tie / undecided


@dataclass
class Analysis:
    points: list[Point]
    crossover: dict[str, str]
    margin_pct: float


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _comps(requests: list[dict]) -> dict[str, float | None]:
    """Mean over the usable decode requests; a component the engine never printed stays None (shown as '-')."""
    dec = [r for r in requests if r["phase"] == "decode" and r.get("usable")]
    return {c: _mean([r[c] for r in dec if r.get(c) is not None]) for c in COMPONENTS}


def _add_start(ap: ArmPoint, start: dict, verdict: dict) -> None:
    ap.starts += 1
    ap.slots.append(start["startup"]["cap_slots"])
    ap.reasons += verdict["reasons"]
    ap.notes += verdict["notes"]
    if not verdict["valid"]:
        return
    ap.valid_starts += 1
    for metric, key in METRICS.items():
        ap.samples[metric] += [r[key] for r in start["requests"] if r["phase"] == metric and r.get("usable")]


def _finish_arm(ap: ArmPoint, starts: list[dict], verdicts: list[dict]) -> None:
    good = [s for s, v in zip(starts, verdicts) if v["valid"]]
    pooled = [r for s in good for r in s["requests"]]
    ap.comps = _comps(pooled)


def _arm_point(cap: int, arm: str, speed: list[dict], kl: dict, cfg: dict) -> ArmPoint:
    ap = ArmPoint(arm)
    starts = [s for s in speed if s["cap"] == cap and s["arm"] == arm]
    ap.kl = kl.get((cap, arm), "NONE")
    verdicts = [pay_gate.start_verdict(s, ap.kl, cfg) for s in starts]
    for s, v in zip(starts, verdicts):
        _add_start(ap, s, v)
    _finish_arm(ap, starts, verdicts)
    return ap


def build_points(results: list[dict], cfg: dict, kl_rows: list[dict]) -> list[Point]:
    kl = pay_gate.kl_status(kl_rows)
    speed = [s for s in results if s["mode"] == "speed"]
    out = []
    for cap in sorted({s["cap"] for s in speed}, reverse=True):
        arms = {a: _arm_point(cap, a, speed, kl, cfg) for a in cfg["sweep"]["arms"]}
        out.append(Point(cap, arms, matched=_matched(arms)))
    return out


def _matched(arms: dict[str, ArmPoint]) -> bool:
    """Every arm ran at the same logged slot count: a cap the engine shrank for one arm only is no longer a matched budget."""
    seen = {n for ap in arms.values() for n in ap.slots}
    return len(seen) == 1 and all(ap.slots for ap in arms.values())


def _bounds(second: list[float], rival: list[float]) -> tuple | None:
    return st.welch_bounds(second, rival) if len(second) >= 2 and len(rival) >= 2 else None


def judge_point(pt: Point, cfg: dict) -> None:
    """Fill pt.compare and pt.status for both metrics (see the module docstring for the status rule)."""
    second, margin = cfg["sweep"]["second_gpu_arm"], cfg["gates"]["margin_pct"]
    rivals = [a for a in cfg["sweep"]["arms"] if a != second]
    for metric in METRICS:
        pt.compare[metric] = {r: _bounds(pt.arms[second].samples[metric], pt.arms[r].samples[metric]) for r in rivals}
        pt.status[metric] = _status(pt, list(pt.compare[metric].values()), margin)


def _status(pt: Point, bounds: list, margin: float) -> str:
    if not pt.matched or any(b is None for b in bounds):
        return "undecided"
    if all(b[0] > margin for b in bounds):
        return "pays"
    return "loses" if any(b[1] < -margin for b in bounds) else "tie"


def _caps(rows: list[tuple[int, str]]) -> str:
    return ", ".join(str(c) for c, _ in rows) or "none"


def _boundary(decided: list[tuple[int, str]], pays: list[bool]) -> str:
    flips = [i for i in range(1, len(pays)) if pays[i] != pays[i - 1]]
    if len(flips) == 1:
        (c0, s0), (c1, s1) = decided[flips[0] - 1], decided[flips[0]]
        return f"the boundary lies between cap {c0} ({s0}) and cap {c1} ({s1}); each side is measured, nothing inside the gap is."
    yes = _caps([d for d, p in zip(decided, pays) if p])
    no = _caps([d for d, p in zip(decided, pays) if not p])
    return f"no single crossover: pays at cap {yes}, not at {no}."


def _decided_sentence(decided: list[tuple[int, str]]) -> str:
    pays = [s == "pays" for _, s in decided]
    if all(pays):
        return f"the second GPU beats every rival at every decided cap ({_caps(decided)}), including the largest, {decided[0][0]}: no crossover inside the measured range."
    if not any(pays):
        return f"the second GPU does not beat both rivals at any decided cap ({_caps(decided)}): no crossover inside the measured range."
    return _boundary(decided, pays)


def crossover_statement(metric: str, rows: list[tuple[int, str]]) -> str:
    """One sentence on where the second GPU starts to pay, from (cap, status) rows ordered from the largest cap down."""
    caps = [c for c, _ in rows] or [0]
    decided = [(c, s) for c, s in rows if s != "undecided"]
    undecided = [(c, s) for c, s in rows if s == "undecided"]
    body = _decided_sentence(decided) if decided else "no point could be decided."
    tail = f" Undecided (not interpolated): cap {_caps(undecided)}." if undecided else ""
    return f"{metric}: within the measured caps {min(caps)}..{max(caps)}, {body}{tail}"


def analyze(results: list[dict], cfg: dict, kl_rows: list[dict]) -> Analysis:
    points = build_points(results, cfg, kl_rows)
    for pt in points:
        judge_point(pt, cfg)
    cross = {m: crossover_statement(m, [(p.cap, p.status[m]) for p in points]) for m in METRICS}
    return Analysis(points, cross, cfg["gates"]["margin_pct"])
