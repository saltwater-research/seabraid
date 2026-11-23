# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Plain-text tables for SUMMARY.txt: per cap point the speeds, the time components and the second-GPU comparison.

OWNER: this module owns the layout only; every number is computed in pay_analyze.py (presentation never computes: the one
exception, a share of decode time, is a division of two numbers pay_analyze already holds and is labelled as such).
CONTRACT: ``render(analysis, cfg) -> str``; ``verdicts(analysis) -> list[dict]`` in qual_report's verdict shape for the HTML.
FAILURE: pure; a missing value prints ``-``.
CHECKER: compat/strata/tests/payoff/test_payoff_analyze.py (the text a reader sees is asserted).
"""

from __future__ import annotations

import pay_analyze as pa
import qual_stats as st


def _f(x, spec: str = ".1f") -> str:
    return "-" if x is None else format(x, spec)


def _sd(xs: list[float]) -> float | None:
    return None if len(xs) < 2 else (sum((x - st.mean(xs)) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def _speed_rows(pt: pa.Point) -> list[str]:
    out = ["arm  starts valid  kl       decode tok/s (mean sd n)     prefill tok/s (mean sd n)"]
    for arm, ap in pt.arms.items():
        d, p = ap.samples["decode"], ap.samples["prefill"]
        out.append(f"{arm:<4} {ap.starts:>6} {ap.valid_starts:>5}  {ap.kl:<8} {_f(st.mean(d)):>9} {_f(_sd(d)):>6} {len(d):>3}"
                   f"            {_f(st.mean(p)):>9} {_f(_sd(p)):>6} {len(p):>3}")
    return out


def _comp_rows(pt: pa.Point) -> list[str]:
    out = ["arm  per decode request, ms (and % of decode_ms; the parts overlap, they do not sum to 100%):"]
    out.append("     " + " ".join(f"{c.replace('_ms', ''):>17}" for c in pa.COMPONENTS))
    for arm, ap in pt.arms.items():
        total = ap.comps.get("decode_ms")
        cells = []
        for c in pa.COMPONENTS:
            v = ap.comps.get(c)
            share = f" ({100 * v / total:.0f}%)" if v is not None and total and c != "decode_ms" else ""
            cells.append(f"{_f(v, '.0f') + share:>17}")
        out.append(f"{arm:<4} " + " ".join(cells))
    return out


def _compare_rows(pt: pa.Point, margin: float) -> list[str]:
    out = []
    for metric, by_rival in pt.compare.items():
        parts = [f"vs {r}: " + ("n<2" if b is None else f"{b[2]:+.1f}% [lower {b[0]:+.1f}%, upper {b[1]:+.1f}%]") for r, b in by_rival.items()]
        out.append(f"second GPU {metric}: {pt.status[metric].upper():<9} " + "; ".join(parts) + f"   (pays needs lower > +{margin}% vs every rival)")
    return out


def _problems(pt: pa.Point) -> list[str]:
    out = []
    for arm, ap in pt.arms.items():
        out += [f"  {arm} INVALID: {r}" for r in dict.fromkeys(ap.reasons)] + [f"  {arm} note: {n}" for n in dict.fromkeys(ap.notes)]
    return out


def render(an: pa.Analysis, cfg: dict) -> str:
    sw = cfg["sweep"]
    lines = [f"Second-GPU payoff, {len(an.points)} cap points, arms {','.join(sw['arms'])}; second GPU = {sw['second_gpu_arm']}; n = {sw['n_requests']} counted requests per phase after {sw['warmup']} warm-up.",
             "LIMIT: samples are requests inside ONE engine start per arm per point: the bounds cover request-to-request noise, not start-to-start variance.", ""]
    for pt in an.points:
        slots = ", ".join(f"{a} {'/'.join(str(n) for n in sorted(set(ap.slots))) or '-'}" for a, ap in pt.arms.items())
        lines += [f"=== cap {pt.cap}: logged expert-cache slots {slots}; {'matched' if pt.matched else 'NOT MATCHED'} ==="]
        lines += _speed_rows(pt) + [""] + _comp_rows(pt) + [""] + _compare_rows(pt, an.margin_pct) + _problems(pt) + [""]
    lines += ["CROSSOVER (measured range only):"] + [f"  {s}" for s in an.crossover.values()]
    return "\n".join(lines)


def _point_status(pt: pa.Point) -> str:
    if any(ap.kl == "FAIL" for ap in pt.arms.values()):
        return "FAIL"
    weak = not pt.matched or any(ap.valid_starts == 0 or ap.kl in ("NONE", "INVALID") for ap in pt.arms.values())
    return "WARN" if weak else "PASS"


def verdicts(an: pa.Analysis) -> list[dict]:
    """One row per cap point (data-gate status; the payoff statuses are the detail) and one per metric (the crossover)."""
    out = [{"id": f"cap{pt.cap}", "title": f"cap {pt.cap}: data gate", "status": _point_status(pt),
            "detail": "; ".join(f"{m} {s}" for m, s in pt.status.items())} for pt in an.points]
    out += [{"id": f"X-{m}", "title": f"crossover ({m})", "status": "WARN" if "could be decided" in s else "PASS", "detail": s}
            for m, s in an.crossover.items()]
    return out
