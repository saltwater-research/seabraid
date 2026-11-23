# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""The four SEPARATE result verdicts of the placement qualification (audit decision 1) plus the host-RAM-headroom comparison.

OWNER: this module owns D1 (stall-inclusive decode), Q1 (per-text quality), H1 (headroom: moves on vs off at a fixed RAM
budget) and the aggregate AGG.  Convergence (C4) and helper-loss recovery (C9) stay in qual_judge.py and decode-path quality
(C10) in qual_judge_decode.py; AGG only reads them.
CONTRACT: ``judge_all(cfg, results, kl) -> list[verdict]`` = ``qual_judge.judge`` plus C10, D1, Q1, H1, AGG; same verdict shape
({id, title, status, detail}; PASS / FAIL / WARN).  The five results are never merged: each keeps its own verdict, and AGG
is PASS only when all five are PASS (FAIL when any is FAIL, else WARN) and lists each one's status.
STALL: requests are not streamed, so a pause inside a request is measured as wall time minus what the engine accounts for
(prompt_ms + completion tokens / decode tok/s).  Pauses >= ``stall_long_s`` (10 s) are long; >= ``stall_warn_s`` and
below that are short warnings, counted separately; both thresholds are in arms.json and PROVISIONAL.
FAILURE: pure, no I/O; a request without the fields is left out of the counts, and an arm with no measurable request is
not decidable (WARN), never a pass.
"""

from __future__ import annotations

import qual_judge as qj
import qual_judge_decode as qd
import qual_judge_file as qf
import qual_stats as st

_RANK = qj._RANK


def _wall_tps(r: dict) -> float | None:
    """Stall-INCLUSIVE decode: completion tokens over the request's wall time (the engine's own tok/s excludes pauses), skipping the first request like C1."""
    xs = [q["completion_n"] / q["wall_s"] for q in r["phases"].get("decode", [])[1:] if q.get("completion_n") and q.get("wall_s")]
    return st.mean(xs)


def _stall_s(q: dict) -> float | None:
    """Seconds of one request the engine's own timings do not account for (None when a field is missing)."""
    n, tps, wall, prompt_ms = q.get("completion_n"), q.get("decode_tps"), q.get("wall_s"), q.get("ttft_ms")
    if not (n and tps and wall) or prompt_ms is None:
        return None
    return max(0.0, wall - prompt_ms / 1000.0 - n / tps)


def _stall_counts(res: list[dict], key: str, t: dict) -> dict:
    xs = [s for r in qj._runs(res, "W1", key) for s in map(_stall_s, r["phases"].get("decode", [])) if s is not None]
    long_ = sum(s >= t["stall_long_s"] for s in xs)
    return {"n": len(xs), "long": long_, "warn": sum(t["stall_warn_s"] <= s < t["stall_long_s"] for s in xs)}


def _stall_pair(res: list[dict], arm: str, ref: str, t: dict) -> tuple[str, str]:
    """(status, note): a live arm with a higher long-pause rate than its static reference fails; nothing measured is not decidable."""
    a, b = _stall_counts(res, arm, t), _stall_counts(res, ref, t)
    note = f"{arm} long {a['long']}/{a['n']} warn {a['warn']}, {ref} long {b['long']}/{b['n']} warn {b['warn']}"
    if not a["n"] or not b["n"]:
        return "WARN", note + " (no measurable request)"
    return ("FAIL" if a["long"] * b["n"] > b["long"] * a["n"] else "PASS"), note


def d1(cfg, res) -> dict:
    t = cfg["thresholds"]
    pairs = (("P2", "P0"), ("N2", "N1"))
    noninf = [qj._noninferior("D1", "", qj._series(res, "W1", a, _wall_tps), qj._series(res, "W1", r, _wall_tps), t["c1_margin_pct"], f"{a} vs {r} stall-inclusive") for a, r in pairs]
    stalls = [_stall_pair(res, a, r, t) for a, r in pairs]
    status = max([v["status"] for v in noninf] + [s[0] for s in stalls], key=_RANK.get)
    notes = [v["detail"] for v in noninf] + [s[1] for s in stalls]
    return qj._v("D1", "Stall-inclusive decode", status, "; ".join(notes) + f" (long pause >= {t['stall_long_s']} s, short warning >= {t['stall_warn_s']} s)")


def _combine(sts: set[str]) -> str:
    """FAIL if any, PASS only if all PASS, else WARN."""
    return "FAIL" if "FAIL" in sts else ("PASS" if sts == {"PASS"} else "WARN")


def _start_status(kl: list[dict], text: str, idx: int) -> str:
    recs = [k for k in kl if k.get("text") == text and str(k.get("pair", "")).startswith(f"P2-s{idx}-vs-")]
    return max((qj._kl_status(k) for k in recs), key=_RANK.get) if recs else "WARN"


def _text_status(kl: list[dict], text: str, starts: list[int]) -> tuple[str, str]:
    """One text over the required live starts: FAIL when it fails in EVERY start (even if pooled passes), PASS when it passes in all, else not decidable."""
    got = [_start_status(kl, text, i) for i in starts]
    status = got[0] if got and len(set(got)) == 1 and got[0] != "WARN" else "WARN"
    return status, f"{text} {got}"


def q1(cfg, res, kl) -> dict:
    texts = cfg["workloads"].get("W4", {}).get("texts", [])
    starts = sorted({r["idx"] for r in res if r["workload"] == "W4" and r["arm"] == "P2"})
    if not texts or not starts:
        return qj._v("Q1", "Per-text quality", "WARN", "no W4 texts or no live W4 start")
    got = [_text_status(kl, t, starts) for t in texts]
    return qj._v("Q1", "Per-text quality", _combine({s for s, _ in got}), "per-text gate (KL and top-1) over live starts: " + "; ".join(n for _, n in got) + "; FAIL = the text fails in every live start, PASS = passes in all")


def h1(cfg, res) -> dict:
    """Moves on (P2) vs off (P0) at each fixed RAM budget, one variable per comparison; the budget effect with the arm fixed is shown, not gated."""
    budgets = [v for v in cfg["variants"] if v.startswith("ram_")]
    ser = lambda arm, b: qj._series(res, "W6", f"{arm}/{b}", qj._decode)  # noqa: E731
    if not budgets:
        return qj._v("H1", "RAM headroom", "WARN", "no ram_* variants in the config")
    cells = [qj._noninferior("H1", "", ser("P2", b), ser("P0", b), cfg["thresholds"]["c1_margin_pct"], f"P2 vs P0 at {b}") for b in budgets]
    fixed = [f"{arm} {budgets[-1]} vs {budgets[0]}: {_diff(ser(arm, budgets[-1]), ser(arm, budgets[0]))}" for arm in ("P0", "P2")]
    return qj._v("H1", "RAM headroom", max((c["status"] for c in cells), key=_RANK.get), "; ".join(c["detail"] for c in cells) + "; budget varied, arm fixed (informational): " + "; ".join(fixed))


def _diff(a: list[float], b: list[float]) -> str:
    w = st.welch_bounds(a, b)
    return "n/a" if w is None else f"{w[2]:+.2f}%"


def agg(vs: list[dict]) -> dict:
    by = {v["id"]: v["status"] for v in vs}
    four = [("stall-inclusive decode", "D1"), ("per-text quality", "Q1"), ("convergence", "C4"), ("helper-loss recovery", "C9"), ("decode-path quality", "C10")]
    sts = [by.get(cid, "WARN") for _, cid in four]
    status = "FAIL" if "FAIL" in sts else ("PASS" if set(sts) == {"PASS"} else "WARN")
    return qj._v("AGG", "Separate verdicts", status, "; ".join(f"{n} {cid}: {s}" for (n, cid), s in zip(four, sts, strict=True)))


def judge_all(cfg: dict, results: list[dict], kl: list[dict]) -> list[dict]:
    base = qj.judge(cfg, results, kl) + [qd.c10(cfg, results, kl)]
    own = [d1(cfg, results), q1(cfg, results, kl), h1(cfg, results)]
    # F1 (file-moves persona) is opt-in evidence: shown when the persona was selected, never part of AGG (a run of the five separate verdicts must not change because it exists).
    f1 = qf.f1(cfg, results)
    return base + own + [agg(base + own)] + ([f1] if f1 else [])
