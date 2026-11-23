# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""C10, decode-path quality: every live W7 start must have an explicit pooled PASS against the main-card-alone reference.

OWNER: this module owns the C10 decision.  Thresholds live in arms.json (``kl_max`` / ``top1_min``, applied by kl.py when
qual_decode.compare ran the gate); the exit-code reading is qual_judge._kl_status, shared with C8 so the two cannot drift.
WHY A SECOND KL CRITERION: C8 scores a prefill, where the helper tier never runs; only a forced decode with the tier engaged
exercises the path the tier changes.  C8 says so in its detail and C10 is the verdict for the decode path.
CONTRACT: ``c10(cfg, results, kl) -> verdict``.  Required evidence: a W7 reference (M) start, and for each live arm of W7 and each
start 0..n-1 a result and a ``decode-pooled`` record.  Any missing piece is WARN (never PASS); a record whose exit code is outside
{0, 2}, or exit 0 without the gate's own PASS line, is not a pass (qual_judge._kl_status); exit 2 (INVALID, including targets that
differ) is WARN; the worst status over all required starts wins.  The noise floor (M vs M) is reported in the detail, not gated.
FAILURE: pure, no I/O; a missing key is missing data.
"""

from __future__ import annotations

import qual_judge as qj

TITLE = "Decode-path quality (KL)"


def _ref_idx(res: list[dict]) -> int | None:
    idx = sorted(r["idx"] for r in res if r["workload"] == "W7" and r["arm"] == "M")
    return idx[0] if idx else None


def _start(kl: list[dict], have: set, arm: str, i: int, ref: int) -> tuple[str, str]:
    """(status, note) of one required live start: the worst of its pooled decode records."""
    if (arm, i) not in have:
        return "WARN", f"{arm} s{i}: no start result"
    recs = [k for k in kl if k.get("text") == "decode-pooled" and k.get("pair") == f"D-{arm}-s{i}-vs-M-s{ref}"]
    if not recs:
        return "WARN", f"{arm} s{i}: no pooled decode record"
    worst = max((qj._kl_status(k) for k in recs), key=qj._RANK.get)
    return worst, f"{arm} s{i}: rc {[k.get('rc') for k in recs]} {worst}"


def _noise(kl: list[dict]) -> str:
    lines = [str(k.get("line", "")) for k in kl if k.get("text") == "decode-noise"]
    return f"noise floor (M vs M, report only): {lines[0]}" if lines else "noise floor (M vs M) not measured"


def c10(cfg: dict, res: list[dict], kl: list[dict]) -> dict:
    wl, ref = cfg["workloads"].get("W7"), _ref_idx(res)
    if wl is None or ref is None:
        return qj._v("C10", TITLE, "WARN", "W7 is not selected or has no main-card-alone (M) start: nothing to compare the decode path against")
    have = {(r["arm"], r["idx"]) for r in res if r["workload"] == "W7"}
    got = [_start(kl, have, a, i, ref) for a in wl["arms"] if a != "M" for i in range(wl["n"])]
    return _decide(cfg["thresholds"], got, ref, kl) if got else qj._v("C10", TITLE, "WARN", "W7 has no live arm")


def _decide(th: dict, got: list[tuple[str, str]], ref: int, kl: list[dict]) -> dict:
    status = max((g[0] for g in got), key=qj._RANK.get)
    return qj._v("C10", TITLE, status, f"required live starts {len(got)} against M s{ref}: {'; '.join(g[1] for g in got)} "
                 f"(max KL {th['kl_max']}, top-1 >= {th['top1_min']}); {_noise(kl)}")
