# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""The data gate of the payoff harness: did each start run the arm it claims, and did its output pass the quality gate?

OWNER: this module owns "is this start's speed allowed to count".  Speeds are computed in pay_analyze.py and only from
starts this module calls valid.  The KL arithmetic and its thresholds belong to tools/kl.py (called as a subprocess, exit 0
PASS / 1 FAIL / 2 INVALID); a threshold is passed only when the config sets one, so kl.py's defaults stay the one source.
CONTRACT: ``config_problems(arm, start, gates) -> (invalid, notes)``; ``kl_gates(results, cfg, run) -> rows``;
``kl_status(rows) -> {(cap, arm): PASS|FAIL|INVALID|NONE}``; ``start_verdict(start, kl, cfg) -> {valid, reasons, notes}``.
WHY the log is the witness: a flag that the launcher silently drops (a base config that already carries --pcie-frac, a
clamped 1.0) would make arms A and B the same arm.  The engine's own per-request lines prove what actually happened:
A must show no experts read over PCIe, B must show the misses moved to PCIe, C must show the helper channel up.
FAILURE: pure except ``run_kl_py`` (one subprocess per comparison); an unusable input is INVALID (exit 2), never a pass.
CHECKER: compat/strata/tests/payoff/test_payoff_gate.py (planted wrong-arm logs, real kl.py exit codes on synthetic rows).
"""

from __future__ import annotations

import subprocess
import sys

import pay_paths

KL = pay_paths.KL_PY


def _decode(reqs: list[dict]) -> list[dict]:
    return [r for r in reqs if r.get("phase") == "decode" and r.get("ok")]


def _share(r: dict) -> float | None:
    return r.get("pcie_share_of_misses")


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _a_problems(start: dict, gates: dict) -> list[str]:
    reqs, out = _decode(start["requests"]), []
    shares = [s for s in map(_share, reqs) if s is not None]
    if shares and max(shares) > gates["a_pcie_max_share"]:
        out.append(f"arm A read {max(shares):.3f} of its missed experts over PCIe (max {gates['a_pcie_max_share']}): --pcie-frac 0 did not hold")
    if start["startup"]["helper_up"] or any(r.get("has_remote") for r in reqs):
        out.append("arm A shows a helper channel or a CUDA<r> request line: it is not a one-GPU arm")
    return out


def _b_problems(start: dict, gates: dict) -> tuple[list[str], list[str]]:
    shares = [s for s in map(_share, _decode(start["requests"])) if s is not None]
    if not shares:
        return [], ["arm B had no missed experts (the cache holds everything): A and B are the same run at this point"]
    m = _mean(shares)
    if m < gates["b_pcie_min_share"]:
        return [f"arm B moved only {m:.3f} of its missed experts to PCIe (min {gates['b_pcie_min_share']}): --pcie-frac 1 did not hold"], []
    return [], []


def _c_problems(start: dict) -> tuple[list[str], list[str]]:
    if not start["startup"]["helper_up"]:
        return ["arm C: no 'seabraid: helper .. on channel' line: the second GPU never joined"], []
    used = [r.get("helper_requests") or 0 for r in _decode(start["requests"])]
    return [], ([] if any(used) else ["arm C: the helper served no layer request (idle second GPU: expect a tie with A at this point)"])


def config_problems(arm: str, start: dict, gates: dict) -> tuple[list[str], list[str]]:
    """(reasons the start is not the arm it claims, notes that do not invalidate it)."""
    if start["startup"]["cap_slots"] is None:
        return ["no 'expert cache N slots' line in the engine log: the cap was not confirmed"], []
    if arm == "A":
        return _a_problems(start, gates), []
    if arm == "B":
        return _b_problems(start, gates)
    return _c_problems(start) if arm == "C" else ([], [])


def run_kl_py(args: list[str]) -> tuple[int, str]:
    """One kl.py call: (exit code, its last output line).  An OSError is exit 2 (INVALID) with the reason as the line."""
    try:
        p = subprocess.run([sys.executable, "-B", KL, *args], capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.SubprocessError) as exc:
        return 2, f"kl.py could not run: {exc}"
    lines = (p.stdout + p.stderr).strip().splitlines()
    return p.returncode, (lines[-1] if lines else "")


def _usable(start: dict, text: str) -> str | None:
    f = start.get("kl_files", {}).get(text)
    return f["path"] if f and f["ok"] and f["rows"] > 0 else None


def _thresholds(cfg: dict) -> list[str]:
    t, out = cfg.get("thresholds", {}), []
    if "kl_max" in t:
        out += ["--max-kl", str(t["kl_max"])]
    if "top1_min" in t:
        out += ["--min-top1", str(t["top1_min"])]
    return out


def _row(s: dict, text: str, kind: str, rc: int, line: str) -> dict:
    return {"cap": s["cap"], "arm": s["arm"], "idx": s["idx"], "text": text, "kind": kind, "rc": rc, "line": line}


def _start_rows(s: dict, base: dict | None, top: int, cfg: dict, run) -> list[dict]:
    rows = []
    for text in cfg["sweep"]["kl_texts"]:
        arm_path, ref = _usable(s, text), (_usable(base, text) if base else None)
        if s is base:
            rows.append(_row(s, text, "baseline", 0, "baseline start: the reference of every gate"))
        elif arm_path is None or ref is None:
            rows.append(_row(s, text, "gate", 2, "no usable rows (start failed, empty file, or no baseline)"))
        elif s["cap"] == top and s["arm"] == base["arm"]:
            rows.append(_row(s, text, "noise", *run([ref, arm_path])))
        else:
            rows.append(_row(s, text, "gate", *run(["--baseline", ref, "--arm", arm_path, *_thresholds(cfg)])))
    return rows


def kl_gates(results: list[dict], cfg: dict, run=run_kl_py) -> list[dict]:
    """Every KL start against the baseline start (baseline arm at the largest cap, repeat 0), one row per text.

    The baseline's own repeat starts give report-only noise-floor rows (kind ``noise``).  No usable baseline = every
    gate row is rc 2 (INVALID), so a lost baseline can never turn into passes."""
    kls = [s for s in results if s["mode"] == "kl"]
    top = max((s["cap"] for s in results), default=0)
    base = next((s for s in kls if s["cap"] == top and s["arm"] == cfg["sweep"]["baseline_arm"] and s["idx"] == 0), None)
    return [row for s in kls for row in _start_rows(s, base, top, cfg, run)]


def kl_status(rows: list[dict]) -> dict[tuple[int, str], str]:
    """Worst gate verdict per (cap, arm) over its texts: FAIL beats INVALID beats PASS; no gate rows = NONE."""
    out: dict[tuple[int, str], str] = {}
    for r in rows:
        if r["kind"] == "noise":
            continue
        key = (r["cap"], r["arm"])
        v = {0: "PASS", 1: "FAIL"}.get(r["rc"], "INVALID")
        order = ("FAIL", "INVALID", "PASS")
        out[key] = min((out.get(key, v), v), key=order.index)
    return out


def _count_reasons(start: dict, n: int) -> list[str]:
    out = []
    for phase in ("decode", "prefill"):
        usable = [r for r in start["requests"] if r["phase"] == phase and r["counted"] and r.get("usable")]
        if len(usable) < n:
            out.append(f"only {len(usable)} of {n} usable {phase} requests")
    return out


def start_verdict(start: dict, kl: str, cfg: dict) -> dict:
    """May this speed start's numbers count?  ``kl`` is its (cap, arm) gate status (NONE when the gate was not run)."""
    gates = cfg["gates"]
    reasons, notes = config_problems(start["arm"], start, gates) if not start.get("error") else ([start["error"]], [])
    if start.get("aborted"):
        reasons.append(f"aborted after {gates['max_consecutive_failures']} consecutive failed requests")
    reasons += _count_reasons(start, cfg["sweep"]["n_requests"])
    if kl in ("FAIL", "INVALID"):
        reasons.append(f"quality gate {kl} (output not shown to match the baseline: speed does not count)")
    if kl == "NONE":
        notes.append("UNGATED: no teacher-forced KL result for this arm at this cap")
    return {"valid": not reasons, "reasons": reasons, "notes": notes}
