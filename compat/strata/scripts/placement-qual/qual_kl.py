# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""W4 quality gate: compare each P2 start's teacher-forced rows against the main-card-alone start, with compat/strata/tools/kl.py.

OWNER: kl.py owns the KL arithmetic and its gate (``--baseline --arm --max-kl --min-top1``, exit 0 PASS, 1 FAIL, 2
INVALID); this module only pairs the files and records the exit codes.  The baseline is the first ``M`` start; a second
``M`` start against the first is the noise floor (report-only mode, always exit 0).
CONTRACT: ``compare(cfg, results, rec) -> list[{"pair", "text", "rc", "line"}]``, every comparison a recorded step.
FAILURE: a missing or empty rows file is rc 2 (INVALID), never a pass; a kl.py that cannot run, is killed, or returns an
undocumented exit code is a FAIL step whose recorded rc is the raw code (None if it never finished), which qual_judge reads as FAIL.
"""

from __future__ import annotations

import os
import subprocess
import sys

KL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools", "kl.py")


def _usable(r: dict, name: str) -> str | None:
    f = r.get("kl_files", {}).get(name)
    return f["path"] if f and f["ok"] and f["rows"] > 0 else None


def _run(args: list[str]) -> tuple[int | None, str]:
    """(exit code, last output line).  A checker that did not finish (timeout, cannot start) has exit code None: the judge reads it as a checker failure, never a pass."""
    try:
        p = subprocess.run([sys.executable, "-B", KL, *args], capture_output=True, text=True, timeout=600)
    except (subprocess.TimeoutExpired, OSError) as e:
        return None, f"kl.py did not finish: {type(e).__name__}"
    lines = (p.stdout + p.stderr).strip().splitlines()
    return p.returncode, (lines[-1] if lines else "")


def _pair(cfg: dict, rec, ref: str | None, a: dict, base0: dict, name: str) -> dict:
    t = cfg["thresholds"]
    label, path = f"P2-s{a['idx']}-vs-M-s{base0['idx']}", _usable(a, name)
    if ref is None or path is None:
        rec.mark(f"kl-{name}-{label}", "WARN", "rows file missing or empty: INVALID, not a pass")
        return {"pair": label, "text": name, "rc": 2, "line": "no rows"}
    args = ["--baseline", ref, "--arm", path, "--max-kl", str(t["kl_max"]), "--min-top1", str(t["top1_min"])]
    rc, line = _gate_step(rec, f"kl-{name}-{label}", args)
    return {"pair": label, "text": name, "rc": rc, "line": line}


def _pool(r: dict, names: list[str]) -> str | None:
    """One rows file with every text of a start: the position column of text k is offset by k * 1,000,000 so kl.py sees no repeated position."""
    paths = [_usable(r, n) for n in names]
    if None in paths:
        return None
    out = os.path.join(os.path.dirname(paths[0]), "pooled.tsv")
    with open(out, "w", encoding="utf-8") as fh:
        for k, path in enumerate(paths):
            with open(path, encoding="utf-8") as src:
                for line in src.read().splitlines():
                    pos, _, rest = line.partition("\t")
                    fh.write(f"{int(pos) + k * 1_000_000}\t{rest}\n")
    return out


def _pooled(cfg: dict, rec, base0: dict, a: dict, names: list[str]) -> dict:
    """The protocol's gate (DESIGN.md 5.4 C8): KL and top-1 pooled over all texts of one start against the baseline start."""
    t, label = cfg["thresholds"], f"P2-s{a['idx']}-vs-M-s{base0['idx']}"
    ref, path = _pool(base0, names), _pool(a, names)
    if ref is None or path is None:
        rec.mark(f"kl-pooled-{label}", "WARN", "a text has no rows: INVALID, not a pass")
        return {"pair": label, "text": "pooled", "rc": 2, "line": "no rows"}
    rc, line = _gate_step(rec, f"kl-pooled-{label}", ["--baseline", ref, "--arm", path, "--max-kl", str(t["kl_max"]), "--min-top1", str(t["top1_min"])])
    return {"pair": label, "text": "pooled", "rc": rc, "line": line}


def compare(cfg: dict, results: list[dict], rec) -> list[dict]:
    """Per-text gates (diagnostic) and the pooled gate per P2 start (the criterion), plus the main-card noise floor."""
    if "W4" not in cfg["workloads"]:
        return []  # W4 not selected: nothing to compare (C8 then reports WARN, never a pass)
    base = [r for r in results if r["workload"] == "W4" and r["arm"] == "M"]
    arms = [r for r in results if r["workload"] == "W4" and r["arm"] == "P2"]
    texts = cfg["workloads"]["W4"]["texts"]
    out = [_pair(cfg, rec, _usable(base[0], n), a, base[0], n) for n in texts for a in arms] if base else []
    out += [_pooled(cfg, rec, base[0], a, texts) for a in arms] if base else []
    _noise_floor(rec, base, texts)
    return out


def _noise_floor(rec, base: list[dict], texts: list[str]) -> None:
    """Report-only kl.py of the second main-card start against the first, per text and pooled: the run-to-run noise the gate sits on."""
    if len(base) < 2:
        return
    for t in texts:
        a, b = _usable(base[0], t), _usable(base[1], t)
        rec.step(f"kl-noise-floor-M1-vs-M0-{t}", lambda a=a, b=b: (None, _run([a or "", b or ""])[1]))
    pa, pb = _pool(base[0], texts), _pool(base[1], texts)
    rec.step("kl-noise-floor-M1-vs-M0-pooled", lambda: (None, _run([pa or "", pb or ""])[1]))


def _gate_step(rec, name: str, args: list[str]) -> tuple[int | None, str]:
    """One recorded kl.py gate call; exit 1 (gate FAIL) and any exit code kl.py does not document (a signal such as -9, a crash,
    None for a run that never finished) show as FAIL steps, exit 2 (INVALID) as PASS-with-note for C8 to WARN on."""
    seen: dict = {}

    def fn():
        seen["v"] = _run(args)
        if seen["v"][0] not in (0, 2):
            raise RuntimeError(f"kl gate FAIL (exit {seen['v'][0]}): {seen['v'][1]}")
        return None, f"kl.py exit {seen['v'][0]}: {seen['v'][1]}"

    res = rec.step(name, fn)
    return seen.get("v") or (None, res.err)
