# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""W7 decode-path quality: capture one forced-decode start, pool its rows, and gate each live start against the main-card-alone one.

OWNER: this module owns "what W7 runs and which kl.py pairs it makes".  kl.py owns the KL arithmetic and the gate (exit
0 PASS, 1 FAIL, 2 INVALID, and INVALID for two files that did not force the same targets); qual_judge_decode.py owns the verdict.
WHY W7 EXISTS: W4 teacher-forces a prefill, where the Seabraid helper tier never runs, so its KL cannot see a decode-path change
(live run G1, 2026-11-19: tier on and off gave byte-identical W4 rows).  Here the engine decodes a FIXED continuation with the tier
engaged (Strata force_check.py ``score --path decode``: the verify windows take the forced token, the head logits come out as top-32 rows).
CONTRACT: ``capture(ctx, run) -> (result, note)`` or raises (the caller records the FAIL step; no result means no evidence).
``compare(cfg, results, rec) -> list[{"pair", "text", "rc", "line"}]``: pair ``D-<arm>-s<i>-vs-M-s<ref>``, text ``decode-pooled`` (the
decision) or ``decode-noise`` (a second M start against the reference, report only).
FAILURE: a capture that cannot start, exits nonzero or times out raises; a missing or empty rows file is rc 2 (INVALID), never a pass;
a kl.py that crashes is a FAIL step whose raw exit code is recorded (qual_kl._gate_step), which the judge reads as FAIL.
SAFETY: argv list, never a shell; on a timeout the capture's own process group is killed (it started its own engine).
"""

from __future__ import annotations

import json
import os
import signal
import subprocess

import qual_cfg
import qual_kl

POOL_STRIDE = 1_000_000  # the position column of item k is offset by k * POOL_STRIDE so kl.py sees no repeated position (as qual_kl._pool)
KEY = "forced"  # the kl_files entry of a W7 result


def pool(tsv: str, out: str) -> int:
    """Pool the items of one force_check ``score`` output (OUT.tsv + OUT.tsv.meta.json) into ``out``; returns the row count.

    WHY the last row per position: a window re-read after a rejected draft rewrites the row, and the last one is the committed one."""
    with open(tsv + ".meta.json", encoding="utf-8") as fh:
        items = json.load(fh)["items"]
    with open(tsv, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    n = 0
    with open(out, "w", encoding="utf-8") as fh:
        for k, it in enumerate(items):
            last = {}
            for ln in lines[it["first_line"]:it["end_line"]]:
                pos, _, rest = ln.partition("\t")
                last[int(pos)] = rest
            if not last:
                raise RuntimeError(f"item {k} of {tsv} scored no positions")
            fh.writelines(f"{pos + k * POOL_STRIDE}\t{last[pos]}\n" for pos in sorted(last))
            n += len(last)
    return n


def _argv(ctx, run, rundir: str) -> list[str]:
    cfg = ctx.cfg
    vars_ = {**cfg["vars"], "rundir": rundir}
    vars_["forced_cfg"] = qual_cfg.subst(cfg["arms"][run.arm]["forced_cfg"], vars_)
    return qual_cfg.subst(cfg["workloads"][run.workload]["capture"]["argv"], vars_)


def _need_files(ctx, run) -> None:
    """Fail before running anything when the capture tool, the forced set or the arm's engine config is not a file."""
    for name in ("force_check", "forced_set", f"forced_cfg_{run.arm}"):
        path = ctx.cfg["vars"].get(name)
        if not path or not os.path.isfile(path):
            raise RuntimeError(f"W7 needs var {name!r} to name an existing file (got {path!r}); pass --var {name}=<path>")


def _spawn(argv: list[str], timeout: float) -> tuple[int, str]:
    """(exit code, output tail).  A timeout kills the capture's whole process group (it owns an engine) and raises."""
    with subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True) as p:
        try:
            out, _ = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.communicate()
            raise RuntimeError(f"capture did not finish in {timeout:.0f} s: process group killed") from None
    return p.returncode, "\n".join(out.strip().splitlines()[-6:])


def capture(ctx, run) -> tuple[dict, str]:
    """One W7 start: run the capture command, pool its rows, return the result record the judge and compare() read."""
    rundir = os.path.join(ctx.outdir, "runs", run.tag)
    base = {"workload": run.workload, "arm": run.arm, "variant": run.variant, "key": run.key, "idx": run.idx, "kind": "forced", "phases": {}, "rounds": []}  # rounds: the shape every result consumer (tables, C3) reads
    if ctx.dry:
        return {**base, "ok": True, "dry": True, "kl_files": {}}, "DRY-RUN would run: " + " ".join(_argv(ctx, run, rundir))
    _need_files(ctx, run)
    os.makedirs(rundir, exist_ok=True)
    argv = _argv(ctx, run, rundir)
    rc, tail = _spawn(argv, ctx.cfg["workloads"][run.workload]["capture"].get("timeout_s", 7200))
    if rc != 0:
        raise RuntimeError(f"capture exit {rc}: {tail}")
    out = os.path.join(rundir, "pooled.tsv")
    rows = pool(os.path.join(rundir, "forced.tsv"), out)
    return {**base, "ok": rows > 0, "kl_files": {KEY: {"path": out, "rows": rows, "ok": True}}}, f"{rows} pooled rows; {tail.splitlines()[-1] if tail else ''}"


def _gate(cfg: dict, rec, ref: dict, r: dict) -> dict:
    """One live start against the reference, with the same thresholds as the prefill gate (G1 used them for the decode path too)."""
    t, label = cfg["thresholds"], f"D-{r['arm']}-s{r['idx']}-vs-M-s{ref['idx']}"
    a, b = qual_kl._usable(ref, KEY), qual_kl._usable(r, KEY)
    if a is None or b is None:
        rec.mark(f"kl-decode-{label}", "WARN", "rows file missing or empty: INVALID, not a pass")
        return {"pair": label, "text": "decode-pooled", "rc": 2, "line": "no rows"}
    rc, line = qual_kl._gate_step(rec, f"kl-decode-{label}", ["--baseline", a, "--arm", b, "--max-kl", str(t["kl_max"]), "--min-top1", str(t["top1_min"])])
    return {"pair": label, "text": "decode-pooled", "rc": rc, "line": line}


def _noise(rec, ref: dict, other: dict) -> list[dict]:
    """Report-only kl.py of the second M start against the reference: the run-to-run noise the gate sits on."""
    a, b = qual_kl._usable(ref, KEY), qual_kl._usable(other, KEY)
    if a is None or b is None:
        return []
    label = f"D-noise-M-s{other['idx']}-vs-M-s{ref['idx']}"

    def run():
        rc, line = qual_kl._run([a, b])
        return (rc, line), line

    got = rec.step(f"kl-decode-{label}", run)
    return [{"pair": label, "text": "decode-noise", "rc": got.value[0], "line": got.value[1]}] if got.ok else []


def compare(cfg: dict, results: list[dict], rec) -> list[dict]:
    """The pooled gate per live start against the lowest M start, plus the M-vs-M noise floor.  No W7 or no reference: no records (C10 then WARNs)."""
    if "W7" not in cfg["workloads"]:
        return []
    runs = [r for r in results if r["workload"] == "W7"]
    refs = sorted((r for r in runs if r["arm"] == "M"), key=lambda r: r["idx"])
    if not refs:
        return []
    out = [_gate(cfg, rec, refs[0], r) for r in runs if r["arm"] != "M"]
    return out + (_noise(rec, refs[0], refs[1]) if len(refs) > 1 else [])
