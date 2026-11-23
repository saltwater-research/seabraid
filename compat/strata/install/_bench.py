# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""seabraid installer, PLAN input: measure each usable card's expert decode rate.  UNTESTED-ON-HARDWARE.

OWNER     : this file is the ONE place that runs the per-card benchmark and parses its output.  The benchmark itself is
            not a separate binary: it is `seabraid-helper` started with SEABRAID_STRATA_CARD_BENCH=1 (strata-ext/src/card_bench_run.cpp),
            which times the helper's real RemoteExperts path on one card, prints ONE `SEABRAID-CARD-BENCH v1 ...` line on
            stderr, and then carries on serving.  The line's format is owned by strata-ext/src/card_bench.cpp (format_line);
            parse() below is its Python reader and must change with it.
CONTRACT  : measure(report, cands, helper_exes, pack, native, layers, experts, runner) -> BenchResult(rates, why, lead_ms, lead_why);
            skipped(reason) -> BenchResult;  parse(text) -> (rows_per_s | None, why);  parse_lead(text, cls) -> (ms | None, why).
            rates = {(vendor, index): decode rows/s} for EVERY candidate, or None (with why) - never a partial map.
            lead_ms = {1 | 4: {(vendor, index): ms per decode window of the LEAD job}} holding only the classes (1 stream,
            >= 4 streams) that EVERY candidate reported; BenchResult.lead_for(streams) picks one.  The seabraid-helper binary prints no
            lead figure (it has no attention path; the engine-side probe prints it) leaves lead_ms empty and says why.
            Imports: build (Runner), services (card mask), _build_rocm (lib dirs).  Import direction: plan <- build <- here.
FAILURE   : fail-open-loud.  Anything that stops one card being measured (no helper binary, dry-run, non-zero exit, no
            bench line, unparseable or unmeasured line, timeout) voids the WHOLE map: a mix of measured and guessed
            numbers would rank on noise.  The caller then plans by heuristic and prints `why`.  Nothing here raises.
CHECKER   : tests/install/test_bench.py (fixture of the real line format; failure paths; three guards sabotaged).
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional
from urllib.parse import unquote

from ._build_rocm import rocm_lib_dirs
from seabraid.ranking import lead_class
from .build import Cmd, Runner
from .services import _visibility   # the one owner of "mask a process to one card"; a copy here would drift

MARKER = "SEABRAID-CARD-BENCH v1 "
# Bounds: the helper caps each regime at 1.5 s (card_bench_run.cpp kCapSeconds), so the measured part is ~3 s; the rest is
# process start, runtime init and loading 10 experts.  ITERS is the decode-regime call count (the helper clamps it to 30..5000).
TIMEOUT_S = 120.0
ITERS = 100
NOT_OPTED_IN = "pass --bench to measure (it briefly loads every card)"   # no "not measured:" here: rank_note adds that prefix


@dataclass(frozen=True)
class BenchResult:
    rates: Optional[Mapping]        # {(vendor, index): decode rows/s} for every candidate, or None
    why: str                        # "" when measured; otherwise the reason there is no measurement
    lead_ms: Optional[Mapping] = None   # {1 | 4: {(vendor, index): ms per window}}, only classes every candidate reported
    lead_why: str = ""              # why a class is missing (empty when both are present)

    def lead_for(self, streams: int) -> Optional[Mapping]:
        """The lead-job map for the configured concurrency, or None (2-3 streams have no regime; see _plan_rank.lead_class)."""
        k = lead_class(streams)
        return (self.lead_ms or {}).get(k) if k else None


def skipped(reason: str) -> BenchResult:
    return BenchResult(None, reason)


def _last_bench_fields(text: str) -> Optional[dict]:
    """The key=value fields of the LAST bench line in `text`, or None when there is no such line."""
    line = next((ln for ln in reversed(text.splitlines()) if MARKER in ln), None)
    if line is None:
        return None
    return dict(tok.split("=", 1) for tok in line[line.index(MARKER) + len(MARKER):].split() if "=" in tok)


def parse(text: str) -> tuple:
    """Decode rows/s from the LAST bench line in `text`, or (None, why).  Only the decode regime is read: it is the
    one that sets MAIN's per-token cost; prefill is the largest batch a helper can serve and is not a ranking input."""
    kv = _last_bench_fields(text)
    if kv is None:
        return None, "no SEABRAID-CARD-BENCH line in the helper output"
    note = unquote(kv.get("note", ""))
    try:
        rate = float(kv["decode_rows_per_s"])
    except (KeyError, ValueError):
        return None, "bench line has no readable decode_rows_per_s" + (f" (note: {note})" if note else "")
    if not math.isfinite(rate) or rate <= 0:
        return None, "decode regime was not measured" + (f" (note: {note})" if note else "")
    return rate, ""


def parse_lead(text: str, cls: int) -> tuple:
    """ms per decode window of the lead job (non-expert path, all experts resident) for concurrency class `cls` (1 | 4), from
    the LAST bench line, or (None, why).  Lower is better; the opposite sign of rows/s, so it never shares a code path with it."""
    kv = _last_bench_fields(text)
    if kv is None:
        return None, "no SEABRAID-CARD-BENCH line"
    try:
        ms = float(kv[f"lead{cls}_window_ms"])
    except (KeyError, ValueError):
        return None, f"bench line has no lead{cls}_window_ms"
    return (ms, "") if math.isfinite(ms) and ms > 0 else (None, f"lead{cls} was not measured")


def _cmd(c: Any, exe: str, libs: Mapping, pack: str, native: str, layers: int, experts: int) -> Cmd:
    # `--device 0` inside a one-card visibility mask, exactly like the services' helper units, so the number describes the
    # same process shape that will serve.  A private shm name: never collide with a running /seabraidN channel.
    env = {**_visibility(c.vendor, c.index, c.uuid), "SEABRAID_STRATA_CARD_BENCH": "1", "SEABRAID_STRATA_CARD_BENCH_ITERS": str(ITERS)}
    if libs.get(c.vendor):
        env["LD_LIBRARY_PATH"] = libs[c.vendor]
    argv = (exe, "--device", "0", "--shm", f"/seabraid-bench-{c.vendor}{c.index}", "--pack", pack, "--native", native,
            "--layers", str(layers), "--experts", str(experts))
    return Cmd(argv, env=env, label=f"card bench {c.vendor} gpu{c.index}")


def _helper_libs(report: Any, cands: tuple) -> dict:
    """{vendor: LD_LIBRARY_PATH} for each candidate vendor's toolkit ("" when it was not found)."""
    libs = {}
    for v in {c.vendor for c in cands}:
        tk = report.toolkit_for(v)
        libs[v] = ":".join(rocm_lib_dirs(tuple(tk.lib_dirs), os.path.isdir)) if tk is not None else ""
    return libs


def _bench_text(runner: Runner, c: Any, cmd: Cmd, timeout_s: float) -> tuple:
    """(helper output holding the bench line, "") or ("", why this card was not measured)."""
    cap = runner.run_capture(cmd, MARKER, timeout_s)
    if cap.found:
        return cap.text, ""
    tail = next((ln.strip() for ln in reversed(cap.text.splitlines()) if ln.strip()), "no output")[:160]
    what = f"timed out after {timeout_s:.0f}s" if cap.timed_out else f"exited {cap.rc} without a bench line"
    return "", f"{c.name} ({c.vendor} gpu{c.index}) {what}: {tail}"


def _lead_result(leads: dict, n_cands: int) -> tuple:
    """({class: {card: ms}} for the classes every candidate has, why the rest is missing)."""
    lead = {k: m for k, m in leads.items() if len(m) == n_cands}
    why = "" if len(lead) == 2 else "the lead-job figure (lead1/lead4_window_ms) is missing for at least one card; seabraid-helper does not time it"
    return lead, why


def measure(report: Any, cands, helper_exes: Mapping[str, str], pack: str, native: str, layers: int, experts: int,
            runner: Runner, timeout_s: float = TIMEOUT_S) -> BenchResult:
    """Run the benchmark once per candidate, one at a time (concurrent runs would share PCIe and host cores and perturb
    each other's number), through `runner` so dry-run and tests execute nothing."""
    cands = tuple(cands)
    if not cands:
        return skipped("no usable cards to measure")
    if runner.dry_run:
        return skipped("dry-run: the benchmark was not executed")
    libs = _helper_libs(report, cands)
    rates, leads = {}, {1: {}, 4: {}}
    for c in cands:
        exe = helper_exes.get(c.vendor)
        if not exe:
            return skipped("no seabraid-helper built yet (first install); rerun with --bench after the build, or with --no-build")
        text, why = _bench_text(runner, c, _cmd(c, exe, libs, pack, native, layers, experts), timeout_s)
        if why:
            return skipped(why)
        rate, why = parse(text)
        if rate is None:
            return skipped(f"{c.name} ({c.vendor} gpu{c.index}): {why}")
        rates[(c.vendor, c.index)] = rate
        for k in leads:
            ms, _ = parse_lead(text, k)
            if ms is not None:
                leads[k][(c.vendor, c.index)] = ms
    lead, why = _lead_result(leads, len(cands))
    return BenchResult(rates, "", lead, why)
