# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""What one engine start measures: matched decode and prefill requests, each joined to the engine's own log lines.

OWNER: this module owns "what is sent to a running engine and what is kept per request".  It decides nothing about
validity (pay_gate.py) or speed (pay_analyze.py).  HTTP is qual_work.send (the shipped loopback client), the teacher-forced
rows are qual_work.run_kl, the recorded chain is qual_report.Recorder.step: this file calls them, it does not copy them.
CONTRACT: ``measure_speed(ctx, eng, run, cfg) -> start record``; ``measure_kl(ctx, eng, run, cfg) -> start record``.  ``eng``
needs only ``rundir`` and ``log_text()``; ``ctx`` is a qual_work.Ctx.  A request record is a flat dict of numbers (see
``build_record``); ``usable`` is True only for a counted request the engine answered with its prompt line and, for prefill,
that did not reuse a cached prefix (a reused prefix would measure the cache, not the prefill).
WHY the request stream is the same for every arm: decode requests are one fixed prompt with greedy sampling and prefill
requests are the same long text with a distinct first line, in the same order, so the arms differ in nothing the user sends.
WHY decode and prefill alternate inside a start: slow drift (heat, page cache) then hits both phases alike.
FAILURE: fail-open-loud.  A failed request is a FAIL step and a record with ``ok`` False; ``max_consecutive_failures``
failed requests end the start early (a hung GPU would otherwise burn the whole request timeout per request).
CHECKER: compat/strata/tests/payoff/test_payoff_run.py (synthetic engine, monkeypatched qual_work.send).
"""

from __future__ import annotations

import time

import pay_parse
import qual_work


def settle(eng, start: int, gates: dict, sleep=time.sleep, clock=time.time) -> str:
    """The engine log text after offset ``start``, once the request's lines are in: its ``prompt`` line has appeared and
    the log has been quiet for ``settle_s`` (the CUDA<r> and helper lines print after it).  A timeout returns what exists;
    the caller then sees no prompt line and refuses the request instead of guessing."""
    t0 = last = clock()
    seen = len(eng.log_text())
    while True:
        text = eng.log_text()
        if len(text) != seen:
            seen, last = len(text), clock()
        sl = text[start:]
        if pay_parse.PROMPT.search(sl) and clock() - last >= gates["settle_s"]:
            return sl
        if clock() - t0 >= gates["settle_timeout_s"]:
            return sl
        sleep(0.25)


def _tier_delta(cur: pay_parse.Tiers | None, prev: pay_parse.Tiers | None) -> dict:
    if cur is None or prev is None:
        return {"ram_blobs": None, "file_blobs": None, "file_mb": None}
    return {"ram_blobs": cur.ram_blobs - prev.ram_blobs, "file_blobs": cur.file_blobs - prev.file_blobs, "file_mb": cur.file_mb - prev.file_mb}


def _pcie_share(log: pay_parse.RequestLog) -> float | None:
    """Share of the experts the GPU cache missed that went over PCIe (per layer-window terms of the decode timing line)."""
    t = log.timing
    return t.pcie / (t.cpu_experts + t.pcie) if t and (t.cpu_experts + t.pcie) > 0 else None


def _problems(p: pay_parse.Prompt | None, phase: str, gates: dict) -> list[str]:
    if p is None:
        return ["no 'strata serve: prompt' line for this request in the engine log"]
    if phase == "prefill" and p.reused > gates["prefill_reuse_max_frac"] * p.prompt_tokens:
        return [f"{p.reused} of {p.prompt_tokens} prompt tokens came from the prompt cache: not a prefill measurement"]
    return []


def _engine_fields(log: pay_parse.RequestLog) -> dict:
    """The log-derived numbers of a request; every one is None when its line was absent."""
    t = log.timing
    return {**(vars(log.prompt) if log.prompt else dict.fromkeys(pay_parse.PROMPT_KEYS)), "windows": t.windows if t else None,
            "lookups": log.lookups, "hits": log.hits, "offload": log.offload, "pcie_share_of_misses": _pcie_share(log),
            "has_remote": bool(log.remote), "helper_requests": log.helper_requests, "helper_us_each": log.helper_us_each,
            "prefill_helper_mib": log.prefill_helper_mib, "os_read_mb": log.os_read_mb, **vars(pay_parse.components(log))}


def build_record(phase: str, k: int, counted: bool, resp: dict, log: pay_parse.RequestLog, prev: pay_parse.Tiers | None, gates: dict) -> dict:
    """One request as numbers.  Throughput comes from the engine's own ``prompt`` line (the same clock for every arm)."""
    problems, answered = _problems(log.prompt, phase, gates), bool(resp.get("ok"))
    return {"phase": phase, "k": k, "counted": counted, "ok": answered and log.prompt is not None, "problems": problems,
            "usable": counted and answered and not problems, "wall_s": resp.get("wall_s"), "sha": resp.get("sha"),
            **_engine_fields(log), **_tier_delta(log.tiers, prev)}


def _request(ctx, eng, name: str, phase: str, k: int, counted: bool, prompt: str, max_tokens: int, prev: list) -> dict:
    gates, offset, kept = ctx.cfg["gates"], len(eng.log_text()), {}

    def fn():
        resp = qual_work.send(ctx, prompt, max_tokens)
        if not resp.get("ok"):
            raise RuntimeError(resp.get("error", "request failed"))
        log = pay_parse.parse_request(settle(eng, offset, gates))
        kept["rec"] = build_record(phase, k, counted, resp, log, prev[0], gates)
        if log.tiers is not None:
            prev[0] = log.tiers
        if not kept["rec"]["ok"]:
            raise RuntimeError("; ".join(kept["rec"]["problems"]))
        r = kept["rec"]
        return r, f"{phase} {r['prompt_tokens']} tok in, {r['generated']} out: prefill {r['prefill_tps']} tok/s, decode {r['decode_tps']} tok/s"

    sr = ctx.rec.step(name, fn)
    return sr.value if sr.ok else {**kept.get("rec", {"phase": phase, "k": k, "counted": counted, "usable": False, "problems": []}),
                                   "ok": False, "usable": False, "error": sr.err}


def start_record(run, cfg: dict, startup: pay_parse.Startup) -> dict:
    wl = cfg["workloads"][run.workload]
    return {"workload": run.workload, "cap": wl["cap"], "mode": wl["mode"], "arm": run.arm, "idx": run.idx, "tag": run.tag,
            "startup": vars(startup), "requests": [], "kl_files": {}, "aborted": False, "error": None}


def measure_speed(ctx, eng, run, cfg: dict) -> dict:
    """The matched request stream of one speed start: (warmup + n) rounds of one decode and one prefill request."""
    sw, gates = cfg["sweep"], cfg["gates"]
    out = start_record(run, cfg, pay_parse.parse_startup(eng.log_text()))
    decode_prompt = qual_work.prompt_text(cfg, sw["decode_prompt"])
    long_text, _ = qual_work.build_long(cfg, sw["prefill_tokens"], False)
    prev, fails = [pay_parse.last_tiers(eng.log_text())], 0
    for k in range(sw["warmup"] + sw["n_requests"]):
        counted = k >= sw["warmup"]
        for phase, prompt, mt in (("decode", decode_prompt, sw["decode_tokens"]), ("prefill", f"Probe {run.tag} {k}.\n\n{long_text}", 1)):
            r = _request(ctx, eng, f"{run.tag}-{phase}-r{k + 1}", phase, k, counted, prompt, mt, prev)
            out["requests"].append(r)
            fails = 0 if r["ok"] else fails + 1
            if fails >= gates["max_consecutive_failures"]:
                out["aborted"] = True
                return out
    return out


def measure_kl(ctx, eng, run, cfg: dict) -> dict:
    """The quality start: each KL text teacher-forced through decode windows (qual_work.run_kl keeps the rows of the last request)."""
    out = start_record(run, cfg, pay_parse.parse_startup(eng.log_text()))
    holder = {"workload": run.workload, "phases": {}}
    qual_work.run_kl(ctx, eng, run.tag, {"texts": cfg["workloads"][run.workload]["texts"]}, holder)
    out["kl_files"] = holder["kl_files"]
    return out
