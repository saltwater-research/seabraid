# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Workloads W0 to W5 of the placement qualification protocol (DESIGN.md section 5.3): requests, prompts, raw metrics.

OWNER: this module owns "what is sent and what raw numbers come back" per start.  It decides nothing: thresholds and
verdicts are qual_judge.py's.  CONTRACT: ``run_workload(ctx, eng, run) -> dict`` (one result per engine start; every
request is its own recorded step); ``build_long`` makes the long and needle prompts.
METRICS: decode tok/s, prompt tok/s and TTFT come from the response ``timings`` block (llama-server style:
prompt_ms, prompt_per_second, predicted_per_second); when absent they fall back to wall clock over ``usage`` and the
request is flagged ``approx``.  Round lines, file-tier I/O and D4 lines come from the engine log.
FAILURE: a failed request is a FAIL step and a ``{"ok": False, "error": ...}`` entry, never an exception out of here.
"""

from __future__ import annotations

import hashlib
import os
import signal
import threading
import time
from dataclasses import dataclass

import qual_engine
import qual_stats as st
from qual_engine import post_json


@dataclass
class Ctx:
    cfg: dict
    url: str
    rec: object  # qual_report.Recorder
    dry: bool
    outdir: str


def prompt_text(cfg: dict, name: str) -> str:
    with open(os.path.join(cfg["_dir"], "prompts", name), encoding="utf-8") as fh:
        return fh.read()


def build_long(cfg: dict, tokens: int, needle: bool) -> tuple[str, str | None]:
    """Deterministic filler of about ``tokens`` tokens (chars_per_token estimate); the needle sits at the midpoint."""
    lim = cfg["limits"]
    paras = [p for p in prompt_text(cfg, "filler.txt").split("\n\n") if p.strip()]
    target, out, i, size = int(tokens * lim["chars_per_token"]), [], 0, 0
    while size < target:
        para = f"Entry {i}. {paras[i % len(paras)].strip()}"
        out.append(para)
        size += len(para) + 2
        i += 1
    secret = lim["needle_secret"] if needle else None
    if secret:
        out.insert(len(out) // 2, f"Remember this: the vault passphrase is {secret}.")
    ask = "What is the vault passphrase? Answer with the passphrase only." if needle else "Summarize the entries above in three sentences."
    return "\n\n".join(out) + "\n\n" + ask, secret


def send(ctx: Ctx, prompt: str, max_tokens: int) -> dict:
    """One greedy chat request; returns the metrics dict (ok False with an error string on failure)."""
    body = {"model": "x", "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.time()
    try:
        resp = post_json(ctx.url, body, ctx.cfg["limits"]["request_timeout_s"])
    except (OSError, ValueError) as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "t_start": t0, "t_end": time.time(), "wall_s": time.time() - t0}
    wall = time.time() - t0
    text = ((resp.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    tm, us = resp.get("timings") or {}, resp.get("usage") or {}
    out = {"ok": True, "t_start": t0, "t_end": t0 + wall, "wall_s": wall, "sha": hashlib.sha256(text.encode()).hexdigest()[:16],
           "completion_n": us.get("completion_tokens", tm.get("predicted_n")), "prompt_n": us.get("prompt_tokens", tm.get("prompt_n")),
           "decode_tps": tm.get("predicted_per_second"), "prompt_tps": tm.get("prompt_per_second"), "ttft_ms": tm.get("prompt_ms"),
           "approx": "predicted_per_second" not in tm, "text": text}
    if out["approx"] and out["completion_n"]:
        out["decode_tps"] = out["completion_n"] / wall
    return out


def _request_step(ctx: Ctx, name: str, prompt: str, max_tokens: int, secret: str | None = None) -> dict:
    holder: dict = {}

    def fn():
        r = send(ctx, prompt, max_tokens)
        holder.update(r)
        if not r["ok"]:
            raise RuntimeError(r["error"])
        if secret is not None:
            r["needle_ok"] = secret in r["text"]
        tps = r.get("decode_tps")
        return r, f"{r['wall_s']:.2f} s, {r.get('completion_n')} tok, decode {tps if tps is None else round(tps, 1)} tok/s, ttft_ms {r.get('ttft_ms')}"

    sr = ctx.rec.step(name, fn)
    res = sr.value if sr.ok else {**holder, "ok": False, "error": sr.err}
    return res


def run_phase(ctx: Ctx, eng, tag: str, ph: dict, result: dict) -> list[dict]:
    """``count`` requests of one phase, each a recorded step; per request the round count of the log afterwards."""
    secret = None
    if "tokens" in ph:
        text, secret = build_long(ctx.cfg, ph["tokens"], bool(ph.get("needle")))
    else:
        text = prompt_text(ctx.cfg, ph["prompt"])
    reqs = []
    hp = eng.helper_pid()
    io0 = helper_read_bytes(hp)
    for k in range(ph["count"]):
        r = _request_step(ctx, f"{tag}-{ph['name']}-r{k + 1}", text, ph["max_tokens"], secret)
        text_out = r.pop("text", None)
        if text_out is not None:
            with open(os.path.join(eng.rundir, f"{ph['name']}-r{k + 1}.txt"), "w", encoding="utf-8") as fh:
                fh.write(text_out)
        r["rounds_after"] = len(st.parse_round_lines(eng.log_text(), ctx.cfg["round_marker"]))
        reqs.append(r)
    io1 = helper_read_bytes(hp)
    result["phases"][ph["name"]] = reqs
    if io0 is not None and io1 is not None:
        result.setdefault("helper_read_delta", {})[ph["name"]] = io1 - io0
    return reqs


def helper_read_bytes(pid: int | None) -> int | None:
    """read_bytes of /proc/<pid>/io (read-only), or None without a helper or off Linux."""
    if pid is None:
        return None
    for line in qual_engine._read(f"/proc/{pid}/io").splitlines():
        if line.startswith("read_bytes:"):
            return int(line.split()[1])
    return None


def finish_log_metrics(ctx: Ctx, eng, result: dict) -> None:
    text = eng.log_text()
    result["rounds"] = st.parse_round_lines(text, ctx.cfg["round_marker"])
    result["io"] = st.parse_io(text, ctx.cfg.get("io_patterns", {}))
    hashes = [r.get("table_hash") for r in result["rounds"] if r.get("table_hash") is not None]
    result["table_hash_first"], result["table_hash_last"] = (hashes[0], hashes[-1]) if hashes else (None, None)
    result["quarantine_seen"] = st.quarantined(result["rounds"])
    result["driver_errors"] = st.driver_errors(text)


def run_kl(ctx: Ctx, eng, tag: str, wl: dict, result: dict) -> None:
    """Teacher-forced texts (max_tokens 1): after each, keep the last position-restart segment of STRATA_LOGPOS rows."""
    rows_path = os.path.join(eng.rundir, "rows.tsv")
    result["kl_files"] = {}
    for name in wl["texts"]:
        r = _request_step(ctx, f"{tag}-kl-{name}", prompt_text(ctx.cfg, name), 1)
        r.pop("text", None)
        rows = qual_engine._read(rows_path).splitlines() if os.path.exists(rows_path) else []
        seg = st.last_segment(rows)
        out = os.path.join(eng.rundir, f"{name}.tsv")
        with open(out, "w", encoding="utf-8") as fh:
            fh.write("\n".join(seg) + ("\n" if seg else ""))
        result["kl_files"][name] = {"path": out, "rows": len(seg), "ok": r["ok"]}


def run_fault(ctx: Ctx, eng, tag: str, ph: dict, result: dict) -> None:
    """W5: SIGKILL the helper (its own pid, inside the engine pgid) after a committed round, mid-request."""
    th_cfg, text = ctx.cfg["thresholds"], prompt_text(ctx.cfg, ph["prompt"])
    holder: dict = {}
    worker = threading.Thread(target=lambda: holder.update(send(ctx, text, ph["max_tokens"])))
    worker.start()
    rec = {"dry": ctx.dry}
    if not ctx.dry:
        rec.update(_kill_helper_after_commit(ctx, eng, th_cfg))
    worker.join(ctx.cfg["limits"]["request_timeout_s"])
    t_kill = rec.get("t_kill")
    rec["fail_after_s"] = (holder["t_end"] - t_kill) if t_kill and not holder.get("ok", True) else None
    rec["request_failed"] = bool(holder) and not holder.get("ok")
    ctx.rec.step(f"{tag}-fault-request", lambda: (None, f"killed={t_kill is not None} failed={rec['request_failed']} fail_after_s={rec['fail_after_s']}"))
    if not ctx.dry:
        rec.update(_after_fault(ctx, eng, tag, th_cfg, t_kill))
    result["fault"] = rec


def _kill_helper_after_commit(ctx: Ctx, eng, th: dict) -> dict:
    deadline = time.time() + th["fault_quarantine_rounds_s"] * 5
    while time.time() < deadline:
        rounds = st.parse_round_lines(eng.log_text(), ctx.cfg["round_marker"])
        if st.committed(rounds) > 0:
            pid = eng.helper_pid()
            if pid is None:
                return {"error": "no helper process found in the engine pgid"}
            t = time.time()
            os.kill(pid, signal.SIGKILL)
            return {"t_kill": t, "helper_pid": pid, "rounds_at_kill": len(rounds)}
        time.sleep(1)
    return {"error": "no committed round before the deadline"}


def _recovery_s(nxt: dict, t_kill: float | None) -> float | None:
    return nxt["t_end"] - t_kill if nxt.get("ok") and t_kill else None


def _after_fault(ctx: Ctx, eng, tag: str, th: dict, t_kill: float | None) -> dict:
    """The next request is a long one (600 tokens, about 100+ decode windows each, so at least one placement round ticks): the driver only
    runs at decode-window boundaries, so waiting idle for the quarantine round line (the first version) can never see it."""
    prompt = prompt_text(ctx.cfg, ctx.cfg["workloads"][eng.run.workload]["phases"][0]["prompt"])
    out = {}
    for k in (1, 2):  # two requests = about 4 rounds at --retier-every 128; the first one is the "next request" of the criterion
        nxt = _request_step(ctx, f"{tag}-fault-next-request-{k}", prompt, 600)
        out["next_served"] = bool(nxt.get("ok")) if k == 1 else out["next_served"] and bool(nxt.get("ok"))
        if k == 1:  # time to recovery: from the kill to the end of the first request served afterwards (None: not served, or no kill)
            out["recovery_s"] = _recovery_s(nxt, t_kill)
    deadline = time.time() + th["fault_quarantine_rounds_s"]
    while time.time() < deadline and not out.get("quarantined"):
        out["quarantined"] = st.quarantined(st.parse_round_lines(eng.log_text(), ctx.cfg["round_marker"]))
        out["quarantined"] = out["quarantined"] or any("placement degraded" in e for e in st.driver_errors(eng.log_text()))
        time.sleep(1)
    out["driver_errors"] = st.driver_errors(eng.log_text())[:4]
    out["zombies"] = eng.zombies()
    return out


def run_workload(ctx: Ctx, eng, run) -> dict:
    """All phases of one start, then the log-derived metrics.  ``result["ok"]`` is False if any request failed."""
    wl = ctx.cfg["workloads"][run.workload]
    tag = run.tag
    result = {"workload": run.workload, "arm": run.arm, "variant": run.variant, "key": run.key, "idx": run.idx, "kind": wl["kind"], "phases": {}}
    for ph in wl["phases"]:
        if wl["kind"] == "fault":
            run_fault(ctx, eng, tag, ph, result)
        else:
            run_phase(ctx, eng, tag, ph, result)
    if wl["kind"] == "kl":
        run_kl(ctx, eng, tag, wl, result)
    finish_log_metrics(ctx, eng, result)
    result["ok"] = all(r.get("ok") for reqs in result["phases"].values() for r in reqs)
    return result
