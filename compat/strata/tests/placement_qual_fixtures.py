# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Shared synthetic results for the placement qualification verdict tests (test_placement_qual_verdicts.py,
test_placement_qual_separate.py): one start set that passes every criterion, which each absence test then breaks."""
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
QUAL = HERE.parent / "scripts" / "placement-qual"
sys.path.insert(0, str(QUAL))
sys.path.insert(0, str(HERE))

import qual_separate as qs  # noqa: E402

CFG = json.loads((QUAL / "arms.json").read_text(encoding="utf-8"))

# ---- synthetic results: one start set that passes every criterion (the baseline each absence test then breaks) ----
def _rounds(committed, share=None, fence=40000.0):
    return [{"committed": float(c), "fence_us": fence + i, "plan_us": 300.0, **({} if share is None else {"helper_share": share})}
            for i, c in enumerate(committed)]


def _run(wl, key, idx, **kw):
    arm = key.split("/")[0]
    return {"workload": wl, "arm": arm, "key": key, "idx": idx, "ok": True, "phases": {}, "rounds": [], "driver_errors": [], **kw}


def _req(tps, extra_s=0.0):
    """A decode request whose wall time is what the engine accounts for (2000 tokens at ``tps``, prompt 500 ms) plus ``extra_s`` of pause."""
    return {"decode_tps": tps, "sha": "x", "completion_n": 2000, "ttft_ms": 500.0, "wall_s": 2000 / tps + 0.5 + extra_s}


def _w1(key, idx, tps, first_extra=0.0):
    return _run("W1", key, idx, phases={"decode": [_req(tps, first_extra), _req(tps), _req(tps)]}, rounds=_rounds([1, 0, 0]))


def _w6():
    return [_run("W6", f"{a}/{b}", i, phases={"decode": [{"decode_tps": 100.0 + JIT[i]}] * 3}) for a in ("P0", "P2") for b in ("ram_hi", "ram_mid", "ram_lo") for i in range(3)]


SETTLED = [2, 2, 1, 1] + [0] * 10


def _w2(key, idx, commits, share=None):
    return _run("W2", key, idx, phases={"A": [{"rounds_after": 0}], "B": [{"decode_tps": 100.0}] * 4}, rounds=_rounds(commits, share))


def _w3(key, idx, ttft, os_mb, d4=()):
    return _run("W3", key, idx, phases={"p8k": [{"ttft_ms": ttft}]}, io={"os_read_mb": [os_mb], "d4_chunks": list(d4)})


JIT = [0.0, 0.2, -0.2]


def _w3_set():
    off = [_w3(k, i, 500.0 + JIT[i], 40.0 + JIT[i]) for k in ("N1/off", "N2/off", "P0/off", "P2/off") for i in range(3)]
    return off + [_w3(k, i, 500.0 + JIT[i], 40.0, d4=[64]) for k in ("N1/on", "N2/on") for i in range(3)]


def _w4_w5_set():
    fault = {"fail_after_s": 0.2, "quarantined": True, "next_served": True, "zombies": 0, "recovery_s": 5.0}
    return ([_run("W4", k, i) for k in ("M", "P2") for i in range(2)] + [_run("W5", "P2", i, fault=fault) for i in range(3)])


def _w7_set():
    """W7 decode-path capture: two starts of the reference (M) and of each live arm, each with one usable pooled rows file."""
    kf = lambda k, i: {"forced": {"path": f"{k}-s{i}-pooled.tsv", "rows": 2400, "ok": True}}  # noqa: E731
    return [_run("W7", k, i, kl_files=kf(k, i)) for k in ("M", "P0", "P2") for i in range(2)]


def good_results():
    res = [_run("W0", "P2", 0, rounds=_rounds([2]))]
    res += [_w1(k, i, 100.0 + JIT[i]) for k in ("P0", "P2", "P2z", "N1", "N2") for i in range(3)]
    res += [_w2(k, i, SETTLED, 0.3 if k == "N1" else None) for k in ("P2", "N1", "N2", "P0", "N0") for i in range(2)]
    return res + _w3_set() + _w4_w5_set() + _w6() + _w7_set()


def kl_rec(i, rc, text="pooled", line="KL-GATE PASS: n=1"):
    return {"pair": f"P2-s{i}-vs-M-s0", "text": text, "rc": rc, "line": line}


TEXTS = CFG["workloads"]["W4"]["texts"]


def dec_rec(arm, i, rc, ref=0, text="decode-pooled", line="KL-GATE PASS: n=2400"):
    """One W7 comparison record (qual_decode.compare): live arm start i against the main-card-alone start ``ref``."""
    return {"pair": f"D-{arm}-s{i}-vs-M-s{ref}", "text": text, "rc": rc, "line": line}


NOISE = {"pair": "D-noise-M-s1-vs-M-s0", "text": "decode-noise", "rc": 0, "line": "positions 2400  top-1 agree 98.5%  KL(top32) mean 0.00102"}
GOOD_DECODE_KL = [dec_rec(a, i, 0) for a in ("P0", "P2") for i in (0, 1)] + [NOISE]
GOOD_KL = [kl_rec(0, 0), kl_rec(1, 0)] + [kl_rec(i, 0, text=t) for i in (0, 1) for t in TEXTS] + GOOD_DECODE_KL


def verdicts(res, kl=None, cfg=CFG):
    return qs.judge_all(cfg, res, GOOD_KL if kl is None else kl)


def status(vs, cid):
    return [v["status"] for v in vs if v["id"] == cid]
