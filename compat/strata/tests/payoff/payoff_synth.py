# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Synthetic engine log text for the payoff tests, written in the exact line formats the parsers read.

OWNER: this file owns the test-side copy of the engine's line formats (pay_parse.py owns the real regexes); a parser change
that breaks a format breaks the tests that build text here.  Each function returns plain text; nothing here runs an engine.
Formats: Strata v0.2.15 src/program/generate.cpp (prompt, hit rate, tiers, file tier I/O, decode timing, expert cache) and
compat/strata/patches/strata/0001 (CUDA<r> request line, seabraid helper line).
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.normpath(os.path.join(HERE, "..", "..", "scripts", "payoff"))
sys.path.insert(0, SCRIPTS)
import pay_paths  # noqa: E402,F401


def timing_line(windows: int = 100, cpu: float = 6.0, stage: float = 2.0, cpu_experts: float = 1.5, pcie: float = 0.0, pipelined: bool = False) -> str:
    head = "adapt wait 0.00 + " if not pipelined else ""
    return (f"strata decode timing: {windows} windows, avg T 4.30, 2.80 tokens/window, 26.20 ms/window = {head}verify 23.89 "
            f"(GPU-reach wait 18.40 + per-layer host 5.10 [plan 0.50 actq 0.40 jobs 0.30 CPU {cpu:.2f}] + stage {stage:.2f}) + commit/emit 1.00 + "
            f"draft 0.50 + adapt join 0.00 + other 0.80; per layer-window: CPU experts {cpu_experts:.2f} (2.10 entries), VRAM hits 6.20, "
            f"PCIe {pcie:.2f}; 0.0 swaps/window\n")


def prompt_line(prompt_tokens: int, reused: int, prefill_tps: float, generated: int, decode_tps: float) -> str:
    read = prompt_tokens - reused
    return (f"strata serve: prompt {prompt_tokens} tokens = {reused} reused + {read} read in {int(1000 * read / prefill_tps)} ms "
            f"({prefill_tps:.1f} tok/s), {generated} generated in {int(1000 * generated / decode_tps)} ms ({decode_tps:.1f} tok/s), "
            f"drafts accepted 100 of 200, 3 checkpoints\n")


def hit_line(offload: int = 0) -> str:
    extra = f"; {offload} more read by the GPU over PCIe or from another GPU (6.0% of all {1000 + offload} routed)" if offload else ""
    return f"strata serve: decode expert cache hit rate: 90.0% (900 hits / 1000 lookups){extra}\n"


def tiers_line(ram: int, files: int, mb: float) -> str:
    return f"strata serve: expert tiers: GPU 900 hits this request; since the start RAM {ram} blobs, files {files} blobs {mb:.1f} MB read\n"


def io_line(wait_ms: float = 7.5) -> str:
    return (f"strata serve: file tier I/O this request: the OS read 12.5 MB from storage (3 major faults) for 40.0 MB of expert reads; "
            f"io prefetch: 5 read ahead (1.0 MB by pread), 4 used, 1 unused, 0 skipped (cached), 0 dropped; decode waited {wait_ms:.1f} ms in 12 fetches\n")


def remote_lines(staging_ms: float = 310.0, wait_ms: float = 940.0, helper_requests: int = 1200) -> str:
    return (f"strata serve: CUDA1: 4000 expert entries, {helper_requests} active layer launches, 10.5 MiB returned (20.0 MiB with full rows) in this request; "
            f"host {staging_ms:.0f} ms staging+launching, {wait_ms:.0f} ms waiting for it in this request\n"
            f"strata serve: seabraid helper: {helper_requests} layer requests, 74.8 us each on average in the helper "
            f"(16.9 us staging+launching, 57.9 us waiting for its GPU)\n")


def startup_text(slots: int, helper: bool = False) -> str:
    out = f"strata generate: expert cache {slots} slots, 21.50 GiB of VRAM; policy is\n"
    if helper:
        out += "seabraid: helper NVIDIA GeForce RTX 3090 on channel /seabraid-payoff, 22.10 GiB free\n"
    return out + "strata serve: resident RAM: 9.80 GiB of experts in RAM, 0 exchanged with the VRAM tier, 0 blob reads from the file\n"


def request_text(arm: str, phase: str, decode_tps: float, prefill_tps: float, ram: int = 1000, files: int = 0, prompt_tokens: int = 3500,
                 reused: int = 5) -> str:
    """One request's log block in the engine's print order (decode timing first, then the prompt line, then the per-request lines)."""
    if phase == "prefill":
        return prompt_line(prompt_tokens, reused, prefill_tps, 1, decode_tps) + hit_line() + tiers_line(ram, files, 0.0)
    pcie = {"A": 0.0, "B": 1.5, "C": 0.0}[arm]
    cpu_experts = {"A": 1.5, "B": 0.0, "C": 0.4}[arm]
    text = timing_line(windows=100, cpu=0.0 if arm == "B" else 6.0, stage=9.0 if arm == "B" else 2.0, cpu_experts=cpu_experts, pcie=pcie)
    text += prompt_line(12, 12 if reused else 0, prefill_tps, 400, decode_tps) + hit_line(60 if arm == "B" else 0) + tiers_line(ram, files, 0.0) + io_line()
    return text + (remote_lines() if arm == "C" else "")


TAB, NL = chr(9), chr(10)


def write_logpos(path, seed: int = 1, noise: float = 0.0, n: int = 60, vocab: int = 40) -> str:
    """A STRATA_LOGPOS-shaped TSV (pos, target, logp_target, top, logp_top, top==target, 0, 0, then 32 id:logprob pairs) from seeded
    logits; ``noise`` adds an independent gaussian to every logit, so noise 0 reproduces the baseline and a large noise breaks the gate."""
    import math
    import random

    rng, rng2 = random.Random(seed), random.Random(seed + 1000)
    lines = []
    for pos in range(n):
        base = [rng.gauss(0, 3) for _ in range(vocab)]
        z = [b + (rng2.gauss(0, noise) if noise else 0.0) for b in base]
        m = max(z)
        lse = m + math.log(sum(math.exp(x - m) for x in z))
        lp = [x - lse for x in z]
        top = max(range(vocab), key=lp.__getitem__)
        target = pos % vocab
        pairs = sorted(range(vocab), key=lp.__getitem__, reverse=True)[:32]
        lines.append(TAB.join([str(pos), str(target), f"{lp[target]:.6f}", str(top), f"{lp[top]:.6f}", str(int(top == target)), "0", "0"]
                               + [f"{i}:{lp[i]:.6f}" for i in pairs]))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(NL.join(lines) + NL)
    return str(path)
