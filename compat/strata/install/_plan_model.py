# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Model facts for the planner: what a GGUF set holds and where it lives at run time.  UNTESTED-ON-HARDWARE.

OWNER     : ModelFacts and the classification of every tensor (expert / PLE / host-mapped / head / HC / dense), split out of
            plan.py for size.  The HC (hyper-connection) Q8_0 on-card rule and its source live here.
CONTRACT  : read_model_facts(paths) -> ModelFacts;  facts_from_tensors(...) is the same without touching files (tests).
            ModelFacts.dense_bytes EXCLUDES hc_* tensors: plan adds hc_stored_bytes or hc_q8_bytes depending on the main card.
FAILURE   : an unreadable or non-MoE GGUF set raises ModelError; nothing is defaulted.
CHECKER   : tests/install/test_plan.py (real tensor groups of the test system's IQ2_XS; synthetic GGUFs).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Sequence

from ._plan_gguf import ModelError, scan_gguf

# Strata src/core/native_dense.cpp:85-91 hc_q8_name(): these tensors are held as Q8_0 on the card when HC_Q8 is on (lines 45-77:
# on by default for some HIP archs - see _plan_opts.HC_Q8_DEFAULT_ARCHS - and off on every other backend including CUDA).  Q8_0 = 34 bytes per 32 weights.
HC_NAME = re.compile(r"(blk\.\d+\.hc_(attn|ffn)_(down|up|inject)\.weight|output_hc_(down|up)\.weight)$")


def q8_0_bytes(n_elements: int) -> int:
    return -(-n_elements // 32) * 34


@dataclass(frozen=True)
class ModelFacts:
    layers: int
    experts: int            # routed experts per layer
    expert_bytes: int       # all expert tensors (names containing "_exps")
    dense_bytes: int        # non-expert tensors that go on the MAIN card, excluding hc_*, the head, the PLE table and token_embd
    hidden: Optional[int] = None    # embedding length H (sizes the shm channel); None = unknown
    blob_max_bytes: Optional[int] = None    # largest per-layer expert blob (gate+up+down of one expert); None = unknown
    files: tuple = ()
    head_bytes: Optional[int] = None    # output.weight (None: PlanOptions.head_bytes applies)
    ple_bytes: int = 0                  # per_layer_token_embd table: streamed from SSD, never in VRAM
    host_bytes: int = 0                 # token_embd: host-mapped
    hc_stored_bytes: int = 0            # hc_* as stored in the GGUF (bf16/f32)
    hc_q8_bytes: int = 0                # the same tensors as Q8_0 (what an RDNA4-class HIP main holds)


def _tensor_kind(name: str) -> str:
    """Which byte total a tensor counts toward: ple | host | head | exps | hc | dense.  The first rule that applies wins."""
    if name.startswith("per_layer_token_embd"):
        return "ple"
    if name.startswith("token_embd"):
        return "host"
    if name == "output.weight":
        return "head"
    if "_exps" in name:
        return "exps"
    return "hc" if HC_NAME.search(name) else "dense"


def _scan_shards(shards: Sequence[Sequence[tuple]]) -> tuple:
    """(bytes per kind, kinds seen, Q8_0 bytes of the hc tensors, {layer: expert bytes})."""
    total = dict.fromkeys(("ple", "host", "head", "exps", "hc", "dense"), 0)
    seen, hc_q8, per_layer = set(), 0, {}      # per_layer: layer -> bytes of its expert tensors (one expert's blob = that / experts)
    for sizes in shards:
        # Swift co-locates its PLE table with dense tensors; only the table itself stays off-card.
        for name, sz, n in sizes:
            kind = _tensor_kind(name)
            total[kind] += sz
            seen.add(kind)
            if kind == "hc":
                hc_q8 += q8_0_bytes(n)
            elif kind == "exps":
                m = re.search(r"blk\.(\d+)\.", name)
                if m:
                    per_layer[int(m.group(1))] = per_layer.get(int(m.group(1)), 0) + sz
    return total, seen, hc_q8, per_layer


def facts_from_tensors(layers, experts, hidden, shards: Sequence[Sequence[tuple]], paths: Sequence[str] = ()) -> ModelFacts:
    """shards: per file, [(tensor_name, size_bytes, n_elements)]."""
    total, seen, hc_q8, per_layer = _scan_shards(shards)
    if not layers or not experts or not total["exps"]:
        raise ModelError(f"not an MoE GGUF set (layers={layers}, experts={experts}, expert bytes={total['exps']}): {list(paths)}")
    blob = -(-max(per_layer.values()) // int(experts)) if per_layer else None
    head_b = total["head"] if "head" in seen else None
    return ModelFacts(int(layers), int(experts), total["exps"], total["dense"], int(hidden) if hidden else None, blob, tuple(paths), head_b,
                      total["ple"], total["host"], total["hc"], hc_q8)


def read_model_facts(paths: Sequence[str]) -> ModelFacts:
    """Layers, routed experts per layer, tensor-class byte totals from the GGUF files of one model (every shard plus any
    separate PLE file).  Refuses (ModelError) when the architecture keys are absent: nothing is defaulted."""
    layers = experts = hidden = None
    shards = []
    for p in paths:
        kv, sizes = scan_gguf(p)
        arch = kv.get("general.architecture")
        if arch:
            layers = kv.get(f"{arch}.block_count", layers)
            experts = kv.get(f"{arch}.expert_count", experts)
            hidden = kv.get(f"{arch}.embedding_length", hidden)
        shards.append(sizes)
    return facts_from_tensors(layers, experts, hidden, shards, paths)
