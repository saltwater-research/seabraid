# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""PlanOptions and the main/helper VRAM reserve formula (split out of plan.py for size).  UNTESTED-ON-HARDWARE.

OWNER     : every tunable of the planner and the one formula for what a card holds back from expert placement.
CONTRACT  : PlanOptions (frozen defaults);  main_reserve(opts, dense, head, ctx) -> int;  helper_reserve(opts) -> int.
            stdlib only; imports nothing from plan.
FAILURE   : pure arithmetic.  An unknown head size falls back to the documented default, never to zero.
CHECKER   : tests/install/test_plan.py reproduces the test system's measured IQ2_XS split within a stated tolerance.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

GIB = 1 << 30
MIB = 1 << 20


@dataclass(frozen=True)
class PlanOptions:
    # ---- the MAIN card's non-expert VRAM.  Measured on the R9700, IQ2_XS, 131072 ctx, int8 KV (2026-11-06/07): native
    # projections 2,112 MiB + pack weights 266 MiB (both inside the GGUF dense figure, with hc_* at Q8_0) + Q5_K head 322 MiB + DFlash2
    # drafter 550 MiB + KV/workspace ~2.5 GiB + ~0.8 GiB margin; Strata's auto expert cache then got 18,733 slots = 25.17 of 31.86 GiB.
    head_bytes: int = 322 * MIB                # output head, used only when the GGUF has no output.weight to measure
    drafter_bytes: int = 550 * MIB             # DFlash2 drafter; set 0 when no drafter pack is used
    # KV + workspace per context token.  A per-token constant, not a head-count formula: ModelFacts has no head counts and the
    # model is a hybrid (GDN/QSA).  Source: 2.5 GiB measured at 131072 tokens.
    kv_workspace_bytes_per_token: int = int(2.5 * GIB) // 131072
    margin_bytes: int = int(0.8 * GIB)
    hc_q8: Optional[bool] = None               # None = Strata's per-backend default (hc_q8_on_card); True/False overrides
    # ---- a HELPER holds experts only, plus its runtime context and staging: ~0.5-0.6 GiB measured (the 3060 held 10.57 of 12.00 GiB).
    helper_reserve_bytes: int = int(0.6 * GIB)
    ram_headroom_floor_bytes: int = 8 * GIB    # host keeps >= max(this, 25% of total RAM)
    ram_headroom_fraction: float = 0.25
    window_max_t: Optional[int] = None         # None = the MAIN backend's Strata default (HIP 16, CUDA 8); applies to main and helpers
    # GTT must exceed the AMD RAM tier by this.  5 GiB, not 6: the verified IQ3_S run (README: RAM tier 10 GiB, ~96-99 tok/s,
    # 10/10, 0 hangs, KL 0.0098) had a 15.13 GiB GTT cap = 5.13 GiB of headroom.
    gtt_margin_bytes: int = 5 * GIB
    # None = the mode default (CONTEXT_ALL_VRAM / CONTEXT_TIERED, plan.py); an explicit value overrides.
    assumed_max_context: Optional[int] = None
    mlock: bool = False                        # an mlock option is requested: only a note today
    model_move_headroom: float = 0.10          # advice: move the model to an NVMe fs with free >= size * 1.1
    # ---- card ranking (N-CARD-LADDER 2).  streams = the configured concurrency (picks the lead-job regime: 1 or >= 4);
    # main_card = the USER's choice of card 1 (vendor, index), which beats every ranking; admission = apply R3 to cards 2..N
    # (off by default: it removes cards from a plan, and whether a slow card should stay as a blob store is a product call);
    # floors = {(vendor, index): _plan_rank.Terms} measured per-layer terms for R3b (None = R3b is reported as not evaluated).
    streams: int = 1
    main_card: Optional[Tuple[str, int]] = None
    admission: bool = False
    floors: Optional[Mapping] = None


def main_reserve(o: PlanOptions, dense: int, head: Optional[int], ctx: int) -> int:
    """Everything the main card holds besides experts: dense (hc_* included) + head + drafter + KV/workspace(ctx) + margin."""
    return dense + (head or o.head_bytes) + o.drafter_bytes + ctx * o.kv_workspace_bytes_per_token + o.margin_bytes


def helper_reserve(o: PlanOptions) -> int:
    """A helper holds NO dense weights, KV, drafter or workspace: only its runtime context and staging."""
    return o.helper_reserve_bytes


# Strata src/core/native_dense.cpp:45-77: HC_Q8 defaults ON for gfx12 (RDNA4) and gfx1151 under HIP, OFF on CUDA and elsewhere.
HC_Q8_DEFAULT_ARCHS = {"rocm": ("gfx12", "gfx1151")}  # seabraid-generic: vendor-wide taxonomy, not this machine


def hc_q8_on_card(o: PlanOptions, kind: str, arch: str) -> bool:
    return o.hc_q8 if o.hc_q8 is not None else arch.startswith(HC_Q8_DEFAULT_ARCHS.get(kind, ()))


def dense_on_card(o: PlanOptions, hc_stored: int, hc_q8: int, dense: int, kind: str, arch: str) -> int:
    """Non-expert tensors on the main card: dense plus the hc_* tensors at Q8_0 when the main backend converts them."""
    return dense + (hc_q8 if hc_q8_on_card(o, kind, arch) else hc_stored)
