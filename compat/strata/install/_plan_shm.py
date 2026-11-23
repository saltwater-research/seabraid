# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Size of one helper's /dev/shm channel, mirrored from the wire's Layout::of (adapters/strata/wire/strata/seabraid/channel_layout.hpp).  UNTESTED-ON-HARDWARE.

OWNER     : the constants of the shm channel layout and the one formula that totals them (split out of plan.py for size).
CONTRACT  : channel_bytes(layers, experts, hidden, blob_max) -> int | None (None = a needed fact is unknown).  stdlib only.
FAILURE   : unknown input -> None, never a guess; plan.py turns None into a refusal.
CHECKER   : tests/install/test_plan.py pins the v3/v4 formula to the 26,038,272 B / 27,672,576 B measured on the test system, and the v6
            formula + CHANNEL_HEADER_BYTES to tests/channel_v6_golden.txt, which seabraid_layout_print (the real compiler, the real
            Layout::of) writes and tests/channel_v6_test.cpp verifies from the C++ side: both sides read the SAME file.
"""
from __future__ import annotations

from typing import Optional

# ---- the helper shm channel's size, mirrored from adapters/strata/wire/strata/seabraid/channel_layout.hpp (Layout::of).
# Every constant below names its C++ symbol; tests/install/test_plan_shm_citations.py reads those symbols and fails on a drift.
PAGE = 4096                       # Layout::of up(): rounds the running offset to 4 KiB
# v7 sizeof(Header) (v7 = the D4 prefill fields appended to v6's 2240), MEASURED (g++ and nvcc, seabraid_layout_print) on the
# committed header.  Header size cannot tell one version from another (`version` does), and it does not move any region: every
# header under a page starts the first region at PAGE, so channel_bytes() below is the same for v4 .. v7 (the optional D4 prefill
# region is appended AFTER the total and exists only when the helper is started with --prefill-tokens).
CHANNEL_HEADER_BYTES = 2752
LAYER_SPEC_BYTES = 112            # expert_spec.hpp: static_assert(sizeof(LayerSpec) == 112)
SPEC_MAX_FILES = 64               # expert_spec.hpp: kSpecMaxFiles
SPEC_PATH_MAX = 512               # expert_spec.hpp: kSpecPathMax
KBLOBBATCH = 16                   # channel_layout.hpp: kBlobBatch
KTIERMAXOPS = 64                  # channel_layout.hpp: kTierMaxOps
TIER_WIRE_OP_BYTES = 20           # channel_layout.hpp: static_assert(sizeof(TierWireOp) == 20)
# kWindowMaxT (strata/kernels/window_limits.hpp) is set by the Strata CMake cache variable STRATA_WINDOW_MAX_T, whose "auto"
# default is per backend (Strata CMakeLists.txt:109): HIP 16, CUDA 8.  main and helper MUST be built with the same value
# (Channel::connect compares window_max_t: channel.hpp, Channel::geometry_matches), so the plan resolves the MAIN backend's default here, passes it
# explicitly (never "auto"), and every helper build gets the main's value.  The "window up to 8 tokens" in the engine log is
# the DFlash2 drafter's window, a different thing.  PlanOptions.window_max_t overrides for main and helpers alike.
BACKEND_WINDOW_MAX_T = {"rocm": 16, "cuda": 8}      # keyed by Build.kind
CAP_PAIRS_PER_TOKEN = 10          # strata-ext/src/helper.cpp admit(): `cap = kWindowMaxT * legacy_picks` (named channel, top-10 pairs)


def channel_bytes(layers: int, experts: int, hidden: Optional[int], blob_max: Optional[int], window_max_t: int,
                  tier_ops: bool = True, version: int = 6, cap_pairs_per_token: int = CAP_PAIRS_PER_TOKEN) -> Optional[int]:
    """Layout::of(hidden, cap, L, E, blob_max).total: each region starts on a PAGE boundary.
    version 6 (default; 4 or 5 = the v4 layout) adds the weights region (cap floats) and the spec region (layers * 112 B + 64 paths * 512 B);
    version 4 is the Qwen-era layout the test system measured; tier_ops=False with version 4 gives the v3 layout.
    cap = window_max_t * cap_pairs_per_token: 10 for the Qwen helper (top-10 routing pairs per token); the golden file's unit-test geometry uses 8.
    A Qwen caller that does not know blob_max still gets None, never a guess."""
    if not (hidden and window_max_t > 0 and blob_max):
        return None
    cap = window_max_t * cap_pairs_per_token
    up = lambda v: -(-v // PAGE) * PAGE  # noqa: E731
    at = up(CHANNEL_HEADER_BYTES)
    regions = [layers * experts * 2 * 4, layers * experts, cap * 4, cap * hidden * 4, cap * hidden * 4,
               KBLOBBATCH * blob_max, KTIERMAXOPS * TIER_WIRE_OP_BYTES * tier_ops]            # plan, held, ids, x, out, blobs, tier ops
    if version >= 6:
        regions += [cap * 4, layers * LAYER_SPEC_BYTES + SPEC_MAX_FILES * SPEC_PATH_MAX]    # weights, spec
    for n in regions:
        at = up(at + n)
    return at
