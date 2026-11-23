// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
#pragma once
// seabraid D8 GPU half: times the helper's real expert path (RemoteExperts::begin/finish) on this card.
// Host-side maths, ordering and the stderr line live in include/strata/seabraid/card_bench.hpp.
#include "strata/seabraid/card_bench.hpp"
#include "strata/core/expert_source.hpp"

#include <cstdint>
#include <string>

namespace seabraid {

/// SEABRAID_STRATA_CARD_BENCH=1 (opt-in; anything else, or unset, is off).
bool card_bench_enabled();

/// Measure `device` and fill `out` (vendor/name/free VRAM always; the regimes stay NaN unless measured).
/// Never throws and never fails the caller: any problem leaves the regime unmeasured with out.note saying why.
/// Loads a handful of real experts from `src`, times them, and releases everything, so call it before the helper
/// reports kHelperUp and before the engine's plan arrives.
void run_card_bench(int device, strata::core::ExpertSource& src, int64_t layers, int64_t experts,
                    uint64_t free_vram_bytes, const std::string& device_name, CardBench& out);

}  // namespace seabraid
