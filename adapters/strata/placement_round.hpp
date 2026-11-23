// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// The driver's round bookkeeping: counters, the pure table comparisons and the round line.
// OWNER: placement_round_log.cpp. CONTRACT: pure functions of their arguments (no ABI call, no state); only the
//   driver and its test include this. SIZE BUDGET: 80 lines. FAILURE CONTRACT: nothing here can fail or throw.
// CHECKER: placement_driver_test exercises every function through the driver's rounds.
#pragma once

#include "placement_trace.hpp"

#include "seabraid/placement.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace seabraid::strata_adapter {

// Per-round counters and timings.
struct RoundStats {
    uint64_t round = 0, epoch = 0, qmask = 0, table_hash = 0;
    uint64_t plan_us = 0, fence_us = 0, stage_us = 0, replace_us = 0, publish_us = 0, follow_us = 0;
    uint32_t planned = 0, committed = 0, rejected = 0, restored = 0, adopted = 0, spares = 0, deferred = 0;
    uint32_t quarantined = 0;    // QUARANTINED outcomes
    uint32_t skipped = 0;        // SKIPPED climb hops (policy v2 max_hops_per_round > 1)
    uint32_t file_asked = 0, file_acked = 0;   // by-reference writes the library asked for / the rung ACKed (a REFUSED ask took the byte path)
    bool qual = false;           // the qualification heat bias steered this round
    double helper_share = 0.0;   // heat-weighted fraction of the heat on helper rungs
    bool worked = false;         // a round with work or a state change writes a round line
};
void log_round(const RoundStats& stats) noexcept;
// What a log call adds to the trace record: the round, and the per-outcome stage and write times (may be null).
struct MoveContext {
    uint64_t round = 0, epoch = 0;
    const std::vector<trace::PairTiming>* timing = nullptr;
    bool verify_writes = true;   // PlacementDriverConfig::verify_writes: whether helper writes are read back
};
// One INFO line per COMMITTED hop: "placement move L/E r->r src=card|fallback bytes=N verified=V". V is 1 only when both
// writes of the hop were read back by their writer (the lead's verify_slot; a helper's kTierSwapBlob read-back, which
// PlacementDriverConfig::verify_writes can switch off); a hop whose lower rung is the fallback, or whose helper write
// was not read back, prints verified=0 (the fallback's copy has no read-back), then
// "placement evict L/D u->l dst=card|fallback" for the expert it pushed down. The move line's format is parsed by
// ov_parse.py and placement-qual and must not change. Each also writes a trace "move" record when the trace is on.
void log_moves(const std::vector<seabraid_placement_outcome_v1>& outcomes, const std::vector<size_t>& slot_bytes,
               uint8_t fallback, const MoveContext& context = {}) noexcept;
// One INFO line per outcome that did not commit: "placement fail L/E r->r kind=... stage=... result=...".
void log_failures(const std::vector<seabraid_placement_outcome_v1>& outcomes, const std::vector<size_t>& slot_bytes,
                  uint8_t fallback, const MoveContext& context = {}) noexcept;
// The engine's RAM copy follows a committed hop only across the lead <-> helper 1 boundary (upper_rung 0): that is the
// only boundary that changes the lead's complement.  A fallback-boundary hop is the FallbackRung's own RAM decision.
bool ram_follows(const seabraid_placement_outcome_v1& outcome) noexcept;
void tally_outcomes(const std::vector<seabraid_placement_outcome_v1>& outcomes, RoundStats& stats) noexcept;
double helper_share(const uint8_t* executors, const float* heat, size_t cells, uint8_t fallback) noexcept;

struct Adoption {
    uint32_t moved = 0;
    int conflict = -1;   // the helper rung whose cell the host moved, or -1
};
// Fills `next` with the owners after the host's own lead moves. A moved cell that a helper held is a conflict.
Adoption scan_adoption(const std::vector<uint8_t>& published, const std::vector<uint8_t>& lead, uint8_t fallback,
                       std::vector<uint8_t>& next) noexcept;
// Cells where the table's lead and the host's lead differ. A size mismatch counts every cell.
size_t lead_mismatches(const std::vector<uint8_t>& executors, const std::vector<uint8_t>& lead) noexcept;

struct HelperCheck {
    size_t missing = 0;   // the table says `rung`, the helper does not hold it
    size_t spares = 0;    // the helper holds a cell the table gives elsewhere: harmless, counted
};
HelperCheck check_helper(const std::vector<uint8_t>& executors, const std::vector<uint8_t>& held,
                         uint8_t rung) noexcept;

}  // namespace seabraid::strata_adapter
