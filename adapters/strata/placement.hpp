// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "placement_trace.hpp"

#include "seabraid/placement.h"

#include <string>
#include <vector>

namespace strata::core {
class RungIO;
}

namespace seabraid::strata_adapter {

struct PlacementRungs {
    const std::vector<strata::core::RungIO*>& cards;
    const std::vector<size_t>& layer_slot_bytes;
    // The fallback rung (rung == cards.size(): the engine's RAM copy and the pack on NVMe), or nullptr when the handle
    // never plans a fallback swap (policy v2 fallback_boundary 0).  WHY a separate field and not a last card: the
    // fallback is not a card (the core never quarantines it and shape checks count cards), so it cannot hide among them.
    strata::core::RungIO* fallback = nullptr;
};

// The provider's rung mask (placement.h provider_v2.from_fallback_rungs): bit r is set when card r reports reads_files(), i.e. its
// replace_from_fallback may take the promoted expert's bytes from the model file.  Never the fallback and never a rung that is not a
// card.  Computed before EVERY apply, so a helper that turned file moves off (a bad sum) leaves the mask at once.
uint64_t from_fallback_mask(const PlacementRungs& rungs) noexcept;

// The card for `rung`, the fallback for rung == cards.size() when one is bound, or nullptr (logged) when the runtime
// names a rung the engine never bound. Exposed so the fail-closed bound is directly testable.
strata::core::RungIO* checked_rung(const PlacementRungs& rungs, uint8_t rung) noexcept;

struct PlacementRound {
    // The engine sizes this staging table to the layout's owner count.
    std::vector<uint8_t> executors;
    seabraid_placement_snapshot_v1 snapshot{};
    std::vector<seabraid_placement_outcome_v1> outcomes;
    std::string first_io_error;
    // Time spent inside the cards' stage and replace calls during the last apply, summed over its pairs.
    uint64_t stage_us = 0;
    uint64_t replace_us = 0;
    // By-reference writes of the last apply (provider_v2): how many times the library asked a masked rung to load an expert from the
    // model file, and how many of those it ACKed.  A REFUSED ask is not a failure (the byte path followed in the same swap).
    uint32_t file_asked = 0, file_acked = 0;
    // Every provider call of the last apply, in order; filled only while the trace is on (placement_trace.hpp).
    std::vector<trace::CallSample> calls;
};

// Fills round.executors and round.snapshot from the placement. Shared with the driver so the header setup that the
// ABI requires (version and size) has one owner.
int32_t take_snapshot(seabraid_placement* placement, PlacementRound& round) noexcept;

// The engine thread must hold admission closed and drain every affected queue
// before calling. Each RungIO must report uncertain completion as timeout
// and finish accessing its borrowed buffers before returning or throwing.
// Card bindings and verified slot extents must stay fixed. The staging table
// must be separate from the currently published route.
// The caller may publish the result only on SEABRAID_OK and while still fenced.
// A quarantined card must be excluded from all paths that read its slots.
// On a bridge error the engine must keep admission closed until the verified
// executor table is republished or the engine has stopped.
int32_t apply_placement_at_fence(seabraid_placement* placement,
                                 const seabraid_placement_ticket_v1& ticket,
                                 const PlacementRungs& rungs,
                                 PlacementRound& round) noexcept;

}  // namespace seabraid::strata_adapter
