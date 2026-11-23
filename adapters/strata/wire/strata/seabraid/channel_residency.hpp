// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
#pragma once
// seabraid channel, the tier-residency publish/check pair (helper publishes what VRAM really holds; main checks it against
// the ops the round reported applied).  WHY a file of its own: these are free functions over the layout's TierWireOp and
// need nothing of Channel, so they sit apart to keep channel.hpp under the size split point.  Callers keep including
// "strata/seabraid/channel.hpp", which includes this; including this alone also works.
#include "strata/seabraid/channel_layout.hpp"

namespace seabraid {

/// True for all three swap kinds (pack-read, v8 blob, v9 file): each displaces (layer_old, expert_old).  One predicate, so a blob or file swap can never be
/// replayed as a load (which would leave the displaced cell marked resident).
inline constexpr bool tier_op_displaces(const TierWireOp& op) {
    return op.kind == kTierSwap || op.kind == kTierSwapBlob || op.kind == kTierSwapFile;
}

/// Helper: after a tier round's ops are applied AND synced, write the TRUE residency of every cell the round touched
/// into the channel's held[] (the bitmap main copies as its routing view).  `holds(l, e)` is the helper cache's own
/// answer, so a failed op, a partly applied round or a failed sync all publish what VRAM really holds, not what was
/// asked for.  Must run before tier_ack: the ack's release-store is what makes these writes visible to main.
/// (Without this, held[] kept the plan-time view: main routed swapped-out experts to a helper that skips experts it
/// does not hold -- their rows silently dropped -- and re-proposed swapping them out again. D7 arms, 2026-11-06.)
template <class HoldsFn>
inline void publish_tier_residency(uint8_t* held, int64_t layers, int64_t experts, const TierWireOp* ops, int n,
                                   HoldsFn&& holds) {
    auto put = [&](int32_t l, int32_t e) {
        if (l < 0 || e < 0 || l >= layers || e >= experts) return;
        held[(size_t) l * (size_t) experts + (size_t) e] = holds(l, e) ? 1 : 0;
    };
    for (int i = 0; i < n; ++i) {
        if (tier_op_displaces(ops[i])) put(ops[i].layer_old, ops[i].expert_old);
        put(ops[i].layer_new, ops[i].expert_new);
    }
}

namespace detail {
/// Where a round's APPLIED ops leave each cell they touch, replayed in order (a later op on a cell overwrites an
/// earlier one).  WHY a fixed array: a round has at most kTierMaxOps ops of at most two cells each, so the check
/// allocates nothing on the engine thread.
struct ResidencyReplay {
    static constexpr int kMax = 2 * kTierMaxOps;
    int64_t cell[kMax];
    uint8_t want[kMax];
    int m = 0;

    void set(int64_t layers, int64_t experts, int32_t l, int32_t e, uint8_t v) {
        if (l < 0 || e < 0 || l >= layers || e >= experts) return;
        const int64_t c = (int64_t) l * experts + e;
        for (int j = 0; j < m; ++j) if (cell[j] == c) { want[j] = v; return; }
        if (m < kMax) { cell[m] = c; want[m] = v; ++m; }
    }
};

inline std::string residency_contradiction(int64_t cell, int64_t experts, bool want, bool held) {
    return "layer " + std::to_string(cell / experts) + " expert " + std::to_string(cell % experts) +
           ": the round applied it as " + (want ? "resident" : "displaced") + " but held[] says " +
           (held ? "resident" : "not resident");
}
}  // namespace detail

/// Main: the held[] a round published must agree with the ops it reported APPLIED (replayed in order, so a round
/// that moves an expert twice is judged by where it ends up).  A contradiction means the helper's bitmap is stale,
/// and routing from it silently drops experts.  Returns false and names the first contradicting cell.
inline bool tier_residency_consistent(const uint8_t* held, int64_t layers, int64_t experts, const TierWireOp* ops,
                                      const uint8_t* res, int n, std::string& why) {
    detail::ResidencyReplay replay;
    for (int i = 0; i < n; ++i) {
        if (res[i] != kTierApplied) continue;
        if (tier_op_displaces(ops[i])) replay.set(layers, experts, ops[i].layer_old, ops[i].expert_old, 0);
        replay.set(layers, experts, ops[i].layer_new, ops[i].expert_new, 1);
    }
    for (int j = 0; j < replay.m; ++j) {
        const bool want = replay.want[j] != 0, has = held[replay.cell[j]] != 0;
        if (has != want) {
            why = detail::residency_contradiction(replay.cell[j], experts, want, has);
            return false;
        }
    }
    return true;
}

} // namespace seabraid
