// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
#pragma once
// A fake helper cache for the tier-round tests (blob_swap_test.cpp, channel_blob_swap_test.cpp): the RoundCache that
// strata-ext/include/strata/seabraid/tier_round.hpp documents, over plain vectors, with fault hooks.  Host only, no Strata.
// It models exactly what the real cache promises and no more: a swap writes the slot of the displaced expert, its residency
// moves only at sync, a failed sync quarantines the slots written, a read-back compares the slot to the bytes.
#include "strata/seabraid/tier_round.hpp"

#include <cstring>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace seabraid::test {

using Cell = std::pair<int32_t, int32_t>;

/// What the pack holds for (layer, expert): the bytes a plain swap loads.
inline uint8_t pack_byte(int32_t l, int32_t e, int64_t i) { return (uint8_t) (31 * l + 7 * e + 3 * i + 1); }

struct FakeCache {
    int64_t bytes = 64;                       ///< blob size of every layer
    std::vector<std::vector<uint8_t>> slots;  ///< slot storage
    std::map<Cell, int> slot_of;              ///< cell -> slot
    std::set<Cell> quarantined;
    std::vector<std::pair<TierWireOp, int>> queued;   ///< swaps waiting for sync (op, slot)
    std::vector<Cell> written;
    // fault hooks
    bool sync_fails = false;
    int corrupt_expert_new = -1;              ///< after a good sync, flip a byte in the slot that now holds this expert
    int applies = 0, applies_from_blob = 0;
    std::function<void()> during_apply;      ///< a peer writing the shared channel while the helper is mid-round

    void put(int32_t l, int32_t e) {          ///< startup fill from the pack
        slots.emplace_back((size_t) bytes);
        for (int64_t i = 0; i < bytes; ++i) slots.back()[(size_t) i] = pack_byte(l, e, i);
        slot_of[{l, e}] = (int) slots.size() - 1;
    }
    bool apply(const TierWireOp& op, const uint8_t* blob, std::string& err) {
        ++applies;
        if (during_apply) during_apply();
        if (blob) ++applies_from_blob;
        if (op.kind != kTierSwap) { err = "fake: only swaps"; return false; }
        const auto it = slot_of.find({op.layer_old, op.expert_old});
        if (it == slot_of.end()) { err = "fake: the displaced expert is not resident"; return false; }
        if (quarantined.count({op.layer_old, op.expert_old})) { err = "fake: quarantined slot"; return false; }
        auto& s = slots[(size_t) it->second];
        for (int64_t i = 0; i < bytes; ++i) s[(size_t) i] = blob ? blob[i] : pack_byte(op.layer_new, op.expert_new, i);
        queued.emplace_back(op, it->second);
        written.emplace_back(op.layer_old, op.expert_old);
        return true;
    }
    bool sync(std::string& err) {
        if (sync_fails) {
            for (const Cell& c : written) quarantined.insert(c);
            queued.clear(); written.clear();
            err = "fake: sync failed";
            return false;
        }
        for (const auto& q : queued) {
            slot_of.erase({q.first.layer_old, q.first.expert_old});
            slot_of[{q.first.layer_new, q.first.expert_new}] = q.second;
            if (q.first.expert_new == corrupt_expert_new) slots[(size_t) q.second][0] ^= 0xFF;
        }
        queued.clear(); written.clear();
        return true;
    }
    bool suspect(int32_t l, int32_t e) const { return quarantined.count({l, e}) != 0; }
    bool holds(int32_t l, int32_t e) const { return slot_of.count({l, e}) != 0 && !suspect(l, e); }
    bool verify(int32_t l, int32_t e, const uint8_t* blob, std::string& err) {
        const auto it = slot_of.find({l, e});
        if (it == slot_of.end()) { err = "fake: not resident"; return false; }
        if (std::memcmp(slots[(size_t) it->second].data(), blob, (size_t) bytes) != 0) {
            quarantined.insert({l, e});
            err = "fake: slot differs from the blob";
            return false;
        }
        return true;
    }
    int64_t blob_bytes(int32_t) const { return bytes; }
    void totals(Header& h) { h.resident = (int64_t) slot_of.size(); }
    int64_t resident() const { return (int64_t) slot_of.size(); }
};

/// Every failure `run_tier_round` reports, in order.
struct Reports {
    std::vector<std::pair<int, bool>> seen;   ///< (op, first)
    std::vector<std::string> why;
    void operator()(int op, bool first, const std::string& w) { seen.emplace_back(op, first); why.push_back(w); }
};

inline TierWireOp swap_op(uint8_t kind, int32_t l, int32_t eo, int32_t en, uint16_t slot = 0) {
    TierWireOp o{};
    o.kind = kind; o.layer_old = l; o.expert_old = eo; o.layer_new = l; o.expert_new = en; o.blob_slot = slot;
    return o;
}

}  // namespace seabraid::test
