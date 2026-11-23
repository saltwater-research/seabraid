// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// tests/blob_swap_test.cpp - the helper's tier round with the blob swap (wire v8 kTierSwapBlob), over a fake cache.  CPU only.
//
// What each block guards (and what it would look like if it broke):
//   round trip   : the bytes in the channel's blob slot, not the pack's, end up in the displaced expert's slot, and held[]
//                  moves the old expert out and the new one in (a blob swap judged as a plain swap).
//   refusals     : a bad slot index, a cross-layer swap, an oversize blob, a stray blob_slot -> kTierRejected, slot untouched
//                  (main fails open); a bad op never reaches the cache.
//   read-back    : a slot that does not hold the bytes -> kTierFailed + quarantined + published as holding nothing (main closes
//                  the card).  Without the read-back the round would ACK bytes nobody checked.
//   accounting   : swaps keep the resident count and held[]'s popcount, whatever the round did or refused.
// House rule: each guard was sabotaged once and watched to FAIL (tests/SABOTAGE.md).
#include "tier_round_fake.hpp"

#include <cstdio>

using namespace seabraid;
using namespace seabraid::test;

static int g_fail = 0;
#define CHECK(cond, what)                                                                       \
    do {                                                                                        \
        if (!(cond)) { std::printf("FAIL: %s (%s:%d)\n", (what), __FILE__, __LINE__); ++g_fail; } \
    } while (0)

static constexpr int64_t L = 2, E = 8, BYTES = 64;

// A helper holding experts 0..3 of both layers; the channel's blob slots carry `fill`-patterned blobs, one per slot.
struct Rig {
    FakeCache cache;
    std::vector<uint8_t> held = std::vector<uint8_t>((size_t) (L * E), 0);
    std::vector<std::vector<uint8_t>> blobs;
    TierWireOp ops[kTierMaxOps];
    uint8_t res[kTierMaxOps];
    int n = 0;
    int64_t blob_max = BYTES;
    Rig() {
        cache.bytes = BYTES;
        for (int32_t l = 0; l < L; ++l)
            for (int32_t e = 0; e < 4; ++e) { cache.put(l, e); held[(size_t) (l * E + e)] = 1; }
        for (int s = 0; s < kBlobBatch; ++s) blobs.emplace_back((size_t) BYTES, (uint8_t) (0xA0 + s));
    }
    void add(const TierWireOp& o) { ops[n] = o; res[n] = kTierPending; ++n; }
    int count_held() const { int c = 0; for (uint8_t v : held) c += v != 0; return c; }
    bool run(Reports& rep) {
        const detail::RoundIO io{ops, n, res, held.data(), L, E, blob_max};
        return run_tier_round(cache, io, [&](int slot) { return (const uint8_t*) blobs[(size_t) slot].data(); }, rep);
    }
    bool slot_is_blob(int32_t l, int32_t e, int s) const {
        const auto it = cache.slot_of.find({l, e});
        return it != cache.slot_of.end() && std::memcmp(cache.slots[(size_t) it->second].data(), blobs[(size_t) s].data(), BYTES) == 0;
    }
    bool held_at(int32_t l, int32_t e) const { return held[(size_t) (l * E + e)] != 0; }
};

static void t_round_trip() {
    Rig r;
    r.add(swap_op(kTierSwapBlob, 1, 0, 6, 3));   // expert 6 replaces 0 in layer 1, bytes from blob slot 3
    Reports rep;
    CHECK(r.run(rep), "a good blob swap is a good round");
    CHECK(r.res[0] == kTierApplied && rep.seen.empty(), "applied, nothing reported");
    CHECK(r.cache.applies_from_blob == 1, "the cache was handed the channel's bytes");
    CHECK(r.slot_is_blob(1, 6, 3), "the slot holds the BLOB's bytes, not the pack's");
    CHECK(r.held_at(1, 6) && !r.held_at(1, 0), "held[]: the new expert in, the displaced one out (judged as a swap)");
    CHECK(r.cache.resident() == 8 && r.count_held() == 8, "slot accounting: a swap keeps the count");
}

static void t_plain_swap_unchanged() {
    Rig r;
    r.add(swap_op(kTierSwap, 0, 1, 5));
    Reports rep;
    CHECK(r.run(rep) && r.res[0] == kTierApplied && r.cache.applies_from_blob == 0, "a plain swap still loads from the pack");
    CHECK(r.held_at(0, 5) && !r.held_at(0, 1), "and publishes the same way");
}

static void t_refusals_touch_nothing() {
    Rig r;
    r.add(swap_op(kTierSwapBlob, 0, 0, 6, kBlobBatch));      // slot index one past the end
    TierWireOp cross = swap_op(kTierSwapBlob, 0, 0, 6, 1); cross.layer_new = 1;
    r.add(cross);                                            // cross-layer
    r.add(swap_op(kTierSwapBlob, 0, 2, 2, 1));               // an expert with itself
    r.add(swap_op(kTierSwap, 0, 3, 7, 5));                   // a plain op carrying a stray slot
    r.add(swap_op(kTierSwapBlob, 0, 6, 7, 1));               // displaced expert not resident
    Reports rep;
    CHECK(!r.run(rep), "the round is not ok");
    for (int i = 0; i < 5; ++i) CHECK(r.res[i] == kTierRejected, "refused: kTierRejected (untouched), never kTierFailed");
    CHECK(r.cache.applies == 1, "only the last op reached the cache (the first four failed validation)");
    CHECK(rep.seen.size() == 5 && rep.seen[0].second && !rep.seen[1].second, "every failure is reported, the first is marked first");
    CHECK(r.cache.quarantined.empty() && r.cache.resident() == 8 && r.count_held() == 8, "no slot quarantined, no count change");
    CHECK(r.slot_is_blob(0, 0, 0) == false && r.held_at(0, 0), "the displaced expert is still there");
}

static void t_oversize_blob_refused() {
    Rig r;
    r.blob_max = BYTES - 1;   // the channel's slots are smaller than a blob of this layer
    r.add(swap_op(kTierSwapBlob, 0, 0, 6, 0));
    Reports rep;
    CHECK(!r.run(rep) && r.res[0] == kTierRejected && r.cache.applies == 0, "a blob that does not fit the slot is refused unread");
    CHECK(!rep.why.empty() && rep.why[0].find("blob slots hold") != std::string::npos, "and the reason names the sizes");
}

static void t_readback_mismatch() {
    Rig r;
    r.cache.corrupt_expert_new = 6;   // the slot ends up not holding the bytes
    r.add(swap_op(kTierSwapBlob, 0, 0, 6, 2));
    r.add(swap_op(kTierSwapBlob, 0, 1, 7, 4));   // an innocent neighbour in the same round
    Reports rep;
    CHECK(!r.run(rep), "a mismatch makes the round not ok (never ACKed as good)");
    CHECK(r.res[0] == kTierFailed, "the op whose slot did not read back is kTierFailed");
    CHECK(r.res[1] == kTierApplied && r.slot_is_blob(0, 7, 4), "the neighbour was read back and stays applied");
    CHECK(r.cache.suspect(0, 6) && !r.held_at(0, 6) && !r.held_at(0, 0), "quarantined, published as holding nothing, displaced one gone");
    CHECK(!rep.why.empty() && rep.why[0].find("read-back") != std::string::npos && rep.seen[0] == std::make_pair(0, true), "reported as the first failure");
}

static void t_sync_failure_is_failed_not_applied() {
    Rig r;
    r.cache.sync_fails = true;
    r.add(swap_op(kTierSwapBlob, 1, 2, 5, 0));
    Reports rep;
    CHECK(!r.run(rep) && r.res[0] == kTierFailed, "a copy that did not complete is FAILED");
    CHECK(r.cache.suspect(1, 2) && !r.held_at(1, 2) && !r.held_at(1, 5), "its slot is quarantined and nothing is published for it");
    CHECK(rep.seen.size() == 1 && rep.seen[0].first == -1, "the round-level failure is reported once, as op -1");
}

static void t_two_ops_read_back_each() {
    Rig r;
    r.add(swap_op(kTierSwapBlob, 0, 0, 6, 1));
    r.add(swap_op(kTierSwapBlob, 1, 2, 7, 2));
    Reports rep;
    CHECK(r.run(rep) && r.res[0] == kTierApplied && r.res[1] == kTierApplied, "two blob swaps in one round");
    CHECK(r.slot_is_blob(0, 6, 1) && r.slot_is_blob(1, 7, 2), "each slot holds its own blob");
    CHECK(r.count_held() == 8 && r.cache.resident() == 8, "count constant");
}

static void t_as_swap_view() {
    const TierWireOp b = swap_op(kTierSwapBlob, 1, 2, 3, 9);
    const TierWireOp p = as_swap(b);
    CHECK(p.kind == kTierSwap && p.blob_slot == 0 && p.layer_old == 1 && p.expert_old == 2 && p.expert_new == 3, "blob swap -> plain swap, cells kept");
    const TierWireOp l = swap_op(kTierLoad, 1, -1, 3);
    CHECK(as_swap(l).kind == kTierLoad, "a load is untouched");
}

// The residency promise after EVERY round: whatever the helper reported applied, held[] says resident (tier_residency_consistent).
static bool consistent(const Rig& r) {
    std::string why;
    return tier_residency_consistent(r.held.data(), L, E, r.ops, r.res, r.n, why);
}

// FZFIX KQ (fuzz 0x30c5dbe6fb647ddd): an expert the cache holds QUARANTINED must not be re-admitted: the new blob would get no
// read-back (a quarantined cell is not held) yet be reported kTierApplied.
static void t_kq_quarantined_expert_not_readmitted() {
    Rig r;
    r.cache.corrupt_expert_new = 5;
    r.add(swap_op(kTierSwapBlob, 1, 2, 5, 0));
    Reports rep0;
    CHECK(!r.run(rep0) && r.res[0] == kTierFailed && r.cache.suspect(1, 5), "round 0: expert 5 fails its read-back and is quarantined");
    CHECK(consistent(r), "round 0 is residency-consistent");
    r.cache.corrupt_expert_new = -1;
    r.n = 0;
    r.add(swap_op(kTierSwapBlob, 1, 3, 5, 1));   // the same expert again, now into expert 3's slot
    Reports rep1;
    CHECK(!r.run(rep1) && r.res[0] == kTierRejected, "round 1: re-admitting a quarantined expert is refused, not applied");
    CHECK(r.cache.applies == 1, "round 1 never reached the cache (no slot write)");
    CHECK(r.held_at(1, 3) && !r.held_at(1, 5), "expert 3 still resident, 5 still not");
    CHECK(!rep1.why.empty() && rep1.why[0].find("quarantined") != std::string::npos, "the reason says quarantined");
    CHECK(consistent(r), "round 1 is residency-consistent");
}

// FZFIX KD (fuzz 0x2012c9cd21ef2cca): two ops naming one destination cell.  The one-op rule is for FILE swaps only, so two blob
// swaps still reach the round.  The round is refused whole before any slot write.
static void t_kd_duplicate_destination_refused() {
    Rig r;
    r.add(swap_op(kTierSwapBlob, 1, 3, 7, 0));
    r.add(swap_op(kTierSwapBlob, 1, 1, 7, 1));
    Reports rep;
    CHECK(!r.run(rep), "a round naming one destination twice is not ok");
    CHECK(r.res[0] == kTierRejected && r.res[1] == kTierRejected, "every op of it is rejected");
    CHECK(r.cache.applies == 0 && r.cache.queued.empty(), "before any slot write");
    CHECK(r.held_at(1, 3) && r.held_at(1, 1) && !r.held_at(1, 7) && r.count_held() == 8, "held[] unchanged: the truth");
    CHECK(rep.seen.size() == 1 && rep.seen[0].first == -1 && rep.why[0].find("destination") != std::string::npos, "one round-level reason");
    CHECK(consistent(r), "residency-consistent");
    Rig q;   // the same expert into two LAYERS is two cells, not a duplicate
    q.add(swap_op(kTierSwapBlob, 0, 0, 6, 0));
    q.add(swap_op(kTierSwapBlob, 1, 0, 6, 1));
    Reports rep2;
    CHECK(q.run(rep2) && q.res[1] == kTierApplied && consistent(q), "distinct destination cells are still served");
}

int main() {
    t_round_trip(); t_plain_swap_unchanged(); t_refusals_touch_nothing(); t_oversize_blob_refused();
    t_readback_mismatch(); t_sync_failure_is_failed_not_applied(); t_two_ops_read_back_each(); t_as_swap_view();
    t_kq_quarantined_expert_not_readmitted(); t_kd_duplicate_destination_refused();
    std::printf(g_fail ? "blob_swap_test: %d FAILED checks\n" : "blob_swap_test: all checks passed\n", g_fail);
    return g_fail != 0;
}
