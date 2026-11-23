// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// tests/channel_blob_swap_test.cpp - a kTierSwapBlob served over a REAL shared-memory channel: main writes the new expert's
// bytes into blob slot 0 and submits one op, an in-process helper (its own thread, its own mapping of the same object) runs
// serve_tier_channel against a fake cache.  CPU only; no Strata.
//
// What it guards: the bytes cross the process line through the blob slot and land in the displaced expert's slot; the ack
// comes only after the read-back (a corrupted slot is kTierFailed with tier_ok 0, never an ACKed good round); a bad slot index
// or a bad count is refused with the slot untouched; main's own residency check (tier_residency_consistent) accepts the
// published held[] of a blob round.  House rule: each guard sabotaged once (tests/SABOTAGE.md).
#include "tier_round_fake.hpp"

#include <atomic>
#include <cstdio>
#include <thread>
#include <unistd.h>

using namespace seabraid;
using namespace seabraid::test;

static int g_fail = 0;
#define CHECK(cond, what)                                                                       \
    do {                                                                                        \
        if (!(cond)) { std::printf("FAIL: %s (%s:%d)\n", (what), __FILE__, __LINE__); ++g_fail; } \
    } while (0)

static constexpr int64_t H = 64, CAP = 32, NL = 4, NE = 8, BLOB = 256;
static constexpr uint32_t WMT = 4;

// One helper loop: serve every tier round main publishes until told to stop.
static void helper_loop(Channel& ch, FakeCache& cache, std::atomic<bool>& stop, Reports& rep) {
    while (!stop.load()) {
        const uint64_t seq = ch.tier_pending();
        if (seq == 0) { usleep(200); continue; }
        serve_tier_channel(ch, seq, cache, rep);
    }
}

struct Rig {
    Channel helper, main_side;
    FakeCache cache;
    Reports rep;
    std::atomic<bool> stop{false};
    std::thread th;
    std::string name = "/seabraid_blobswap_" + std::to_string(getpid());
    bool up() {
        std::string err;
        if (!helper.create(name, H, CAP, NL, NE, WMT, BLOB, err) || !main_side.connect(name, H, CAP, NL, NE, WMT, err)) {
            std::printf("FAIL: channel setup: %s\n", err.c_str()); ++g_fail; return false;
        }
        cache.bytes = BLOB;
        for (int32_t l = 0; l < NL; ++l)
            for (int32_t e = 0; e < 4; ++e) { cache.put(l, e); helper.held()[l * NE + e] = 1; }
        th = std::thread(helper_loop, std::ref(helper), std::ref(cache), std::ref(stop), std::ref(rep));
        return true;
    }
    ~Rig() { stop = true; if (th.joinable()) th.join(); }
    // Main: write the blob, submit one op, wait for the ack.
    bool round(TierWireOp op, uint8_t fill, uint8_t* res) {
        std::memset(main_side.blob(op.blob_slot < kBlobBatch ? op.blob_slot : 0), fill, BLOB);
        const uint64_t seq = main_side.tier_submit(&op, 1);
        if (!Channel::wait([&] { return main_side.tier_done(seq); }, 10)) { std::printf("FAIL: the helper never acked\n"); ++g_fail; return false; }
        *res = main_side.header()->tier_res[0];
        return main_side.header()->tier_ok != 0;
    }
    bool main_sees_held(int32_t l, int32_t e) { return main_side.held()[l * NE + e] != 0; }
};

static void t_good_swap(Rig& r) {
    uint8_t res = 0;
    CHECK(r.round(swap_op(kTierSwapBlob, 1, 0, 6, 0), 0x5C, &res) && res == kTierApplied, "a blob swap over the channel is applied");
    const auto it = r.cache.slot_of.find({1, 6});
    CHECK(it != r.cache.slot_of.end(), "the new expert is resident");
    CHECK(r.cache.slots[(size_t) it->second][0] == 0x5C && r.cache.slots[(size_t) it->second][BLOB - 1] == 0x5C,
          "the helper's slot holds the bytes main put in the channel");
    CHECK(r.main_sees_held(1, 6) && !r.main_sees_held(1, 0), "held[] as main reads it: new in, displaced out");
    const TierWireOp plain = as_swap(swap_op(kTierSwapBlob, 1, 0, 6, 0));
    std::string why;
    CHECK(tier_residency_consistent(r.main_side.held(), NL, NE, &plain, &res, 1, why), "main's residency check accepts the round");
}

static void t_corrupt_slot(Rig& r) {
    uint8_t res = 0;
    r.cache.corrupt_expert_new = 7;
    CHECK(!r.round(swap_op(kTierSwapBlob, 2, 1, 7, 0), 0x11, &res) && res == kTierFailed, "a slot that does not read back is kTierFailed, tier_ok 0");
    CHECK(!r.main_sees_held(2, 7) && !r.main_sees_held(2, 1) && r.cache.suspect(2, 7), "and published as holding nothing (main closes the card)");
    r.cache.corrupt_expert_new = -1;
}

static void t_refused_and_bad_count(Rig& r) {
    uint8_t res = 0;
    const int before = r.cache.applies;
    CHECK(!r.round(swap_op(kTierSwapBlob, 0, 0, 6, kBlobBatch), 0x22, &res) && res == kTierRejected, "a bad slot index is kTierRejected");
    CHECK(r.cache.applies == before && r.main_sees_held(0, 0), "refused before the cache or the slot was touched");
    // a round with a count of zero (a hostile or broken peer) must still be answered: publish it by hand, tier_submit clamps
    Header* hd = r.main_side.header();
    hd->tier_n = 0;
    hd->tier_ok = 1;
    const uint64_t seq = __atomic_load_n(&hd->tier_seq, __ATOMIC_RELAXED) + 1;
    __atomic_store_n(&hd->tier_seq, seq, __ATOMIC_RELEASE);
    CHECK(Channel::wait([&] { return r.main_side.tier_done(seq); }, 10) && r.main_side.header()->tier_ok == 0, "a bad count is acked as failed, not left unanswered");
}

// SOLFIX 1 (TOCTOU): a peer rewrites the op in shared memory AFTER the helper validated it, while the slot is being written: the blob
// swap becomes a plain swap.  The helper must judge and read back the op it validated (its own copy), so the torn slot is still caught.
static void t_op_rewritten_mid_round(Rig& r) {
    uint8_t res = 0;
    r.cache.corrupt_expert_new = 5;
    r.cache.during_apply = [&] { r.main_side.tier_ops()[0].kind = kTierSwap; };
    CHECK(!r.round(swap_op(kTierSwapBlob, 3, 2, 5, 0), 0x33, &res) && res == kTierFailed, "an op rewritten mid-round still gets its read-back: kTierFailed");
    CHECK(!r.main_sees_held(3, 5) && r.cache.suspect(3, 5), "and the torn slot is published as holding nothing");
    r.cache.during_apply = nullptr;
    r.cache.corrupt_expert_new = -1;
}

// SOLFIX 1 (geometry): a peer inflates the Header's n_layers / n_expert.  held[] is laid out by the geometry the helper MAPPED, so
// held[] is still indexed with the mapped stride, and a blob op in a layer the mapping does not have is refused before the cache is
// asked (nothing is written past held[]).
static void t_header_geometry_ignored(Rig& r) {
    Header* hd = r.main_side.header();
    const int64_t layers = hd->n_layers, experts = hd->n_expert;
    hd->n_expert = NE * 4;
    uint8_t res = 0;
    CHECK(r.round(swap_op(kTierSwap, 2, 2, 6, 0), 0, &res) && res == kTierApplied, "a plain swap under an inflated n_expert is applied");
    CHECK(r.main_sees_held(2, 6) && !r.main_sees_held(2, 2), "and held[] is published with the mapped stride, where main reads it");
    hd->n_layers = NL + 4;
    r.cache.put(NL, 0);   // a cell the fake would happily swap, in a layer the channel does not have
    const int before = r.cache.applies;
    CHECK(!r.round(swap_op(kTierSwapBlob, NL, 0, 1, 0), 0x44, &res) && res == kTierRejected, "a blob op beyond the mapped layers is refused whatever the Header says");
    CHECK(r.cache.applies == before, "before the cache was asked");
    hd->n_layers = layers;
    hd->n_expert = experts;
    r.cache.slot_of.erase({NL, 0});
}

int main() {
    Rig r;
    if (!r.up()) return 1;
    t_good_swap(r);
    t_corrupt_slot(r);
    t_refused_and_bad_count(r);
    t_op_rewritten_mid_round(r);
    t_header_geometry_ignored(r);
    CHECK(r.cache.resident() == 16, "slot accounting over the whole session: 4 layers x 4 experts, unchanged");
    std::printf(g_fail ? "channel_blob_swap_test: %d FAILED checks\n" : "channel_blob_swap_test: all checks passed\n", g_fail);
    return g_fail != 0;
}
