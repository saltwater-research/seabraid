// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// tests/channel_v4_race_test.cpp - the v4 tier handshake under two concurrent threads.
//
//   (from compat/strata; tests/CMakeLists.txt registers it, -DSEABRAID_STRATA_TSAN=ON adds the sanitizer)
//   g++ -std=c++17 -Wall -Wextra -O1 -fsanitize=thread -I../../adapters/strata/wire
//       tests/channel_v4_race_test.cpp -o /tmp/channel_v4_race_test -lrt -pthread
//
// WHY THIS EXISTS BESIDE channel_v4_test.cpp.  That test drives the same handshake, but from ONE thread, so
// it can only check the SEQUENCE of the protocol and never its VISIBILITY: a thread's own stores are
// trivially visible to itself, so every acquire/release pair in the channel could be a plain load and store
// and all 38 of its checks would still pass.  The bug this guards against is the other one -- the helper
// publishing `tier_resp` before the residency it describes is readable, so the main engine wakes up and
// routes against a `held[]` that has not landed.
//
// Both threads use ONE mapping on purpose.  Two mappings of the same segment live at different addresses, so
// ThreadSanitizer would treat them as unrelated memory and see nothing; one mapping is the arrangement under
// which the fences are actually checked, and the fences are the same code either way.
//
// On x86 the hardware hides most store/load reordering, so a missing fence often will NOT show up as a wrong
// value here.  ThreadSanitizer is the real detector: it reasons about the happens-before edges rather than
// what this CPU happens to do, so it flags the unsynchronised access even when the value came out right.
// Run it under TSan or it is worth much less.
//
// House rule: the ordering check was sabotaged (ack moved BEFORE the writes it is supposed to publish) and
// watched to fail before this was committed.

#include "strata/seabraid/channel.hpp"

#include <atomic>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

using namespace seabraid;

static int g_fail = 0;
#define CHECK(c, why) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, why); ++g_fail; } } while (0)

// Ack before the writes it publishes: the mistake a helper author makes once.
static bool g_ack_early = false;

static const int kRounds = 400;
static const int kCells = 64;      // held[] cells each round stamps

int main(int argc, char** argv) {
    g_ack_early = argc > 1 && std::string(argv[1]) == "--sabotage-ack-early";

    const char* name = "/seabraid_v4_race_test";
    Channel ch;
    std::string err;
    // Small but real geometry: 4 layers x 16 experts gives held[] 64 cells, which is what the threads stamp.
    if (!ch.create(name, /*hidden*/ 8, /*cap*/ 8, /*n_layers*/ 4, /*n_expert*/ 16,
                   /*window_max_t*/ 8, /*blob_max*/ 64, err)) {
        std::printf("FAIL: could not create the channel: %s\n", err.c_str());
        return 1;
    }

    std::atomic<bool> stop{false};
    std::atomic<int> served{0};

    // ---- the fake helper: apply, record, publish.  In that order, unless sabotaged.
    std::thread helper([&] {
        uint64_t last = 0;
        while (!stop.load(std::memory_order_relaxed)) {
            const uint64_t seq = ch.tier_pending();
            if (seq == 0 || seq == last) { std::this_thread::yield(); continue; }
            last = seq;
            const uint8_t stamp = (uint8_t) (seq & 0x7f) | 0x80;   // never 0, so "unwritten" is detectable
            if (g_ack_early) ch.tier_ack(seq, true);
            uint8_t* held = ch.held();
            for (int i = 0; i < kCells; ++i) held[i] = stamp;
            const int n = ch.header()->tier_n;
            for (int i = 0; i < n; ++i) ch.header()->tier_res[i] = kTierApplied;
            if (!g_ack_early) ch.tier_ack(seq, true);
            served.fetch_add(1, std::memory_order_relaxed);
        }
    });

    // ---- the main engine: submit, wait, then read what the round published.
    TierWireOp ops[2];
    ops[0].kind = kTierLoad; ops[0].helper = 0;
    ops[0].layer_old = -1; ops[0].expert_old = -1; ops[0].layer_new = 1; ops[0].expert_new = 5;
    ops[1].kind = kTierSwap; ops[1].helper = 0;
    ops[1].layer_old = 2; ops[1].expert_old = 11; ops[1].layer_new = 2; ops[1].expert_new = 6;

    int bad_held = 0, bad_res = 0;
    for (int r = 0; r < kRounds; ++r) {
        const uint64_t seq = ch.tier_submit(ops, 2);
        const uint8_t want = (uint8_t) (seq & 0x7f) | 0x80;
        if (!Channel::wait([&] { return ch.tier_done(seq); }, 30)) {
            std::printf("FAIL: round %llu was never acked\n", (unsigned long long) seq);
            ++g_fail;
            break;
        }
        // The point of the whole handshake: once tier_done is true, everything the helper wrote for THIS
        // round must be readable.  A stale or half-written held[] here is the engine routing to an expert
        // whose bytes have not landed.
        const uint8_t* held = ch.held();
        for (int i = 0; i < kCells; ++i) if (held[i] != want) { ++bad_held; break; }
        if (ch.header()->tier_res[0] != kTierApplied || ch.header()->tier_res[1] != kTierApplied) ++bad_res;
    }

    stop.store(true, std::memory_order_relaxed);
    helper.join();

    CHECK(bad_held == 0, "held[] was stale or half-written after tier_done reported the round complete");
    CHECK(bad_res == 0, "per-op results were not readable after tier_done reported the round complete");
    CHECK(served.load() >= kRounds, "the helper did not see every round");

    ch.unmap();
    (void) name;   // unmap() unlinks: this process is the owner

    if (g_fail == 0) std::printf("channel_v4_race_test: all checks passed (%d rounds)\n", kRounds);
    else std::printf("channel_v4_race_test: %d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
