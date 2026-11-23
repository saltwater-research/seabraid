// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// tests/channel_v4_test.cpp - the seabraid channel v4 wire contract (D7 live re-tiering).
//
// Self-contained: builds against the wire alone (adapters/strata/wire), no Strata code, no GPU runtime.  Registered
// in tests/CMakeLists.txt; by hand, from compat/strata:
//   g++ -std=c++17 -Wall -Wextra -O1 -I../../adapters/strata/wire tests/channel_v4_test.cpp -o /tmp/channel_v4_test -lrt
//
// What this is for.  The helper process is built by nvcc + gcc and the main engine by clang.  If those two
// disagree about `Header` - padding under `alignas(64)`, a field inserted on one side only, a different
// kTierMaxOps - the failure mode is NOT a crash: both sides map the same bytes and read shifted fields, so a
// LOAD lands on the wrong expert and a layer quietly loses a contribution.  Every check below exists to turn
// that class of bug into a refusal at connect() time.
//
// House rule: each check here was sabotaged once and watched to fail before being committed.  See
// SABOTAGE.md beside this file for the recorded output.

#include "strata/seabraid/channel.hpp"

#include <cstdio>
#include <cstring>
#include <string>

static int g_fail = 0;
#define CHECK(cond, what)                                                                  \
    do {                                                                                   \
        if (!(cond)) { std::printf("FAIL: %s (%s:%d)\n", (what), __FILE__, __LINE__); ++g_fail; } \
    } while (0)

using namespace seabraid;

// Geometry small enough to keep the segment tiny; the contract does not depend on the values.
static constexpr int64_t H = 64, CAP = 32, NL = 4, NE = 8, BLOB = 256;
static constexpr uint32_t WMT = 8;

int main() {
    const std::string name = "/seabraid_v4_test_" + std::to_string((long) getpid());
    std::string err;

    // ---- 1. the preamble is where both sides think it is.  The static_asserts in the header prove this at
    // compile time in EVERY translation unit that includes it; asserting at runtime too means a build that
    // somehow disabled them still fails here.
    CHECK(offsetof(Header, magic) == 0, "magic at 0");
    CHECK(offsetof(Header, version) == 8, "version at 8");
    CHECK(offsetof(Header, hdr_bytes) == 12, "hdr_bytes at 12");
    CHECK(offsetof(Header, window_max_t) == 16, "window_max_t at 16");
    CHECK(offsetof(Header, tier_max_ops) == 20, "tier_max_ops at 20");
    CHECK(sizeof(TierWireOp) == 20, "TierWireOp is 20 bytes");
    CHECK(alignof(TierWireOp) == 4, "TierWireOp is 4-aligned");

    // ---- 2. create -> connect round trip.
    Channel helper, main_side;
    CHECK(helper.create(name, H, CAP, NL, NE, WMT, BLOB, err), ("create: " + err).c_str());
    CHECK(helper.header()->hdr_bytes == (uint32_t) sizeof(Header), "create stamps hdr_bytes");
    CHECK(helper.header()->tier_max_ops == (uint32_t) kTierMaxOps, "create stamps tier_max_ops");
    CHECK(main_side.connect(name, H, CAP, NL, NE, WMT, err), ("connect: " + err).c_str());

    // ---- 3. the tier round trip, in the order the contract specifies.
    // A load (no displaced resident), a same-layer swap, and a cross-layer swap - the three shapes the helper
    // must tell apart.  The cross-layer one is the shape whose per-slot size check is the difference between a
    // rejected op and a GPU buffer overrun, so it has to survive the wire intact.
    TierWireOp ops[3] = {};
    ops[0].kind = kTierLoad; ops[0].helper = 0;
    ops[0].layer_old = -1; ops[0].expert_old = -1; ops[0].layer_new = 1; ops[0].expert_new = 5;
    ops[1].kind = kTierSwap; ops[1].helper = 0;
    ops[1].layer_old = 2; ops[1].expert_old = 11; ops[1].layer_new = 2; ops[1].expert_new = 6;
    ops[2].kind = kTierSwap; ops[2].helper = 0;
    ops[2].layer_old = 3; ops[2].expert_old = 12; ops[2].layer_new = 1; ops[2].expert_new = 7;

    CHECK(helper.tier_pending() == 0, "nothing pending before submit");
    const uint64_t seq = main_side.tier_submit(ops, 3);
    CHECK(seq == 1, "first round is sequence 1");
    CHECK(!main_side.tier_done(seq), "not done before the helper acks");

    const uint64_t seen = helper.tier_pending();
    CHECK(seen == seq, "helper sees the submitted sequence");
    CHECK(helper.header()->tier_n == 3, "op count crosses");
    for (int i = 0; i < 3; ++i) {
        CHECK(helper.tier_ops()[i].kind == ops[i].kind, "op kind crosses");
        CHECK(helper.tier_ops()[i].layer_old == ops[i].layer_old, "op layer_old crosses");
        CHECK(helper.tier_ops()[i].expert_old == ops[i].expert_old, "op expert_old crosses");
        CHECK(helper.tier_ops()[i].layer_new == ops[i].layer_new, "op layer_new crosses");
        CHECK(helper.tier_ops()[i].expert_new == ops[i].expert_new, "op expert_new crosses");
        CHECK(helper.header()->tier_res[i] == kTierPending, "ops start pending");
    }
    // the helper applies, records per-op results, then acks
    helper.header()->tier_res[0] = kTierApplied;
    helper.header()->tier_res[1] = kTierFailed;      // main must be able to tell these apart, for note_failed()
    helper.header()->tier_res[2] = kTierApplied;
    helper.tier_ack(seen, false);
    CHECK(main_side.tier_done(seq), "done after ack");
    CHECK(main_side.header()->tier_ok == 0, "a failed op clears tier_ok");
    CHECK(main_side.header()->tier_res[1] == kTierFailed, "per-op result crosses, so note_failed() is exact");
    CHECK(helper.tier_pending() == 0, "nothing pending after ack");

    // a second round must not be mistaken for the first
    const uint64_t seq2 = main_side.tier_submit(ops, 1);
    CHECK(seq2 == 2, "sequences advance");
    CHECK(!main_side.tier_done(seq2), "a stale ack does not satisfy a new round");

    // ---- 4. epoch advances and is readable from both sides.
    const uint64_t e0 = main_side.epoch();
    main_side.bump_epoch();
    CHECK(main_side.epoch() == e0 + 1, "epoch advances");
    CHECK(helper.epoch() == e0 + 1, "epoch is visible across the mapping");

    // ---- 5. THE ABI CHECK.  This is the one that matters: a peer whose Header differs must be REFUSED, not
    // allowed to read shifted fields.  Simulating it by poking hdr_bytes is exactly what a mismatched
    // toolchain would produce, and it is the only way to test it from one build.
    {
        const uint32_t good = helper.header()->hdr_bytes;
        helper.header()->hdr_bytes = good + 64;        // as if the other side padded differently
        Channel bad;
        std::string berr;
        CHECK(!bad.connect(name, H, CAP, NL, NE, WMT, berr), "connect REFUSES a mismatched sizeof(Header)");
        CHECK(berr.find("ABI mismatch") != std::string::npos, "and says it is an ABI mismatch");
        helper.header()->hdr_bytes = good;
    }
    {
        const uint32_t good = helper.header()->tier_max_ops;
        helper.header()->tier_max_ops = good + 1;      // as if one side changed kTierMaxOps
        Channel bad;
        std::string berr;
        CHECK(!bad.connect(name, H, CAP, NL, NE, WMT, berr), "connect REFUSES a mismatched tier_max_ops");
        helper.header()->tier_max_ops = good;
    }
    {
        const uint32_t good = helper.header()->version;
        helper.header()->version = good - 1;           // an older helper against a newer main
        Channel bad;
        std::string berr;
        CHECK(!bad.connect(name, H, CAP, NL, NE, WMT, berr), "connect REFUSES an older version");
        CHECK(berr.find("magic/version") != std::string::npos, "and names the version");
        helper.header()->version = good;
    }
    // and the geometry check still works, so v4 did not break v3's contract
    {
        Channel bad;
        std::string berr;
        CHECK(!bad.connect(name, H, CAP + 1, NL, NE, WMT, berr), "connect REFUSES a cap mismatch");
    }

    // ---- 6. a submit larger than the block is clamped, never written past the end.
    {
        TierWireOp many[kTierMaxOps + 8] = {};
        for (int i = 0; i < kTierMaxOps + 8; ++i) { many[i].kind = kTierSwap; many[i].layer_new = i; }
        main_side.tier_submit(many, kTierMaxOps + 8);
        CHECK(main_side.header()->tier_n == kTierMaxOps, "an oversized round is clamped to kTierMaxOps");
    }

    main_side.unmap();
    helper.unmap();

    if (g_fail == 0) std::printf("channel_v4_test: all checks passed\n");
    else std::printf("channel_v4_test: %d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
