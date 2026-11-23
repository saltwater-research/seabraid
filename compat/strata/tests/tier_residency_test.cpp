// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// seabraid D7: the helper must publish post-round residency into held[], and main must refuse a round whose held[]
// contradicts what it reported applied.  Host only, no GPU, no gtest:
//   g++ -std=c++17 -O1 -Wall -Wextra -I<strata>/include tests/tier_residency_test.cpp -o tier_residency_test -lrt
// Regression for the 2026-11-06 D7 arms: held[] was written only at plan load, so after every applied SWAP main kept
// routing the displaced expert to a helper that skips experts it does not hold (rows silently dropped), and the next
// round re-proposed displacing it ("SWAP but the displaced expert is not resident here", 21 of 47 rounds).
#include "strata/seabraid/channel.hpp"

#include <cstdio>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace seabraid;

static int g_fail = 0;
static const char* g_test = "?";
#define CHECK(c) do { if (!(c)) { ++g_fail; std::printf("  FAIL [%s] %s:%d  %s\n", g_test, __FILE__, __LINE__, #c); } } while (0)

static constexpr int L = 4, E = 8;
static size_t at(int l, int e) { return (size_t) l * E + (size_t) e; }

static TierWireOp swap(int l, int eo, int en) { TierWireOp o{}; o.kind = kTierSwap; o.layer_old = l; o.expert_old = eo; o.layer_new = l; o.expert_new = en; return o; }
static TierWireOp load(int l, int e) { TierWireOp o{}; o.kind = kTierLoad; o.layer_old = -1; o.expert_old = -1; o.layer_new = l; o.expert_new = e; return o; }

// A fake helper cache: the set of (layer, expert) really in VRAM.
struct Cache {
    std::set<std::pair<int, int>> in;
    bool holds(int l, int e) const { return in.count({l, e}) != 0; }
};

static void t_stale_bitmap_is_caught() {
    g_test = "stale held[] (the bug)";
    std::vector<uint8_t> held(L * E, 0);
    held[at(1, 2)] = 1;                       // plan time: the helper holds (1,2)
    TierWireOp ops[1] = {swap(1, 2, 5)};
    uint8_t res[1] = {kTierApplied};          // the helper applied the swap ... but never rewrote held[]
    std::string why;
    CHECK(!tier_residency_consistent(held.data(), L, E, ops, res, 1, why));
    CHECK(why.find("layer 1 expert") != std::string::npos);
}

static void t_publish_then_consistent() {
    g_test = "publish -> consistent";
    std::vector<uint8_t> held(L * E, 0);
    held[at(1, 2)] = 1;
    Cache c; c.in = {{1, 2}};
    TierWireOp ops[2] = {swap(1, 2, 5), load(3, 7)};
    c.in.erase({1, 2}); c.in.insert({1, 5}); c.in.insert({3, 7});   // what tier_apply + tier_sync did
    publish_tier_residency(held.data(), L, E, ops, 2, [&](int32_t l, int32_t e) { return c.holds(l, e); });
    CHECK(held[at(1, 2)] == 0 && held[at(1, 5)] == 1 && held[at(3, 7)] == 1);
    uint8_t res[2] = {kTierApplied, kTierApplied};
    std::string why;
    CHECK(tier_residency_consistent(held.data(), L, E, ops, res, 2, why));
}

static void t_failed_op_publishes_truth() {
    g_test = "failed op";
    std::vector<uint8_t> held(L * E, 0);
    held[at(0, 1)] = 1;
    Cache c; c.in = {{0, 1}};                 // the swap failed: VRAM unchanged
    TierWireOp ops[1] = {swap(0, 1, 6)};
    publish_tier_residency(held.data(), L, E, ops, 1, [&](int32_t l, int32_t e) { return c.holds(l, e); });
    CHECK(held[at(0, 1)] == 1 && held[at(0, 6)] == 0);
    uint8_t res[1] = {kTierFailed};
    std::string why;
    CHECK(tier_residency_consistent(held.data(), L, E, ops, res, 1, why));   // failed ops make no claim
}

static void t_double_move_judged_by_final_state() {
    g_test = "A->B then B->C";
    std::vector<uint8_t> held(L * E, 0);
    held[at(2, 0)] = 1;
    Cache c; c.in = {{2, 3}};                 // final: only C
    TierWireOp ops[2] = {swap(2, 0, 1), swap(2, 1, 3)};
    publish_tier_residency(held.data(), L, E, ops, 2, [&](int32_t l, int32_t e) { return c.holds(l, e); });
    CHECK(held[at(2, 0)] == 0 && held[at(2, 1)] == 0 && held[at(2, 3)] == 1);
    uint8_t res[2] = {kTierApplied, kTierApplied};
    std::string why;
    CHECK(tier_residency_consistent(held.data(), L, E, ops, res, 2, why));
}

static void t_out_of_range_ignored() {
    g_test = "out of range";
    std::vector<uint8_t> held(L * E, 7);
    TierWireOp ops[2] = {load(L, 0), swap(0, -1, E)};
    publish_tier_residency(held.data(), L, E, ops, 2, [](int32_t, int32_t) { return true; });
    bool untouched = true;
    for (uint8_t v : held) untouched &= v == 7;
    CHECK(untouched);
    uint8_t res[2] = {kTierApplied, kTierApplied};
    std::string why;
    CHECK(tier_residency_consistent(held.data(), L, E, ops, res, 2, why));
}

int main() {
    t_stale_bitmap_is_caught();
    t_publish_then_consistent();
    t_failed_op_publishes_truth();
    t_double_move_judged_by_final_state();
    t_out_of_range_ignored();
    std::printf("%s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
