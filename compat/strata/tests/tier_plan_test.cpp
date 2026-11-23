// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// seabraid D7 policy tests: one test per rule of include/strata/seabraid/tier_plan.hpp (the "Rn" in each name is the
// rule number from the roadmap brief).  Standalone, no GPU, no gtest:
//   g++ -std=c++17 -O2 -Wall -Wextra -Istrata-ext/include tests/tier_plan_test.cpp strata-ext/src/tier_plan.cpp -o tier_plan_test
//
// Sim plays the transport: it hands the policy authoritative on_helper / free_slots each round, applies the
// returned ops, and CHECKS the per-round invariants (rules 1, 3, 4, 5 and op sanity) on EVERY step of EVERY test, so
// a violation fails whichever test was running.  The rule-specific assertions sit in the tests themselves.
#include "strata/seabraid/tier_plan.hpp"

#include <cmath>
#include <cstdio>
#include <set>
#include <string>
#include <vector>

using namespace seabraid;

static int g_fail = 0;
static const char* g_test = "?";
#define CHECK(c) do { if (!(c)) { ++g_fail; std::printf("  FAIL [%s] %s:%d  %s\n", g_test, __FILE__, __LINE__, #c); } } while (0)

static TierPolicyConfig no_ceiling() { TierPolicyConfig c; c.max_helper_layer_share = 1.0; return c; }

struct Sim {
    int64_t L, E, slots;                 // slots: the helper's capacity in experts
    TierPolicyConfig cfg;
    TierPolicyState st;
    std::vector<uint8_t> on;
    std::vector<int64_t> last;           // the TEST's own record of each expert's last move round (independent of the policy)
    int64_t round = 0, moves = 0, free_override = -1;
    Sim(int64_t l, int64_t e, int64_t s, TierPolicyConfig c) : L(l), E(e), slots(s), cfg(c), on((size_t) (l * e), 0), last((size_t) (l * e), INT64_MIN / 2) {}

    int64_t n_on() const { int64_t n = 0; for (uint8_t v : on) n += v != 0; return n; }
    double share(int64_t l, const std::vector<uint8_t>& o) const {
        double t = 0, h = 0;
        for (int64_t e = 0; e < E; ++e) { t += st.heat[(size_t) (l * E + e)]; if (o[(size_t) (l * E + e)]) h += st.heat[(size_t) (l * E + e)]; }
        return t > 0 ? h / t : 0.0;
    }
    std::vector<TierOp> step(const std::vector<double>& heat, bool apply = true) {
        const int64_t free_slots = free_override >= 0 ? free_override : slots - n_on();
        const std::vector<uint8_t> before = on;
        std::string err;
        auto ops = plan_tier_round(L, E, on, free_slots, heat, cfg, st, &err);
        CHECK(err.empty());
        int64_t loads = 0;
        std::set<int64_t> seen;
        for (const TierOp& o : ops) {
            const int64_t i = (int64_t) o.layer * E + o.expert;
            CHECK(o.layer >= 0 && o.layer < L && o.expert >= 0 && o.expert < E && o.helper == 0);
            CHECK(seen.insert(i).second);                                         // one op per expert per round
            CHECK(round - last[(size_t) i] > cfg.cooldown_rounds);                // R4
            if (o.kind == TierOp::kLoad) { ++loads; CHECK(!before[(size_t) i]); }
            else { CHECK(o.kind == TierOp::kEvict); CHECK(before[(size_t) i]); }
        }
        CHECK(loads <= free_slots);                                               // R1
        CHECK((int64_t) ops.size() <= cfg.max_moves_per_round);                   // R3
        std::vector<uint8_t> after = before;
        for (const TierOp& o : ops) {
            const size_t i = (size_t) ((int64_t) o.layer * E + o.expert);
            after[i] = o.kind == TierOp::kLoad;
            last[i] = round;
        }
        for (int64_t l = 0; l < L; ++l) {                                         // R5, on the policy's own smoothed heat
            const double b = share(l, before), a = share(l, after);
            if (b > cfg.max_helper_layer_share + 1e-9) CHECK(a <= b + 1e-9);      // already over: may only lower
            else CHECK(a <= cfg.max_helper_layer_share + 1e-9);
        }
        moves += (int64_t) ops.size();
        ++round;
        if (apply) on = after;
        return ops;
    }
};

// Deterministic noise in [-1, 1) so the multi-round tests are repeatable.
static uint64_t g_lcg = 88172645463325252ull;
static double noise() { g_lcg = g_lcg * 6364136223846793005ull + 1442695040888963407ull; return (double) (g_lcg >> 11) / 4503599627370496.0 - 1.0; }

static int count_kind(const std::vector<TierOp>& ops, TierOp::Kind k) { int n = 0; for (auto& o : ops) n += o.kind == k; return n; }

// R1: LOADs only target slots free at the START of the round.  EVICTs never enable a same-round LOAD.
static void t_R1_loads_within_free_slots() {
    g_test = "R1 loads <= free_slots";
    Sim s(4, 8, 100, no_ceiling()); s.cfg.max_moves_per_round = 50;
    std::vector<double> heat(32, 5.0);
    s.free_override = 2;
    CHECK(count_kind(s.step(heat, false), TierOp::kLoad) == 2);
    s.free_override = 0;
    CHECK(count_kind(s.step(heat, false), TierOp::kLoad) == 0);
    // A full helper with a clearly hotter challenger: the swap may EVICT now but cannot LOAD (no free slot).
    Sim t(1, 4, 2, no_ceiling()); t.on = {1, 1, 0, 0};
    auto ops = t.step({1, 1, 50, 50}, false);
    CHECK(count_kind(ops, TierOp::kLoad) == 0 && count_kind(ops, TierOp::kEvict) >= 1);
}

// R2: evict only for a challenger that beats the coldest incumbent by the margin.
static void t_R2_evict_sparingly() {
    g_test = "R2 evict sparingly";
    auto mk = [](double challenger) {
        Sim s(2, 8, 4, no_ceiling()); s.cfg.cooldown_rounds = 0;
        for (int e = 0; e < 4; ++e) s.on[(size_t) e] = 1;                         // layer 0 experts 0..3 resident
        std::vector<double> h(16, 0.0);
        for (int e = 0; e < 4; ++e) h[(size_t) e] = 10.0;
        h[8] = challenger;                                                        // layer 1 expert 0 is the challenger
        return std::make_pair(s, h);
    };
    { auto [s, h] = mk(12.0); CHECK(s.step(h, false).empty()); }                  // 1.2x < 1.25: nobody moves
    { auto [s, h] = mk(13.0); auto ops = s.step(h, false);                        // 1.3x: exactly one EVICT, free_slots = 0
      CHECK(ops.size() == 1 && ops[0].kind == TierOp::kEvict && ops[0].layer == 0 && ops[0].expert == 0); }
    { auto [s, h] = mk(13.0); s.free_override = 1; s.cfg.reserve_slots = 1;       // a reserve slot: LOAD + EVICT in one round
      auto ops = s.step(h, false);
      CHECK(ops.size() == 2 && ops[0].kind == TierOp::kLoad && ops[0].layer == 1 && ops[1].kind == TierOp::kEvict); }
}

// R3: total ops per round <= max_moves_per_round, and the budget really is used when there is work.
static void t_R3_move_budget() {
    g_test = "R3 move budget";
    Sim s(8, 16, 20, no_ceiling()); s.cfg.max_moves_per_round = 3; s.cfg.cooldown_rounds = 0;
    size_t max_ops = 0;
    for (int r = 0; r < 30; ++r) {
        std::vector<double> h(128);
        for (auto& v : h) v = 1.0 + 9.0 * (noise() + 1.0);
        max_ops = std::max(max_ops, s.step(h).size());
    }
    CHECK(max_ops == 3);
    Sim z(2, 4, 4, no_ceiling()); z.cfg.max_moves_per_round = 0;
    CHECK(z.step(std::vector<double>(8, 5.0)).empty());
}

// R4: an expert that moved cannot move again until cooldown_rounds have passed.  One slot, heat flips at round 2.
static void t_R4_cooldown() {
    g_test = "R4 cooldown";
    Sim s(1, 2, 1, no_ceiling()); s.cfg.cooldown_rounds = 3; s.cfg.max_moves_per_round = 4;
    s.on = {1, 0};                                                                // expert 0 resident
    CHECK(s.step({10, 100}).size() == 1);                                         // round 0: EVICT 0 (1 beats 0 by 10x), frees the slot
    CHECK(s.step({10, 100}).size() == 1);                                         // round 1: LOAD 1 into the slot
    int first_move = -1;
    for (int r = 2; r < 12; ++r) {
        if (!s.step({1000, 1}).empty() && first_move < 0) first_move = r;         // 1 is now cold and 0 hot, but both are frozen
    }
    CHECK(first_move == 5);                                                       // 1 moved at round 1: free again at 1 + 3 + 1 = 5
}

// R5: per-layer ceiling.  Hot experts clustered in a few layers must not drag the helper into them, at any cap.
static void t_R5_ceiling_clustering() {
    g_test = "R5 ceiling vs clustering";
    const int64_t L = 12, E = 16, slots = 40;
    const double layer_w[12] = {100, 80, 60, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    for (double cap : {0.08, 0.15, 0.20, 0.30, 1.0}) {
        TierPolicyConfig c; c.max_helper_layer_share = cap; c.max_moves_per_round = 8; c.cooldown_rounds = 2; c.promote_margin = 1.1;
        Sim s(L, E, slots, c);
        for (int r = 0; r < 80; ++r) {
            std::vector<double> h((size_t) (L * E));
            for (int64_t l = 0; l < L; ++l)
                for (int64_t e = 0; e < E; ++e) h[(size_t) (l * E + e)] = layer_w[l] / (double) (e + 1) * (1.0 + 0.1 * noise());
            s.step(h);                                                            // Sim::step asserts the ceiling every round
        }
        std::set<int64_t> layers;
        for (int64_t i = 0; i < L * E; ++i) if (s.on[(size_t) i]) layers.insert(i / E);
        if (cap < 1.0) CHECK(layers.size() >= 10);                                // spread, not clustered
        else CHECK(layers.size() <= 3);                                           // sensitivity: with no ceiling the drift DOES cluster
    }
}

// R5 (second half): a layer already over the ceiling at the start may only be lowered, down to the ceiling.
static void t_R5_over_at_start_only_lowers() {
    g_test = "R5 over at start";
    TierPolicyConfig c; c.max_helper_layer_share = 0.25; c.max_moves_per_round = 2; c.cooldown_rounds = 0;
    Sim s(2, 8, 8, c);
    for (int e = 0; e < 8; ++e) s.on[(size_t) e] = 1;                             // helper holds ALL of layer 0 (share 1.0)
    std::vector<double> h(16, 1.0);
    int evicts = 0;
    for (int r = 0; r < 6; ++r) { auto ops = s.step(h); evicts += count_kind(ops, TierOp::kEvict); CHECK(count_kind(ops, TierOp::kLoad) == 0 || r >= 3); }
    CHECK(evicts >= 6);                                                           // 8 -> 2 residents of 8 (share 0.25)
    CHECK(s.share(0, s.on) <= 0.25 + 1e-9);
}

// R6: hysteresis.  Heat that flips every round must not ping-pong the single slot between two experts.
static void t_R6_no_pingpong() {
    g_test = "R6 hysteresis";
    Sim s(1, 2, 1, no_ceiling()); s.cfg.cooldown_rounds = 0;                      // cooldown OFF: only margin + smoothing hold it
    s.on = {1, 0};
    for (int r = 0; r < 60; ++r) s.step(r % 2 == 0 ? std::vector<double>{10, 3} : std::vector<double>{3, 10});
    CHECK(s.moves == 0);
}

// R7: deterministic; ties break by (layer, expert).
static void t_R7_deterministic_ties() {
    g_test = "R7 deterministic";
    Sim s(2, 4, 8, no_ceiling()); s.cfg.max_moves_per_round = 3;
    auto ops = s.step(std::vector<double>(8, 1.0), false);                        // all equal: the 3 free slots go to (0,0) (0,1) (0,2)
    CHECK(ops.size() == 3);
    for (size_t k = 0; k < ops.size(); ++k) CHECK(ops[k].kind == TierOp::kLoad && ops[k].layer == 0 && ops[k].expert == (int) k);
    auto run = [] {
        Sim a(6, 16, 20, no_ceiling()); a.cfg.cooldown_rounds = 1; g_lcg = 12345;
        std::vector<int64_t> trace;
        for (int r = 0; r < 25; ++r) {
            std::vector<double> h(96); for (auto& v : h) v = 1.0 + 4.0 * (noise() + 1.0);
            for (auto& o : a.step(h)) trace.push_back(((int64_t) o.kind * 100 + o.layer) * 100 + o.expert);
        }
        return trace;
    };
    CHECK(run() == run());
    CHECK(!run().empty());
}

// R8: bad input fails closed: no ops, an error, and the state untouched.
static void t_R8_validation() {
    g_test = "R8 validation";
    TierPolicyConfig cfg;
    auto bad = [&](int64_t L, int64_t E, size_t n_on, size_t n_heat, int64_t fr, double h0, TierPolicyConfig c) {
        TierPolicyState st; std::string err;
        std::vector<double> heat(n_heat, 1.0); if (!heat.empty()) heat[0] = h0;
        auto ops = plan_tier_round(L, E, std::vector<uint8_t>(n_on, 0), fr, heat, c, st, &err);
        CHECK(ops.empty()); CHECK(!err.empty()); CHECK(st.round == 0 && st.heat.empty());
    };
    bad(2, 4, 7, 8, 1, 1.0, cfg);                                                 // on_helper too short
    bad(2, 4, 8, 9, 1, 1.0, cfg);                                                 // heat too long
    bad(2, 4, 8, 7, 1, 1.0, cfg);                                                 // heat too short
    bad(0, 4, 0, 0, 1, 1.0, cfg);                                                 // no layers
    bad(2, 4, 8, 8, -1, 1.0, cfg);                                                // negative free_slots
    bad(2, 4, 8, 8, 1, -1.0, cfg);                                                // negative heat
    bad(2, 4, 8, 8, 1, std::nan(""), cfg);                                        // NaN heat
    TierPolicyConfig c2 = cfg; c2.promote_margin = 0.5; bad(2, 4, 8, 8, 1, 1.0, c2);
    TierPolicyConfig c3 = cfg; c3.max_helper_layer_share = 1.5; bad(2, 4, 8, 8, 1, 1.0, c3);
    TierPolicyState st; std::string err;                                          // a state built for another shape
    plan_tier_round(2, 4, std::vector<uint8_t>(8, 0), 1, std::vector<double>(8, 1.0), cfg, st, &err);
    CHECK(err.empty() && st.round == 1);
    auto ops = plan_tier_round(2, 5, std::vector<uint8_t>(10, 0), 1, std::vector<double>(10, 1.0), cfg, st, &err);
    CHECK(ops.empty() && !err.empty() && st.round == 1);
    plan_tier_round(2, 4, std::vector<uint8_t>(8, 0), 1, std::vector<double>(8, 1.0), cfg, st, nullptr);   // err is optional
    CHECK(st.round == 2);
}

// R9: on_helper / free_slots are authoritative; no membership in the state.  A LOAD the helper dropped re-plans.
static void t_R9_authoritative_membership() {
    g_test = "R9 authoritative";
    Sim s(1, 4, 2, no_ceiling()); s.cfg.cooldown_rounds = 0;
    std::vector<double> h = {9, 1, 1, 1};
    auto a = s.step(h, false);                                                    // LOAD (0,0) (and one more): not applied, as if it failed
    CHECK(!a.empty() && a[0].kind == TierOp::kLoad && a[0].expert == 0);
    auto b = s.step(h, false);
    CHECK(!b.empty() && b[0].kind == TierOp::kLoad && b[0].expert == 0);          // asked again: the state remembered nothing about it
    s.on = {1, 0, 0, 0};                                                          // now the transport says it IS there
    for (auto& o : s.step(h, false)) CHECK(!(o.kind == TierOp::kLoad && o.expert == 0));
}

// Reserve (rule 2): free slots kept back for LOAD-without-EVICT.
static void t_reserve() {
    g_test = "reserve";
    Sim a(1, 8, 8, no_ceiling()); a.cfg.reserve_slots = 1; a.free_override = 1;
    CHECK(a.step(std::vector<double>(8, 5.0), false).empty());                    // the only free slot is the reserve: no fill
    Sim b(1, 8, 8, no_ceiling()); b.cfg.reserve_slots = 1; b.free_override = 2;
    CHECK(count_kind(b.step(std::vector<double>(8, 5.0), false), TierOp::kLoad) == 1);
    Sim c(1, 4, 4, no_ceiling()); c.cfg.reserve_slots = 1; c.cfg.cooldown_rounds = 0; c.on = {1, 1, 1, 1};   // full, nothing to promote
    auto ops = c.step({4, 3, 5, 6}, false);                                       // top-up: evict the coldest resident
    CHECK(ops.size() == 1 && ops[0].kind == TierOp::kEvict && ops[0].expert == 1);
}

// note_failed: a rejected LOAD can be re-planned next round; without the call it waits out the cooldown.
static void t_note_failed() {
    g_test = "note_failed";
    auto first_reload = [](bool call) {
        Sim s(1, 2, 1, no_ceiling()); s.cfg.cooldown_rounds = 5;
        auto ops = s.step({10, 1}, false);                                        // round 0: LOAD expert 0, "rejected" (not applied)
        CHECK(ops.size() == 1 && ops[0].kind == TierOp::kLoad && ops[0].expert == 0);
        std::string err;
        if (call) { CHECK(note_failed(s.st, 0, 0, &err) && err.empty()); s.last[0] = INT64_MIN / 2; }
        for (int r = 1; r < 10; ++r)
            for (auto& o : s.step({10, 1}, false)) if (o.kind == TierOp::kLoad && o.expert == 0) return r;
        return -1;
    };
    CHECK(first_reload(true) == 1);                                               // retried in the very next round
    CHECK(first_reload(false) == 6);                                              // 0 + cooldown 5 + 1
    TierPolicyState fresh; std::string err;
    CHECK(!note_failed(fresh, 0, 0, &err) && !err.empty());                       // unseeded: no-op, error
    Sim s(2, 4, 2, no_ceiling()); s.step(std::vector<double>(8, 1.0));
    const auto before = s.st.last_move;
    for (auto [l, e] : {std::pair<int,int>{-1, 0}, {2, 0}, {0, -1}, {0, 4}}) {
        err.clear(); CHECK(!note_failed(s.st, l, e, &err) && !err.empty()); CHECK(s.st.last_move == before);
    }
}

// Coverage report vs a hand-computed case: 2 layers x 4 experts, heat layer0 = {4,2,1,1}, layer1 = {1,1,1,1}.
static void t_coverage_fields() {
    g_test = "coverage fields";
    Sim s(2, 4, 8, no_ceiling()); s.cfg.max_moves_per_round = 2;
    s.on = {1, 0, 0, 0, 0, 0, 0, 0};                                              // already holds (0,0)
    s.free_override = 2;
    auto ops = s.step({4, 2, 1, 1, 1, 1, 1, 1}, false);                           // loads (0,1) then (0,2)? hottest first: (0,1) heat 2, then 1s by index
    CHECK(ops.size() == 2 && ops[0].layer == 0 && ops[0].expert == 1 && ops[1].layer == 0 && ops[1].expert == 2);
    CHECK(s.st.last_layers_occupied == 1);                                        // post-op: layer 0 only
    CHECK(std::fabs(s.st.last_max_layer_share - 7.0 / 8.0) < 1e-12);              // (4+2+1)/8
    Sim t(2, 4, 8, no_ceiling()); t.cfg.max_moves_per_round = 1; t.on = {1, 0, 0, 0, 0, 0, 0, 0};
    t.free_override = 1;
    t.step({1, 1, 1, 1, 1, 1, 1, 1}, false);                                      // loads (0,1): layer 1 untouched
    CHECK(t.st.last_layers_occupied == 1 && std::fabs(t.st.last_max_layer_share - 0.5) < 1e-12);
    Sim u(2, 4, 8, no_ceiling()); u.on = {0, 0, 0, 0, 0, 0, 0, 0}; u.cfg.max_moves_per_round = 4; u.free_override = 2;
    u.step({1, 1, 1, 1, 9, 1, 1, 1}, false);                                      // loads (1,0) heat 9 and (0,0): both layers
    CHECK(u.st.last_layers_occupied == 2);
}

// Empty heat (what an unset --adapt-every/--adapt-swaps yields) is a loud R8 error, not "nothing to do".
static void t_empty_heat_is_error() {
    g_test = "empty heat";
    TierPolicyState st; std::string err; TierPolicyConfig c;
    auto ops = plan_tier_round(2, 4, std::vector<uint8_t>(8, 0), 1, {}, c, st, &err);
    CHECK(ops.empty() && !err.empty() && st.round == 0);
}

// Same-layer mode (the wire has no EVICT and a SWAP is same-layer): on a FULL helper with a hot challenger in each of two
// layers, every returned move is either a free-slot LOAD or a LOAD paired with an EVICT of its own layer; never a bare EVICT.
static void t_same_layer_swaps() {
    g_test = "same-layer swaps";
    TierPolicyConfig c = no_ceiling(); c.same_layer_swaps = true; c.max_moves_per_round = 4; c.cooldown_rounds = 0;
    TierPolicyState st; std::string err;
    std::vector<uint8_t> on(16, 0); on[0] = on[8] = 1;                        // layer 0 expert 0 and layer 1 expert 0 resident, helper FULL (free 0)
    std::vector<double> h(16, 0.0); h[0] = h[8] = 1.0; h[5] = 50.0; h[13] = 50.0; h[3] = 40.0;   // hot misses: (0,5) (1,5) (0,3)
    auto ops = plan_tier_round(2, 8, on, 0, h, c, st, &err);
    CHECK(err.empty() && ops.size() == 4);                                    // budget 4 = two swaps; (0,3) cannot swap: layer 0's only resident is used
    int64_t loads[2] = {0, 0}, evicts[2] = {0, 0};
    for (const TierOp& o : ops) (o.kind == TierOp::kLoad ? loads : evicts)[o.layer]++;
    CHECK(loads[0] == evicts[0] && loads[1] == evicts[1] && loads[0] + loads[1] == 2);
    for (const TierOp& o : ops) if (o.kind == TierOp::kEvict) CHECK(o.expert == 0);   // the cold incumbents, nobody else
    // the coldest resident in the model lives in ANOTHER layer than the challenger: same-layer mode must still evict its own layer's
    {
        TierPolicyState s4; std::vector<double> h4(16, 0.0); h4[0] = 2.0; h4[8] = 0.1; h4[5] = 50.0;   // challenger (0,5); residents (0,0)=2.0 and (1,0)=0.1
        auto o4 = plan_tier_round(2, 8, on, 0, h4, c, s4, &err);
        CHECK(o4.size() == 2 && o4[0].kind == TierOp::kLoad && o4[0].layer == 0 && o4[1].kind == TierOp::kEvict && o4[1].layer == 0 && o4[1].expert == 0);
    }
    // the default mode on the same input: evict-only (no free slot), which the wire cannot carry
    TierPolicyConfig d = no_ceiling(); d.cooldown_rounds = 0; TierPolicyState st2;
    auto ops2 = plan_tier_round(2, 8, on, 0, h, d, st2, &err);
    CHECK(!ops2.empty() && count_kind(ops2, TierOp::kLoad) == 0);
    // same-layer mode never emits an unpaired EVICT even when a layer is over its ceiling at the start
    TierPolicyConfig k = c; k.max_helper_layer_share = 0.01; TierPolicyState st3;
    auto ops3 = plan_tier_round(2, 8, on, 0, h, k, st3, &err);
    CHECK(count_kind(ops3, TierOp::kEvict) == count_kind(ops3, TierOp::kLoad));
}

// Ceiling = gate, not restorer.  Layer 0 heat {3,5,1,1} (total 10), helper holds (0,0) = 0.30, no free slot, same-layer mode.
// Ceiling 0.08: the layer is over it, so it is reported and FROZEN (the hot-for-cold swap (0,1)<->(0,0) would raise the
// share to 0.5).  Ceiling 0.5: not over it, and the swap is allowed.
static void t_over_ceiling_frozen() {
    g_test = "over-ceiling frozen";
    const std::vector<double> h = {3, 5, 1, 1, 1, 1, 1, 1};
    const std::vector<uint8_t> on = {1, 0, 0, 0, 0, 0, 0, 0};
    for (double ceil : {0.08, 0.5}) {
        TierPolicyConfig c; c.max_helper_layer_share = ceil; c.same_layer_swaps = true; c.cooldown_rounds = 0; c.max_moves_per_round = 4;
        TierPolicyState st; std::string err;
        auto ops = plan_tier_round(2, 4, on, 0, h, c, st, &err);
        CHECK(err.empty());
        if (ceil < 0.1) {
            CHECK(st.last_layers_over_ceiling == 1);                              // 0.30 > 0.08
            CHECK(ops.empty());                                                   // no op touches the frozen layer
            for (const TierOp& o : ops) CHECK(o.layer != 0);
        } else {
            CHECK(st.last_layers_over_ceiling == 0);                              // post-op share 0.5 <= 0.5
            CHECK(ops.size() == 2 && ops[0].layer == 0 && ops[1].layer == 0);     // the swap goes through
        }
    }
}

int main() {
    t_note_failed(); t_coverage_fields(); t_empty_heat_is_error();
    t_R1_loads_within_free_slots(); t_R2_evict_sparingly(); t_R3_move_budget(); t_R4_cooldown();
    t_R5_ceiling_clustering(); t_R5_over_at_start_only_lowers(); t_R6_no_pingpong(); t_R7_deterministic_ties();
    t_R8_validation(); t_R9_authoritative_membership(); t_reserve(); t_same_layer_swaps(); t_over_ceiling_frozen();
    std::printf(g_fail ? "tier_plan_test: %d FAILED checks\n" : "tier_plan_test: all checks passed\n", g_fail);
    return g_fail != 0;
}
