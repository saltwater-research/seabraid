// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// M3b: the N-rung tier policy (plan_tier_round's rung_of overload, strata-ext/src/tier_ladder.cpp).  Host only:
//   g++ -std=c++17 -O1 -Wall -Wextra -Istrata-ext/include tests/tier_ladder_test.cpp strata-ext/src/tier_ladder.cpp
//   (continued) strata-ext/src/tier_plan.cpp -o tier_ladder_test
// Ladder plays the transport: it hands the policy the authoritative rung_of each round, applies the moves, and CHECKS
// the per-round invariants on EVERY step of EVERY test (one rung per move, from == the cell's rung, no cell twice,
// every drop directly after a same-layer promotion it pairs with, the budget, the test's own cooldown record), so a
// violation fails whichever test was running.  Sabotage runs: tests/SABOTAGE.md, "tier_ladder_test".
#include "strata/seabraid/tier_plan.hpp"

#include <cstdio>
#include <set>
#include <string>
#include <vector>

using namespace seabraid;

static int g_fail = 0;
static const char* g_test = "?";
#define CHECK(c) do { if (!(c)) { ++g_fail; std::printf("  FAIL [%s] %s:%d  %s\n", g_test, __FILE__, __LINE__, #c); } } while (0)

static TierPolicyConfig cfg0() { TierPolicyConfig c; c.max_helper_layer_share = 1.0; c.cooldown_rounds = 0; return c; }

struct Ladder {
    int64_t L, E;
    int R;
    TierPolicyConfig cfg;
    TierPolicyState st;
    std::vector<int8_t> rung;
    std::vector<int64_t> free;
    std::vector<int64_t> last;
    int64_t round = 0;
    Ladder(int64_t l, int64_t e, int r, TierPolicyConfig c)
        : L(l), E(e), R(r), cfg(c), rung((size_t) (l * e), (int8_t) (r - 1)), free((size_t) r, 0), last((size_t) (l * e), INT64_MIN / 2) {}
    void at(int l, int e, int r) { rung[(size_t) (l * E + e)] = (int8_t) r; }
    int8_t of(int l, int e) const { return rung[(size_t) (l * E + e)]; }

    std::vector<TierMove> step(const std::vector<double>& heat, bool apply = true) {
        std::string err;
        auto mv = plan_tier_round(L, E, rung, free, heat, cfg, st, &err);
        CHECK(err.empty());
        CHECK((int) mv.size() <= cfg.max_moves_per_round);
        std::set<int64_t> seen;
        for (size_t k = 0; k < mv.size(); ++k) {
            const TierMove& m = mv[k];
            const int64_t i = (int64_t) m.layer * E + m.expert;
            CHECK(m.layer >= 0 && m.layer < L && m.expert >= 0 && m.expert < E);
            CHECK(m.to == m.from - 1 || m.to == m.from + 1);                  // one rung per move
            CHECK(m.from == rung[(size_t) i]);                                // planned against the snapshot
            CHECK(seen.insert(i).second);                                     // at most once per round
            CHECK(round - last[(size_t) i] > cfg.cooldown_rounds);            // cooldown, by the test's own record
            if (m.to == m.from + 1) {                                         // a drop: paired with the promotion just before it
                CHECK(k > 0 && mv[k - 1].to == m.from && mv[k - 1].from == m.to && mv[k - 1].layer == m.layer);
            }
        }
        if (apply)
            for (const TierMove& m : mv) { rung[(size_t) (m.layer * E + m.expert)] = m.to; last[(size_t) (m.layer * E + m.expert)] = round; }
        ++round;
        return mv;
    }
};

static bool has(const std::vector<TierMove>& v, int l, int e, int from, int to) {
    for (const TierMove& m : v) if (m.layer == l && m.expert == e && m.from == from && m.to == to) return true;
    return false;
}

// P1 at the top boundary: card 2's hot expert promotes to card 1, card 1's coldest of that layer drops to card 2.
static void t_promote_to_card1() {
    g_test = "promote card 2 -> card 1";
    Ladder s(2, 4, 3, cfg0());
    s.at(0, 0, 0); s.at(0, 1, 0); s.at(0, 2, 1);
    s.at(1, 0, 0); s.at(1, 1, 1);
    std::vector<double> h = {1, 50, 100, 0,   20, 21, 0, 0};
    auto mv = s.step(h);
    CHECK(mv.size() == 2 && has(mv, 0, 2, 1, 0) && has(mv, 0, 0, 0, 1));
    CHECK(mv.size() == 2 && mv[0].to == 0 && mv[1].to == 1);                 // promotion first, its drop right after
    CHECK(s.of(1, 1) == 1);                                                   // layer 1: 21 < 1.25 x 20, no move
}

// Same-layer only: a cold card-1 expert in ANOTHER layer is never the one displaced.
static void t_same_layer_limit() {
    g_test = "same-layer limit";
    Ladder s(2, 4, 3, cfg0());
    s.at(0, 0, 0); s.at(0, 1, 0); s.at(0, 2, 1);
    s.at(1, 0, 0);
    std::vector<double> h = {90, 95, 100, 0,   1, 0, 0, 0};
    auto mv = s.step(h);
    CHECK(mv.empty());                                                        // 100 < 1.25 x 90, and (1,0) is not eligible
    for (const TierMove& m : mv) CHECK(!(m.layer == 1 && m.expert == 0));
}

// The live-disjointness gap (tier_plan.cpp:90-96): an expert on a higher rung is never a challenger for a lower one.
static void t_no_challenger_from_above() {
    g_test = "no challenger from a higher rung";
    Ladder s(1, 6, 3, cfg0());
    s.at(0, 0, 0); s.at(0, 1, 0);                                             // card 1: two very hot experts
    s.at(0, 2, 1); s.at(0, 3, 1);                                             // card 2: cold ones
    s.free[1] = 2;                                                            // and card 2 has free slots
    std::vector<double> h = {1000, 900, 1, 2, 5, 0};
    for (int r = 0; r < 4; ++r) {
        auto mv = s.step(h);
        for (const TierMove& m : mv) CHECK(!(m.expert <= 1 && m.to == 2) && !(m.expert <= 1 && m.from == 2));
        for (const TierMove& m : mv) CHECK(!(m.expert <= 1 && m.to == 1));   // card 1's experts never "load" onto card 2
    }
    CHECK(s.of(0, 0) == 0 && s.of(0, 1) == 0);
    CHECK(s.of(0, 4) == 1);                                                   // the fallback's hot one filled card 2 instead
}

// Cascade over three cards: one rung per move, so card 3's hottest reaches card 1 in two rounds, and each displaced
// resident lands exactly one rung down.
static void t_cascade_three_cards() {
    g_test = "cascade, card 3 -> card 2 -> card 1";
    Ladder s(1, 8, 4, cfg0());
    s.at(0, 0, 0); s.at(0, 1, 0); s.at(0, 2, 1); s.at(0, 3, 1); s.at(0, 4, 2); s.at(0, 5, 2);
    std::vector<double> h = {10, 60, 40, 2, 100, 1, 0, 0};
    auto mv = s.step(h);
    CHECK(mv.size() == 4);
    CHECK(has(mv, 0, 2, 1, 0) && has(mv, 0, 0, 0, 1));                        // boundary 0 first
    CHECK(has(mv, 0, 4, 2, 1) && has(mv, 0, 3, 1, 2));                        // boundary 1: card 3's hottest to card 2, not card 1
    CHECK(mv.size() == 4 && mv[0].to == 0 && mv[2].to == 1);
    mv = s.step(h);
    CHECK(has(mv, 0, 4, 1, 0) && has(mv, 0, 2, 0, 1));                        // then card 2 -> card 1 (100 > 1.25 x 40)
    CHECK(s.of(0, 4) == 0 && s.of(0, 1) == 0);
}

// Throttle: max_moves_per_round counts MOVES, and a displacing pair needs two of them.
static void t_throttle() {
    g_test = "throttle";
    TierPolicyConfig c = cfg0(); c.max_moves_per_round = 3;
    Ladder s(3, 2, 3, c);
    for (int l = 0; l < 3; ++l) { s.at(l, 0, 0); s.at(l, 1, 1); }
    std::vector<double> h = {1, 10,  1, 20,  1, 30};
    auto mv = s.step(h);
    CHECK(mv.size() == 2 && has(mv, 2, 1, 1, 0));                             // hottest first; a third move cannot pair
    s.cfg.max_moves_per_round = 1;
    mv = s.step(h);
    CHECK(mv.empty());                                                        // a pair never splits to fit the budget
    s.free[0] = 1;
    mv = s.step(h);
    CHECK(mv.size() == 1 && mv[0].to == 0 && mv[0].from == 1);                // a fill is one move
}

// Hysteresis: the margin and the cooldown.
static void t_hysteresis() {
    g_test = "margin and cooldown";
    TierPolicyConfig c = cfg0(); c.cooldown_rounds = 3;
    Ladder a(1, 2, 3, c);                                                     // a fresh state judges the raw heat
    a.at(0, 0, 0); a.at(0, 1, 1);
    for (int r = 0; r < 3; ++r) CHECK(a.step({10, 12}).empty());              // 12 < 1.25 x 10, round after round
    Ladder s(1, 2, 3, c);
    s.at(0, 0, 0); s.at(0, 1, 1);
    CHECK(s.step({10, 13}).size() == 2);                                      // 13 > 12.5: the pair moves (round 0)
    for (int r = 0; r < 3; ++r) CHECK(s.step({100, 1}).empty());              // heat flips: both frozen for cooldown_rounds
    CHECK(s.step({100, 1}).size() == 2);                                      // round 4: thawed, and it moves back
}

// ladder_boundaries: 1 plans card 2 -> card 1 only (the live first move); 0 plans every boundary.
static void t_boundaries() {
    g_test = "ladder_boundaries";
    TierPolicyConfig c = cfg0(); c.ladder_boundaries = 1;
    Ladder s(1, 4, 3, c);
    s.at(0, 0, 0); s.at(0, 1, 1);
    std::vector<double> h = {10, 1, 50, 0};                                   // the fallback's (0,2) out-heats card 2's (0,1)
    CHECK(s.step(h, false).empty());
    s.cfg.ladder_boundaries = 0;
    auto mv = s.step(h, false);
    CHECK(mv.size() == 2 && has(mv, 0, 2, 2, 1) && has(mv, 0, 1, 1, 2));
}

// The ceiling gates card 2 (rung 1) and never card 1.
static void t_ceiling_rungs() {
    g_test = "ceiling: card 2 yes, card 1 no";
    TierPolicyConfig c = cfg0(); c.max_helper_layer_share = 0.10;
    Ladder s(1, 4, 3, c);
    s.at(0, 0, 0); s.at(0, 1, 1);
    auto mv = s.step({1, 80, 0, 19});                                         // card 1 takes 80% of the layer: allowed
    CHECK(has(mv, 0, 1, 1, 0));
    Ladder t(1, 4, 3, c);
    t.at(0, 0, 0); t.at(0, 1, 1);
    mv = t.step({80, 1, 0, 19});                                              // (0,3) to card 2 would make it 19%: blocked
    CHECK(mv.empty());
}

static void t_fail_closed() {
    g_test = "fail closed";
    TierPolicyConfig c = cfg0();
    TierPolicyState st;
    std::string err;
    std::vector<double> h(8, 1.0);
    std::vector<int8_t> r(8, 2);
    auto bad = [&](const std::vector<int8_t>& rr, const std::vector<int64_t>& fr, const std::vector<double>& hh, const TierPolicyConfig& cc) {
        err.clear();
        auto mv = plan_tier_round(2, 4, rr, fr, hh, cc, st, &err);
        CHECK(mv.empty() && !err.empty() && st.round == 0 && st.heat.empty());
    };
    bad(std::vector<int8_t>(7, 2), {0, 0, 0}, h, c);                          // rung_of size
    bad(r, {0}, h, c);                                                        // one rung
    bad(r, {0, -1, 0}, h, c);                                                 // negative free slots
    std::vector<int8_t> r3 = r; r3[3] = 3;
    bad(r3, {0, 0, 0}, h, c);                                                 // a rung that does not exist
    std::vector<int8_t> rn = r; rn[0] = -1;
    bad(rn, {0, 0, 0}, h, c);
    std::vector<double> hn = h; hn[2] = -1.0;
    bad(r, {0, 0, 0}, hn, c);                                                 // negative heat
    TierPolicyConfig cb = c; cb.ladder_boundaries = -1;
    bad(r, {0, 0, 0}, h, cb);
    auto ok = plan_tier_round(2, 4, r, {0, 0, 0}, h, c, st, &err);            // and the same call with good input works
    CHECK(err.empty() && st.round == 1);
    (void) ok;
}

int main() {
    t_promote_to_card1();
    t_same_layer_limit();
    t_no_challenger_from_above();
    t_cascade_three_cards();
    t_throttle();
    t_hysteresis();
    t_boundaries();
    t_ceiling_rungs();
    t_fail_closed();
    std::printf("tier_ladder_test: %s (%d failures)\n", g_fail ? "FAIL" : "all checks passed", g_fail);
    return g_fail ? 1 : 0;
}
