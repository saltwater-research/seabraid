// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// R6 golden generator: drives the legacy C++ ladder (plan_tier_round, rung_of overload) plus to_pairs over seeded
// multi-round cases, all moves acknowledged, and writes inputs AND chosen moves to a text file.  The Rust planner
// (src/placement/tests/parity.rs) replays the file and must choose the same moves.
//   g++ -std=c++17 -O1 -Wall -Wextra -Istrata-ext/include tests/ladder_golden_gen.cpp strata-ext/src/tier_plan.cpp
//   (continued) strata-ext/src/tier_ladder.cpp strata-ext/src/retier_loop.cpp strata-ext/src/_retier_ladder.cpp -pthread
//   (continued) -Istrata-ext/../../../include -o ladder_golden_gen && ./ladder_golden_gen out.txt
//
// WHY these cases: 200 seeds vary shape (1-4 layers, 4-16 experts, 3-5 rungs, so 1-3 card boundaries), every policy knob
// (move budget incl. odd and zero, margin, cooldown, helper-share ceiling, ladder_boundaries) and the heat texture:
// small integer heat gives exact ties (index tie-break), zeros (never promoted), spikes (margin and ceiling bite) and
// a drifting multi-round signal (smoothing, cooldown expiry).  free_slots are all 0: Rust never proposes fills (planner.rs
// header), so the comparison is exactly the paired subset.  Heat is integral so the file round-trips bit-exactly; the
// policy's own smoothing yields the fractions.  Seeds are splitmix64, fully deterministic, no clock, no I/O but the file.
#include "strata/seabraid/retier_loop.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace seabraid;

namespace {

constexpr int kCases = 200, kRounds = 6;

struct Rng {
    uint64_t s;
    uint64_t next() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    int below(int n) { return (int) (next() % (uint64_t) n); }
};

struct Case {
    uint64_t seed;
    int L, E, R;
    TierPolicyConfig cfg;
    std::vector<int8_t> rung;
};

[[noreturn]] void die(const char* why, uint64_t seed) {
    std::fprintf(stderr, "ladder_golden_gen: seed %llu: %s\n", (unsigned long long) seed, why);
    std::exit(1);
}

TierPolicyConfig make_cfg(Rng& r) {
    static const double kMargin[] = {1.0, 1.1, 1.25, 1.5, 2.0}, kShare[] = {0.08, 0.2, 0.5, 1.0};
    static const int kCool[] = {0, 1, 2, 8}, kBound[] = {0, 0, 1, 2};
    TierPolicyConfig c;
    c.max_moves_per_round = r.below(9);
    c.promote_margin = kMargin[r.below(5)];
    c.cooldown_rounds = kCool[r.below(4)];
    c.max_helper_layer_share = kShare[r.below(4)];
    c.ladder_boundaries = kBound[r.below(4)];
    return c;
}

// WHY the clamp: the last rung is the fallback.  C++ also plans fallback <-> card swaps (a card cell displaced to RAM);
// Rust deliberately does not (planner.rs header; validate_promotion rejects hi >= fallback - 1), so those moves would
// differ by design.  Parity is claimed over the card boundaries 0..R-3 only, which is what ladder_boundaries expresses
// (0 would mean "every boundary" here, hence R >= 3).  The unclamped divergence was seen first: seed 10 round 3.
Case make_case(uint64_t seed, Rng& r) {
    Case c{seed, 1 + r.below(4), 4 + r.below(13), 3 + r.below(3), make_cfg(r), {}};
    int& b = c.cfg.ladder_boundaries;
    if (b == 0 || b > c.R - 2) b = c.R - 2;
    for (int i = 0; i < c.L * c.E; ++i) c.rung.push_back((int8_t) r.below(c.R));
    return c;
}

std::vector<double> make_heat(const Case& c, Rng& r, bool coarse) {
    std::vector<double> h((size_t) (c.L * c.E));
    for (double& v : h) {
        const int pick = r.below(16);
        v = pick < 4 ? 0 : coarse ? r.below(4) : pick == 15 ? 200 + r.below(300) : r.below(40);
    }
    return h;
}

void put_ints(std::FILE* f, const char* tag, const std::vector<int8_t>& v) {
    std::fprintf(f, "%s", tag);
    for (int8_t x : v) std::fprintf(f, " %d", x);
    std::fprintf(f, "\n");
}

void put_round(std::FILE* f, const std::vector<double>& heat, const LadderRound& lr) {
    std::fprintf(f, "heat");
    for (double v : heat) std::fprintf(f, " %d", (int) v);
    std::fprintf(f, "\nmoves");
    for (const LadderPair& p : lr.pairs) std::fprintf(f, " %d:%d:%d:%d", p.layer, p.up, p.down, p.hi);
    std::fprintf(f, "\n");
}

// All-ACK application: the pair's up cell now sits on hi, its down cell on hi + 1.
void apply(Case& c, const LadderRound& lr) {
    for (const LadderPair& p : lr.pairs) {
        c.rung[(size_t) (p.layer * c.E + p.up)] = p.hi;
        c.rung[(size_t) (p.layer * c.E + p.down)] = (int8_t) (p.hi + 1);
    }
}

void run_case(std::FILE* f, Case c, Rng& r) {
    const bool coarse = c.seed % 3 == 0;
    std::fprintf(f, "case %llu %d %d %d %d %.17g %d %.17g %d\n", (unsigned long long) c.seed, c.L, c.E, c.R,
                 c.cfg.max_moves_per_round, c.cfg.promote_margin, c.cfg.cooldown_rounds, c.cfg.max_helper_layer_share,
                 c.cfg.ladder_boundaries);
    put_ints(f, "owner", c.rung);
    TierPolicyState st;
    const std::vector<int64_t> no_free((size_t) c.R, 0);
    for (int round = 0; round < kRounds; ++round) {
        const std::vector<double> heat = make_heat(c, r, coarse);
        std::string err;
        const std::vector<TierMove> mv = plan_tier_round(c.L, c.E, c.rung, no_free, heat, c.cfg, st, &err);
        if (!err.empty()) die(err.c_str(), c.seed);
        const LadderRound lr = to_pairs(mv);
        if (!lr.dropped.empty()) die("a move has no live pair form", c.seed);
        put_round(f, heat, lr);
        apply(c, lr);
    }
    put_ints(f, "final", c.rung);
}

}  // namespace

int main(int argc, char** argv) {
    std::FILE* f = argc > 1 ? std::fopen(argv[1], "w") : stdout;
    if (!f) { std::perror("ladder_golden_gen: open"); return 1; }
    for (int i = 1; i <= kCases; ++i) {
        Rng r{(uint64_t) i};
        run_case(f, make_case((uint64_t) i, r), r);
    }
    return std::fclose(f) == 0 ? 0 : 1;
}
