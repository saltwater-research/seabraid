// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// M3b: the tier policy over N rungs (the rung_of overload of plan_tier_round).  Contract and rules: tier_plan.hpp.
// Owner unchanged (tier_plan.hpp's design gate): the policy constants are TierPolicyConfig's, the smoothing and the
// fail-closed checks are _tier_common.hpp's, shared with the two-set round in tier_plan.cpp.
//
// Why a second Round and not a generalised tier_plan.cpp: the two-set round has phases the ladder must NOT have
// (ceiling restoration and the reserve top-up emit unpaired EVICTs, which have no wire form for a card-to-card move), and
// the live two-set path (--retier on) is measured and stays bit-for-bit what it was.  The ladder round is the cascade
// alone: fill, then same-layer displacement, boundary by boundary.
#include "strata/seabraid/tier_plan.hpp"
#include "_tier_common.hpp"

#include <algorithm>

namespace seabraid {
namespace {

using detail::tier_fail;

class Ladder {
public:
    Ladder(int64_t L, int64_t E, const std::vector<int8_t>& rung, const std::vector<int64_t>& free_slots,
           const std::vector<double>& smooth, const TierPolicyConfig& c, const TierPolicyState& st)
        : L_(L), E_(E), R_((int) free_slots.size()), rung_(rung), free_(free_slots), h_(smooth), c_(c), st_(st),
          used_(rung.size(), 0), total_((size_t) L, 0.0), held_((size_t) (R_ * L), 0.0) {
        for (size_t i = 0; i < rung.size(); ++i) {
            total_[i / (size_t) E_] += h_[i];
            held_[(size_t) rung[i] * (size_t) L_ + i / (size_t) E_] += h_[i];
        }
    }

    std::vector<TierMove> run() {
        const int boundaries = R_ - 1;
        const int nb = c_.ladder_boundaries > 0 ? std::min(boundaries, c_.ladder_boundaries) : boundaries;
        // Top boundary first: a promotion into card 1 saves the most expensive miss (design section 4, "promotions first").
        for (int hi = 0; hi < nb && budget() > 0; ++hi) {
            fill(hi);
            displace(hi);
        }
        return moves_;
    }
    const std::vector<int64_t>& moved() const { return moved_; }

private:
    int64_t layer(int64_t i) const { return i / E_; }
    int budget() const { return c_.max_moves_per_round - (int) moves_.size(); }
    double& held(int r, int64_t l) { return held_[(size_t) r * (size_t) L_ + (size_t) l]; }
    // Card 1 (rung 0) is never capped: holding the hottest experts IS its job (run7, design section 3).  The cap is the
    // safety gate the two-set round applies to card 2, kept for every card below card 1; the fallback has no cap.
    bool ceiling_ok(int r, int64_t l, double after) const {
        if (r == 0 || r == R_ - 1) return true;
        return after <= c_.max_helper_layer_share * total_[(size_t) l];
    }
    bool movable(int64_t i) const {
        return !used_[(size_t) i] && st_.round - st_.last_move[(size_t) i] > c_.cooldown_rounds;
    }
    void move(int64_t i, int from, int to) {
        moves_.push_back(TierMove{(int32_t) layer(i), (int32_t) (i % E_), (int8_t) from, (int8_t) to});
        moved_.push_back(i);
        used_[(size_t) i] = 1;
        held(from, layer(i)) -= h_[(size_t) i];
        held(to, layer(i)) += h_[(size_t) i];
        rung_[(size_t) i] = (int8_t) to;
    }
    // Challengers for boundary hi: on rung hi + 1 ONLY (a cell on a higher rung is never a challenger for a lower one),
    // hot enough to matter, free to move; hottest first, ties by index.
    std::vector<int64_t> challengers(int hi) const {
        std::vector<int64_t> v;
        for (size_t i = 0; i < rung_.size(); ++i)
            if (rung_[i] == hi + 1 && h_[i] > 0.0 && movable((int64_t) i)) v.push_back((int64_t) i);
        std::stable_sort(v.begin(), v.end(), [&](int64_t a, int64_t b) { return h_[(size_t) a] > h_[(size_t) b]; });
        return v;
    }

    // Rule 1 as in the two-set round: only slots free at the START of the round take a promotion without a drop.
    void fill(int hi) {
        int64_t n = free_[(size_t) hi];
        for (int64_t c : challengers(hi)) {
            if (n <= 0 || budget() < 1) break;
            if (!ceiling_ok(hi, layer(c), held(hi, layer(c)) + h_[(size_t) c])) continue;
            move(c, hi + 1, hi);
            --n;
        }
    }

    // Per layer, rung hi's movable residents coldest first (built once per displace(): a used one is skipped there).
    std::vector<std::vector<int64_t>> residents_by_layer(int hi) const {
        std::vector<std::vector<int64_t>> inc((size_t) L_);
        for (size_t i = 0; i < rung_.size(); ++i)
            if (rung_[i] == hi && movable((int64_t) i)) inc[(size_t) layer((int64_t) i)].push_back((int64_t) i);
        for (auto& v : inc)
            std::stable_sort(v.begin(), v.end(), [&](int64_t a, int64_t b) { return h_[(size_t) a] < h_[(size_t) b]; });
        return inc;
    }

    // Rule P1: each challenger, hottest first, displaces the coldest movable resident of rung hi in ITS layer, if it
    // out-heats it by the margin.  The pair is emitted adjacent (promotion, then its drop) so the transport can pair it.
    void displace(int hi) {
        const std::vector<std::vector<int64_t>> inc = residents_by_layer(hi);
        for (int64_t c : challengers(hi)) {
            if (budget() < 2) break;
            const int64_t l = layer(c);
            const double hc = h_[(size_t) c];
            for (int64_t i : inc[(size_t) l]) {
                if (used_[(size_t) i]) continue;
                if (!(hc > c_.promote_margin * h_[(size_t) i])) break;   // ascending: no colder resident remains
                if (!ceiling_ok(hi, l, held(hi, l) + hc - h_[(size_t) i])) continue;   // a warmer resident raises it less
                move(c, hi + 1, hi);
                move(i, hi, hi + 1);
                break;
            }
        }
    }

    int64_t L_, E_;
    int R_;
    std::vector<int8_t> rung_;          // updated as moves are chosen (an expert moves at most once: used_)
    std::vector<int64_t> free_;
    const std::vector<double>& h_;
    const TierPolicyConfig& c_;
    const TierPolicyState& st_;
    std::vector<uint8_t> used_;
    std::vector<double> total_, held_;  // per layer: all heat / heat per (rung, layer)
    std::vector<TierMove> moves_;
    std::vector<int64_t> moved_;
};

// The ladder-only input checks (the shared shape/heat/config/state ones are detail::validate_round's).
bool validate_ladder_inputs(int64_t n, const std::vector<int8_t>& rung_of, const std::vector<int64_t>& free_slots,
                            std::string* err) {
    if (rung_of.size() != (size_t) n) return tier_fail(err, "rung_of size != n_layers * n_expert");
    // int8_t rungs; 64 is far past any machine and keeps (rungs x layers) heat arrays small.
    if (free_slots.size() < 2 || free_slots.size() > 64) return tier_fail(err, "the ladder needs 2..64 rungs");
    for (int64_t f : free_slots)
        if (f < 0) return tier_fail(err, "a rung's free_slots is negative");
    for (int8_t r : rung_of)
        if (r < 0 || (size_t) r >= free_slots.size()) return tier_fail(err, "rung_of names a rung outside [0, R)");
    return true;
}

}  // namespace

std::vector<TierMove> plan_tier_round(int64_t n_layers, int64_t n_expert, const std::vector<int8_t>& rung_of,
                                      const std::vector<int64_t>& free_slots, const std::vector<double>& heat,
                                      const TierPolicyConfig& cfg, TierPolicyState& state, std::string* err) {
    if (err) err->clear();
    if (!detail::validate_round(n_layers, n_expert, heat.size(), heat, cfg, state, err)) return {};
    if (!validate_ladder_inputs(n_layers * n_expert, rung_of, free_slots, err)) return {};

    TierPolicyState fresh;
    const bool seeded = !state.heat.empty();
    std::vector<double> smooth = detail::smooth_heat(heat, state, fresh.last_move);
    if (!seeded) fresh.round = state.round;
    Ladder lad(n_layers, n_expert, rung_of, free_slots, smooth, cfg, seeded ? state : fresh);
    std::vector<TierMove> moves = lad.run();

    if (!seeded) state.last_move = std::move(fresh.last_move);
    for (int64_t i : lad.moved()) state.last_move[(size_t) i] = state.round;
    state.n_layers = n_layers;
    state.n_expert = n_expert;
    // Coverage of rung 1 (card 2) after the moves, the same fields and comparison the two-set round reports.
    std::vector<int8_t> after = rung_of;
    for (const TierMove& m : moves) after[(size_t) m.layer * (size_t) n_expert + (size_t) m.expert] = m.to;
    const bool tracks_rung1 = free_slots.size() > 2;
    detail::record_coverage(state, n_layers, n_expert, smooth, cfg, [&](size_t i) { return after[i] == 1 && tracks_rung1; });
    state.heat = std::move(smooth);
    ++state.round;
    return moves;
}

}  // namespace seabraid
