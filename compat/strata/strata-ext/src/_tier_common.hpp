// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Private to tier_plan.cpp and tier_ladder.cpp: the parts of a policy round both planners share, so the heat smoothing,
// the config bounds and the state's shape rule each have ONE definition (a second copy is how two planners drift).
// Not installed, not included by anything outside src/.
#include "strata/seabraid/tier_plan.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace seabraid::detail {

inline bool tier_fail(std::string* err, const char* why) {
    if (err) *err = std::string("plan_tier_round: ") + why;
    return false;
}

// The pieces of validate_round, one per input family (each is its own function to keep every one small and testable by
// reading; validate_round runs them in the order the checks always ran, so the first failure and its message are unchanged).
inline bool check_shape(int64_t L, int64_t E, size_t n_heat, std::string* err) {
    if (L <= 0 || E <= 0) return tier_fail(err, "n_layers and n_expert must be positive");
    if (L > INT32_MAX || E > INT32_MAX || L > (int64_t(1) << 26) / E) return tier_fail(err, "implausible model shape");
    if (n_heat != (size_t) (L * E)) return tier_fail(err, "heat size != n_layers * n_expert");
    return true;
}

inline bool check_heat(const std::vector<double>& heat, std::string* err) {
    for (double h : heat)
        if (!std::isfinite(h) || h < 0.0) return tier_fail(err, "heat has a negative or non-finite entry");
    return true;
}

inline bool check_config(const TierPolicyConfig& c, std::string* err) {
    if (c.max_moves_per_round < 0 || c.cooldown_rounds < 0 || c.reserve_slots < 0 || c.ladder_boundaries < 0)
        return tier_fail(err, "negative config");
    if (!std::isfinite(c.promote_margin) || c.promote_margin < 1.0) return tier_fail(err, "promote_margin must be >= 1");
    if (!std::isfinite(c.max_helper_layer_share) || c.max_helper_layer_share < 0.0 || c.max_helper_layer_share > 1.0)
        return tier_fail(err, "max_helper_layer_share must be in [0, 1]");
    return true;
}

inline bool check_state(size_t n, const TierPolicyState& st, std::string* err) {
    if (!st.heat.empty() && (st.heat.size() != n || st.last_move.size() != n))
        return tier_fail(err, "state was built for a different shape");
    return true;
}

// Fail closed on everything both planners divide, compare or index with: the shape, the heat, the config, the state.
inline bool validate_round(int64_t L, int64_t E, size_t n_heat, const std::vector<double>& heat, const TierPolicyConfig& c,
                           const TierPolicyState& st, std::string* err) {
    return check_shape(L, E, n_heat, err) && check_heat(heat, err) && check_config(c, err) &&
           check_state((size_t) (L * E), st, err);
}

// The coverage report both planners write into `st` from the post-move membership (the same view the ceiling is judged
// on): layers holding any `in_set` cell, the largest held/total share, and the layers over the ceiling.  `in_set(i)` says
// whether flat cell i is on the set being reported (the helper for the two-set round, rung 1 for the ladder).
template <class InSet>
inline void record_coverage(TierPolicyState& st, int64_t L, int64_t E, const std::vector<double>& smooth,
                            const TierPolicyConfig& c, InSet in_set) {
    st.last_layers_occupied = 0;
    st.last_max_layer_share = 0.0;
    st.last_layers_over_ceiling = 0;   // the one place this is computed; same comparison as Planner::cap() and phase 1
    for (int64_t l = 0; l < L; ++l) {
        double tot = 0.0, held = 0.0;
        bool any = false;
        for (int64_t e = 0; e < E; ++e) {
            const size_t i = (size_t) (l * E + e);
            tot += smooth[i];
            if (in_set(i)) { held += smooth[i]; any = true; }
        }
        st.last_layers_occupied += any;
        if (tot > 0.0) st.last_max_layer_share = std::max(st.last_max_layer_share, held / tot);
        st.last_layers_over_ceiling += held > c.max_helper_layer_share * tot;
    }
}

// The round's smoothed heat (kHeatDecay), computed into a local: `state` changes only once the round succeeds.  A fresh
// state seeds with the raw heat and gets a last_move of kNever for every cell in `fresh_last_move`.
inline std::vector<double> smooth_heat(const std::vector<double>& heat, const TierPolicyState& st, std::vector<int64_t>& fresh_last_move) {
    std::vector<double> smooth = heat;
    if (!st.heat.empty())
        for (size_t i = 0; i < smooth.size(); ++i) smooth[i] = kHeatDecay * st.heat[i] + (1.0 - kHeatDecay) * heat[i];
    else
        fresh_last_move.assign(heat.size(), TierPolicyState::kNever);
    return smooth;
}

}  // namespace seabraid::detail
