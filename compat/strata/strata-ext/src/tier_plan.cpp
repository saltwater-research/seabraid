// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// seabraid D7 tier-move policy: see include/strata/seabraid/tier_plan.hpp for the contract and the design gate.
// The round runs four phases in a fixed priority order, each spending the shared per-round move budget:
//   1. restore the per-layer ceiling (EVICT only)   2. fill free slots (LOAD only, keeping the reserve)
//   3. swap hot challengers for cold incumbents       4. top up the reserve (EVICT only)
// Ceiling first because it is a checked invariant; fills before swaps because a LOAD into an empty slot is pure
// gain; the reserve last because it is a convenience that only buys single-round swaps.
#include "strata/seabraid/tier_plan.hpp"
#include "_tier_common.hpp"

#include <algorithm>
#include <cmath>

namespace seabraid {
namespace {

using detail::tier_fail;
bool fail(std::string* err, const char* why) { return tier_fail(err, why); }

// Fail closed: every number the planner will divide, compare or index with is checked before any state moves.  The
// shape/heat/config/state checks are shared with the N-rung planner (_tier_common.hpp); only the two-set inputs are here.
bool validate(int64_t L, int64_t E, size_t n_on, size_t n_heat, int64_t free_slots, const std::vector<double>& heat,
              const TierPolicyConfig& c, const TierPolicyState& st, std::string* err) {
    if (!detail::validate_round(L, E, n_heat, heat, c, st, err)) return false;
    if (n_on != (size_t) (L * E)) return fail(err, "on_helper size != n_layers * n_expert");
    if (free_slots < 0) return fail(err, "free_slots is negative");
    return true;
}

class Round {
public:
    Round(int64_t L, int64_t E, const std::vector<uint8_t>& on, int64_t free_slots, const std::vector<double>& smooth,
          const TierPolicyConfig& c, const TierPolicyState& st)
        : L_(L), E_(E), free_(free_slots), h_(smooth), c_(c), st_(st), res_(on.size()), used_(on.size(), 0),
          total_((size_t) L, 0.0), held_((size_t) L, 0.0) {
        for (size_t i = 0; i < on.size(); ++i) {
            res_[i] = on[i] != 0;
            total_[i / (size_t) E_] += h_[i];
            if (res_[i]) held_[i / (size_t) E_] += h_[i];
        }
    }

    std::vector<TierOp> run() {
        // Why not lower an over-ceiling layer via swaps?  A hot-for-cold swap always RAISES held heat, so `after <= cap` never holds
        // there; restoring needs an unpaired EVICT, which the wire lacks (cross-layer follow-up, DESIGN-LOG).  So such a layer admits
        // no swap until its smoothed heat drifts back under the cap; last_layers_over_ceiling counts them each round.
        if (!c_.same_layer_swaps) restore_ceiling();   // an unpaired EVICT has no wire form in same-layer mode
        fill_free_slots();
        swap_hot_for_cold();
        if (!c_.same_layer_swaps) top_up_reserve();
        std::sort(loads_.begin(), loads_.end());      // (layer, expert) order = flat order
        std::sort(evicts_.begin(), evicts_.end());
        std::vector<TierOp> ops;
        for (int64_t i : loads_) ops.push_back(op(TierOp::kLoad, i));
        for (int64_t i : evicts_) ops.push_back(op(TierOp::kEvict, i));
        return ops;
    }
    const std::vector<int64_t>& loads() const { return loads_; }
    const std::vector<int64_t>& evicts() const { return evicts_; }

private:
    TierOp op(TierOp::Kind k, int64_t i) const { return TierOp{k, 0, (int32_t) (i / E_), (int32_t) (i % E_)}; }
    int64_t layer(int64_t i) const { return i / E_; }
    int budget() const { return c_.max_moves_per_round - (int) (loads_.size() + evicts_.size()); }
    double cap(int64_t l) const { return c_.max_helper_layer_share * total_[(size_t) l]; }

    // Rule 4: a moved expert is frozen for cooldown_rounds rounds; one moved this round is frozen too.
    bool movable(int64_t i) const {
        return !used_[(size_t) i] && st_.round - st_.last_move[(size_t) i] > c_.cooldown_rounds;
    }
    void do_load(int64_t i) { loads_.push_back(i); used_[(size_t) i] = 1; held_[(size_t) layer(i)] += h_[(size_t) i]; }
    void do_evict(int64_t i) { evicts_.push_back(i); used_[(size_t) i] = 1; held_[(size_t) layer(i)] -= h_[(size_t) i]; }

    // Rule 5: the one gate every LOAD passes.  Judged on the ceiling alone, never on "lower than before": a layer
    // that started over the ceiling therefore takes no LOAD until phase 1 has brought it under, which is stricter
    // than "may only lower it" and keeps the rule a single comparison.
    bool ceiling_ok(int64_t l, double new_held) const { return new_held <= cap(l); }

    // Challengers: not on the helper, hot enough to matter, free to move.  Hottest first, ties by index (rule 7).
    std::vector<int64_t> challengers() const {
        std::vector<int64_t> v;
        for (size_t i = 0; i < res_.size(); ++i)
            if (!res_[i] && h_[i] > 0.0 && movable((int64_t) i)) v.push_back((int64_t) i);
        std::stable_sort(v.begin(), v.end(), [&](int64_t a, int64_t b) { return h_[(size_t) a] > h_[(size_t) b]; });
        return v;
    }
    // Incumbents: on the helper and free to move.  Coldest first, ties by index.
    std::vector<int64_t> incumbents() const {
        std::vector<int64_t> v;
        for (size_t i = 0; i < res_.size(); ++i)
            if (res_[i] && movable((int64_t) i)) v.push_back((int64_t) i);
        std::stable_sort(v.begin(), v.end(), [&](int64_t a, int64_t b) { return h_[(size_t) a] < h_[(size_t) b]; });
        return v;
    }

    // Phase 1.  Layers over the ceiling (worst excess first) evict until under it.  The cheapest sufficient
    // resident goes first (fewest moves, least heat lost); if no single one suffices, the hottest goes, which
    // removes the most share per move.  Cooldown wins over the ceiling: a frozen resident is not evicted, and the
    // layer stays over until it thaws (the round still never raises it).
    void restore_ceiling() {
        for (int64_t l : layers_over_ceiling()) evict_until_under(l);
    }

    // Phase 1's worklist: the layers over the ceiling, worst excess first (ties by index).
    std::vector<int64_t> layers_over_ceiling() const {
        std::vector<int64_t> over;
        for (int64_t l = 0; l < L_; ++l)
            if (held_[(size_t) l] > cap(l)) over.push_back(l);
        std::stable_sort(over.begin(), over.end(), [&](int64_t a, int64_t b) {
            return held_[(size_t) a] - cap(a) > held_[(size_t) b] - cap(b);
        });
        return over;
    }

    // Phase 1 for one layer: evict until it is under the ceiling, out of budget, or nothing movable is left.
    void evict_until_under(int64_t l) {
        while (held_[(size_t) l] > cap(l) && budget() > 0) {
            const int64_t pick = pick_evictee(l);
            if (pick < 0) break;
            do_evict(pick);
        }
    }

    // The resident phase 1 evicts from layer l: the coldest one whose eviction alone gets under the ceiling, else the
    // hottest movable one; -1 when no resident is movable.
    int64_t pick_evictee(int64_t l) const {
        int64_t enough = -1, hottest = -1;
        for (int64_t i = l * E_; i < (l + 1) * E_; ++i) {
            if (!res_[(size_t) i] || !movable(i)) continue;
            if (held_[(size_t) l] - h_[(size_t) i] <= cap(l) && (enough < 0 || h_[(size_t) i] < h_[(size_t) enough])) enough = i;
            if (hottest < 0 || h_[(size_t) i] > h_[(size_t) hottest]) hottest = i;
        }
        return enough >= 0 ? enough : hottest;
    }

    // Phase 2.  Rule 1: only slots free at the START of the round are LOAD targets, and the reserve is kept back.
    void fill_free_slots() {
        const int64_t n_fill = free_ - c_.reserve_slots;
        for (int64_t i : challengers()) {
            if (budget() <= 0 || (int64_t) loads_.size() >= n_fill) break;
            if (ceiling_ok(layer(i), held_[(size_t) layer(i)] + h_[(size_t) i])) do_load(i);
        }
    }

    // Phase 3.  Rules 2 and 6: a challenger displaces the COLDEST incumbent it beats by promote_margin (a
    // resident keeps its place unless beaten by the margin; this is NOT rank_learned_profile's comparator
    // (expert_cache.cpp:221), which puts every resident before every non-resident and would freeze the tier: here
    // incumbency is a ratio, so a hot enough challenger always wins), and only if the ceiling would hold with the swap applied.  With a free slot left
    // (the reserve) the pair is LOAD + EVICT in one round; with none, only the EVICT is emitted and the freed slot
    // takes the best challenger next round (rule 1: no same-round dependency).
    void swap_hot_for_cold() {
        const std::vector<int64_t> inc = incumbents();
        for (int64_t c : challengers()) {
            const bool slot = c_.same_layer_swaps || (int64_t) loads_.size() < free_;   // same-layer mode: the evicted slot is the LOAD's
            if (budget() < (slot ? 2 : 1)) break;
            displace_coldest(c, inc, slot);
        }
    }

    // Phase 3 for one challenger: evict the coldest eligible incumbent it beats (and LOAD the challenger if `slot`).
    void displace_coldest(int64_t c, const std::vector<int64_t>& inc, bool slot) {
        const double hc = h_[(size_t) c];
        for (int64_t i : inc) {
            if (c_.same_layer_swaps && layer(i) != layer(c)) continue;   // the wire swaps within one layer only
            if (used_[(size_t) i]) continue;
            if (!(hc > c_.promote_margin * h_[(size_t) i])) break;       // incumbents ascend: none colder remains
            const double after = held_[(size_t) layer(c)] + hc - (layer(i) == layer(c) ? h_[(size_t) i] : 0.0);
            if (!ceiling_ok(layer(c), after)) continue;
            do_evict(i);
            if (slot) do_load(c);
            break;
        }
    }

    // Phase 4.  Evicts that free slots for future rounds count toward the reserve; the coldest go first.
    void top_up_reserve() {
        int64_t free_after = free_ - (int64_t) loads_.size() + (int64_t) evicts_.size();
        for (int64_t i : incumbents()) {
            if (free_after >= c_.reserve_slots || budget() <= 0) break;
            do_evict(i);
            ++free_after;
        }
    }

    int64_t L_, E_, free_;
    const std::vector<double>& h_;
    const TierPolicyConfig& c_;
    const TierPolicyState& st_;
    std::vector<uint8_t> res_, used_;
    std::vector<double> total_, held_;     // per layer: all heat / heat the helper holds (updated as ops are chosen)
    std::vector<int64_t> loads_, evicts_;
};

}  // namespace

std::vector<TierOp> plan_tier_round(int64_t n_layers, int64_t n_expert, const std::vector<uint8_t>& on_helper,
                                    int64_t free_slots, const std::vector<double>& heat,
                                    const TierPolicyConfig& cfg, TierPolicyState& state, std::string* err) {
    if (err) err->clear();
    if (!validate(n_layers, n_expert, on_helper.size(), heat.size(), free_slots, heat, cfg, state, err)) return {};

    // Smooth into a local first: `state` changes only once the round is known to succeed.
    TierPolicyState fresh;
    const bool seeded = !state.heat.empty();
    std::vector<double> smooth = detail::smooth_heat(heat, state, fresh.last_move);
    if (!seeded) fresh.round = state.round;
    const TierPolicyState& basis = seeded ? state : fresh;

    Round r(n_layers, n_expert, on_helper, free_slots, smooth, cfg, basis);
    std::vector<TierOp> ops = r.run();

    if (!seeded) state.last_move = std::move(fresh.last_move);
    for (int64_t i : r.loads()) state.last_move[(size_t) i] = state.round;
    for (int64_t i : r.evicts()) state.last_move[(size_t) i] = state.round;
    state.n_layers = n_layers;
    state.n_expert = n_expert;
    // Coverage report from the post-op membership (the same view the ceiling is judged on).
    std::vector<uint8_t> after(on_helper.size());
    for (size_t i = 0; i < after.size(); ++i) after[i] = on_helper[i] != 0;
    for (const TierOp& o : ops) after[(size_t) o.layer * (size_t) n_expert + (size_t) o.expert] = o.kind == TierOp::kLoad;
    detail::record_coverage(state, n_layers, n_expert, smooth, cfg, [&](size_t i) { return after[i] != 0; });
    state.heat = std::move(smooth);
    ++state.round;
    return ops;
}

bool note_failed(TierPolicyState& state, int32_t layer, int32_t expert, std::string* err) {
    if (err) err->clear();
    if (state.last_move.empty()) return fail(err, "note_failed: state has not planned a round yet");
    if (layer < 0 || expert < 0 || layer >= state.n_layers || expert >= state.n_expert)
        return fail(err, "note_failed: layer/expert out of range");
    state.last_move[(size_t) layer * (size_t) state.n_expert + (size_t) expert] = TierPolicyState::kNever;
    return true;
}

}  // namespace seabraid
