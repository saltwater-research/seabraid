// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
#pragma once
// seabraid D7, the POLICY half of live cross-card expert re-tiering: given the routing heat of EVERY expert
// (resident on the helper or not) and what the helper holds right now, say which experts to LOAD onto the helper
// and which to EVICT from it this round.  Pure and deterministic: no GPU, no runtime, no channel include.  The
// transport half turns a TierOp into a channel command; a TierOp never crosses the process line itself.
//
// REQUIREMENT (roadmap, verbatim): "periodic re-rank of every expert by Strata's usage counters across ALL
// tiers; an expert crossing a tier boundary moves one level ... throttled (a few moves per window)".  Placement
// follows heat LIVE: a hot-but-not-hottest expert can live on the lower-tier card.
//
// DESIGN GATE
//   Owner    : this file and src/seabraid/tier_plan.cpp own the tier-move policy and its constants (decay, margin
//              semantics, ceiling).  Nothing else decides what moves; the transport only applies what is returned.
//   Contract : plan_tier_round() below.  Imports: <cstdint>, <string>, <vector> only (the dependency edge is
//              one-way: the engine and the transport include this; this includes neither).  One helper (op.helper
//              is always 0 until a second helper exists).  `on_helper` and `free_slots` are AUTHORITATIVE each
//              round, taken from the transport; the state holds no membership, so a LOAD that failed helper-side
//              corrects itself on the next round (modulo its cooldown).
//   Size     : each file <= 300 lines.
//   Failure  : correctness-adjacent, so FAIL CLOSED: a size mismatch, a negative or non-finite number, or a bad
//              config returns NO ops, writes the reason to `err`, and leaves the state untouched.  Never guess.
//   Checker  : tests/seabraid/tier_plan_test.cpp, one test per rule, each sabotaged once and seen to fail; the N-rung
//              overload: tests/tier_ladder_test.cpp (cascade, same-layer pairing, no challenger from a higher rung,
//              throttle and hysteresis), each sabotaged once (tests/SABOTAGE.md).
//   Finish   : builds with -Wall -Wextra, every rule test passes, every invariant test fails under its sabotage.
//
// WHERE `heat` COMES FROM: Strata's adaptive tier counts routing per (layer, expert) in
// `DriverState::usage` (include/strata/core/expert_source.hpp, "decayed routing counts"), incremented where the
// window's ids are routed (src/core/expert_source.cpp) and decayed by --adapt-decay between adapt rounds
// (src/program/generate.cpp, the loop that ends `for (float& v : drive.d.usage) v *= o.adapt_decay`).  It is
// n_layers x n_expert, layer-major, across ALL tiers, which is exactly the vector wanted here.  The policy
// smooths it again (kHeatDecay) because that counter is decayed by adapt rounds, not by this policy's rounds.
#include <cstdint>
#include <string>
#include <vector>

namespace seabraid {

struct TierOp {
    enum Kind : uint8_t { kLoad = 1, kEvict = 2 };
    Kind kind;
    uint8_t helper;     // always 0: one helper today
    int32_t layer, expert;
};

struct TierPolicyConfig {
    int     max_moves_per_round = 4;         // LOADs + EVICTs per round
    double  promote_margin = 1.25;           // challenger must out-heat the incumbent it would displace by this factor
    int     cooldown_rounds = 8;             // an expert that moved cannot move again for this many rounds
    double  max_helper_layer_share = 0.08;   // per-layer ceiling on the helper's share of that layer's HEAT; swept 0.08/0.15/0.20/0.30.
                                             // A GATE on admitting LOADs and SWAPs, NOT a restored invariant: with same_layer_swaps a layer
                                             // already over it is FROZEN (a hot-for-cold swap always raises held heat, and lowering it needs
                                             // an unpaired EVICT the wire lacks); last_layers_over_ceiling reports how many are.
    int64_t reserve_slots = 0;               // free helper slots the policy should try to keep for LOAD-without-EVICT
    // Transport constraint, see TierWireOp in channel.hpp: the wire has no EVICT and a SWAP is same-layer only.  With this set
    // every move the policy returns is EXPRESSIBLE: a LOAD into a slot free at the start of the round, or a LOAD paired with
    // an EVICT of the SAME layer in the same round (a swap).  There are no unpaired EVICTs (ceiling restoration and the reserve
    // top-up are skipped: a same-layer swap never changes the layer's held heat share by more than the swap itself, which the
    // ceiling test already judges) and a paired LOAD does not need a free slot, so rule R1 (LOADs <= free_slots) holds only
    // for the unpaired ones.  Default false: the policy alone is unchanged.
    bool    same_layer_swaps = false;
    // N-rung ladder only (the rung_of overload below): plan the top `ladder_boundaries` rung boundaries, 0 = every one.
    // 1 = card 2 -> card 1 promotions alone: the first live move (N-CARD-LADDER.md section 4), one variable for A/B 5,
    // while card 1 <-> RAM stays Strata's own adapt and card 2 <-> RAM stays the two-set round above.
    int     ladder_boundaries = 0;
};

// One expert changing rung by exactly one (rung 0 = card 1, then the other cards in rank order, the last rung = below
// the cards: RAM, then the drive).  `to == from - 1` is a promotion, `to == from + 1` a drop.
struct TierMove {
    int32_t layer, expert;
    int8_t from, to;
};

// Per-round exponential smoothing of the input heat: smoothed = kHeatDecay * smoothed + (1 - kHeatDecay) * heat.
// 0.75 gives a half-life of ~2.4 rounds: a one-window burst reaches only a quarter of its height and is gone
// within ~6 rounds, while a real shift in routing is >90% absorbed after 8.  The first round seeds with the raw
// heat so there is no warm-up bias.  It is a constant, not a config field: promote_margin and cooldown already
// tune responsiveness, and a third knob for the same trade would just make the sweep bigger.
inline constexpr double kHeatDecay = 0.75;

// Everything the policy remembers between rounds.  Opaque to the transport: construct it once, pass it back
// each round.  The fields are public only so the unit tests can read the smoothed heat the ceiling is judged on.
struct TierPolicyState {
    static constexpr int64_t kNever = INT64_MIN / 2;
    int64_t round = 0;                       // rounds planned so far
    std::vector<double> heat;                // smoothed heat per (layer, expert); empty = fresh state
    std::vector<int64_t> last_move;          // round of the expert's last LOAD or EVICT (kNever: none)
    int64_t n_layers = 0, n_expert = 0;      // the shape the state was built for (set by the first successful round)
    // Coverage report, recomputed every successful round from the membership AFTER that round's ops.  Log-only:
    // a per-layer ceiling bounds concentration but does not force spread, so these make clustering by another
    // route visible; deliberately no coverage floor until the logs say one is needed.
    int64_t last_layers_occupied = 0;        // layers in which the helper holds at least one expert
    double  last_max_layer_share = 0.0;      // max over layers of (helper heat / layer heat), smoothed heat
    int64_t last_layers_over_ceiling = 0;    // layers whose post-op held share exceeds max_helper_layer_share THIS round (no swaps admitted there; heat drift moves layers in and out)
};

// Plans one round.  `on_helper` (nonzero = the helper holds it) and `heat` are n_layers * n_expert, layer-major.
// `free_slots` is the helper's free expert slots at the START of the round; the returned LOADs never exceed it
// (an EVICT frees a slot only for a FUTURE round: there is no same-round dependency).  At most
// cfg.max_moves_per_round ops are returned, LOADs first, each group ordered by (layer, expert).
//
// ERROR API: on invalid input (size mismatch, n_layers/n_expert <= 0, free_slots < 0, negative/non-finite heat,
// out-of-range config, or a state built for a different shape) it returns an EMPTY vector, writes the reason to
// `*err` when `err` is non-null, and does not touch `state`.  On success `*err` is cleared, so an empty result
// with an empty `*err` means "nothing to move this round".
std::vector<TierOp> plan_tier_round(int64_t n_layers, int64_t n_expert, const std::vector<uint8_t>& on_helper,
                                    int64_t free_slots, const std::vector<double>& heat,
                                    const TierPolicyConfig& cfg, TierPolicyState& state, std::string* err = nullptr);

// M3b, the same policy generalised to N rungs (src/tier_ladder.cpp; N-CARD-LADDER.md section 4, C1).  `rung_of` is the
// OWNERSHIP snapshot (n_layers * n_expert, values in [0, R)), so the planner sees every card's residency: an expert a
// higher rung already holds is never a challenger for a lower one (the two-set planner could not see card 1 and could
// LOAD onto card 2 an expert card 1 held).  `free_slots[r]` is rung r's free slots at the START of the round; R =
// free_slots.size() >= 2 and the last rung is uncapped (its entry is ignored).
//   Rule P1, the strict heat cascade: at each boundary (hi, hi + 1), top boundary first, challengers on hi + 1 hottest
//   first; one fills a free slot on hi, else displaces the COLDEST movable resident of hi IN ITS OWN LAYER that it
//   out-heats by promote_margin, and that resident drops exactly one rung (to hi + 1).  Cooldown, the move budget and
//   the margin are the hysteresis, as in the two-set round.  The ceiling gates rungs 1..R-2 (cards other than card 1),
//   never card 1: taking the hottest experts is card 1's whole job.
//   Output: moves in decision order; a displacing promotion is IMMEDIATELY followed by its drop (same layer), which is
//   how the transport pairs them.  An expert moves at most once per round.
// Same ERROR API as above (fail closed, empty result, `*err` set, state untouched); plus rung_of/free_slots shape and
// range.  Coverage fields of `state` describe rung 1 (card 2) after the moves.
std::vector<TierMove> plan_tier_round(int64_t n_layers, int64_t n_expert, const std::vector<int8_t>& rung_of,
                                      const std::vector<int64_t>& free_slots, const std::vector<double>& heat,
                                      const TierPolicyConfig& cfg, TierPolicyState& state, std::string* err = nullptr);

// The transport calls this for every op the helper REJECTED: it clears that expert's cooldown so a failed LOAD (or
// EVICT) is re-planned next round instead of waiting out cooldown_rounds.  Fail closed: an unseeded state or an
// out-of-range layer/expert is a no-op that returns false and sets `*err`; true means the cooldown was cleared.
bool note_failed(TierPolicyState& state, int32_t layer, int32_t expert, std::string* err = nullptr);

}  // namespace seabraid
