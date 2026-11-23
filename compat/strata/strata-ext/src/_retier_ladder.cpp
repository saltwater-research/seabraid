// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// seabraid: the promote mode of the retier call site (--retier-promote on): the TierMove -> LadderPair conversion, the
// engine-thread tick and the round line.  Split from retier_loop.cpp to keep both under the 300-line budget; the
// contract, the thread model and the design gate are retier_loop.hpp's.
#include "strata/seabraid/retier_loop.hpp"

#include <algorithm>
#include <chrono>
#include <sstream>

namespace seabraid {

LadderRound to_pairs(const std::vector<TierMove>& moves) {
    LadderRound r;
    for (size_t i = 0; i < moves.size(); ++i) {
        const TierMove& u = moves[i];
        const bool promo = u.to == u.from - 1;
        // The policy emits a displacing promotion and its drop adjacent; pairing by position (not by search) means a
        // move the policy did not pair is never silently paired with some other move of the same layer.
        if (promo && i + 1 < moves.size()) {
            const TierMove& d = moves[i + 1];
            if (d.layer == u.layer && d.from == u.to && d.to == u.from && d.expert != u.expert) {
                r.pairs.push_back(LadderPair{u.layer, u.expert, d.expert, u.to});
                r.from.push_back({(int) i, (int) (i + 1)});
                ++i;
                continue;
            }
        }
        r.dropped.push_back((int) i);   // a fill or a lone drop: no live form (the engine's cache never grows or shrinks)
    }
    return r;
}

bool RetierLoop::tick_ladder(const float* usage, size_t usage_n, const std::vector<int8_t>& rung_of, const std::vector<int64_t>& rung_free,
                             bool quiescent, const ApplyPairsFn& apply) {
    if (!engine_thread()) return false;
    if (c_.mode == RetierMode::kOff) return true;
    if (!c_.ladder()) {
        warn_once(warned_mode_, "retier: tick_ladder() called without --retier-promote on; ignored");
        return false;
    }
    ++st_.windows;
    // A finished plan waits for a quiet boundary: its pairs write card 1's own slots, which an adapt copy in flight may
    // also be writing.  (tick() applies at once: its ops touch card 2 alone, whose slots only the fence writes.)
    if (!quiescent) {
        if (in_flight_ || st_.windows - last_post_window_ >= c_.every_windows) ++st_.skipped_not_quiet;
        return true;
    }
    RetierPlan p;
    if (out_.try_take(p)) { in_flight_ = false; apply_ladder(p, apply); }
    if (!due(true) || !shape_ok(usage, usage_n, rung_of.size())) return true;
    RetierInput in;
    in.heat.assign(usage, usage + usage_n);
    in.rung_of = rung_of;
    in.rung_free = rung_free;
    post(std::move(in));
    return true;
}

namespace {

std::string rung_label(int r, int rungs) {
    return r == rungs - 1 ? std::string("fallback") : "card " + std::to_string(r + 1);
}

using FailedCells = std::vector<std::pair<int32_t, int32_t>>;

// What the pair apply did: per-pair outcome counts, the transition's wall time and the engine's free-text summary.
struct PairOutcome {
    int applied = 0, failed = 0;
    double ms = 0;
    std::string note;
};

// Hands p's pairs to `apply` (the ONLY place pairs reach the cards: engine thread, here, through the ownership map) and
// records the cells of every pair that did not commit in `failed_cells`.  False when there was nothing to apply.
bool run_pairs(const RetierPlan& p, const RetierLoop::ApplyPairsFn& apply, PairOutcome& o, FailedCells& failed_cells) {
    if (p.lr.pairs.empty()) return false;
    std::vector<uint8_t> ok(p.lr.pairs.size(), 0);
    const auto t0 = std::chrono::steady_clock::now();
    apply(p.lr.pairs, ok, o.note);
    o.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    for (size_t k = 0; k < p.lr.pairs.size(); ++k) {
        (ok[k] ? o.applied : o.failed) += 1;
        if (!ok[k]) for (int idx : p.lr.from[k]) failed_cells.emplace_back(p.moves[(size_t) idx].layer, p.moves[(size_t) idx].expert);
    }
    return true;
}

// Name the rung of every move (M3b finish line); bounded so a large round cannot flood the log.
void append_moves(std::ostream& s, const RetierPlan& p) {
    const size_t shown = std::min<size_t>(p.moves.size(), 16);
    if (shown > 0) s << "; moves:";
    for (size_t i = 0; i < shown; ++i) {
        const TierMove& m = p.moves[i];
        s << " L" << m.layer << "/E" << m.expert << " " << rung_label(m.from, p.rungs) << "->" << rung_label(m.to, p.rungs);
    }
    if (p.moves.size() > shown) s << " (+" << (p.moves.size() - shown) << " more)";
}

// The round line after its prefix: what was proposed, what the transition did, the moves by rung and any notes.
void describe_ladder_round(std::ostream& s, const RetierPlan& p, const PairOutcome& o, bool zero_op) {
    s << "proposed " << p.proposed << " moves, pairs " << p.lr.pairs.size() << ", committed " << o.applied << ", failed " << o.failed
      << ", dropped " << p.lr.dropped.size();
    if (zero_op) s << " [zero-op: " << p.proposed << " proposed moves discarded, no transition]";
    else if (p.lr.pairs.empty()) s << " [nothing to apply: no transition]";
    append_moves(s, p);
    s << "; ranker " << p.ranker_ms << " ms, transition " << o.ms << " ms";
    if (!o.note.empty()) s << "; " << o.note;
    if (!p.note.empty()) s << "; note_failed refused: " << p.note;
}

}  // namespace

void RetierLoop::apply_ladder(RetierPlan& p, const ApplyPairsFn& apply) {
    ++st_.rounds; ++round_;
    std::ostringstream s;
    s << "retier round " << round_ << " (window " << st_.windows << ", promote): ";
    if (!p.err.empty()) {   // fail open, loudly, exactly as the two-set round
        ++st_.plan_errors;
        s << "PLANNER ERROR, round skipped: " << p.err << " (ranker " << p.ranker_ms << " ms)";
        log(s.str()); return;
    }
    PairOutcome o;
    if (run_pairs(p, apply, o, failed_)) ++st_.fence_calls;
    for (int idx : p.lr.dropped) failed_.emplace_back(p.moves[(size_t) idx].layer, p.moves[(size_t) idx].expert);
    st_.proposed += p.proposed; st_.applied += o.applied; st_.failed += o.failed; st_.dropped += (int64_t) p.lr.dropped.size();
    describe_ladder_round(s, p, o, c_.mode == RetierMode::kZeroOp);
    log(s.str());
}

}  // namespace seabraid
