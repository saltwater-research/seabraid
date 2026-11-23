// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// seabraid D7 call site: see include/strata/seabraid/retier_loop.hpp for the contract, the thread model and the design gate.
#include "strata/seabraid/retier_loop.hpp"

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <sstream>

namespace seabraid {
namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

bool to_long(const char* s, long& v) { char* e = nullptr; errno = 0; v = std::strtol(s, &e, 10); return errno == 0 && e != s && *e == '\0'; }
bool to_dbl(const char* s, double& v) { char* e = nullptr; errno = 0; v = std::strtod(s, &e); return errno == 0 && e != s && *e == '\0' && std::isfinite(v); }

using NextFn = std::function<const char*(const char*)>;

int bad_value(const char* what, const char* v, std::string& err) { err = std::string(what) + ": cannot read '" + v + "'"; return 2; }

// --retier off|on|zero-op
int parse_mode(const NextFn& next, RetierConfig& c, std::string& err) {
    const std::string v = next("--retier");
    if (v == "off") c.mode = RetierMode::kOff;
    else if (v == "on") c.mode = RetierMode::kOn;
    else if (v == "zero-op") c.mode = RetierMode::kZeroOp;
    else { err = "--retier: expected off, on or zero-op, got '" + v + "'"; return 2; }
    return 1;
}

// --retier-promote legacy|seabraid|off
int parse_promote(const NextFn& next, RetierConfig& c, std::string& err) {
    const std::string v = next("--retier-promote");
    if (v == "legacy" || v == "on") c.promote = RetierPromote::kLegacy;   // `on` is the pre-seabraid spelling of legacy
    else if (v == "seabraid") c.promote = RetierPromote::kSeabraid;
    else if (v == "off") c.promote = RetierPromote::kOff;
    else { err = "--retier-promote: expected legacy, seabraid or off (on = legacy), got '" + v + "'"; return 2; }
    return 1;
}

// The numeric flags as data: one row per flag, so a new knob is one line and not another branch.
struct LongFlag { const char* name; void (*set)(RetierConfig&, long); };
struct DblFlag { const char* name; void (*set)(RetierConfig&, double); };
const LongFlag kLongFlags[] = {
    {"--retier-every", [](RetierConfig& c, long n) { c.every_windows = (int) n; }},
    {"--retier-moves", [](RetierConfig& c, long n) { c.policy.max_moves_per_round = (int) n; }},
    {"--retier-cooldown", [](RetierConfig& c, long n) { c.policy.cooldown_rounds = (int) n; }},
    // clamped, not truncated: a value past int range must fail the range check, not wrap into a valid one
    {"--retier-fallback", [](RetierConfig& c, long n) { c.fallback_boundary = n < -1 ? -1 : n > 2 ? 2 : (int) n; }},
    {"--retier-hops", [](RetierConfig& c, long n) { c.max_hops = n < 0 ? 0 : n > 1000 ? 1000 : (int) n; }},
    {"--retier-file-moves", [](RetierConfig& c, long n) { c.file_moves = n < -1 ? -1 : n > 2 ? 2 : (int) n; }},   // clamped like --retier-fallback
};
const DblFlag kDblFlags[] = {
    {"--retier-margin", [](RetierConfig& c, double d) { c.policy.promote_margin = d; }},
    {"--retier-ceiling", [](RetierConfig& c, double d) { c.policy.max_helper_layer_share = d; }},
};

// Consumes `a` if it names a row of `table`: reads its value with `next`, converts with `conv`, stores with the row's setter.
// 0: not in this table; 1: consumed; 2: the value did not parse (err set).
template <class Flag, size_t N, class Num>
int parse_valued(const Flag (&table)[N], bool (*conv)(const char*, Num&), const std::string& a, const NextFn& next, RetierConfig& c,
                 std::string& err) {
    for (const Flag& f : table) {
        if (a != f.name) continue;
        const char* v = next(f.name);
        Num n{};
        if (!conv(v, n)) return bad_value(f.name, v, err);
        f.set(c, n);
        return 1;
    }
    return 0;
}

}  // namespace

int retier_parse_flag(const std::string& a, const std::function<const char*(const char*)>& next, RetierConfig& c, std::string& err) {
    if (a.rfind("--retier", 0) != 0) return 0;
    if (a == "--retier") return parse_mode(next, c, err);
    if (a == "--retier-promote") return parse_promote(next, c, err);
    if (const int r = parse_valued(kLongFlags, to_long, a, next, c, err)) return r;
    return parse_valued(kDblFlags, to_dbl, a, next, c, err);   // 0 here: some other --retier-* flag, the caller reports it as unknown
}

namespace {

// The adaptive tier's counts must exist and match the model: zero heat would plan nothing forever and look like a working run.
bool check_counts(int adapt_every, int adapt_swaps, size_t usage_size, int64_t cells, std::string& err) {
    if (adapt_every <= 0 || adapt_swaps <= 0 || usage_size == 0) {
        err = "--retier needs the adaptive tier's routing counts, which exist only with --adapt-every > 0 AND --adapt-swaps > 0 "
              "(got --adapt-every " + std::to_string(adapt_every) + ", --adapt-swaps " + std::to_string(adapt_swaps) +
              "; counts " + (usage_size == 0 ? "not allocated" : "allocated") + ")";
        return false;
    }
    if ((int64_t) usage_size != cells) { err = "--retier: the routing counts have " + std::to_string(usage_size) + " cells, the model has " + std::to_string(cells); return false; }
    return true;
}

// The knob ranges (the planner re-checks its own copy; this one fails at startup instead of at the first round).
bool check_policy_bounds(const RetierConfig& c, std::string& err) {
    if (c.every_windows < 1) { err = "--retier-every must be >= 1"; return false; }
    if (c.policy.max_moves_per_round < 1 || c.policy.max_moves_per_round > kTierMaxOps) { err = "--retier-moves must be 1.." + std::to_string(kTierMaxOps); return false; }
    if (!(c.policy.promote_margin >= 1.0)) { err = "--retier-margin must be >= 1"; return false; }
    if (c.policy.cooldown_rounds < 0) { err = "--retier-cooldown must be >= 0"; return false; }
    if (!(c.policy.max_helper_layer_share >= 0.0 && c.policy.max_helper_layer_share <= 1.0)) { err = "--retier-ceiling must be in [0, 1]"; return false; }
    return true;
}

// --retier-file-moves: 0 or 1, and only the seabraid publisher can take an expert's bytes from the model file.
bool check_file_moves(const RetierConfig& c, std::string& err) {
    if (c.file_moves != 0 && c.file_moves != 1) { err = "--retier-file-moves must be 0 or 1"; return false; }
    if (c.file_moves != 0 && c.promote != RetierPromote::kSeabraid) {
        err = "--retier-file-moves 1 needs --retier-promote seabraid (only its swap can take an expert's bytes from the model file)";
        return false;
    }
    return true;
}

// The pyramid knobs: their ranges, and the one publisher that can use them.
bool check_pyramid(const RetierConfig& c, std::string& err) {
    if (c.fallback_boundary != 0 && c.fallback_boundary != 1) { err = "--retier-fallback must be 0 or 1"; return false; }
    if (c.max_hops < 1 || c.max_hops > kMaxHops) { err = "--retier-hops must be 1.." + std::to_string(kMaxHops); return false; }
    if ((c.fallback_boundary != 0 || c.max_hops > 1) && c.promote != RetierPromote::kSeabraid) {
        err = "--retier-fallback 1 and --retier-hops > 1 need --retier-promote seabraid (only its planner moves a cell between a card and the fallback, or climbs)";
        return false;
    }
    // WHY refuse and not clamp: a hop is one swap (a LOAD and an EVICT = 2 ops), so a climb of `hops` rungs needs 2 * hops moves; with
    // fewer the planner silently stops a rung short and the run looks like a working pyramid.  Raising --retier-moves for the operator
    // would change the per-round cost they chose, so the startup error names both values and lets them pick.  hops 1 is v1 and is left alone.
    if (c.max_hops > 1 && 2 * c.max_hops > c.policy.max_moves_per_round) {
        err = "--retier-hops " + std::to_string(c.max_hops) + " needs --retier-moves >= " + std::to_string(2 * c.max_hops) + " (a hop is a LOAD and an EVICT), got --retier-moves " +
              std::to_string(c.policy.max_moves_per_round);
        return false;
    }
    return true;
}

}  // namespace

bool retier_check_helpers(const RetierConfig& c, int helpers, std::string& err) {
    if (helpers <= 1 || c.mode == RetierMode::kOff || c.promote == RetierPromote::kSeabraid) return true;
    err = std::string("--retier ") + (c.promote == RetierPromote::kLegacy ? "with --retier-promote legacy" : "without --retier-promote seabraid") +
          " serves ONE helper, but SEABRAID_STRATA_SHM lists " + std::to_string(helpers) + "; use --retier-promote seabraid";
    return false;
}

bool retier_check_flags(const RetierConfig& c, bool serve, std::string& err) {
    if (!check_pyramid(c, err) || !check_file_moves(c, err)) return false;   // knobs first: a flag without --retier is refused too
    if (c.mode == RetierMode::kOff) {
        if (c.promote != RetierPromote::kOff) { err = "--retier-promote legacy|seabraid needs --retier on: it is a mode of the retier round"; return false; }
        return true;
    }
    // WHY: the placement driver applies real moves; zero-op's contract is that none are sent.
    if (c.promote == RetierPromote::kSeabraid && c.mode != RetierMode::kOn) { err = "--retier-promote seabraid needs --retier on (zero-op sends no ops)"; return false; }
    if (!serve) { err = "--retier requires --serve"; return false; }
    return true;
}

bool retier_check(const RetierConfig& c, int adapt_every, int adapt_swaps, size_t usage_size, int64_t cells, bool helper_open, std::string& err) {
    if (!retier_check_flags(c, true, err)) return false;
    if (c.mode == RetierMode::kOff) return true;
    // The heat IS DriverState::usage, allocated only when both adapt flags are positive.  Zero heat would plan nothing
    // forever and look like a working run, so refuse before the first request.
    if (!check_counts(adapt_every, adapt_swaps, usage_size, cells, err)) return false;
    if (!helper_open) { err = "--retier needs a seabraid helper tier (SEABRAID_STRATA_SHM with --expert-cache-device1); none is open"; return false; }
    return check_policy_bounds(c, err);
}

namespace {

// The first unused EVICT of the same layer and helper as `ld` (ops are ordered, so this is deterministic); -1 if none.
int pair_evict(const std::vector<TierOp>& ops, const std::vector<uint8_t>& used, const TierOp& ld) {
    for (size_t ei = 0; ei < ops.size(); ++ei)
        if (ops[ei].kind == TierOp::kEvict && !used[ei] && ops[ei].layer == ld.layer && ops[ei].helper == ld.helper) return (int) ei;
    return -1;
}

// The wire form of LOAD `ld`: a SWAP when it has an EVICT partner `ev`, a plain LOAD otherwise.
TierWireOp wire_of(const TierOp& ld, const TierOp* ev) {
    TierWireOp w{};
    w.helper = ld.helper; w.layer_new = ld.layer; w.expert_new = ld.expert;
    if (ev) { w.kind = kTierSwap; w.layer_old = ev->layer; w.expert_old = ev->expert; }
    else { w.kind = kTierLoad; w.layer_old = -1; w.expert_old = -1; }
    return w;
}

}  // namespace

WireRound to_wire(const std::vector<TierOp>& ops) {
    WireRound r;
    std::vector<uint8_t> used(ops.size(), 0);
    for (size_t li = 0; li < ops.size(); ++li) {
        if (ops[li].kind != TierOp::kLoad) continue;
        const TierOp& ld = ops[li];
        if (ld.helper != 0 || (int) r.wire.size() >= kTierMaxOps) { r.dropped.push_back((int) li); used[li] = 1; continue; }
        const int ev = pair_evict(ops, used, ld);
        r.wire.push_back(wire_of(ld, ev >= 0 ? &ops[(size_t) ev] : nullptr));
        if (ev >= 0) { used[(size_t) ev] = 1; r.from.push_back({(int) li, ev}); }
        else r.from.push_back({(int) li});
        used[li] = 1;
    }
    for (size_t i = 0; i < ops.size(); ++i) if (!used[i]) r.dropped.push_back((int) i);
    return r;
}

RetierLoop::RetierLoop(const RetierConfig& c, int64_t n_layers, int64_t n_expert, LogFn log)
    : c_(c), L_(n_layers), E_(n_expert), log_(std::move(log)), owner_(std::this_thread::get_id()) {
    // The wire cannot carry an unpaired EVICT or a cross-layer swap, so ask the policy for only what it can apply.
    c_.policy.same_layer_swaps = true;
    // Promote mode plans the card 2 -> card 1 boundary alone: one variable against --retier on's card 2 <-> RAM round.
    if (c_.ladder()) c_.policy.ladder_boundaries = 1;
}

RetierLoop::~RetierLoop() {
    in_.close();
    if (th_.joinable()) th_.join();
    std::ostringstream s;
    s << "retier summary: " << st_.windows << " windows ticked, " << st_.rounds << " rounds applied/logged, proposed " << st_.proposed
      << ", applied " << st_.applied << ", failed " << st_.failed << ", dropped " << st_.dropped << ", fence calls " << st_.fence_calls
      << ", plan errors " << st_.plan_errors << ", skipped (planner busy " << st_.skipped_busy << ", adapt job running " << st_.skipped_not_quiet << ")";
    if (st_.windows == 0) s << " -- WARNING: --retier was on but no decode window ticked it (this decode loop does not host the hook)";
    log(s.str());
}

void RetierLoop::log(const std::string& s) const {
    if (log_) log_(s); else std::fprintf(stderr, "strata serve: %s\n", s.c_str());
}

void RetierLoop::worker() {
    RetierInput in;
    while (in_.wait_take(in)) {
        RetierPlan p;
        p.round = state_.round;
        std::string err;
        for (const auto& f : in.failed) {
            std::string e2;
            if (!note_failed(state_, f.first, f.second, &e2)) p.note += (p.note.empty() ? "" : "; ") + e2;   // loud, but not a lost round
        }
        const auto t0 = Clock::now();
        if (c_.ladder()) {   // M3b over the ownership snapshot; the two-set round below is untouched
            std::vector<TierMove> mv = plan_tier_round(L_, E_, in.rung_of, in.rung_free, in.heat, c_.policy, state_, &err);
            p.ranker_ms = ms_since(t0);
            if (!err.empty()) p.err = err;
            p.proposed = (int) mv.size();
            p.rungs = (int) in.rung_free.size();
            p.layers_occupied = state_.last_layers_occupied;
            p.max_share = state_.last_max_layer_share;
            p.over_ceiling = state_.last_layers_over_ceiling;
            if (c_.mode == RetierMode::kZeroOp) mv.clear();
            p.lr = to_pairs(mv);
            p.moves = std::move(mv);
            out_.post(std::move(p));
            continue;
        }
        std::vector<TierOp> ops = plan_tier_round(L_, E_, in.held, in.free_slots, in.heat, c_.policy, state_, &err);
        p.ranker_ms = ms_since(t0);
        if (!err.empty()) p.err = err;
        p.proposed = (int) ops.size();
        p.layers_occupied = state_.last_layers_occupied;
        p.max_share = state_.last_max_layer_share;
        p.over_ceiling = state_.last_layers_over_ceiling;
        if (c_.mode == RetierMode::kZeroOp) ops.clear();   // machinery runs, nothing is sent
        p.w = to_wire(ops);
        p.ops = std::move(ops);
        out_.post(std::move(p));   // the engine posts a new input only after taking this plan, so the slot is free
    }
}

bool RetierLoop::engine_thread() {
    if (std::this_thread::get_id() == owner_) return true;   // never apply (or even snapshot) off the engine thread
    ++st_.refused_off_thread;
    warn_once(warned_thread_, "retier: tick() called off the engine thread; refused (ops are applied on the engine thread only)");
    return false;
}

void RetierLoop::warn_once(bool& flag, const char* msg) {
    if (flag) return;
    flag = true;
    log(msg);
}

bool RetierLoop::due(bool quiescent) {
    if (st_.windows - last_post_window_ < c_.every_windows) return false;
    if (in_flight_) { ++st_.skipped_busy; return false; }
    if (!quiescent) { ++st_.skipped_not_quiet; return false; }
    return true;
}

bool RetierLoop::shape_ok(const float* usage, size_t usage_n, size_t view_n) {
    if ((int64_t) usage_n == L_ * E_ && (int64_t) view_n == L_ * E_ && usage != nullptr) return true;
    log("retier: snapshot shape mismatch (usage " + std::to_string(usage_n) + ", residency " + std::to_string(view_n) + ", expected " + std::to_string(L_ * E_) + "); round skipped");
    ++st_.plan_errors; last_post_window_ = st_.windows;
    return false;
}

void RetierLoop::post(RetierInput&& in) {
    in.failed.swap(failed_);
    if (!th_.joinable()) th_ = std::thread([this] { worker(); });
    in_.post(std::move(in));
    in_flight_ = true; last_post_window_ = st_.windows; ++st_.posted;
}

bool RetierLoop::tick(const float* usage, size_t usage_n, const std::vector<uint8_t>& held, int64_t free_slots, bool quiescent, const ApplyFn& apply) {
    if (!engine_thread()) return false;
    if (c_.mode == RetierMode::kOff) return true;
    if (c_.ladder()) {   // the planner thread would plan the ladder with no snapshot: a wiring bug, said once
        warn_once(warned_mode_, "retier: tick() called in promote mode (the engine must call tick_ladder); ignored");
        return false;
    }
    ++st_.windows;
    RetierPlan p;
    if (out_.try_take(p)) { in_flight_ = false; apply_plan(p, apply); }
    if (!due(quiescent) || !shape_ok(usage, usage_n, held.size())) return true;
    RetierInput in;
    in.heat.assign(usage, usage + usage_n);   // float -> double copy, at the boundary, on this thread
    in.held = held;
    in.free_slots = free_slots;
    post(std::move(in));
    return true;
}

namespace {

using FailedCells = std::vector<std::pair<int32_t, int32_t>>;

// What the wire apply did: per-op outcome counts, the fence's wall time and the helper's own refusal text.
struct WireOutcome {
    int applied = 0, failed = 0, swaps = 0, loads = 0;
    double fence_ms = 0;
    std::string aerr;
};

// Sends p's wire ops through `apply` (the ONLY place ops reach the helper: engine thread, here) and records the cell of
// every op that did not apply in `failed_cells`.  False when there was nothing to send (the fence was not entered).
bool run_wire(const RetierPlan& p, const RetierLoop::ApplyFn& apply, WireOutcome& o, FailedCells& failed_cells) {
    if (p.w.wire.empty()) return false;
    const int32_t n = (int32_t) p.w.wire.size();
    std::vector<uint8_t> res((size_t) n, (uint8_t) kTierPending);
    const auto t0 = Clock::now();
    const bool ok = apply(p.w.wire.data(), n, res.data(), o.aerr);
    o.fence_ms = ms_since(t0);
    (void) ok;   // per-op results are the truth; a refused round still moved some ops
    for (size_t w = 0; w < p.w.wire.size(); ++w) {
        (p.w.wire[w].kind == kTierSwap ? o.swaps : o.loads) += 1;
        const bool good = res[w] == (uint8_t) kTierApplied;
        (good ? o.applied : o.failed) += 1;
        if (!good) for (int idx : p.w.from[w]) failed_cells.emplace_back(p.ops[(size_t) idx].layer, p.ops[(size_t) idx].expert);
    }
    return true;
}

// The round line after its prefix: what was proposed, what the wire did, the planned coverage and any notes.
void describe_round(std::ostream& s, const RetierPlan& p, const WireOutcome& o, bool zero_op) {
    s << "proposed " << p.proposed << " ops, wire " << p.w.wire.size() << " (swap " << o.swaps << ", load " << o.loads << "), applied " << o.applied
      << ", failed " << o.failed << ", dropped " << p.w.dropped.size();
    if (zero_op) s << " [zero-op: " << p.proposed << " proposed ops discarded, fence not entered]";
    else if (p.w.wire.empty()) s << " [nothing to apply: fence not entered]";
    s << "; ranker " << p.ranker_ms << " ms, fence " << o.fence_ms << " ms; planned coverage: " << p.layers_occupied << " layers occupied, max layer share " << p.max_share
      << ", " << p.over_ceiling << " layers over the ceiling this round (no swaps admitted there while over)";
    if (!o.aerr.empty()) s << "; helper said: " << o.aerr;
    if (!p.note.empty()) s << "; note_failed refused: " << p.note;
}

}  // namespace

void RetierLoop::apply_plan(RetierPlan& p, const ApplyFn& apply) {
    ++st_.rounds; ++round_;
    std::ostringstream s;
    s << "retier round " << round_ << " (window " << st_.windows << "): ";
    if (!p.err.empty()) {   // fail open, loudly: this round is lost, serving goes on, the next round re-plans from fresh state
        ++st_.plan_errors;
        s << "PLANNER ERROR, round skipped: " << p.err << " (ranker " << p.ranker_ms << " ms)";
        log(s.str()); return;
    }
    WireOutcome o;
    if (run_wire(p, apply, o, failed_)) ++st_.fence_calls;
    // An op with no wire form never ran: clear its cooldown so it is not frozen for nothing.
    for (int idx : p.w.dropped) failed_.emplace_back(p.ops[(size_t) idx].layer, p.ops[(size_t) idx].expert);
    st_.proposed += p.proposed; st_.applied += o.applied; st_.failed += o.failed; st_.dropped += (int64_t) p.w.dropped.size();
    describe_round(s, p, o, c_.mode == RetierMode::kZeroOp);
    log(s.str());
}

}  // namespace seabraid
