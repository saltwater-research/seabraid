// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
#pragma once
// seabraid D7, the CALL SITE of live cross-card re-tiering: schedules rounds of the tier policy (tier_plan.hpp) off the
// decode path and hands the resulting ops to the ENGINE thread, which alone may drive the helper (RemoteExperts::
// seabraid_retier asserts it).  The engine (generate.cpp) owns two hooks and nothing else: construct + precheck at
// startup, and `tick()` once per decode window.
//
// DESIGN GATE
//   Owner    : this file + src/retier_loop.cpp own round scheduling, the TierOp -> TierWireOp and TierMove -> LadderPair
//              conversions and the retier CLI flags.  tier_plan.* still owns WHAT moves; RemoteExperts (two-set) and
//              the engine's ownership map (promote mode, strata/core/ownership_map.hpp) own HOW it is applied.
//   Contract : RetierConfig / retier_parse_flag / retier_check / to_wire / RetierLoop below.  Imports only the policy
//              header and the wire header (channel.hpp); NOT remote_experts.hpp: the engine passes an ApplyFn, so the
//              dependency edge is one-way (engine -> this -> policy) and the loop is unit-testable with no GPU.
//   Size     : each file <= 300 lines.
//   Failure  : misconfiguration (bad value, empty heat, no helper) FAILS FAST at startup via retier_check.  A round error
//              (planner error, helper refusal, fence timeout) FAILS OPEN, LOUDLY: logged, the round skipped, serving goes
//              on, failed ops are fed back to note_failed.  Ops are NEVER applied off the engine thread: tick() refuses
//              any other thread and the planner thread only ever produces a plan.
//   Checker  : tests/retier_loop_test.cpp (hand-off thread identity, zero-op, conversion, note_failed feedback, off =
//              no thread), each guard sabotaged once (tests/SABOTAGE.md).
//   Finish   : with the flag off nothing here is constructed; with it on, a round line per interval with ops
//              proposed/applied/failed, ranker ms and fence ms.
//
// THREAD MODEL (three actors, two single-slot mailboxes, no shared mutable state besides them)
//   engine thread : tick() at a window boundary.  (1) takes a finished plan from `out_` and applies it through the
//                   ApplyFn; (2) when a round is due and none is in flight, copies heat + the helper's residency into
//                   a RetierInput and posts it to `in_`.  Both copies are made HERE, on the engine thread, at the
//                   boundary, so the planner never reads a buffer the engine or the adapt thread is writing.
//   planner thread: waits on `in_`, runs note_failed for last round's failures, plan_tier_round (timed), converts to
//                   wire ops, posts a RetierPlan to `out_`.  It owns TierPolicyState exclusively.
//   Failures travel engine -> planner inside the NEXT RetierInput, which is why the state needs no lock.
#include <cstdint>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "strata/seabraid/channel.hpp"
#include "strata/seabraid/tier_plan.hpp"

namespace seabraid {

enum class RetierMode { kOff, kOn, kZeroOp };
// --retier-promote: WHO applies a round.  kOff: the two-set round (card 2 <-> RAM).  kLegacy: the ownership-map ladder
// (tick_ladder; `on` is its old spelling).  kSeabraid: the placement driver (adapters/strata), which reads cadence and
// policy from this config but needs no RetierLoop.  One enum, one reader of it, so no two publishers can both be on.
enum class RetierPromote { kOff, kLegacy, kSeabraid };

// Defaults of the policy knobs are TierPolicyConfig's own (one owner); only the scheduling lives here.
struct RetierConfig {
    RetierMode mode = RetierMode::kOff;
    // Interval in decode WINDOWS, not seconds: the heat counter (DriverState::usage) is decayed per adapt round, and an
    // adapt round is itself every --adapt-every windows, so windows are the unit heat ages in.  Wall seconds would give a
    // fast run more heat decay per round than a slow one, so two arms of an A/B on the same prompts would not be comparable.
    int every_windows = 128;
    TierPolicyConfig policy;   // max_moves_per_round / promote_margin / cooldown_rounds / max_helper_layer_share
    // --retier-promote legacy: the N-rung round (M3b) over the ownership snapshot instead of the two-set round, planning
    // card 2 -> card 1 promotions only (policy.ladder_boundaries = 1), applied through the engine's ownership map (M3)
    // with tick_ladder().  Default off: --retier on is bit-for-bit the measured two-set path.
    RetierPromote promote = RetierPromote::kOff;
    // --retier-fallback / --retier-hops (pyramid): read by the placement driver only, which hands them to Seabraid's planner
    // (policy v2 fallback_boundary / max_hops_per_round).  Defaults are v1's: no fallback swaps, adjacent-only.  Both need
    // --retier-promote seabraid (retier_check): no other publisher can move a cell between a card and the fallback or climb.
    int fallback_boundary = 0;   // 0|1
    int max_hops = 1;            // 1..kMaxHops
    // --retier-file-moves (FR-E): with --retier-promote seabraid a helper may take a promoted expert straight from the model file
    // (wire v9 kTierSwapFile) instead of from bytes main read; a refusal falls back to bytes in the same swap.  Default 0: unmeasured.
    int file_moves = 0;          // 0|1
    bool ladder() const { return promote == RetierPromote::kLegacy; }   // the only mode this file's RetierLoop serves
};

// ONE conversion TierMove -> LadderPair (the ladder's twin of to_wire): a promotion immediately followed by its drop
// in the same layer across the same boundary is a pair; anything else (a fill: no live form yet, the engine's cache
// never grows) is reported in `dropped`.  `from[p]` lists the indices into `moves` that pair p realises.
struct LadderPair {
    int32_t layer, up, down;
    int8_t hi;   // up: rung hi + 1 -> hi; down: hi -> hi + 1
};
struct LadderRound {
    std::vector<LadderPair> pairs;
    std::vector<std::vector<int>> from;
    std::vector<int> dropped;
};
LadderRound to_pairs(const std::vector<TierMove>& moves);

inline constexpr int kMaxHops = 64;   // the policy's own bound (seabraid_placement_policy_v2.max_hops_per_round)

// Consumes `a` if it is a --retier* flag.  0: not ours; 1: consumed; 2: error (err set).  `next(name)` returns the value.
int retier_parse_flag(const std::string& a, const std::function<const char*(const char*)>& next, RetierConfig& c, std::string& err);
const char* retier_help();

// At the end of argv parsing, before opening a model or device. An inactive driver never sees the later resource checks.
bool retier_check_flags(const RetierConfig& c, bool serve, std::string& err);

// Startup check, FAIL FAST.  `usage_size` is drive.d.usage.size() (0 = never allocated); `helper_open` = a seabraid helper is up.
bool retier_check(const RetierConfig& c, int adapt_every, int adapt_swaps, size_t usage_size, int64_t cells, bool helper_open, std::string& err);

// Startup check for MORE than one helper process (SEABRAID_STRATA_SHM lists `helpers` channels), FAIL FAST.  Only --retier-promote
// seabraid publishes one routing table over k rungs; --retier on and --retier-promote legacy are one-helper paths and keep
// exactly one publisher (porting them to k would make two).  Call it beside retier_check; `helpers` <= 1 always passes.
bool retier_check_helpers(const RetierConfig& c, int helpers, std::string& err);

// ONE conversion TierOp -> TierWireOp.  The wire has no EVICT and a SWAP is same-layer, so: a LOAD paired with an EVICT of
// its layer becomes a SWAP; a LOAD alone is a LOAD; an EVICT left unpaired (or anything past kTierMaxOps) has no wire
// form and is reported in `dropped`.  `from[w]` lists the indices into `ops` that wire op w realises (1 or 2).
struct WireRound {
    std::vector<TierWireOp> wire;
    std::vector<std::vector<int>> from;
    std::vector<int> dropped;
};
WireRound to_wire(const std::vector<TierOp>& ops);

struct RetierInput {
    std::vector<double> heat;
    std::vector<uint8_t> held;
    int64_t free_slots = 0;
    std::vector<std::pair<int32_t, int32_t>> failed;   // last round's failures, for note_failed on the planner thread
    std::vector<int8_t> rung_of;                        // promote mode: the ownership snapshot (held/free_slots unused)
    std::vector<int64_t> rung_free;                     // promote mode: free slots per rung
};
struct RetierPlan {
    int64_t round = 0;
    std::vector<TierOp> ops;
    std::vector<TierMove> moves;       // promote mode
    LadderRound lr;                    // promote mode: moves as pairs
    int rungs = 0;                     // promote mode: R, so the round line can name every rung
    int proposed = 0;                  // what the policy returned (kept when zero-op discards them)
    WireRound w;
    double ranker_ms = 0;
    std::string err;
    std::string note;                  // non-fatal: a note_failed that was refused
    int64_t layers_occupied = 0;
    double max_share = 0;
    int64_t over_ceiling = 0;          // layers whose planned held share exceeds the ceiling
};

// Single-slot mailbox: post() refuses when full, so a slow consumer is back-pressure rather than a queue.
template <class T> class Mailbox {
public:
    bool post(T v) { { std::lock_guard<std::mutex> g(m_); if (full_ || closed_) return false; v_ = std::move(v); full_ = true; } cv_.notify_one(); return true; }
    bool try_take(T& out) { std::lock_guard<std::mutex> g(m_); if (!full_) return false; out = std::move(v_); full_ = false; return true; }
    bool wait_take(T& out) {   // false once closed
        std::unique_lock<std::mutex> g(m_);
        cv_.wait(g, [&] { return full_ || closed_; });
        if (!full_) return false;
        out = std::move(v_); full_ = false; return true;
    }
    void close() { { std::lock_guard<std::mutex> g(m_); closed_ = true; } cv_.notify_all(); }
private:
    std::mutex m_; std::condition_variable cv_; T v_{}; bool full_ = false, closed_ = false;
};

struct RetierStats {
    int64_t windows = 0, posted = 0, rounds = 0, skipped_busy = 0, skipped_not_quiet = 0, plan_errors = 0, refused_off_thread = 0;
    int64_t proposed = 0, applied = 0, failed = 0, dropped = 0, fence_calls = 0;
};

class RetierLoop {
public:
    using ApplyFn = std::function<bool(const TierWireOp*, int32_t, uint8_t*, std::string&)>;   // = RemoteExperts::seabraid_retier
    using LogFn = std::function<void(const std::string&)>;
    // Constructed ON THE ENGINE THREAD: the constructing thread is the only one tick() will serve.  No thread starts here.
    RetierLoop(const RetierConfig& c, int64_t n_layers, int64_t n_expert, LogFn log = nullptr);
    ~RetierLoop();   // joins the planner thread (if one ever started), logs a summary
    RetierLoop(const RetierLoop&) = delete;
    RetierLoop& operator=(const RetierLoop&) = delete;

    // ENGINE THREAD, once per decode window, between windows.  `usage`/`held` are read here and copied; `quiescent` false
    // (an adapt job is writing usage) skips the snapshot.  Returns false only on misuse (wrong thread).
    bool tick(const float* usage, size_t usage_n, const std::vector<uint8_t>& held, int64_t free_slots, bool quiescent, const ApplyFn& apply);

    // Promote mode (RetierConfig::promote).  `ok[p]` is set by the engine per pair (nonzero = committed); `note` is a
    // free-text summary for the round line (the ownership map's outcome per pair).  The engine applies every pair
    // through the ownership map's prepare/commit, on this thread, at this boundary.
    using ApplyPairsFn = std::function<bool(const std::vector<LadderPair>&, std::vector<uint8_t>& ok, std::string& note)>;
    // ENGINE THREAD, once per window.  `rung_of` (the ownership snapshot) and `rung_free` are copied when a round is
    // posted.  Unlike tick(), a finished plan waits while `quiescent` is false: a promotion writes card 1's own slots,
    // which the engine's adapt copies may also be writing, so it is applied only at a quiet boundary.
    bool tick_ladder(const float* usage, size_t usage_n, const std::vector<int8_t>& rung_of, const std::vector<int64_t>& rung_free,
                     bool quiescent, const ApplyPairsFn& apply);
    // Engine thread: will the NEXT tick post a round if the boundary is quiet?  Lets the engine build the (L x E)
    // ownership snapshot only on the windows that use it instead of every window.
    bool snapshot_due() const { return c_.mode != RetierMode::kOff && !in_flight_ && st_.windows + 1 - last_post_window_ >= c_.every_windows; }
    const RetierStats& stats() const { return st_; }   // engine thread
    bool thread_started() const { return th_.joinable(); }

private:
    bool engine_thread();                        // false (and one log line) off the constructing thread
    void warn_once(bool& flag, const char* msg);   // log `msg` the first time only (flag is the caller's latch)
    bool due(bool quiescent);                    // a round is due, none in flight, and the boundary is quiet
    bool shape_ok(const float* usage, size_t usage_n, size_t view_n);
    void post(RetierInput&& in);
    void apply_plan(RetierPlan& p, const ApplyFn& apply);
    void apply_ladder(RetierPlan& p, const ApplyPairsFn& apply);
    void worker();
    void log(const std::string& s) const;
    RetierConfig c_;
    int64_t L_, E_;
    LogFn log_;
    std::thread::id owner_;
    std::thread th_;
    Mailbox<RetierInput> in_;
    Mailbox<RetierPlan> out_;
    TierPolicyState state_;                       // planner thread only
    std::vector<std::pair<int32_t, int32_t>> failed_;   // engine thread, travels in the next input
    RetierStats st_;
    int64_t last_post_window_ = 0, round_ = 0;
    bool in_flight_ = false, warned_thread_ = false, warned_mode_ = false;
};

}  // namespace seabraid
