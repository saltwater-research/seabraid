// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// seabraid D7 call-site tests (host only, no GPU, no gtest).  Needs Strata's include dir (patch 0001's channel.hpp) and
// strata-ext's:
//   g++ -std=c++17 -O1 -Wall -Wextra -pthread -I<strata>/include -Istrata-ext/include tests/retier_loop_test.cpp
//       strata-ext/src/retier_loop.cpp strata-ext/src/tier_plan.cpp -o retier_loop_test -lrt
// The ApplyFn below plays RemoteExperts::seabraid_retier; it records which thread called it and what it was given.
#include "strata/seabraid/retier_loop.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <set>
#include <thread>

using namespace seabraid;

static int g_fail = 0;
static const char* g_test = "?";
#define CHECK(c) do { if (!(c)) { ++g_fail; std::printf("  FAIL [%s] %s:%d  %s\n", g_test, __FILE__, __LINE__, #c); } } while (0)

static TierOp ld(int l, int e) { return TierOp{TierOp::kLoad, 0, l, e}; }
static TierOp ev(int l, int e) { return TierOp{TierOp::kEvict, 0, l, e}; }

// ---- the one conversion function ------------------------------------------------------------------------------------
static void t_conversion() {
    g_test = "to_wire";
    // same-layer EVICT+LOAD -> one SWAP carrying both halves; the wire layout is (old = the evicted, new = the loaded)
    auto r = to_wire({ld(2, 7), ev(2, 1)});
    CHECK(r.wire.size() == 1 && r.dropped.empty());
    CHECK(r.wire[0].kind == kTierSwap && r.wire[0].layer_old == 2 && r.wire[0].expert_old == 1 && r.wire[0].layer_new == 2 && r.wire[0].expert_new == 7);
    CHECK(r.from[0].size() == 2);
    // a LOAD alone is a LOAD with -1 for the old half
    r = to_wire({ld(0, 3)});
    CHECK(r.wire.size() == 1 && r.wire[0].kind == kTierLoad && r.wire[0].layer_old == -1 && r.wire[0].expert_old == -1 && r.wire[0].expert_new == 3);
    // cross-layer pair: the swap is same-layer only, so the LOAD goes alone and the EVICT has no wire form (the wire has no EVICT)
    r = to_wire({ld(1, 4), ev(5, 0)});
    CHECK(r.wire.size() == 1 && r.wire[0].kind == kTierLoad && r.dropped.size() == 1 && r.dropped[0] == 1);
    // an EVICT never pairs twice
    r = to_wire({ld(3, 1), ld(3, 2), ev(3, 9)});
    CHECK(r.wire.size() == 2 && r.wire[0].kind == kTierSwap && r.wire[1].kind == kTierLoad && r.dropped.empty());
    // nothing in, nothing out
    r = to_wire({});
    CHECK(r.wire.empty() && r.dropped.empty());
    // the capacity: past kTierMaxOps is dropped, never truncated silently
    std::vector<TierOp> many;
    for (int i = 0; i < kTierMaxOps + 3; ++i) many.push_back(ld(i, 0));
    r = to_wire(many);
    CHECK((int) r.wire.size() == kTierMaxOps && r.dropped.size() == 3);
    // a second helper has no wire form yet
    TierOp h1 = ld(0, 0); h1.helper = 1;
    r = to_wire({h1});
    CHECK(r.wire.empty() && r.dropped.size() == 1);
}

// ---- the mailbox ----------------------------------------------------------------------------------------------------
static void t_mailbox() {
    g_test = "mailbox";
    Mailbox<int> m;
    int v = 0;
    CHECK(!m.try_take(v));
    CHECK(m.post(1));
    CHECK(!m.post(2));                  // single slot: a second post is refused, not queued
    CHECK(m.try_take(v) && v == 1);
    CHECK(!m.try_take(v));
    m.close();
    CHECK(!m.post(3) && !m.wait_take(v));
}

// ---- the loop -------------------------------------------------------------------------------------------------------
struct Fixture {
    static constexpr int64_t L = 4, E = 8;
    RetierConfig cfg;
    std::vector<std::string> lines;
    std::vector<uint8_t> held = std::vector<uint8_t>(L * E, 0);
    std::vector<float> usage = std::vector<float>(L * E, 0.f);
    std::thread::id engine = std::this_thread::get_id();
    std::atomic<int> calls{0}, off_thread_calls{0};
    std::vector<TierWireOp> last;
    uint8_t result = kTierApplied;
    Fixture() {
        cfg.mode = RetierMode::kOn; cfg.every_windows = 1;
        cfg.policy.max_helper_layer_share = 1.0; cfg.policy.cooldown_rounds = 8; cfg.policy.promote_margin = 1.25;
        for (int l = 0; l < L; ++l) { held[(size_t) (l * E + 0)] = 1; usage[(size_t) (l * E + 0)] = 1.f; usage[(size_t) (l * E + 7)] = 50.f; }   // cold resident 0, hot miss 7
    }
    RetierLoop::ApplyFn apply() {
        return [this](const TierWireOp* ops, int32_t n, uint8_t* res, std::string&) {
            ++calls;
            if (std::this_thread::get_id() != engine) ++off_thread_calls;
            last.assign(ops, ops + n);
            for (int i = 0; i < n; ++i) res[i] = result;
            if (result == kTierApplied)   // play the helper: the residency view moves
                for (int i = 0; i < n; ++i) { if (ops[i].kind == kTierSwap) held[(size_t) (ops[i].layer_old * E + ops[i].expert_old)] = 0; held[(size_t) (ops[i].layer_new * E + ops[i].expert_new)] = 1; }
            return result == kTierApplied;
        };
    }
    RetierLoop::LogFn logger() { return [this](const std::string& s) { lines.push_back(s); }; }
};

// Drive tick() as the engine would, a window at a time, until `done` or timeout.
template <class F> static void pump(RetierLoop& lp, Fixture& f, F done) {
    auto ap = f.apply();
    for (int i = 0; i < 4000 && !done(); ++i) {
        lp.tick(f.usage.data(), f.usage.size(), f.held, 0, true, ap);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

static void t_applied_on_engine_thread_only() {
    g_test = "engine-thread hand-off";
    Fixture f;
    {
        RetierLoop lp(f.cfg, Fixture::L, Fixture::E, f.logger());
        pump(lp, f, [&] { return f.calls.load() >= 1; });
        CHECK(f.calls.load() >= 1);
        CHECK(f.off_thread_calls.load() == 0);          // the apply callback ran on the thread that constructed the loop
        CHECK(!f.last.empty() && f.last[0].kind == kTierSwap && f.last[0].layer_new <= 3 && f.last[0].expert_new == 7 && f.last[0].expert_old == 0);
        // a tick from ANOTHER thread is refused and applies nothing
        const int before = f.calls.load();
        bool ret = true;
        std::thread t([&] { auto ap = f.apply(); ret = lp.tick(f.usage.data(), f.usage.size(), f.held, 0, true, ap); });
        t.join();
        CHECK(!ret && f.calls.load() == before && lp.stats().refused_off_thread == 1);
        bool saw_line = false;
        for (auto& s : f.lines) saw_line = saw_line || (s.find("retier round 1") != std::string::npos && s.find("ranker") != std::string::npos && s.find("fence") != std::string::npos && s.find("applied") != std::string::npos && s.find("layers occupied") != std::string::npos);
        CHECK(saw_line);
    }
}

static void t_zero_op() {
    g_test = "zero-op";
    Fixture f; f.cfg.mode = RetierMode::kZeroOp;
    RetierLoop lp(f.cfg, Fixture::L, Fixture::E, f.logger());
    pump(lp, f, [&] { return lp.stats().rounds >= 2; });
    CHECK(lp.stats().rounds >= 2);
    CHECK(lp.thread_started());                         // the machinery ran...
    CHECK(lp.stats().proposed > 0);                     // ...the ranker did propose moves...
    CHECK(f.calls.load() == 0);                         // ...and seabraid_retier was never called (n=0 is refused by the fence; the loop skips)
    bool saw = false;
    for (auto& s : f.lines) saw = saw || s.find("zero-op") != std::string::npos;
    CHECK(saw);
}

static void t_off_runs_nothing() {
    g_test = "off";
    Fixture f; f.cfg.mode = RetierMode::kOff;
    RetierLoop lp(f.cfg, Fixture::L, Fixture::E, f.logger());
    auto ap = f.apply();
    for (int i = 0; i < 50; ++i) lp.tick(f.usage.data(), f.usage.size(), f.held, 0, true, ap);
    CHECK(!lp.thread_started() && lp.stats().windows == 0 && lp.stats().posted == 0 && f.calls.load() == 0);
}

static void t_failed_ops_feed_note_failed() {
    g_test = "note_failed feedback";
    // Applied: the swapped experts are on cooldown, so the next rounds do not re-propose them.
    {
        Fixture f;
        RetierLoop lp(f.cfg, Fixture::L, Fixture::E, f.logger());
        pump(lp, f, [&] { return lp.stats().rounds >= 3; });
        CHECK(lp.stats().applied >= 1);
        int64_t calls_applied = f.calls.load();
        CHECK(calls_applied <= 4);                      // 4 layers can each swap once; then cooldown + nothing hot left
    }
    // Failed: every op reports kTierFailed, cooldown is cleared for it, and the SAME swap is proposed again next round.
    {
        Fixture f; f.result = kTierFailed; f.cfg.policy.max_moves_per_round = 2;   // one swap per round
        RetierLoop lp(f.cfg, Fixture::L, Fixture::E, f.logger());
        std::vector<std::pair<int, int>> seen;
        auto ap = f.apply();
        auto rec = [&](const TierWireOp* o, int32_t n, uint8_t* r, std::string& e) { bool ok = ap(o, n, r, e); seen.emplace_back(o[0].layer_new, o[0].expert_new); return ok; };
        for (int i = 0; i < 4000 && seen.size() < 2; ++i) { lp.tick(f.usage.data(), f.usage.size(), f.held, 0, true, rec); std::this_thread::sleep_for(std::chrono::milliseconds(1)); }
        CHECK(seen.size() >= 2 && seen[0] == seen[1]);
        CHECK(lp.stats().failed >= 2);
    }
}

static void t_planner_error_fails_open() {
    g_test = "planner error";
    Fixture f;
    f.usage[3] = -1.f;                                  // a negative heat: the policy refuses it (fail closed)
    RetierLoop lp(f.cfg, Fixture::L, Fixture::E, f.logger());
    pump(lp, f, [&] { return lp.stats().plan_errors >= 1; });
    CHECK(lp.stats().plan_errors >= 1 && f.calls.load() == 0);
    bool saw = false;
    for (auto& s : f.lines) saw = saw || s.find("PLANNER ERROR") != std::string::npos;
    CHECK(saw);
    CHECK(lp.tick(f.usage.data(), f.usage.size(), f.held, 0, true, f.apply()));   // and it keeps serving
}

static void t_startup_check() {
    g_test = "retier_check";
    RetierConfig c; c.mode = RetierMode::kOn;
    std::string err;
    CHECK(!retier_check(c, 0, 0, 0, 32, true, err) && err.find("--adapt-every") != std::string::npos && err.find("--adapt-swaps") != std::string::npos);
    CHECK(!retier_check(c, 4, 96, 0, 32, true, err) && err.find("not allocated") != std::string::npos);   // empty heat with the feature on
    CHECK(!retier_check(c, 4, 0, 32, 32, true, err));
    CHECK(!retier_check(c, 4, 96, 32, 32, false, err) && err.find("helper") != std::string::npos);
    CHECK(retier_check(c, 4, 96, 32, 32, true, err));
    RetierConfig off;
    CHECK(retier_check(off, 0, 0, 0, 32, false, err));  // off: no requirements at all
    off.promote = RetierPromote::kSeabraid;
    CHECK(!retier_check(off, 0, 0, 0, 32, false, err) && err.find("needs --retier on") != std::string::npos);   // a mode of the round
    RetierConfig sb = c; sb.promote = RetierPromote::kSeabraid;
    CHECK(retier_check(sb, 4, 96, 32, 32, true, err));
    sb.mode = RetierMode::kZeroOp;
    CHECK(!retier_check(sb, 4, 96, 32, 32, true, err) && err.find("zero-op") != std::string::npos);   // the driver sends real moves
    RetierConfig b = c; b.policy.max_moves_per_round = 65;
    CHECK(!retier_check(b, 4, 96, 32, 32, true, err));
    b = c; b.policy.promote_margin = 0.5; CHECK(!retier_check(b, 4, 96, 32, 32, true, err));
    b = c; b.policy.max_helper_layer_share = 1.5; CHECK(!retier_check(b, 4, 96, 32, 32, true, err));
    b = c; b.every_windows = 0; CHECK(!retier_check(b, 4, 96, 32, 32, true, err));
}

static void t_flags() {
    g_test = "flags";
    RetierConfig c;
    CHECK(c.mode == RetierMode::kOff);                  // DEFAULT OFF
    CHECK(c.policy.max_helper_layer_share == 0.08 && c.every_windows == 128);
    std::string err;
    const char* val = nullptr;
    auto next = [&](const char*) { return val; };
    val = "on";          CHECK(retier_parse_flag("--retier", next, c, err) == 1 && c.mode == RetierMode::kOn);
    val = "zero-op";     CHECK(retier_parse_flag("--retier", next, c, err) == 1 && c.mode == RetierMode::kZeroOp);
    val = "maybe";       CHECK(retier_parse_flag("--retier", next, c, err) == 2);
    val = "16";          CHECK(retier_parse_flag("--retier-every", next, c, err) == 1 && c.every_windows == 16);
    val = "6";           CHECK(retier_parse_flag("--retier-moves", next, c, err) == 1 && c.policy.max_moves_per_round == 6);
    val = "1.5";         CHECK(retier_parse_flag("--retier-margin", next, c, err) == 1 && c.policy.promote_margin == 1.5);
    val = "3";           CHECK(retier_parse_flag("--retier-cooldown", next, c, err) == 1 && c.policy.cooldown_rounds == 3);
    val = "0.15";        CHECK(retier_parse_flag("--retier-ceiling", next, c, err) == 1 && c.policy.max_helper_layer_share == 0.15);
    val = "x";           CHECK(retier_parse_flag("--retier-every", next, c, err) == 2);
    // --retier-promote: `on` is the old spelling of legacy; seabraid is its own publisher; anything else is refused, value kept.
    CHECK(c.promote == RetierPromote::kOff && !c.ladder());
    val = "on";          CHECK(retier_parse_flag("--retier-promote", next, c, err) == 1 && c.promote == RetierPromote::kLegacy && c.ladder());
    val = "seabraid";    CHECK(retier_parse_flag("--retier-promote", next, c, err) == 1 && c.promote == RetierPromote::kSeabraid && !c.ladder());
    val = "legacy";      CHECK(retier_parse_flag("--retier-promote", next, c, err) == 1 && c.promote == RetierPromote::kLegacy);
    val = "both";        CHECK(retier_parse_flag("--retier-promote", next, c, err) == 2 && c.promote == RetierPromote::kLegacy && err.find("seabraid") != std::string::npos);
    val = "off";         CHECK(retier_parse_flag("--retier-promote", next, c, err) == 1 && c.promote == RetierPromote::kOff);
    CHECK(retier_parse_flag("--adapt-every", next, c, err) == 0);
    CHECK(retier_parse_flag("--retier-bogus", next, c, err) == 0);
}

// --retier-fallback / --retier-hops: defaults are v1's, ranges are checked, and only the seabraid publisher may use them.
static void pyramid_ranges(RetierConfig& c);
static void t_pyramid_flags() {
    g_test = "pyramid_flags";
    RetierConfig c;
    CHECK(c.fallback_boundary == 0 && c.max_hops == 1);   // DEFAULT: today's behaviour
    std::string err;
    const char* val = nullptr;
    auto next = [&](const char*) { return val; };
    val = "1";   CHECK(retier_parse_flag("--retier-fallback", next, c, err) == 1 && c.fallback_boundary == 1);
    val = "3";   CHECK(retier_parse_flag("--retier-hops", next, c, err) == 1 && c.max_hops == 3);
    val = "x";   CHECK(retier_parse_flag("--retier-hops", next, c, err) == 2);
    c.mode = RetierMode::kOn;
    CHECK(!retier_check(c, 4, 96, 32, 32, true, err) && err.find("--retier-promote seabraid") != std::string::npos);   // legacy/off publisher
    c.promote = RetierPromote::kLegacy;
    CHECK(!retier_check(c, 4, 96, 32, 32, true, err));
    c.promote = RetierPromote::kSeabraid;
    CHECK(!retier_check(c, 4, 96, 32, 32, true, err) && err.find("--retier-hops 3") != std::string::npos && err.find("--retier-moves >= 6") != std::string::npos &&
          err.find("got --retier-moves 4") != std::string::npos);   // hops 3 with the default 4 moves would stop a rung short: refused, both values named
    c.policy.max_moves_per_round = 6;
    CHECK(retier_check(c, 4, 96, 32, 32, true, err));   // 2 * hops moves: allowed
    c.max_hops = 32;   c.policy.max_moves_per_round = 64;   CHECK(retier_check(c, 4, 96, 32, 32, true, err));   // the largest climb the op capacity allows
    pyramid_ranges(c);
}

static void pyramid_ranges(RetierConfig& c) {
    std::string err;
    const char* val = nullptr;
    auto next = [&](const char*) { return val; };
    c.max_hops = 65;   CHECK(!retier_check(c, 4, 96, 32, 32, true, err) && err.find("1..64") != std::string::npos);
    c.max_hops = 0;    CHECK(!retier_check(c, 4, 96, 32, 32, true, err));
    c.max_hops = 1; c.fallback_boundary = 2;   CHECK(!retier_check(c, 4, 96, 32, 32, true, err) && err.find("0 or 1") != std::string::npos);
    val = "4294967297"; c.max_hops = 1; CHECK(retier_parse_flag("--retier-hops", next, c, err) == 1 && !retier_check(c, 4, 96, 32, 32, true, err));   // no wrap into range
    RetierConfig off; off.max_hops = 2;   // flags without --retier at all: still refused
    CHECK(!retier_check(off, 0, 0, 0, 32, false, err));
}

// --retier-file-moves (FR-E): off by default, 0|1 only, and only with the seabraid publisher (the one swap that can take its bytes from the file).
static void t_file_moves_flag() {
    g_test = "file_moves_flag";
    RetierConfig c;
    CHECK(c.file_moves == 0);   // DEFAULT: off (its speed is unmeasured)
    std::string err;
    const char* val = nullptr;
    auto next = [&](const char*) { return val; };
    val = "1";   CHECK(retier_parse_flag("--retier-file-moves", next, c, err) == 1 && c.file_moves == 1);
    val = "x";   CHECK(retier_parse_flag("--retier-file-moves", next, c, err) == 2);
    c.mode = RetierMode::kOn;
    CHECK(!retier_check(c, 4, 96, 32, 32, true, err) && err.find("--retier-file-moves 1 needs --retier-promote seabraid") != std::string::npos);
    c.promote = RetierPromote::kLegacy;
    CHECK(!retier_check(c, 4, 96, 32, 32, true, err));   // the legacy publisher cannot use it either
    c.promote = RetierPromote::kSeabraid;
    CHECK(retier_check(c, 4, 96, 32, 32, true, err));
    c.file_moves = 2;   CHECK(!retier_check(c, 4, 96, 32, 32, true, err) && err.find("0 or 1") != std::string::npos);
    c.file_moves = -1;  CHECK(!retier_check(c, 4, 96, 32, 32, true, err));
    val = "4294967297"; CHECK(retier_parse_flag("--retier-file-moves", next, c, err) == 1 && !retier_check(c, 4, 96, 32, 32, true, err));   // no wrap into range
    RetierConfig off; off.file_moves = 1;   // the flag without --retier at all: still refused
    CHECK(!retier_check(off, 0, 0, 0, 32, false, err));
    CHECK(std::string(retier_help()).find("--retier-file-moves 0|1") != std::string::npos);
}

// k > 1 helpers: only seabraid serves them; the legacy paths name the flag to use.
static void t_helper_count() {
    g_test = "helper_count";
    RetierConfig c; c.mode = RetierMode::kOn;
    std::string err;
    CHECK(retier_check_helpers(c, 1, err) && retier_check_helpers(c, 0, err));   // one helper: legacy keeps working
    CHECK(!retier_check_helpers(c, 2, err) && err.find("--retier-promote seabraid") != std::string::npos && err.find("2") != std::string::npos);
    c.promote = RetierPromote::kLegacy;
    CHECK(!retier_check_helpers(c, 3, err) && err.find("legacy") != std::string::npos);
    c.promote = RetierPromote::kSeabraid;
    CHECK(retier_check_helpers(c, 3, err));
    RetierConfig off;
    CHECK(retier_check_helpers(off, 3, err));   // no retier: nothing to refuse
}

int main() {
    t_pyramid_flags(); t_file_moves_flag(); t_helper_count();
    t_conversion(); t_mailbox(); t_off_runs_nothing(); t_applied_on_engine_thread_only(); t_zero_op();
    t_failed_ops_feed_note_failed(); t_planner_error_fails_open(); t_startup_check(); t_flags();
    std::printf(g_fail ? "retier_loop_test: %d FAILED checks\n" : "retier_loop_test: all checks passed\n", g_fail);
    return g_fail != 0;
}
