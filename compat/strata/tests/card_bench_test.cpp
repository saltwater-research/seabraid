// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// seabraid D8 checker: the pyramid ordering, the sample statistics and the stderr line (host only, no GPU).
//   g++ -std=c++17 -Istrata-ext/include tests/card_bench_test.cpp strata-ext/src/card_bench.cpp -o card_bench_test
// Every rule in order_tiers() has one test that fails when the rule is removed (see the D8 commit message).
#include "strata/seabraid/card_bench.hpp"

#include <cstdio>
#include <vector>

using namespace seabraid;

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++g_fail; } } while (0)

static CardBench helper(const char* name, double dec, double pre, uint64_t free_b = 1) {
    CardBench c;
    c.name = name; c.vendor = "cuda"; c.kind = TierKind::Helper; c.free_vram_bytes = free_b;
    if (dec > 0) c.decode.rows_per_s = dec;
    if (pre > 0) c.prefill.rows_per_s = pre;
    return c;
}
static CardBench tier(TierKind k) { CardBench c; c.name = k == TierKind::Main ? "main" : k == TierKind::Ram ? "ram" : "nvme"; c.kind = k; return c; }
static bool is(const std::vector<int>& got, std::vector<int> want) { return got == want; }

static void test_pyramid_shape() {   // main > helpers > RAM > NVMe, whatever order the caller listed them
    std::vector<CardBench> v = {tier(TierKind::Nvme), helper("a", 100, 100), tier(TierKind::Ram), tier(TierKind::Main)};
    CHECK(is(order_tiers(v, Regime::Decode), {3, 1, 2, 0}));
}
static void test_rate_order() {
    std::vector<CardBench> v = {helper("slow", 10, 10), helper("fast", 30, 30), helper("mid", 20, 20)};
    CHECK(is(order_tiers(v, Regime::Decode), {1, 2, 0}));
}
static void test_regimes_are_independent() {   // the whole point of D8: the two orders may disagree
    std::vector<CardBench> v = {helper("decode_king", 50, 100), helper("prefill_king", 20, 400)};
    CHECK(is(order_tiers(v, Regime::Decode), {0, 1}));
    CHECK(is(order_tiers(v, Regime::Prefill), {1, 0}));
}
static void test_tie_breaks() {   // equal rate: more free VRAM first; equal both: lower index (deterministic)
    std::vector<CardBench> v = {helper("a", 10, 10, 100), helper("b", 10, 10, 300), helper("c", 10, 10, 300)};
    CHECK(is(order_tiers(v, Regime::Decode), {1, 2, 0}));
    CHECK(is(order_tiers(v, Regime::Decode), order_tiers(v, Regime::Decode)));
}
static void test_unmeasured_last_and_flagged() {
    std::vector<CardBench> v = {helper("nan", 0, 10), helper("ok", 5, 5), tier(TierKind::Ram), tier(TierKind::Nvme)};
    v[0].decode.rows_per_s = kNaN;   // NaN, not merely absent
    std::vector<int> flagged;
    CHECK(is(order_tiers(v, Regime::Decode, &flagged), {1, 2, 3, 0}));   // even below NVMe: never guessed
    CHECK(is(flagged, {0}));
    CHECK(is(order_tiers(v, Regime::Prefill, &flagged), {0, 1, 2, 3}));   // measured in prefill, so flagged list empty
    CHECK(flagged.empty());
}
static void test_bad_numbers_are_unmeasured() {
    std::vector<CardBench> v = {helper("inf", 0, 1), helper("neg", 0, 1), helper("zero", 0, 1), helper("ok", 1, 1)};
    v[0].decode.rows_per_s = std::numeric_limits<double>::infinity();
    v[1].decode.rows_per_s = -5;
    v[2].decode.rows_per_s = 0;
    std::vector<int> flagged;
    CHECK(is(order_tiers(v, Regime::Decode, &flagged), {3, 0, 1, 2}));
    CHECK(flagged.size() == 3);
}
static void test_main_pinned_first_even_unmeasured() {
    std::vector<CardBench> v = {helper("fast", 1000, 1000), tier(TierKind::Main)};
    std::vector<int> flagged;
    CHECK(is(order_tiers(v, Regime::Decode, &flagged), {1, 0}));
    CHECK(is(flagged, {1}));   // first, but still reported as unmeasured
}
static void test_empty_and_input_untouched() {
    CHECK(order_tiers({}, Regime::Decode).empty());
    std::vector<CardBench> v = {helper("a", 1, 1), helper("b", 2, 2)};
    order_tiers(v, Regime::Decode);
    CHECK(v[0].name == "a" && v[1].name == "b");
}
static void test_summarize() {
    SampleStats s = summarize({5, 1, 3, 2, 4});
    CHECK(s.n == 5 && s.median == 3 && s.min == 1 && s.max == 5 && s.q1 == 2 && s.q3 == 4);
    s = summarize({4, 1, 2, 3});
    CHECK(s.median == 2.5);
    s = summarize({1, kNaN, -2, 0, 3});   // faults dropped, not averaged in
    CHECK(s.n == 2 && s.median == 2);
    CHECK(summarize({}).n == 0 && std::isnan(summarize({}).median));
}
static void test_make_stat() {
    std::vector<double> t(20, 0.001);   // 1 ms per call, 40 rows -> 40000 rows/s
    RegimeStat r = make_stat(40, 4, t, 10);
    CHECK(r.valid() && r.rows_per_s > 39999 && r.rows_per_s < 40001 && r.iters == 20 && r.spread == 0);
    CHECK(r.rows_per_call == 40 && r.rows_per_expert == 4);
    RegimeStat few = make_stat(40, 4, std::vector<double>(3, 0.001), 10);   // 3 samples < min_iters: no guess
    CHECK(!few.valid() && few.iters == 3);
    CHECK(!make_stat(0, 0, t, 1).valid());
}
static void test_line_roundtrip() {
    CardBench c = helper("NVIDIA GeForce RTX 3060 = x%", 28000.5, 105000, 11ull << 30);
    c.layer = 24; c.note = "prefill: a b";
    c.decode.iters = 200; c.decode.rows_per_call = 40; c.decode.rows_per_expert = 4; c.decode.spread = 0.05;
    const std::string line = format_line(c);
    CHECK(line.find('\n') == std::string::npos);
    CHECK(line.find("status=ok") != std::string::npos);
    CardBench d;
    CHECK(parse_line(line, d));
    CHECK(d.name == c.name && d.vendor == "cuda" && d.free_vram_bytes == c.free_vram_bytes && d.layer == 24);
    CHECK(d.decode.rows_per_s == 28000.5 && d.prefill.rows_per_s == 105000 && d.decode.rows_per_expert == 4 && d.note == c.note);
    // an unmeasured card prints nan and says so; it must not parse back as a number
    CardBench u = helper("u", 0, 0);
    const std::string ul = format_line(u);
    CHECK(ul.find("status=unmeasured") != std::string::npos && ul.find("decode_rows_per_s=nan") != std::string::npos);
    CardBench pu;
    CHECK(parse_line(ul, pu) && !pu.measured(Regime::Decode) && !pu.measured(Regime::Prefill));
    CHECK(!parse_line("seabraid-helper: something else", pu));
}
// A %XX at the very end of a value (the last token of a line, nothing after it) must decode like any other.  The
// audit flagged unesc() for dropping it; this pins the boundary in both directions (a lone '%' or "%4" stays literal).
static void test_trailing_escape_decodes() {
    CardBench c;
    CHECK(parse_line("SEABRAID-CARD-BENCH v1 name=abc%41 vendor=cuda note=end%20", c));
    CHECK(c.name == "abcA" && c.note == "end ");
    CHECK(parse_line("SEABRAID-CARD-BENCH v1 name=%41", c) && c.name == "A");
    CHECK(parse_line("SEABRAID-CARD-BENCH v1 name=x%4", c) && c.name == "x%4");
    CHECK(parse_line("SEABRAID-CARD-BENCH v1 name=x%", c) && c.name == "x%");
    CHECK(parse_line("SEABRAID-CARD-BENCH v1 name=x%zz", c) && c.name == "x%zz");
}

// ---- lead job (N-CARD-LADDER module M2): card 1's own job, ms per decode window, lower is better, one class per concurrency
static CardBench lead_card(const char* name, double one, double many, uint64_t free_b = 1) {
    CardBench c = helper(name, 100, 100, free_b);
    if (one > 0) c.lead_one.window_ms = one;
    if (many > 0) c.lead_many.window_ms = many;
    return c;
}
static void test_lead_order_ranks_by_ms_ascending_per_class() {   // the 2026-11-09 test system: 3090 first at 1 stream, R9700 at 4
    std::vector<CardBench> v = {lead_card("r9700", 26.7, 24.0), lead_card("3090", 24.0, 26.0)};
    CHECK(is(order_lead(v, LeadClass::One), {1, 0}));
    CHECK(is(order_lead(v, LeadClass::Many), {0, 1}));
}
static void test_lead_unmeasured_last_flagged_and_tiers_are_not_candidates() {
    std::vector<CardBench> v = {lead_card("nan", 0, 0), lead_card("ok", 30, 30), tier(TierKind::Ram), tier(TierKind::Nvme)};
    v[0].lead_one.window_ms = kNaN;
    std::vector<int> flagged;
    CHECK(is(order_lead(v, LeadClass::One, &flagged), {1, 0}));   // RAM and NVMe never appear: they cannot be card 1
    CHECK(is(flagged, {0}));
    v[2].lead_one.window_ms = 1;   // a "measured" RAM tier is still not a candidate
    CHECK(is(order_lead(v, LeadClass::One), {1, 0}));
    v[1].lead_one.window_ms = -4;   // bad numbers are unmeasured, never ranked first
    CHECK(is(order_lead(v, LeadClass::One, &flagged), {0, 1}) && is(flagged, {0, 1}));   // both unmeasured: input order
}
static void test_lead_tie_breaks() {   // equal ms: more free VRAM first; equal both: lower index
    std::vector<CardBench> v = {lead_card("a", 25, 25, 100), lead_card("b", 25, 25, 300), lead_card("c", 25, 25, 300)};
    CHECK(is(order_lead(v, LeadClass::One), {1, 2, 0}));
}
static void test_lead_does_not_move_the_expert_order_and_expert_rate_does_not_move_the_lead_order() {
    std::vector<CardBench> v = {helper("fast_rows", 300, 300), helper("fast_lead", 100, 100)};
    v[0].lead_one.window_ms = 30;
    v[1].lead_one.window_ms = 20;
    CHECK(is(order_tiers(v, Regime::Decode), {0, 1}));
    CHECK(is(order_lead(v, LeadClass::One), {1, 0}));
}
static void test_lead_class_mapping() {
    LeadClass c = LeadClass::Many;
    CHECK(lead_class_for(1, c) && c == LeadClass::One);
    CHECK(lead_class_for(4, c) && c == LeadClass::Many);
    CHECK(lead_class_for(16, c) && c == LeadClass::Many);
    CHECK(!lead_class_for(2, c) && !lead_class_for(3, c) && !lead_class_for(0, c) && !lead_class_for(-1, c));
}
static void test_lead_timing_boundary_excludes_the_expert_kernels() {
    std::vector<WindowSample> w(40, {0.020, 0.004});   // 20 ms non-expert + 4 ms of routed-expert kernels per window
    std::vector<WindowSample> slow_experts(40, {0.020, 0.008});   // the same card with its expert time doubled
    const LeadStat a = make_lead(1, w, 10), b = make_lead(1, slow_experts, 10);
    CHECK(a.valid() && a.window_ms > 19.999 && a.window_ms < 20.001 && a.iters == 40 && a.streams == 1 && a.spread == 0);
    CHECK(a.window_ms == b.window_ms);   // expert time cannot leak into the lead figure
    std::vector<WindowSample> slow_lead(40, {0.040, 0.004});
    CHECK(make_lead(1, slow_lead, 10).window_ms > 39.999);   // ...while the non-expert part is what it reads
}
static void test_make_lead_unmeasured_cases() {
    std::vector<WindowSample> few(3, {0.02, 0.004});
    CHECK(!make_lead(1, few, 10).valid() && make_lead(1, few, 10).iters == 3);   // 3 samples < min_iters: no guess
    std::vector<WindowSample> bad(20, {kNaN, 0.004});
    CHECK(!make_lead(1, bad, 5).valid() && make_lead(1, bad, 5).iters == 0);      // expert time alone is not data
    CHECK(!make_lead(0, std::vector<WindowSample>(20, {0.02, 0.004}), 5).valid());   // no stream count, no regime
    CHECK(!make_lead(1, {}, 1).valid());
}
static void test_lead_line_roundtrip_and_old_readers() {
    CardBench c = lead_card("RTX 3090", 24.0, 26.5);
    c.lead_one.spread = 0.04; c.lead_one.iters = 60; c.lead_one.streams = 1;
    c.lead_many.iters = 30; c.lead_many.streams = 4;
    const std::string line = format_line(c);
    CHECK(line.find("lead1_window_ms=24 ") != std::string::npos && line.find("lead4_window_ms=26.5") != std::string::npos);
    CardBench d;
    CHECK(parse_line(line, d) && d.lead_one.window_ms == 24.0 && d.lead_many.window_ms == 26.5 && d.lead_one.iters == 60);
    CHECK(d.lead_one.streams == 1 && d.lead_many.streams == 4 && d.lead_one.spread == 0.04);
    CardBench u = helper("u", 10, 10);   // never measured: prints nan, parses back unmeasured, status is about expert regimes only
    const std::string ul = format_line(u);
    CHECK(ul.find("lead1_window_ms=nan") != std::string::npos && ul.find("status=ok") != std::string::npos);
    CardBench pu;
    CHECK(parse_line(ul, pu) && !pu.lead_measured(LeadClass::One) && !pu.lead_measured(LeadClass::Many));
    CardBench old;   // a line from before the lead regime existed
    CHECK(parse_line("SEABRAID-CARD-BENCH v1 name=x vendor=cuda kind=helper decode_rows_per_s=5 prefill_rows_per_s=6", old));
    CHECK(old.measured(Regime::Decode) && !old.lead_measured(LeadClass::One));
}

int main() {
    test_pyramid_shape(); test_rate_order(); test_regimes_are_independent(); test_tie_breaks();
    test_unmeasured_last_and_flagged(); test_bad_numbers_are_unmeasured(); test_main_pinned_first_even_unmeasured();
    test_empty_and_input_untouched(); test_summarize(); test_make_stat(); test_line_roundtrip(); test_trailing_escape_decodes();
    test_lead_order_ranks_by_ms_ascending_per_class(); test_lead_unmeasured_last_flagged_and_tiers_are_not_candidates();
    test_lead_tie_breaks(); test_lead_does_not_move_the_expert_order_and_expert_rate_does_not_move_the_lead_order();
    test_lead_class_mapping(); test_lead_timing_boundary_excludes_the_expert_kernels(); test_make_lead_unmeasured_cases();
    test_lead_line_roundtrip_and_old_readers();
    std::printf(g_fail ? "card_bench_test: %d FAILED\n" : "card_bench_test: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
