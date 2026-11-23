// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
#pragma once
// seabraid D8: the per-card startup benchmark's host-side half - the result record, the statistics, the one-line
// stderr format and the pyramid ordering.  No GPU runtime is referenced here (the GPU half is
// src/seabraid/card_bench_run.cpp), so the ordering is unit-testable on any host and any vendor's build includes it.
//
// DESIGN GATE
//   Owner:    this header + src/seabraid/card_bench.cpp own "what a card measured, and how cards are ranked".  The
//             constants of the measurement (batch sizes, iteration counts, time cap) are owned by the runner, the
//             only place that knows the real expert path; nothing here duplicates them.
//   Contract: CardBench / RegimeStat (plain data, the struct a future channel version puts on the wire),
//             summarize(), make_stat(), order_tiers(), format_line(), parse_line().  Imports: std only.  The
//             runner and helper.cpp depend on this; this depends on neither.
//   Size:     header <= 300 lines, card_bench.cpp <= 300 (split before 240).
//   Failure:  fail-open-loudly.  A missing / NaN / non-positive / too-few-samples measurement is "unmeasured": it is
//             printed as nan, flagged by order_tiers(), and ranked last - never guessed, never an exception.
//   Checker:  tests/seabraid/card_bench_test.cpp (each ordering rule has a test that was sabotaged to fail once).
//   Finish:   the test builds with host g++ and passes; the helper TUs compile for CUDA sm_86; a run on real cards
//             times real cards (seabraid.md "D8").
//
// LEAD JOB (N-CARD-LADDER.md section 2, rule R1, module M2).  Expert rows/s is the job of cards 2..N; card 1 does something else:
// the non-expert path (attention, dense layers, router, drafter step, verify) with every expert resident.  It is measured as
// milliseconds per decode window (LOWER is better, the opposite sign of rows/s, so it is its own struct, LeadStat, and its own
// ordering, order_lead(), not a third Regime), once per concurrency class: 1 stream, and >= 4 streams.  Classes are never
// blended and 2-3 streams have no class.  Timing boundary: a WindowSample splits one window into non-expert and expert seconds
// and make_lead() reads ONLY the non-expert part, so a slower expert path cannot change a card's lead figure (tested).
// Nothing in the helper times attention: the engine-side probe that fills WindowSample is not part of this change, and until it
// exists every lead figure prints as nan and the installer plans card 1 by its proxy, saying so.
//
// The two regimes are NEVER blended: per-row expert cost swings ~3x with batch size (RTX 3060: ~28 us/row at 3 rows
// per call, ~9.5 us/row at 31), so a card that wins at decode batches can lose at prefill batches.  Callers ask
// order_tiers() once per regime.
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace seabraid {

enum class Regime { Decode, Prefill };

/// Pyramid role.  Main is the engine's own card (pinned first); Helper is a measured card; Ram and Nvme are the
/// fixed tiers below VRAM (pinned in that order; they are not timed by this benchmark).
enum class TierKind { Main, Helper, Ram, Nvme };

inline constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

/// One regime's result.  rows_per_s is NaN when unmeasured.
struct RegimeStat {
    double rows_per_s = kNaN;     ///< rows / median wall time of one begin()+finish() call
    double spread = kNaN;         ///< (q3 - q1) / median of the per-call times: how noisy the median is
    int iters = 0;                ///< measured calls that went into the median (warm-up excluded)
    int rows_per_call = 0;        ///< expert rows (token x k pairs) in one call
    int rows_per_expert = 0;      ///< rows each expert saw in one call: the batch size that sets per-row cost
    bool valid() const { return std::isfinite(rows_per_s) && rows_per_s > 0; }
};

/// Lead-job concurrency class.  One = 1 stream; Many = >= 4 streams.
enum class LeadClass { One, Many };
/// Map a stream count to its class; false for 2-3 (and < 1) streams: no regime exists there.
inline bool lead_class_for(int streams, LeadClass& out) {
    if (streams == 1) { out = LeadClass::One; return true; }
    if (streams >= 4) { out = LeadClass::Many; return true; }
    return false;
}

/// One lead-job result: median wall ms per decode window of the non-expert path.  NaN when unmeasured.  LOWER is better.
struct LeadStat {
    double window_ms = kNaN;
    double spread = kNaN;         ///< (q3 - q1) / median of the per-window times
    int iters = 0;                ///< windows that went into the median
    int streams = 0;              ///< concurrency the figure was taken at
    bool valid() const { return std::isfinite(window_ms) && window_ms > 0; }
};

/// One decode window split at the routed-expert kernels (seconds).  Only nonexpert_s is the lead job.
struct WindowSample { double nonexpert_s = kNaN; double expert_s = kNaN; };

struct CardBench {
    std::string name;             ///< device name as the driver reports it
    std::string vendor;           ///< "cuda" | "hip" | ... (the build the helper was made with)
    TierKind kind = TierKind::Helper;
    uint64_t free_vram_bytes = 0; ///< free before the helper loads anything
    int layer = -1;               ///< the layer whose experts were timed (rows/s differs by layer quant)
    RegimeStat decode, prefill;
    LeadStat lead_one, lead_many; ///< card-1 job at 1 stream / >= 4 streams (see LEAD JOB above)
    std::string note;             ///< why a regime is unmeasured, when the runner knows
    const RegimeStat& stat(Regime r) const { return r == Regime::Decode ? decode : prefill; }
    bool measured(Regime r) const { return stat(r).valid(); }
    const LeadStat& lead(LeadClass c) const { return c == LeadClass::One ? lead_one : lead_many; }
    bool lead_measured(LeadClass c) const { return lead(c).valid(); }
};

/// Median and quartiles of finite, positive samples (others are dropped, not counted).  n == 0 -> all NaN.
struct SampleStats { int n = 0; double min = kNaN, q1 = kNaN, median = kNaN, q3 = kNaN, max = kNaN; };
SampleStats summarize(std::vector<double> samples);

/// Turn per-call wall times (seconds) into a RegimeStat.  Fewer than `min_iters` usable samples -> unmeasured
/// (iters still reports how many there were), because a median of three calls is a guess.
RegimeStat make_stat(int rows_per_call, int rows_per_expert, const std::vector<double>& seconds, int min_iters);

/// Lead figure from per-window samples: median of nonexpert_s only (expert_s is ignored by construction).  Fewer than
/// `min_iters` usable samples (finite, positive nonexpert_s) -> unmeasured.  `streams` is recorded, not interpreted.
LeadStat make_lead(int streams, const std::vector<WindowSample>& windows, int min_iters);

/// Card-1 order for one class, as indices into `cards`: every card of kind `Main` or `Helper` (`Ram`/`Nvme` are not candidates) by window_ms
/// ASCENDING (ties: more free VRAM, then lower index); cards with no valid figure LAST, in input order.  `unmeasured`
/// (optional) receives their indices.  Unlike order_tiers() no card is pinned first: the lead job decides who is card 1.
std::vector<int> order_lead(const std::vector<CardBench>& cards, LeadClass c, std::vector<int>* unmeasured = nullptr);

/// Pyramid order for one regime, as indices into `cards`, best first:
///   Main(s) -> measured Helpers by rows/s desc (ties: more free VRAM, then lower index) -> Ram -> Nvme
///   -> UNMEASURED Helpers (last, so no plan counts on an unknown card).
/// `unmeasured` (optional) receives the indices of Main/Helper entries with no valid rate in this regime, in
/// output order.  A Main with no rate stays first (it is the main engine whatever it measured) but is still listed.
std::vector<int> order_tiers(const std::vector<CardBench>& cards, Regime r, std::vector<int>* unmeasured = nullptr);

/// One parseable stderr line per card: `SEABRAID-CARD-BENCH v1 key=value ...`, values percent-escaped (no spaces),
/// numbers as %.9g, NaN as "nan".  `status` (ok | partial | unmeasured) is derived, never stored.
std::string format_line(const CardBench& c);
/// Inverse of format_line (unknown keys ignored so a v1 reader survives added fields).  False if not a v1 line.
bool parse_line(const std::string& line, CardBench& out);

}  // namespace seabraid
