// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// The opt-in move ledger and latency trace: one JSON object per line, written only when SEABRAID_TRACE=<path>.
// OWNER: the record schema (the "rec" kinds and their keys), the on/off switch and the window statistics. The diag
//   lines stay with placement_round_log.cpp; the trace report tool (a stdlib script) reads what this file writes.
// CONTRACT: callers ask on() first (one relaxed atomic load) and build a record only when it is true. The records are
//   "move", "round", "window" and "helper". Depends on diag and the placement ABI header only; no engine include.
//   SEABRAID_TRACE_EVERY (default 4096) is the number of collects per "window" record.
// SIZE BUDGET: header 110, placement_trace.cpp 330. FAILURE CONTRACT: fail-open-loud. An open or write error logs ONE
//   warning, turns the trace off for the rest of the process and returns to the caller as if nothing happened; no
//   function here throws, and none can change a serving or placement outcome.
// CHECKER: placement_trace_test (format, fail-open, every outcome kind, window math, trace off writes nothing), the
//   driver trace fixture (every outcome kind) and the report tool's own test on a synthetic trace in this format.
#pragma once

#include "seabraid/placement.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace seabraid::strata_adapter::trace {

constexpr uint32_t kDefaultEvery = 4096;

// The switch. on() reads SEABRAID_TRACE on its first call; configure() is for a host or test that sets it directly.
bool on() noexcept;
uint32_t every() noexcept;                         // collects per window record
void configure(const char* path, uint32_t every_collects) noexcept;   // null or "" turns the trace off
void close_trace() noexcept;                       // flush and close; on() is false afterwards until configure()

// Mean and nearest-rank percentiles of microsecond samples.
struct Stats {
    uint64_t n = 0, p50 = 0, p99 = 0;
    double mean = 0.0;
};
class Sampler {
public:
    void add(uint64_t us) noexcept;
    Stats stats() const noexcept;                  // sorts a copy; called once per window
    void clear() noexcept { values_.clear(); sum_ = 0; }
private:
    std::vector<uint64_t> values_;
    uint64_t sum_ = 0;
};

// One stage or replace call of a round, recorded by the provider callbacks in placement.cpp.
struct CallSample {
    bool replace = false;
    uint8_t rung = 0;
    uint32_t layer = 0, expert = 0;
    uint64_t us = 0;
    bool by_ref = false;                           // a replace_from_fallback the rung ACKed: the helper read the expert from the file
};
struct PairTiming {
    uint64_t stage_us = 0, write_us = 0;
    bool timed = false;                            // false: the calls could not be matched to this outcome
    bool by_ref = false;                           // the promoted expert was loaded by file reference (record src "file")
};
// One PairTiming per outcome, in outcome order. A call group starts at a stage that follows a replace; SKIPPED
// outcomes made no call, so they get an untimed entry.
std::vector<PairTiming> attribute_calls(const std::vector<seabraid_placement_outcome_v1>& outcomes,
                                        const std::vector<CallSample>& calls);

struct MoveFields {
    uint64_t round = 0, epoch = 0;
    const seabraid_placement_outcome_v1* outcome = nullptr;
    uint8_t fallback = 0;                          // the fallback rung number: a hop to or from it is src/dst=fallback
    size_t bytes = 0;
    bool verified = true;                          // both writes of a committed hop were read back (see log_moves)
    PairTiming timing;                             // timing.by_ref makes the record's src "file"
};
struct RoundFields {
    uint64_t round = 0, epoch = 0, round_us = 0, qmask = 0;
    uint32_t planned = 0, committed = 0, rejected = 0, restored = 0, quarantined = 0, skipped = 0;
    bool fallback_counted = false;                 // the host gave counters: the two deltas below are real
    int64_t fallback_ram_hits = 0, fallback_file_reads = 0;   // this round's change in FallbackRung's counters
};
struct WindowFields {
    uint32_t rung = 0;                             // the helper's label (see QwenRuntime::set_trace_rung)
    uint64_t layers = 0, rows_submitted = 0, rows_shipped = 0, failures = 0, timeouts = 0, window_us = 0;
    Stats submit_to_collect, blocked;
};

// Each writes one line when on(). helper events: "up", "down", "closed", "served_after_loss".
void move(const MoveFields& fields) noexcept;
void round(const RoundFields& fields) noexcept;
void window(const WindowFields& fields) noexcept;
void helper(uint32_t rung, const char* event, const char* reason, uint64_t after_loss_us) noexcept;

// Names used by the diag lines and the records, one owner: "stage|replace|restore|none", "refused|failed|...|none".
const char* phase_name(uint32_t phase) noexcept;
const char* issue_name(uint32_t error) noexcept;
const char* kind_name(uint32_t kind) noexcept;

}  // namespace seabraid::strata_adapter::trace
