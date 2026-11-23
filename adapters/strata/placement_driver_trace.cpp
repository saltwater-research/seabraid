// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// PlacementDriver's trace wiring: what the driver tells placement_trace.hpp, and when.
// OWNER: which driver events become trace records (round, helper up/closed/served_after_loss, per-move times).
//   The record format is placement_trace.cpp's; the round itself is placement_driver.cpp's.
// FAILURE CONTRACT: every function is a no-op while the trace is off and cannot change a round's outcome; a host
//   callback that throws only loses the fallback counters of that round.
// CHECKER: placement_trace_test drives a real round through the driver and reads the file back.
#include "placement_driver.hpp"

#include <chrono>
#include <exception>

namespace seabraid::strata_adapter {
namespace {

uint64_t steady_us() noexcept {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

}  // namespace

// WHY here and not at open: open() runs before the host decides on a trace path in some embeddings, and a tick is
// the first moment the driver is certain to be serving. The "up" events say which rungs the run started with.
void PlacementDriver::trace_tick() noexcept {
    if (!trace::on()) return;
    if (!announced_) {
        announced_ = true;
        for (uint8_t rung = 1; rung < fallback(); ++rung) trace::helper(rung, "up", "driver started", 0);
    }
    if (lost_at_us_ == 0) return;
    // The first tick after a loss is the next window the engine serves: with the rung closed its cells run on the
    // fallback, so this is the "time from loss to the next served request" as the placement driver can see it.
    trace::helper(lost_rung_, "served_after_loss", "next window", steady_us() - lost_at_us_);
    lost_at_us_ = 0;
}

void PlacementDriver::note_loss(uint8_t rung, const char* why) noexcept {
    if (!trace::on()) return;
    trace::helper(rung, "closed", why, 0);
    lost_at_us_ = steady_us();
    lost_rung_ = rung;
}

bool PlacementDriver::read_fallback(int64_t& ram_hits, int64_t& file_reads) const noexcept {
    if (!config_.fallback_counters) return false;
    try {
        config_.fallback_counters(ram_hits, file_reads);
        return true;
    } catch (const std::exception&) {   // allow-error-path: the round record says "null" instead of a count
        return false;
    }
}

void PlacementDriver::time_outcomes() noexcept {
    timings_.clear();
    if (!trace::on()) return;
    try {
        timings_ = trace::attribute_calls(round_.outcomes, round_.calls);
    } catch (const std::exception&) {   // allow-error-path: moves are written untimed
        timings_.clear();
    }
}

void PlacementDriver::trace_round(bool counted, int64_t ram0, int64_t file0) noexcept {
    if (!trace::on()) return;
    trace::RoundFields fields;
    fields.round = stats_.round;
    fields.epoch = stats_.epoch;
    fields.round_us = stats_.fence_us;
    fields.qmask = stats_.qmask;
    fields.planned = stats_.planned;
    fields.committed = stats_.committed;
    fields.rejected = stats_.rejected;
    fields.restored = stats_.restored;
    fields.quarantined = stats_.quarantined;
    fields.skipped = stats_.skipped;
    int64_t ram1 = 0, file1 = 0;
    if (counted && read_fallback(ram1, file1)) {
        fields.fallback_counted = true;
        fields.fallback_ram_hits = ram1 - ram0;
        fields.fallback_file_reads = file1 - file0;
    }
    trace::round(fields);
}

}  // namespace seabraid::strata_adapter
