// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
#include "placement_driver.hpp"

#include "diag.hpp"

#include <algorithm>
#include <chrono>
#include <utility>

namespace seabraid::strata_adapter {
namespace {

using diag::kError;
using diag::kWarn;
using Clock = std::chrono::steady_clock;

uint64_t since_us(Clock::time_point start) noexcept {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start).count());
}

unsigned long long wide(uint64_t value) noexcept { return static_cast<unsigned long long>(value); }

}  // namespace

bool PlacementDriver::due(bool quiet) noexcept {
    if (++window_ < config_.every_windows) return false;
    // WHY the window counter keeps running while busy: a deferred round runs at the next quiet window, not a full
    // interval later, so a busy server cannot starve placement.
    if (!quiet) {
        ++deferred_;
        return false;
    }
    window_ = 0;
    return true;
}

bool PlacementDriver::tick(const float* heat, size_t n, bool quiet, PlacementHost& host) noexcept {
    if (state_ == State::kDead) return true;
    trace_tick();
    check_liveness(host);   // F11: every tick, whatever the cadence and whatever closed before (placement_liveness.cpp)
    if (state_ == State::kDead || !due(quiet)) return true;
    stats_ = RoundStats{};
    stats_.round = ++rounds_;
    stats_.deferred = static_cast<uint32_t>(deferred_);
    deferred_ = 0;
    const auto start = Clock::now();
    int64_t ram0 = 0, file0 = 0;
    const bool counted = trace::on() && read_fallback(ram0, file0);
    const Flow flow = run_round(heat, n, host);
    stats_.fence_us = since_us(start);
    stats_.epoch = epoch_;
    stats_.qmask = closed_mask();
    stats_.table_hash = view_.table_hash();
    if (n == published_.size()) stats_.helper_share = helper_share(published_.data(), heat, n, fallback());
    if (stats_.worked) {
        log_round(stats_);
        trace_round(counted, ram0, file0);
        host.after_round();
    }
    return flow != Flow::kStop;
}

PlacementDriver::Flow PlacementDriver::run_round(const float* heat, size_t n, PlacementHost& host) noexcept {
    Flow flow = adopt(host);
    if (flow != Flow::kGo) return flow;
    if ((flow = plan(heat, n)) != Flow::kGo) return flow;
    if (stats_.planned == 0) return Flow::kDone;   // nothing to move: publish nothing
    if ((flow = apply(host)) != Flow::kGo) return flow;
    flow = verify(host);
    if (flow == Flow::kGo) flow = publish_round();
    // WHY follow even after a degrade: the hops this round committed are physical fact. A rung lost at verify has its
    // table published by degrade(), the committed lead hops stay, and the engine's RAM copy must still follow them and
    // the move lines must still be written. Only a stop (lead lost) or a dead driver has no published table to follow.
    if (flow != Flow::kStop && state_ != State::kDead) follow(host);
    return flow == Flow::kGo ? Flow::kDone : flow;
}

PlacementDriver::Flow PlacementDriver::adopt(PlacementHost& host) noexcept {
    host.lead_cells(lead_);
    if (lead_.size() != published_.size()) {
        diag::logf(kError, SEABRAID_INVALID, "placement: host lead table has %zu cells, expected %zu", lead_.size(),
                   published_.size());
        stats_.worked = true;
        return Flow::kStop;
    }
    const Adoption found = scan_adoption(published_, lead_, fallback(), next_);
    if (found.conflict >= 0) return degrade(static_cast<uint8_t>(found.conflict), "adopt would move a helper cell");
    if (found.moved == 0) return Flow::kGo;
    // WHY update(ADOPT) from the executor table, closed rungs included: an open rung's executors equal its owners, and a
    // closed rung's cells already execute on the fallback (the bytes that card held are unreachable, and this driver
    // never REBUILDs a lost helper), so handing them to the fallback states the physical truth.
    int32_t status = seabraid_placement_update(handle_, epoch_, SEABRAID_PLACEMENT_ADOPT, next_.data(), next_.size());
    if (status == SEABRAID_OK) status = snapshot_staging();
    if (status != SEABRAID_OK) return die(status, "adopt");
    stats_.adopted = found.moved;
    stats_.worked = true;
    return publish_staged() ? Flow::kGo : Flow::kDone;
}

PlacementDriver::Flow PlacementDriver::plan(const float* heat, size_t n) noexcept {
    if (n != heat_.size()) return skip(kPlanSkip, SEABRAID_INVALID, "heat has the wrong cell count");
    std::copy(heat, heat + n, heat_.begin());
    stats_.qual = qual_.apply(heat_, rounds_);   // qualification only: steers placement, never routing
    ticket_ = seabraid_placement_ticket_v1{SEABRAID_ABI_VERSION, sizeof ticket_, 0, 0, 0};
    const auto start = Clock::now();
    const int32_t status = seabraid_placement_plan(handle_, epoch_, heat_.data(), heat_.size(), &ticket_);
    stats_.plan_us = since_us(start);
    if (status == SEABRAID_INTERNAL) return die(status, "plan");
    if (status != SEABRAID_OK) return skip(kPlanSkip, status, "plan refused");
    skips_[kPlanSkip] = 0;
    stats_.planned = static_cast<uint32_t>(ticket_.promotion_count);
    return Flow::kGo;
}

PlacementDriver::Flow PlacementDriver::apply(PlacementHost& host) noexcept {
    std::string err;   // the host's text may name files, so only the refusal itself is logged
    if (!host.drain(err)) return skip(kDrainSkip, SEABRAID_BUSY_OR_STATE, "drain refused");
    skips_[kDrainSkip] = 0;
    stats_.worked = true;
    const int32_t status = apply_placement_at_fence(handle_, ticket_, rungs_, round_);
    stats_.stage_us = round_.stage_us;
    stats_.replace_us = round_.replace_us;
    stats_.file_asked = round_.file_asked;
    stats_.file_acked = round_.file_acked;
    if (status == SEABRAID_BUSY_OR_STATE) return skip(kPlanSkip, status, "ticket refused");
    if (status != SEABRAID_OK) return die(status, "apply");
    const uint64_t closed_before = last_good_.quarantined_rungs;
    epoch_ = round_.snapshot.epoch;
    last_good_ = round_.snapshot;
    tally_outcomes(round_.outcomes, stats_);
    time_outcomes();   // before the early returns below: a round that closes a rung still has its outcomes on record
    log_failures(round_.outcomes, slot_bytes_, fallback(), move_context());
    const uint64_t closed = closed_mask();
    if (closed & 1u) {
        diag::logf(kError, SEABRAID_BUSY_OR_STATE, "placement lead rung quarantined: mask 0x%llx epoch %llu",
                   wide(closed), wide(epoch_));
        return Flow::kStop;
    }
    // The core already moved the table to the fallback for the closed rung; verify and the helper tier skip it.  Only a
    // rung this round closed is news: an earlier closure was reported when it happened.
    if ((closed & ~closed_before) != 0) {
        note_degraded("a helper rung was closed by the round");
        for (uint8_t rung = 1; rung < fallback(); ++rung)
            if (((closed & ~closed_before) >> rung) & 1u) note_loss(rung, "closed by a round");
    }
    return Flow::kGo;
}

// R2: the staged table is checked against the physical state before the view sees it. A helper that disagrees is
// closed and the fallback table is what gets published.
PlacementDriver::Flow PlacementDriver::verify(PlacementHost& host) noexcept {
    host.lead_cells(lead_);
    if (const size_t bad = lead_mismatches(round_.executors, lead_)) {
        diag::logf(kError, SEABRAID_INTERNAL, "placement lead disagrees with the table: %zu cells, epoch %llu", bad,
                   wide(epoch_));
        return Flow::kStop;
    }
    for (uint8_t rung = 1; rung < fallback(); ++rung) {
        if (rung_closed(rung)) continue;   // its cells already execute on the fallback
        if (!host.helper_cells(rung, held_)) return degrade(rung, "helper is gone");
        const HelperCheck check = check_helper(round_.executors, held_, rung);
        stats_.spares += static_cast<uint32_t>(check.spares);
        if (check.missing != 0) return degrade(rung, "helper does not hold a cell the table gives it");
    }
    return Flow::kGo;
}

PlacementDriver::Flow PlacementDriver::publish_round() noexcept {
    if (round_.snapshot.epoch != published_epoch_ && !publish_staged()) return Flow::kDone;
    return Flow::kGo;
}

void PlacementDriver::follow(PlacementHost& host) noexcept {
    // WHY after the publish: the RAM copy follows the committed pairs only once routing already reflects them.
    const auto start = Clock::now();
    for (const auto& outcome : round_.outcomes)
        if (outcome.kind == SEABRAID_PLACEMENT_COMMITTED) host.after_commit(outcome);
    stats_.follow_us = since_us(start);
    log_moves(round_.outcomes, slot_bytes_, fallback(), move_context());
}

bool PlacementDriver::publish_staged() noexcept {
    const auto start = Clock::now();
    const bool ok = view_.publish(round_.executors.data(), round_.executors.size(), round_.snapshot);
    stats_.publish_us += since_us(start);
    if (!ok) {
        // A stale view only makes cells local (slow, never wrong), so the driver stops rather than guess.
        die(SEABRAID_INTERNAL, "the view refused the table");
        return false;
    }
    published_ = round_.executors;
    published_epoch_ = round_.snapshot.epoch;
    return true;
}

int32_t PlacementDriver::snapshot_staging() noexcept {
    const int32_t status = take_snapshot(handle_, round_);
    if (status == SEABRAID_OK) {
        epoch_ = round_.snapshot.epoch;
        last_good_ = round_.snapshot;
    }
    return status;
}

PlacementDriver::Flow PlacementDriver::skip(Skip kind, int32_t status, const char* what) noexcept {
    // WHY one record per 16: a refusal that repeats every round would otherwise flood the log.
    if (skips_[kind]++ % 16 == 0)
        diag::logf(kWarn, status, "placement round skipped: %s (%s, %u in a row)", what, diag::status_name(status),
                   skips_[kind]);
    return Flow::kDone;
}

}  // namespace seabraid::strata_adapter
