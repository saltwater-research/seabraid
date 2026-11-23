// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// PlacementDriver loss handling: the per-tick liveness check and the three ways a rung or the driver goes down
// (degrade = one helper rung closed, die = the core can no longer be asked).
// OWNER: when a helper counts as lost, and what the published table becomes after a loss.  placement_driver.cpp owns the
//   round; this file owns everything that closes something.
// CONTRACT: check_liveness(host) runs on EVERY tick in kLive and kDegraded (F11, audit 2026-11-19); it asks
//   PlacementHost::helper_alive(rung) for each open helper rung (cheap: no table copy) and quarantines each newly dead
//   one.  kDegraded means "a rung is closed", nothing more: planning continues among the open rungs.
// SIZE BUDGET: <=120 lines.
// FAILURE CONTRACT: per-rung quarantine, one error record per newly closed rung; an ABI error on quarantine kills the
//   driver (kDead, one record); a lost lead is not handled here (verify/apply stop the engine).
// CHECKER: placement_liveness_test (three helpers, two successive losses) and placement_driver_test.
#include "placement_driver.hpp"

#include "diag.hpp"

namespace seabraid::strata_adapter {
namespace {

using diag::kError;

unsigned long long wide(uint64_t value) noexcept { return static_cast<unsigned long long>(value); }

}  // namespace

bool PlacementDriver::rung_closed(uint8_t rung) const noexcept { return ((closed_mask() >> rung) & 1u) != 0; }

// WHY every tick and not inside the round: before F11 the only liveness probe ran at the start of a planning round,
// and tick() returned at once unless kLive, so after the first quarantine no surviving helper was ever checked again.
// The core never plans or routes a closed rung, so stopping all planning protected nothing and blinded the monitor.
// WHY publishing here is safe outside a fenced round: a quarantine only sends cells to the fallback (slow, never wrong),
// and tick() runs on the engine thread at a window boundary, so no layer is in flight on this side.
void PlacementDriver::check_liveness(PlacementHost& host) noexcept {
    for (uint8_t rung = 1; rung < fallback() && state_ != State::kDead; ++rung) {
        if (rung_closed(rung) || host.helper_alive(rung)) continue;
        degrade(rung, "helper is gone");
    }
}

void PlacementDriver::note_degraded(const char* why) noexcept {
    state_ = State::kDegraded;
    stats_.worked = true;
    diag::logf(kError, SEABRAID_BUSY_OR_STATE, "placement degraded: %s (mask 0x%llx epoch %llu)", why,
               wide(closed_mask()), wide(epoch_));
}

PlacementDriver::Flow PlacementDriver::degrade(uint8_t rung, const char* why) noexcept {
    int32_t status = seabraid_placement_quarantine(handle_, epoch_, rung);
    if (status == SEABRAID_OK) status = snapshot_staging();
    if (status != SEABRAID_OK) return die(status, "quarantine");
    note_degraded(why);
    note_loss(rung, why);
    publish_staged();   // on failure it has already moved the driver to kDead
    return Flow::kDone;
}

PlacementDriver::Flow PlacementDriver::die(int32_t status, const char* what) noexcept {
    state_ = State::kDead;
    stats_.worked = true;
    diag::logf(kError, status, "placement driver dead: %s: %s", what, diag::status_name(status));
    // The core can no longer be asked, so the closed table is built here: helper cells to the fallback, the lead
    // unchanged (it routes from its own residency), and every helper rung marked closed for rung_open().
    round_.executors = published_;
    uint64_t closed = 0;
    for (uint8_t rung = 1; rung < fallback(); ++rung) closed |= uint64_t{1} << rung;
    for (auto& executor : round_.executors)
        if (executor != 0) executor = fallback();
    // A failed snapshot call zeroes the header, so the last good one supplies the shape.
    round_.snapshot = last_good_;
    round_.snapshot.epoch = published_epoch_ + 1;
    round_.snapshot.quarantined_rungs = closed;
    if (!view_.publish(round_.executors.data(), round_.executors.size(), round_.snapshot))
        diag::logf(kError, SEABRAID_INTERNAL, "placement: the view refused the closed table; it keeps the last one");
    return Flow::kDone;
}

}  // namespace seabraid::strata_adapter
