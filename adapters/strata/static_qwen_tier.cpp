// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// StaticQwenProvider::tier_swap: helper-side expert maintenance through the one lease.
// OWNER: this TU owns the lease's tier round. CONTRACT: the round, the fence and the
//   checks mirror RemoteExperts::seabraid_retier (Strata patches 0001, 0005, 0012); the
//   helper serves it in strata-ext helper.cpp. FAILURE: REFUSED changes nothing; FAILED
//   and TIMEOUT retire the lease (an uncertain helper op quarantines). Every peer-written
//   field is validated before it is believed. Each failure branch logs exactly once.
#include "static_qwen.hpp"

#include "diag.hpp"

#include "seabraid/placement.h"

#include "strata/seabraid/channel.hpp"
#include "strata/core/geometry.hpp"

#include <cstring>
#include <string>

namespace seabraid::strata_adapter {
namespace {

constexpr auto& kGeometry = ::strata::core::kCompiledGeometry;

int32_t refuse(uint32_t layer, int32_t old_e, int32_t new_e, int32_t status, const char* why) noexcept {
    diag::logf(diag::kWarn, status, "static Qwen layer %u tier swap %d -> %d refused: %s", layer, old_e, new_e, why);
    return SEABRAID_PLACEMENT_IO_REFUSED;
}

const char* args_fault(uint32_t layer, int32_t old_e, int32_t new_e, uint32_t timeout_ms) noexcept {
    const auto bad = [](int32_t e) { return e < 0 || e >= kGeometry.n_expert; };
    if (layer >= kGeometry.n_layers) return "layer out of range";
    if (bad(old_e) || bad(new_e)) return "expert id out of range";
    // WHY: the wire has no in-place form, so a torn slot could not be told from a good one.
    if (old_e == new_e) return "old and new expert are the same";
    if (timeout_ms == 0 || timeout_ms > kMaxLeaseDeadlineMs) return "deadline must be 1..30000 ms";
    return nullptr;
}

const char* residency_fault(const uint8_t* held, uint32_t layer, int32_t old_e, int32_t new_e) noexcept {
    const auto* row = held + size_t{layer} * kGeometry.n_expert;
    if (row[old_e] != 1) return "displaced expert is not resident";
    if (row[new_e] == 1) return "incoming expert is already resident";
    return nullptr;
}

bool unloaded(const seabraid::Channel& channel) noexcept {
    return channel.header()->state.load(std::memory_order_acquire) != seabraid::kLoaded;
}

struct Reply {
    const char* fault = nullptr;   // static text, else the helper's result is well formed
    uint8_t res = seabraid::kTierPending;
};

// Copies each peer-written field once, then judges the copy. Call only after tier_done.
Reply read_reply(const seabraid::Channel& channel, uint64_t seq) noexcept {
    const auto* header = channel.header();
    // WHY exact match: tier_done accepts any resp >= seq; a peer that jumps ahead is lying.
    if (__atomic_load_n(&header->tier_resp, __ATOMIC_ACQUIRE) != seq) return {"tier reply sequence mismatch"};
    const int32_t count = header->tier_n;
    const int32_t ok = header->tier_ok;
    const uint8_t res = header->tier_res[0];
    if (count != 1) return {"malformed tier reply: op count"};
    const bool known = res == seabraid::kTierApplied || res == seabraid::kTierFailed || res == seabraid::kTierRejected;
    if (!known) return {"malformed tier reply: result code"};
    // WHY: an applied op under a failed round means a sync failed after the copy.
    if ((ok != 0) != (res == seabraid::kTierApplied)) return {"malformed tier reply: ok flag disagrees"};
    return {nullptr, res};
}

// The helper's held[] must show the swap it reported applied. false also when the
// diagnostic string cannot be built: an unjudgeable round is a failed round.
bool residency_consistent(const uint8_t* held, const seabraid::TierWireOp& op) noexcept {
    const uint8_t res = seabraid::kTierApplied;
    try {
        std::string why;
        return seabraid::tier_residency_consistent(held, kGeometry.n_layers, kGeometry.n_expert, &op, &res, 1, why);
    } catch (...) {  // allow-error-path: the caller retires the lease and logs once
        return false;
    }
}

}  // namespace

// WHY exact size when the engine told us the per-layer blob size (set_blob_bytes): a short blob would leave the tail of the
// helper's slot stale and a long one overruns it; "fits the slot" is the weaker test and is the fallback only while the
// host has not supplied the sizes.  A negative blob_max is rejected before any cast to unsigned (it would wrap to huge).
bool blob_fits(TierBlob blob, int64_t blob_max, size_t exact) noexcept {
    if (blob.bytes == nullptr) return true;
    if (blob_max < 0 || blob.size == 0 || blob.size > static_cast<uint64_t>(blob_max)) return false;
    return exact == 0 || blob.size == exact;
}

namespace {

// The one op of the round.  WHY blob slot 0 under the lease's fence: the channel's rule puts main-supplied bytes in the
// blob slots, and no cache read can be in flight while this lease holds the wire (preflight_fault checks maintenance).
seabraid::TierWireOp stage_op(seabraid::Channel& channel, uint32_t layer, int32_t old_e, int32_t new_e, TierBlob blob) noexcept {
    seabraid::TierWireOp op{seabraid::kTierSwap, 0, 0, static_cast<int32_t>(layer), old_e, static_cast<int32_t>(layer), new_e};
    if (blob.bytes == nullptr) return op;
    std::memcpy(channel.blob(0), blob.bytes, blob.size);
    op.kind = seabraid::kTierSwapBlob;
    return op;
}

}  // namespace

int32_t StaticQwenProvider::tier_swap(uint32_t layer, int32_t old_e, int32_t new_e,
                                      uint32_t timeout_ms, TierBlob blob) noexcept {
    // No log: retire() logged the transition; later calls are refused by design.
    if (retired()) return SEABRAID_PLACEMENT_IO_FAILED;
    if (const char* why = args_fault(layer, old_e, new_e, timeout_ms))
        return refuse(layer, old_e, new_e, SEABRAID_INVALID, why);
    // WHY the same flag as decode: it is the fence. Every layer request is published
    // under it, so while we hold it none can slip between the drained response and our
    // tier_submit, which the channel leaves entirely to the caller.
    if (in_callback_.test_and_set(std::memory_order_acquire))
        return refuse(layer, old_e, new_e, SEABRAID_BUSY_OR_STATE, "a decode callback holds the lease");
    struct Release {
        std::atomic_flag& busy;
        ~Release() { busy.clear(std::memory_order_release); }
    } release{in_callback_};
    return tier_locked(layer, old_e, new_e, timeout_ms, blob);
}

int32_t StaticQwenProvider::tier_locked(uint32_t layer, int32_t old_e, int32_t new_e,
                                        uint32_t timeout_ms, TierBlob blob) noexcept {
    if (const char* why = residency_fault(channel_->held(), layer, old_e, new_e))
        return refuse(layer, old_e, new_e, SEABRAID_BUSY_OR_STATE, why);
    if (!blob_fits(blob, channel_->limits().blob_max, blob_bytes_for(layer)))
        return refuse(layer, old_e, new_e, SEABRAID_INVALID, "the blob does not fit the channel's blob slot");
    if (const char* fault = preflight_fault()) return tier_fail(layer, fault);  // as a decode does
    const seabraid::TierWireOp op = stage_op(*channel_, layer, old_e, new_e, blob);
    const uint64_t seq = channel_->tier_submit(&op, 1);
    if (!tier_await(seq, timeout_ms)) return tier_lost(layer);
    const Reply reply = read_reply(*channel_, seq);
    if (reply.fault) return tier_fail(layer, reply.fault);
    if (const char* fault = generation_fault()) return tier_fail(layer, fault);
    // REJECTED: refused before the slot was touched, so held[] and the epoch stay.
    if (reply.res == seabraid::kTierRejected)
        return refuse(layer, old_e, new_e, SEABRAID_PROVIDER, "the helper rejected the swap");
    if (reply.res == seabraid::kTierFailed) return tier_fail(layer, "helper failed the swap after its copy started");
    if (!residency_consistent(channel_->held(), op)) return tier_fail(layer, "helper residency is stale after an applied swap");
    // WHY here and only here: the epoch advertises a placement this lease has already
    // adopted. holds() reads held[] live, so adoption is the consistency check above;
    // generation_fault just proved nobody else moved the epoch, so the re-capture is ours.
    channel_->bump_epoch();
    epoch_ = channel_->epoch();
    return SEABRAID_PLACEMENT_IO_ACK;
}

// True once the helper acked; false on a dead helper, an unloaded channel or the deadline.
bool StaticQwenProvider::tier_await(uint64_t seq, uint32_t timeout_ms) noexcept {
    uint32_t polls = 0;
    const auto answered = [&] { return channel_->tier_done(seq) || unloaded(*channel_) || helper_gone(polls); };
    return seabraid::Channel::wait(answered, timeout_ms / 1000.0) && channel_->tier_done(seq);
}

int32_t StaticQwenProvider::tier_lost(uint32_t layer) noexcept {
    if (gone_) return tier_fail(layer, "helper exited");
    if (unloaded(*channel_)) return tier_fail(layer, "helper is no longer loaded");
    retire(layer, "tier round timed out");
    return SEABRAID_PLACEMENT_IO_TIMEOUT;
}

int32_t StaticQwenProvider::tier_fail(uint32_t layer, const char* reason) noexcept {
    retire(layer, reason);
    return SEABRAID_PLACEMENT_IO_FAILED;
}

}  // namespace seabraid::strata_adapter
