// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
#include "qwen_decode.hpp"

#include "diag.hpp"

#include "seabraid/placement.h"

#include "strata/core/geometry.hpp"

#include <algorithm>
#include <utility>

namespace seabraid::strata_adapter {
namespace {

using diag::kError;
using diag::kInfo;
using diag::kWarn;

constexpr uint32_t kLayers = ::strata::core::kCompiledGeometry.n_layers;
// Room for the dispatch thread to wake after the callback's shorter deadline.
constexpr uint32_t kCollectGraceMs = 1000;

}  // namespace

std::unique_ptr<QwenDecodeTier> QwenDecodeTier::start(
    const PrivateQwenLaunch& launch, const PrivateQwenPlan& plan,
    uint32_t decode_timeout_ms, std::string& error) {
    auto runtime = QwenRuntime::start(launch, plan, decode_timeout_ms, error);
    if (!runtime) return nullptr;
    return std::unique_ptr<QwenDecodeTier>(new QwenDecodeTier(std::move(runtime), plan, decode_timeout_ms));
}

QwenDecodeTier::QwenDecodeTier(std::unique_ptr<QwenRuntime> runtime, const PrivateQwenPlan& plan,
                               uint32_t provider_timeout_ms)
    : runtime_(std::move(runtime)), config_(runtime_->config()),
      collect_timeout_ms_(provider_timeout_ms + kCollectGraceMs),
      held_(size_t{kLayers} * config_.experts), ids_(size_t{config_.max_tokens} * config_.top_k),
      weights_(ids_.size()), rows_(size_t{config_.max_tokens} * config_.hidden) {
    // Publish a read-only residency view for engine threads. Only the serial
    // owner touches the runtime, so it can tear down the worker after a failure.
    for (const auto& [layer, expert] : plan.ranked)
        held_[size_t(layer) * config_.experts + expert] = runtime_->holds(layer, expert);
}

bool QwenDecodeTier::held(int64_t layer, int32_t expert) const noexcept {
    return layer >= 0 && layer < kLayers && expert >= 0 && static_cast<uint32_t>(expert) < config_.experts &&
           held_[size_t(layer) * config_.experts + expert] != 0;
}

// WHY one predicate: a claim is (view AND physical AND alive), written once so no caller can
// test two of the three; a demoted cell (view names another rung) must stay local even
// though the helper still holds its bytes. WHY held() is a separate member: the placement
// driver verifies physical residency against it, and verifying against holds() would be
// tautological (holds() is false for every cell the table does not name).
// WHY no view means "route by residency": the legacy publisher never attaches one.
// No host test covers the engine's use of this gate; R1 in private_qwen_test pins it.
bool QwenDecodeTier::holds(int64_t layer, int32_t expert) const noexcept {
    return !retired() && (!view_ || view_->routes_to(layer, expert, view_rung_)) && held(layer, expert);
}

int32_t QwenDecodeTier::tier_swap(uint32_t layer, int32_t old_e, int32_t new_e, uint32_t timeout_ms,
                                  TierBlob blob) noexcept {
    // WHY refuse: a pending layer is a ticket the helper is serving, and the lease's one
    // wire cannot carry a tier op beside it. Nothing was touched, so REFUSED.
    if (ticket_ != 0) {
        diag::logf(kWarn, SEABRAID_BUSY_OR_STATE, "Seabraid Qwen tier swap at layer %u refused: a layer is pending", layer);
        return SEABRAID_PLACEMENT_IO_REFUSED;
    }
    // WHY FAILED: a disabled tier has no helper; the caller must quarantine the rung.
    if (!runtime_) {
        diag::logf(kError, SEABRAID_PROVIDER, "Seabraid Qwen tier swap at layer %u failed: tier disabled", layer);
        return SEABRAID_PLACEMENT_IO_FAILED;
    }
    const int32_t status = runtime_->tier_swap(layer, old_e, new_e, timeout_ms, blob);
    // WHY refresh the whole row on ACK: the lease adopted the helper's held[] for the layer,
    // and the row is the unit it adopts. WHY plain writes: the fence is held, so no engine
    // thread is reading holds() (the same assumption the constructor's fill makes).
    if (status == SEABRAID_PLACEMENT_IO_ACK && layer < kLayers)
        for (uint32_t e = 0; e < config_.experts; ++e) held_[size_t(layer) * config_.experts + e] = runtime_->holds(layer, e);
    return status;
}

bool QwenDecodeTier::owns(size_t pick) const noexcept {
    return pick < picks_ && ids_[pick] != SEABRAID_SKIP_EXPERT_ID;
}

bool QwenDecodeTier::retired() const noexcept {
    return retired_.load(std::memory_order_acquire);
}

void QwenDecodeTier::disable() noexcept {
    // WHY exchange: finish() may have retired the tier already and logged why.
    if (!retired_.exchange(true, std::memory_order_acq_rel))
        diag::logf(kWarn, SEABRAID_OK, "Seabraid Qwen tier disabled by its owner; later windows run locally");
    ticket_ = 0;
    picks_ = 0;
    runtime_.reset();
}

void QwenDecodeTier::note_local_fallback(bool missing_weights) noexcept {
    // WHY two flags: the first missing-weights call and the first call after
    // retirement are different transitions, each worth exactly one record.
    bool& logged = missing_weights ? logged_no_weights_ : logged_retired_local_;
    if (logged) return;
    logged = true;
    if (missing_weights)
        diag::logf(kInfo, SEABRAID_OK, "Seabraid Qwen tier: caller offers no router weights, work stays local");
    else
        diag::logf(kWarn, SEABRAID_PROVIDER, "Seabraid Qwen tier retired: work now runs locally");
}

bool QwenDecodeTier::shape_fits(int64_t layer, int64_t tokens, int64_t top_k) const noexcept {
    return layer >= 0 && layer < kLayers && tokens > 0 && tokens <= config_.max_tokens && top_k == config_.top_k;
}

bool QwenDecodeTier::mask_picks(uint32_t layer, size_t picks, const int32_t* ids, const float* weights,
                                const int32_t* kind, const int32_t* primary) {
    const auto* primary_layer = primary ? primary + size_t{layer} * config_.experts : nullptr;
    bool needed = false;
    for (size_t i = 0; i < picks; ++i) {
        const auto expert = ids[i];
        // Check the ID through holds() before indexing the engine's residency table.
        const bool assigned = (!kind || kind[i] == -1) && holds(layer, expert) &&
                              (!primary_layer || primary_layer[expert] < 0);
        ids_[i] = assigned ? expert : SEABRAID_SKIP_EXPERT_ID;
        weights_[i] = assigned ? weights[i] : 0.0f;
        needed |= assigned;
    }
    return needed;
}

// WHY local fallback, not an error: a stale epoch means this routing was decided against an older
// table than one the session already accepted. Nothing was queued (no ticket, no helper work), and
// running the layer locally is always correct, so the tier stays alive and the engine is not failed.
// WHY the tier's own mirror decides, not the status alone: BUSY_OR_STATE is also the session's busy, closing,
// not-prepared and pending-ticket refusal, so a nonzero epoch proves nothing; only an epoch below the highest
// one the session already accepted (QwenRuntime::accepted_epoch, kept by the session's only gateway) is provably
// the stale case.  Anything else falls through to submit_masked's hard failure.
bool QwenDecodeTier::is_stale_refusal(int32_t status, uint64_t epoch, uint64_t accepted) noexcept {
    return status == SEABRAID_BUSY_OR_STATE && epoch != 0 && epoch < accepted;
}

bool QwenDecodeTier::refused_as_stale(int32_t status, uint64_t epoch, int64_t layer) noexcept {
    if (!is_stale_refusal(status, epoch, accepted_epoch())) return false;
    if (!logged_stale_epoch_) {
        logged_stale_epoch_ = true;
        diag::logf(kWarn, status, "Seabraid Qwen layer %lld refused: stale placement epoch %llu; work stays local",
                   static_cast<long long>(layer), static_cast<unsigned long long>(epoch));
    }
    return true;
}

// WHY local fallback, not an error: the session refuses epoch 0 once it has accepted a nonzero one (INVALID,
// "a versioning engine must version every batch"). Here the unversioned batch is the HOST's doing: it detached
// the view or attached one that never published. That is a routing-source change, not a malformed batch; running
// the layer locally is always correct, so the tier stays alive instead of failing the engine's layer.
// WHY the tier checks before submitting rather than mapping INVALID afterwards: INVALID also means a NaN weight or
// a count fault, which must stay a hard failure, and accepted_epoch() is the session's own high mark (the
// runtime keeps it at the session's only gateway). Logged once, distinct from the stale-epoch record.
bool QwenDecodeTier::unversioned_after_versioned(uint64_t epoch, int64_t layer) noexcept {
    const uint64_t accepted = accepted_epoch();
    if (epoch != 0 || accepted == 0) return false;
    if (!logged_unversioned_) {
        logged_unversioned_ = true;
        diag::logf(kWarn, SEABRAID_BUSY_OR_STATE,
                   "Seabraid Qwen layer %lld unversioned after placement epoch %llu (no table published to the attached view); work stays local",
                   static_cast<long long>(layer), static_cast<unsigned long long>(accepted));
    }
    return true;
}

bool QwenDecodeTier::begin(int64_t layer, const float* input, const int32_t* ids, const float* router_weights,
                           int64_t tokens, int64_t top_k, const int32_t* kind, const int32_t* primary,
                           std::string& error) {
    if (ticket_ != 0) {
        error = "Seabraid Qwen already has a pending layer";
        diag::logf(kError, SEABRAID_BUSY_OR_STATE, "Seabraid Qwen begin refused: a layer is already pending");
        return false;
    }
    picks_ = 0;
    // Strata's per-pick callers don't offer router weights or a weighted merge.
    // They keep all work local, as do later windows once the helper is retired.
    if (!router_weights || retired()) {
        note_local_fallback(router_weights == nullptr);
        return true;
    }
    if (!input || !ids || !shape_fits(layer, tokens, top_k)) {
        error = "Seabraid Qwen received invalid layer or window geometry";
        diag::logf(kError, SEABRAID_INVALID, "Seabraid Qwen begin refused: invalid layer %lld, tokens %lld or top-k %lld",
                   static_cast<long long>(layer), static_cast<long long>(tokens), static_cast<long long>(top_k));
        return false;
    }
    const size_t picks = static_cast<size_t>(tokens * top_k);
    // WHY read before masking: the batch must carry the epoch its routing was decided against.
    const uint64_t epoch = placement_epoch();
    if (!mask_picks(static_cast<uint32_t>(layer), picks, ids, router_weights, kind, primary)) return true;
    return submit_masked(layer, input, static_cast<size_t>(tokens), picks, epoch, error);
}

// The masked ids_/weights_ go to the session. WHY a member of its own: begin() sits at the complexity limit, and
// the three outcomes here (stays local unversioned, stays local stale, hard failure) belong together.
bool QwenDecodeTier::submit_masked(int64_t layer, const float* input, size_t tokens, size_t picks, uint64_t epoch,
                                   std::string& error) {
    if (unversioned_after_versioned(epoch, layer)) return true;   // picks_ is still 0 from the top of begin()
    output_count_ = tokens * config_.hidden;
    const seabraid_batch_v1 batch{SEABRAID_ABI_VERSION, static_cast<uint32_t>(tokens), input, output_count_,
                              ids_.data(), picks, weights_.data(), picks};
    const int32_t status = runtime_->submit(static_cast<uint32_t>(layer), batch, SEABRAID_SOURCE_TEXT, epoch, ticket_);
    if (refused_as_stale(status, epoch, layer)) return true;   // picks_ is still 0 from the top of begin()
    if (status != SEABRAID_OK) {
        error = std::string("Seabraid Qwen could not submit layer: ") + diag::status_name(status);
        diag::logf(kError, status, "Seabraid Qwen could not submit layer %lld: %s", static_cast<long long>(layer),
                   diag::status_name(status));
        return false;
    }
    picks_ = picks;   // the runtime raised the session's accepted-epoch mark with this submit
    return true;
}

bool QwenDecodeTier::finish(const float*& weighted_rows, std::string& error) {
    weighted_rows = nullptr;
    if (ticket_ == 0) return true;
    const int32_t status = runtime_->collect(ticket_, collect_timeout_ms_, rows_.data(), output_count_);
    ticket_ = 0;
    if (status != SEABRAID_OK) {
        // On a collect timeout the runtime still holds the ticket internally.
        // Closing that session drains it. No new layer may reuse its channel.
        error = std::string("Seabraid Qwen could not finish layer: ") + diag::status_name(status);
        if (!retired_.exchange(true, std::memory_order_acq_rel))
            diag::logf(kError, status, "Seabraid Qwen tier retired: collect failed with %s; later windows run locally",
                       diag::status_name(status));
        return false;
    }
    weighted_rows = rows_.data();
    return true;
}

}  // namespace seabraid::strata_adapter
