// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "executor_view.hpp"
#include "runtime_qwen.hpp"

#include <atomic>
#include <vector>

namespace seabraid::strata_adapter {

// The serial verifier keeps ownership of the source rows and local merge.
// A successful begin exposes only picks assigned to this private helper.
class QwenDecodeTier {
public:
    static std::unique_ptr<QwenDecodeTier> start(
        const PrivateQwenLaunch& launch, const PrivateQwenPlan& plan,
        uint32_t decode_timeout_ms, std::string& error);

    // Claim test: published table names this rung, the helper holds the cell, the tier is alive.
    bool holds(int64_t layer, int32_t expert) const noexcept;
    // Per-layer blob size for the exact-size check of every blob swap (see QwenRuntime::set_blob_bytes).
    void set_blob_bytes(std::vector<size_t> per_layer) { runtime_->set_blob_bytes(std::move(per_layer)); }
    // Physical residency only (no view, no liveness): what the placement driver verifies against.
    bool held(int64_t layer, int32_t expert) const noexcept;
    // Opt in to placement routing: before this, or with nullptr, the tier answers from residency alone.
    // Call once before engine threads read holds(); the pointer must outlive the tier.
    void attach_view(const ExecutorView* view, uint8_t rung) noexcept { view_ = view; view_rung_ = rung; }
    // One helper-side swap (SEABRAID_PLACEMENT_IO_* code). Engine thread, fence held, no layer pending.
    int32_t tier_swap(uint32_t layer, int32_t old_e, int32_t new_e, uint32_t timeout_ms, TierBlob blob = {}) noexcept;
    bool owns(size_t pick) const noexcept;
    bool retired() const noexcept;
    void disable() noexcept;
    bool begin(int64_t layer, const float* input, const int32_t* ids, const float* router_weights,
               int64_t tokens, int64_t top_k, const int32_t* kind, const int32_t* primary,
               std::string& error);
    bool finish(const float*& weighted_rows, std::string& error);

    // True only when `status` is the session's stale-epoch refusal as this tier can prove it: BUSY_OR_STATE for a
    // nonzero `epoch` below `accepted`, the highest epoch the tier's session took. Public so a test can feed it
    // the session's real refusals; every other BUSY_OR_STATE or INVALID is an ordinary submit failure.
    static bool is_stale_refusal(int32_t status, uint64_t epoch, uint64_t accepted) noexcept;

private:
    QwenDecodeTier(std::unique_ptr<QwenRuntime> runtime, const PrivateQwenPlan& plan,
                   uint32_t provider_timeout_ms);
    // 0 (unversioned) until a placement table is published to the attached view.
    uint64_t placement_epoch() const noexcept { return view_ ? view_->epoch() : 0; }
    bool refused_as_stale(int32_t status, uint64_t epoch, int64_t layer) noexcept;
    bool unversioned_after_versioned(uint64_t epoch, int64_t layer) noexcept;
    bool submit_masked(int64_t layer, const float* input, size_t tokens, size_t picks, uint64_t epoch,
                       std::string& error);
    bool shape_fits(int64_t layer, int64_t tokens, int64_t top_k) const noexcept;
    void note_local_fallback(bool missing_weights) noexcept;
    bool mask_picks(uint32_t layer, size_t picks, const int32_t* ids, const float* weights,
                    const int32_t* kind, const int32_t* primary);

    std::unique_ptr<QwenRuntime> runtime_;
    const seabraid_config_v1 config_;
    const uint32_t collect_timeout_ms_;
    std::vector<uint8_t> held_;
    const ExecutorView* view_ = nullptr;
    uint8_t view_rung_ = 0;
    std::vector<int32_t> ids_;
    std::vector<float> weights_;
    std::vector<float> rows_;
    size_t picks_ = 0;
    size_t output_count_ = 0;
    uint64_t ticket_ = 0;
    // The session's highest accepted epoch, read from its one gateway (no mirror here: a mirror held only while this
    // tier was the session's sole submitter, an assumption nothing checked).  0 once the tier is disabled.
    uint64_t accepted_epoch() const noexcept { return runtime_ ? runtime_->accepted_epoch() : 0; }
    // Serial-owner state: each local-execution fallback is logged once per
    // transition, not once per layer call.
    bool logged_no_weights_ = false;
    bool logged_retired_local_ = false;
    bool logged_stale_epoch_ = false;
    bool logged_unversioned_ = false;
    std::atomic<bool> retired_{false};
};

}  // namespace seabraid::strata_adapter
