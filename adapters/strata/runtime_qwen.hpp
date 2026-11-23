// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "private_qwen.hpp"

#include <chrono>
#include <memory>
#include <type_traits>

namespace seabraid::strata_adapter {

// Owns the Rust lease and its helper as a unit. The lease closes first so
// no queued callback can access the mapping after the helper is stopped.
class QwenRuntime {
public:
    ~QwenRuntime();
    QwenRuntime(const QwenRuntime&) = delete;
    QwenRuntime& operator=(const QwenRuntime&) = delete;

    static std::unique_ptr<QwenRuntime> start(
        const PrivateQwenLaunch& launch, const PrivateQwenPlan& plan,
        uint32_t decode_timeout_ms, std::string& error);

    seabraid_config_v1 config() const noexcept;
    bool holds(uint32_t layer, int32_t expert) const noexcept;
    bool retired() const noexcept;

    // placement_epoch is ExecutorView::epoch() when the caller routed by a published table, else 0.
    // A stale (lower than an accepted) epoch returns SEABRAID_BUSY_OR_STATE and leaves `ticket` untouched.
    int32_t submit(uint32_t layer, const seabraid_batch_v1& batch, uint32_t source_modalities,
                   uint64_t placement_epoch, uint64_t& ticket) noexcept;
    // The highest epoch the session accepted (0 = none yet): the mark below which it refuses a batch as stale and
    // after which it refuses epoch 0.  OWNER (L14, R8 review): this object, because session_ is created in start()
    // and never leaves it, so every submit the session ever sees passes through submit() above.  The decode tier
    // reads this instead of keeping a mirror that was right only while the tier was the session's sole submitter.
    uint64_t accepted_epoch() const noexcept { return accepted_epoch_; }
    int32_t collect(uint64_t ticket, uint32_t timeout_ms, float* output,
                    size_t output_count) noexcept;

    // One helper-side swap through the lease; SEABRAID_PLACEMENT_IO_*. REFUSED while a
    // session ticket is outstanding. Serial owner only, like submit and collect.
    int32_t tier_swap(uint32_t layer, int32_t old_e, int32_t new_e, uint32_t timeout_ms, TierBlob blob = {}) noexcept;

    // Per-layer expert blob size (the engine's expert_layout().blob_bytes): with it a helper swap accepts only a blob of exactly
    // that size (StaticQwenProvider::set_blob_bytes).  Call before the first swap; the engine does, from open_placement.
    void set_blob_bytes(std::vector<size_t> per_layer) { worker_->adapter().set_blob_bytes(std::move(per_layer)); }

    // The rung number this helper's trace "window" records carry (SEABRAID_TRACE). Defaults to the order the runtimes
    // started in (1, 2, ...), which is the rung order when the host starts them in rung order; a host that knows better
    // sets it. No effect while the trace is off.
    void set_trace_rung(uint32_t rung) noexcept { trace_rung_ = rung; }

private:
    struct TraceWindow;   // the trace's per-window counters, allocated on first use only while the trace is on
    void trace_submit() noexcept;
    void trace_collect(std::chrono::steady_clock::time_point began, int32_t status) noexcept;
    void trace_flush() noexcept;

    QwenRuntime(std::unique_ptr<PrivateQwenWorker> worker, seabraid_session* session,
                uint32_t close_ms) noexcept;

    // Declaration order is not teardown order: the destructor closes session_ first
    // (bounded by close_ms_) and may deliberately leak worker_ if that close times out.
    std::unique_ptr<PrivateQwenWorker> worker_;
    seabraid_session* session_;
    uint32_t close_ms_;
    uint64_t pending_ = 0;   // the submitted ticket not yet collected; serial owner only
    uint64_t accepted_epoch_ = 0;   // raised only by an accepted submit; serial owner only
    uint32_t trace_rung_ = 0;
    std::unique_ptr<TraceWindow> trace_;   // serial owner only, like submit and collect
};

// The sole-gateway claim behind accepted_epoch(): the session handle cannot be duplicated with its owner.
static_assert(!std::is_copy_constructible_v<QwenRuntime> && !std::is_move_constructible_v<QwenRuntime> &&
                  !std::is_copy_assignable_v<QwenRuntime> && !std::is_move_assignable_v<QwenRuntime>,
              "QwenRuntime is the only holder of its session; copying or moving it would add a second submitter");

}  // namespace seabraid::strata_adapter
