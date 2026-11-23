// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
#include "runtime_qwen.hpp"

#include "diag.hpp"
#include "diag_seabraid.hpp"
#include "placement_trace.hpp"

#include "seabraid/placement.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <exception>
#include <string>
#include <utility>

namespace seabraid::strata_adapter {
namespace {

// WHY a bounded close: a provider callback waits on the helper for at most the
// decode deadline, so a close that outlasts it plus this slack means the callback
// is stuck, and an unbounded close would hang engine shutdown behind it.
constexpr uint32_t kCloseSlackMs = 5000;

// Where an intentionally leaked worker is parked, so it stays reachable and out of LeakSanitizer reports.
PrivateQwenWorker* leaked_worker = nullptr;

// Closes the session within `close_ms`. WHY the worker is leaked on timeout: the
// ABI says the detached worker may still call the provider, whose context lives
// in `worker`, so freeing it would turn a slow shutdown into a use-after-free.
void close_bounded(seabraid_session* handle, uint32_t close_ms,
                   std::unique_ptr<PrivateQwenWorker>& worker) noexcept {
    if (handle == nullptr) return;
    const int32_t status = seabraid_close_timeout(handle, close_ms);
    if (status == SEABRAID_TIMEOUT) {
        diag::logf(diag::kError, status, "Seabraid Qwen session close timed out after %u ms; helper worker leaked", close_ms);
        leaked_worker = worker.release();
    } else if (status != SEABRAID_OK) {
        diag::logf(diag::kError, status, "Seabraid Qwen session close failed: %s", diag::status_name(status));
    }
}

uint32_t next_trace_rung() noexcept {
    static std::atomic<uint32_t> next{1};
    return next.fetch_add(1);
}

}  // namespace

// The trace's counters for one window of `trace::every()` collects. WHY the session's own counters for the row and
// failure totals: the ABI already counts them, so the adapter takes deltas instead of keeping a second tally.
struct QwenRuntime::TraceWindow {
    using Clock = std::chrono::steady_clock;
    Clock::time_point submitted = Clock::now(), began = Clock::now();
    trace::Sampler submit_to_collect, blocked;
    uint64_t layers = 0;
    seabraid_session_status_v1 last{};
};

std::unique_ptr<QwenRuntime> QwenRuntime::start(
    const PrivateQwenLaunch& launch, const PrivateQwenPlan& plan,
    uint32_t decode_timeout_ms, std::string& error) {
    // Route every adapter record into the runtime sink before the first one can fire.
    install_seabraid_log_sink();
    // WHY refuse here: a library built for another struct layout would misread every
    // batch this adapter submits, so the check runs before the helper is spawned.
    if (seabraid_abi_epoch() != SEABRAID_ABI_EPOCH) {
        error = "Seabraid library ABI epoch " + std::to_string(seabraid_abi_epoch()) +
                " does not match the adapter's " + std::to_string(SEABRAID_ABI_EPOCH);
        diag::logf(diag::kError, SEABRAID_INVALID, "Seabraid ABI epoch mismatch: library %u, adapter %u",
                   seabraid_abi_epoch(), SEABRAID_ABI_EPOCH);
        return nullptr;
    }
    // The multi-operation ABI selects the layer per submission. Layer zero
    // is only the unused legacy default for this controller.
    auto worker = PrivateQwenWorker::start(launch, plan, 0, decode_timeout_ms, error);
    if (!worker) return nullptr;

    const uint32_t close_ms = decode_timeout_ms + kCloseSlackMs;
    // WHY one open call: session_open copies the config and prepares in one step, so there is no
    // half-open session for a failed prepare to close.
    const auto geometry = worker->adapter().config();
    const auto provider = worker->adapter().session_provider(geometry, launch.compact_rows);
    seabraid_session* handle = nullptr;
    const int32_t opened = seabraid_session_open(&provider, &handle);
    if (opened != SEABRAID_OK) {  // *handle is NULL on every failed open; nothing to close
        error = std::string("Cannot open Seabraid Qwen session: ") + diag::status_name(opened);
        diag::logf(diag::kError, opened, "Seabraid Qwen session open failed: %s", diag::status_name(opened));
        return nullptr;
    }
    auto runtime = std::unique_ptr<QwenRuntime>(new QwenRuntime(std::move(worker), handle, close_ms));
    if (trace::on()) trace::helper(runtime->trace_rung_, "up", "session opened", 0);
    return runtime;
}

QwenRuntime::QwenRuntime(std::unique_ptr<PrivateQwenWorker> worker, seabraid_session* session,
                         uint32_t close_ms) noexcept
    : worker_(std::move(worker)), session_(session), close_ms_(close_ms), trace_rung_(next_trace_rung()) {}

// The lease closes first so no queued callback can touch the mapping after the helper stops.
QwenRuntime::~QwenRuntime() {
    if (trace::on()) {
        trace_flush();   // the partial last window is data too
        trace::helper(trace_rung_, "closed", "runtime closed", 0);
    }
    close_bounded(session_, close_ms_, worker_);
}

void QwenRuntime::trace_submit() noexcept {
    try {
        if (!trace_) trace_ = std::make_unique<TraceWindow>();
        trace_->submitted = TraceWindow::Clock::now();
    } catch (const std::exception&) {   // allow-error-path: no window for this layer; serving continues
        return;
    }
}

void QwenRuntime::trace_collect(std::chrono::steady_clock::time_point began, int32_t status) noexcept {
    if (!trace_) return;
    const auto done = TraceWindow::Clock::now();
    const auto us = [](TraceWindow::Clock::duration d) {
        return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(d).count());
    };
    if (status != SEABRAID_OK) {
        // A failed collect retires the tier: write what the window holds and say why the helper stopped serving.
        trace::helper(trace_rung_, "down", diag::status_name(status), 0);
        trace_flush();
        return;
    }
    ++trace_->layers;
    trace_->submit_to_collect.add(us(done - trace_->submitted));
    trace_->blocked.add(us(done - began));
    if (trace_->layers >= trace::every()) trace_flush();
}

void QwenRuntime::trace_flush() noexcept {
    if (!trace_ || trace_->layers == 0) return;
    trace::WindowFields fields;
    fields.rung = trace_rung_;
    fields.layers = trace_->layers;
    fields.window_us = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
        TraceWindow::Clock::now() - trace_->began).count());
    fields.submit_to_collect = trace_->submit_to_collect.stats();
    fields.blocked = trace_->blocked.stats();
    seabraid_session_status_v1 now{};
    now.abi_version = SEABRAID_ABI_VERSION;
    now.struct_size = sizeof now;
    if (session_ != nullptr && seabraid_session_status(session_, &now) == SEABRAID_OK) {
        const auto& before = trace_->last;
        fields.rows_submitted = now.rows_submitted - before.rows_submitted;
        fields.rows_shipped = now.rows_shipped - before.rows_shipped;
        fields.failures = now.provider_failures - before.provider_failures;
        fields.timeouts = now.wait_timeouts - before.wait_timeouts;
        trace_->last = now;
    }
    trace::window(fields);
    trace_->submit_to_collect.clear();
    trace_->blocked.clear();
    trace_->layers = 0;
    trace_->began = TraceWindow::Clock::now();
}

seabraid_config_v1 QwenRuntime::config() const noexcept {
    return worker_->adapter().config();
}

bool QwenRuntime::holds(uint32_t layer, int32_t expert) const noexcept {
    return worker_->adapter().holds(layer, expert);
}

bool QwenRuntime::retired() const noexcept {
    return worker_->adapter().retired();
}

int32_t QwenRuntime::submit(uint32_t layer, const seabraid_batch_v1& batch, uint32_t source_modalities,
                           uint64_t placement_epoch, uint64_t& ticket) noexcept {
    // WHY batch v2 here: it is the only submit that carries the epoch, so the ABI itself refuses
    // routing decided against a table older than one it already accepted (0 = unversioned).
    seabraid_batch_v2 v2{};
    v2.abi_version = SEABRAID_BATCH_V2_VERSION;
    v2.struct_size = sizeof v2;
    v2.operation_id = layer;
    v2.source_modalities = source_modalities;
    v2.placement_epoch = placement_epoch;
    v2.tokens = batch.tokens;
    v2.activation_format = SEABRAID_FORMAT_ROUTED_F32;
    v2.activations = batch.activations;
    v2.activation_count = batch.activation_count;
    v2.expert_ids = batch.expert_ids;
    v2.expert_id_count = batch.expert_id_count;
    v2.weights = batch.weights;
    v2.weight_count = batch.weight_count;
    if (trace::on()) trace_submit();   // one branch when the trace is off
    const int32_t status = seabraid_session_submit(session_, &v2, &ticket);
    if (status == SEABRAID_OK) {
        pending_ = ticket;
        accepted_epoch_ = std::max(accepted_epoch_, placement_epoch);   // only an accepted submit raises the mark
    }
    return status;
}

int32_t QwenRuntime::collect(uint64_t ticket, uint32_t timeout_ms, float* output,
                            size_t output_count) noexcept {
    const bool tracing = trace::on();   // one branch when the trace is off
    const auto began = tracing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const int32_t status = seabraid_collect(session_, ticket, timeout_ms, output, output_count);
    if (tracing) trace_collect(began, status);
    // WHY clear only on OK: a failed collect leaves the ticket with the session, so the wire is
    // not known idle and a tier op must keep refusing until the session is closed.
    if (status == SEABRAID_OK && ticket == pending_) pending_ = 0;
    return status;
}

int32_t QwenRuntime::tier_swap(uint32_t layer, int32_t old_e, int32_t new_e, uint32_t timeout_ms,
                                TierBlob blob) noexcept {
    // WHY refuse here as well as in the tier: the runtime owns the session, so it must not
    // trust a caller to have drained it; nothing was touched, so REFUSED.
    if (pending_ != 0) {
        diag::logf(diag::kWarn, SEABRAID_BUSY_OR_STATE, "Seabraid Qwen tier swap at layer %u refused: a session ticket is outstanding", layer);
        return SEABRAID_PLACEMENT_IO_REFUSED;
    }
    return worker_->adapter().tier_swap(layer, old_e, new_e, timeout_ms, blob);
}

}  // namespace seabraid::strata_adapter
