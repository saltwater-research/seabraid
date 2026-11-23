// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "helper_exit.hpp"

#include "seabraid/seabraid.h"

#include <atomic>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace seabraid { class Channel; }

namespace seabraid::strata_adapter {

// The longest deadline any lease operation (bind, tier_swap) accepts: it bounds seabraid_close after a
// dead peer. The legacy decode path also uses 30 s.
inline constexpr uint32_t kMaxLeaseDeadlineMs = 30000;

// v8 kTierSwapBlob: the new expert's bytes, sent through the channel's blob slot 0 so the helper writes the engine's
// verified copy and reads it back, instead of reading its own pack.  bytes == nullptr = the v4 kTierSwap.
struct TierBlob {
    const uint8_t* bytes = nullptr;
    size_t size = 0;
};

// Whether `blob` may travel in a blob slot of `blob_max` bytes; `exact` is the layer's blob size, 0 when the host gave none.
// Exposed so the fail-closed rules (negative blob_max, exact size) are directly testable.
bool blob_fits(TierBlob blob, int64_t blob_max, size_t exact) noexcept;

// Fail closed. This lease is the only publisher for its connected
// channel. The caller retires that helper generation after a wire failure.
class StaticQwenProvider {
public:
    ~StaticQwenProvider();
    StaticQwenProvider(const StaticQwenProvider&) = delete;
    StaticQwenProvider& operator=(const StaticQwenProvider&) = delete;

    // Takes a connected, fully loaded channel for the same model as the host.
    // Before handing it over, the host must stop all other publishers and drain
    // decode, prefill, blob and tier operations; v7 has no producer arbitration.
    static std::unique_ptr<StaticQwenProvider> bind(
        std::unique_ptr<seabraid::Channel> channel, uint32_t layer,
        uint32_t timeout_ms, std::string& error);

    seabraid_config_v1 config() const noexcept;

    // Bind to one Seabraid session and keep this object alive until seabraid_close joins
    // its worker. The host may then tear down the helper and its channel.
    seabraid_provider_v1 provider() noexcept;

    // Use layer indices as operation IDs. Keep one Seabraid session for this lease.
    seabraid_provider_v2 multi_provider() noexcept;

    // The same lease as a v3 provider for seabraid_session_open (batch v2, versioned placement).
    // `config` must outlive only the open call (the library copies it). `compact_rows` adds
    // SEABRAID_PROVIDER_COMPACT_ROWS: the helper then receives only rows with a real pick, so
    // `tokens` varies per call. WHY that is safe here: run() takes any tokens in 1..window and the
    // helper reply is checked against the published row count, never a fixed one.
    seabraid_provider_v3 session_provider(const seabraid_config_v1& config, bool compact_rows) noexcept;

    // Read-only residency in the leased epoch. The engine still handles a
    // provider failure after routing if the worker exits or misses its deadline.
    bool holds(uint32_t layer, int32_t expert) const noexcept;
    bool retired() const noexcept;

    // One helper-side expert swap in `layer` through this lease, the only publisher on
    // the channel. Returns a SEABRAID_PLACEMENT_IO_* code:
    //   ACK      the helper applied it; holds() already reads the new residency and the
    //            lease adopted the epoch the round moved.
    //   REFUSED  nothing changed on the wire or in the epoch (bad request, a decode callback
    //            is running, or the helper rejected the swap before touching a slot).
    //   FAILED   the helper's state is unknown or torn; the lease is retired.
    //   TIMEOUT  no ack within timeout_ms (1..30000); the lease is retired.
    // The caller drains its own tickets first.  With `blob` the op is kTierSwapBlob (REFUSED when it does not fit the
    // channel's blob slot); the helper verifies its write by read-back before it acks.
    int32_t tier_swap(uint32_t layer, int32_t old_e, int32_t new_e, uint32_t timeout_ms, TierBlob blob = {}) noexcept;

    // Lend the helper's liveness descriptors (Linux) so a dead helper retires the
    // lease within milliseconds instead of at the decode deadline. Borrowed: the
    // caller keeps them open until this object is destroyed. Call before the first
    // request; the standalone fixtures that script the peer in-process never call it.
    void watch_helper(HelperWatch watch) noexcept { watch_ = watch; }

    // Per-layer blob size (Strata's expert_layout().blob_bytes(layer)); call before the first swap.  With it tier_swap accepts
    // only a blob of exactly that size; without it (empty) a blob is accepted when it merely fits the channel's blob slot.
    void set_blob_bytes(std::vector<size_t> per_layer) { blob_bytes_ = std::move(per_layer); }

private:
    StaticQwenProvider(std::unique_ptr<seabraid::Channel> channel, uint32_t layer,
                      uint32_t timeout_ms, uint64_t idle_sequence, uint64_t epoch);

    static int32_t execute(void* context, const seabraid_config_v1* config,
                           const seabraid_batch_v1* batch, float* output,
                           size_t output_count) noexcept;
    // v3 callback: F32 batch v2 -> the v1 view run() consumes; operation_id is the layer.
    static int32_t execute_session(void* context, const seabraid_config_v1* config,
                                   const seabraid_batch_v2* batch, float* output, size_t output_count) noexcept;
    static int32_t execute_operation(void* context, uint32_t layer,
                                     const seabraid_config_v1* config, const seabraid_batch_v1* batch,
                                     float* output, size_t output_count) noexcept;
    int32_t run(uint32_t layer, const seabraid_config_v1& config, const seabraid_batch_v1& batch,
                float* output, size_t output_count) noexcept;
    bool checked_routes(uint32_t layer, const seabraid_batch_v1& batch, bool& has_work) const noexcept;
    void publish(uint32_t layer, const seabraid_batch_v1& batch) noexcept;
    int32_t collect(uint32_t layer, uint32_t tokens, float* output, size_t output_count) noexcept;
    // nullptr on success, else a static reason the provider retires with.
    const char* accept_reply(uint32_t tokens) noexcept;
    bool replied() const noexcept;
    bool helper_gone(uint32_t& polls) noexcept;
    bool await_reply() noexcept;
    const char* preflight_fault() const noexcept;
    const char* generation_fault() const noexcept;
    // tier_swap steps (static_qwen_tier.cpp); tier_swap holds the callback flag around them.
    int32_t tier_locked(uint32_t layer, int32_t old_e, int32_t new_e, uint32_t timeout_ms, TierBlob blob) noexcept;
    bool tier_await(uint64_t seq, uint32_t timeout_ms) noexcept;
    int32_t tier_lost(uint32_t layer) noexcept;
    int32_t tier_fail(uint32_t layer, const char* reason) noexcept;
    int32_t retire(uint32_t layer, const char* reason) noexcept;

    std::unique_ptr<seabraid::Channel> channel_;
    const uint32_t layer_;
    const uint32_t timeout_ms_;
    uint64_t epoch_;       // moves only in tier_swap, after its own ACK
    uint64_t sequence_;
    std::atomic_flag in_callback_ = ATOMIC_FLAG_INIT;
    std::atomic<bool> retired_{false};
    HelperWatch watch_;    // borrowed; see watch_helper
    // The layer's exact blob size, 0 when unknown (set_blob_bytes not called, or layer past its list).
    size_t blob_bytes_for(uint32_t layer) const noexcept { return layer < blob_bytes_.size() ? blob_bytes_[layer] : 0; }
    std::vector<size_t> blob_bytes_;   // see set_blob_bytes; empty = size unknown
    bool gone_ = false;    // the watched helper exited (written only inside the serial callback)
    HelperExit exit_;
};

}  // namespace seabraid::strata_adapter
