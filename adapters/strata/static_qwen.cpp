// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
#include "static_qwen.hpp"

#include "diag.hpp"
#include "helper_exit.hpp"

#include "strata/seabraid/channel.hpp"
#include "strata/seabraid/finite_rows.hpp"
#include "strata/core/geometry.hpp"
#include "strata/kernels/window_limits.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <utility>

namespace seabraid::strata_adapter {
namespace {

// v8 requalified: v8 adds kTierSwapBlob and renames pad_ to blob_slot at the same offset; no layer-request byte moved.
// v9 requalified: v9 appends Header file fields and adds kTierSwapFile (a FileRef in a blob slot); the adapter sends neither and no layer-request byte moved.
// v11 (only the magic tag changed) and v10 requalified: v10 appends Header::helper_start (the helper's identity); the provider neither reads nor writes it and no earlier offset moved.
static_assert(seabraid::kVersion == 11, "Requalify the adapter for a new channel ABI");
constexpr auto& kGeometry = ::strata::core::kCompiledGeometry;
constexpr auto kWindow = ::strata::kernels::kWindowMaxT;
constexpr auto kPicks = kGeometry.n_expert_used;
// The wait predicate runs at spin speed; probing the pidfd costs a syscall, so do
// it once per this many predicate calls. In the 20 us nap phase that is ~5 ms
// nominal; nanosleep's timer slack stretches it to roughly 15-20 ms.
constexpr uint32_t kLivenessStride = 256;

bool loaded(const seabraid::Channel& channel) noexcept {
    return channel.header()->state.load(std::memory_order_acquire) == seabraid::kLoaded;
}

bool maintenance_idle(const seabraid::Channel& channel) noexcept {
    const auto* header = channel.header();
    return channel.tier_pending() == 0 && channel.pf_pending() == 0 &&
           header->blob_req_seq.load(std::memory_order_acquire) ==
               header->blob_resp_seq.load(std::memory_order_acquire);
}

bool fits_qwen(const seabraid::Channel& channel, uint32_t layer) noexcept {
    const auto& limits = channel.limits();
    const auto* header = channel.header();
    // The helper's compiled geometry owns tensor dimensions; there is no
    // negotiated decode width in v7. Its decode capacity is window * top-k.
    return header->engine == seabraid::kEngineQwen && header->window_max_t == kWindow &&
           limits.hidden == kGeometry.n_embd && limits.experts == kGeometry.n_expert &&
           limits.layers == kGeometry.n_layers && limits.cap == kWindow * kPicks &&
           limits.valid_decode(layer, kWindow, kPicks, kWindow);
}

bool matches(seabraid_config_v1 cfg, seabraid_config_v1 expected) noexcept {
    return cfg.abi_version == expected.abi_version && cfg.hidden == expected.hidden &&
           cfg.experts == expected.experts && cfg.max_tokens == expected.max_tokens &&
           cfg.top_k == expected.top_k;
}

bool extents_match(const seabraid_batch_v1& batch, size_t output_count) noexcept {
    if (batch.tokens == 0 || batch.tokens > kWindow) return false;
    const auto rows = size_t{batch.tokens} * kGeometry.n_embd;
    const auto picks = size_t{batch.tokens} * kPicks;
    return batch.activations && batch.expert_ids && batch.weights &&
           rows == batch.activation_count && rows == output_count &&
           picks == batch.expert_id_count && picks == batch.weight_count;
}

// WHY one helper: a bind refusal sets the caller's error string and writes its
// single log record together, so neither can be forgotten on a new branch.
std::nullptr_t refuse_bind(std::string& error, const char* message) {
    error = message;
    diag::logf(diag::kError, SEABRAID_INVALID, "static Qwen bind refused: %s", message);
    return nullptr;
}

}  // namespace

std::unique_ptr<StaticQwenProvider> StaticQwenProvider::bind(
    std::unique_ptr<seabraid::Channel> channel, uint32_t layer,
    uint32_t timeout_ms, std::string& error) {
    error.clear();
    if (!channel || !channel->valid() || !loaded(*channel))
        return refuse_bind(error, "Strata provider requires a connected, loaded channel");
    if (!fits_qwen(*channel, layer))
        return refuse_bind(error, "Strata provider requires matching Qwen v7 geometry and a valid layer");
    if (timeout_ms == 0 || timeout_ms > kMaxLeaseDeadlineMs)
        return refuse_bind(error, "Strata provider deadline must be 1..30000 ms");
    const auto* header = channel->header();
    const auto sequence = header->req_seq.load(std::memory_order_acquire);
    if (sequence != header->resp_seq.load(std::memory_order_acquire) || !maintenance_idle(*channel))
        return refuse_bind(error, "Strata provider cannot take over with requests or maintenance in flight");
    const auto epoch = channel->epoch();
    return std::unique_ptr<StaticQwenProvider>(
        new StaticQwenProvider(std::move(channel), layer, timeout_ms, sequence, epoch));
}

StaticQwenProvider::StaticQwenProvider(std::unique_ptr<seabraid::Channel> channel,
                                       uint32_t layer, uint32_t timeout_ms,
                                       uint64_t idle_sequence, uint64_t epoch)
    : channel_(std::move(channel)), layer_(layer), timeout_ms_(timeout_ms), epoch_(epoch),
      sequence_(idle_sequence) {}

StaticQwenProvider::~StaticQwenProvider() = default;

seabraid_config_v1 StaticQwenProvider::config() const noexcept {
    return {SEABRAID_ABI_VERSION, kGeometry.n_embd, kGeometry.n_expert, kWindow, kPicks};
}

seabraid_provider_v1 StaticQwenProvider::provider() noexcept {
    return {SEABRAID_ABI_VERSION, this, execute};
}

seabraid_provider_v2 StaticQwenProvider::multi_provider() noexcept {
    return {SEABRAID_MULTI_PROVIDER_VERSION, kGeometry.n_layers,
            SEABRAID_PROVIDER_PARTIAL_ROUTES, this, execute_operation};
}

seabraid_provider_v3 StaticQwenProvider::session_provider(const seabraid_config_v1& config,
                                                          bool compact_rows) noexcept {
    seabraid_provider_v3 provider{};
    provider.abi_version = SEABRAID_PROVIDER_V3_VERSION;
    provider.struct_size = sizeof provider;
    provider.operation_count = kGeometry.n_layers;
    provider.flags = SEABRAID_PROVIDER_PARTIAL_ROUTES | (compact_rows ? SEABRAID_PROVIDER_COMPACT_ROWS : 0u);
    provider.config = &config;
    provider.context = this;
    provider.execute = execute_session;
    return provider;
}

int32_t StaticQwenProvider::execute_session(void* context, const seabraid_config_v1* config,
                                            const seabraid_batch_v2* batch, float* output,
                                            size_t output_count) noexcept {
    // WHY refuse non-F32: this provider never declares ACCEPTS_F16/BF16, so the library cannot
    // hand it one; the check keeps a direct caller from reading half-floats as floats.
    if (!batch || batch->activation_format != SEABRAID_FORMAT_ROUTED_F32) return SEABRAID_INVALID;
    const seabraid_batch_v1 view{SEABRAID_ABI_VERSION, batch->tokens,
                                 static_cast<const float*>(batch->activations), batch->activation_count,
                                 batch->expert_ids, batch->expert_id_count, batch->weights, batch->weight_count};
    return execute_operation(context, batch->operation_id, config, &view, output, output_count);
}

bool StaticQwenProvider::holds(uint32_t layer, int32_t expert) const noexcept {
    if (layer >= kGeometry.n_layers || expert < 0 || expert >= kGeometry.n_expert || retired()) {
        return false;
    }
    const auto* held = channel_->held() + size_t{layer} * kGeometry.n_expert;
    return held[expert] == 1;
}

bool StaticQwenProvider::retired() const noexcept {
    return retired_.load(std::memory_order_acquire);
}

int32_t StaticQwenProvider::execute(void* context, const seabraid_config_v1* config,
                                    const seabraid_batch_v1* batch, float* output,
                                    size_t output_count) noexcept {
    if (!context) return SEABRAID_INVALID;
    const auto* self = static_cast<StaticQwenProvider*>(context);
    return execute_operation(context, self->layer_, config, batch, output, output_count);
}

int32_t StaticQwenProvider::execute_operation(
    void* context, uint32_t layer, const seabraid_config_v1* config, const seabraid_batch_v1* batch,
    float* output, size_t output_count) noexcept {
    if (!context || !config || !batch || !output || layer >= kGeometry.n_layers) return SEABRAID_INVALID;
    auto& self = *static_cast<StaticQwenProvider*>(context);
    // The callback is serial per Seabraid session; refuse accidental parallel use
    // of the same descriptor before either caller can overwrite the mapping.
    if (self.in_callback_.test_and_set(std::memory_order_acquire)) return SEABRAID_BUSY_OR_STATE;
    struct Release {
        std::atomic_flag& busy;
        ~Release() { busy.clear(std::memory_order_release); }
    } release{self.in_callback_};
    return self.run(layer, *config, *batch, output, output_count);
}

int32_t StaticQwenProvider::run(uint32_t layer, const seabraid_config_v1& config, const seabraid_batch_v1& batch,
                                float* output, size_t output_count) noexcept {
    // No log here: retire() already logged the transition, and a retired
    // provider is refused on every later call by design.
    if (retired()) return SEABRAID_PROVIDER;
    if (!matches(config, this->config())) {
        diag::logf(diag::kError, SEABRAID_INVALID, "static Qwen layer %u refused: config does not match the leased geometry", layer);
        return SEABRAID_INVALID;
    }
    if (!extents_match(batch, output_count)) {
        diag::logf(diag::kError, SEABRAID_INVALID,
                   "static Qwen layer %u refused: extent mismatch (tokens %u, activations %zu, ids %zu, weights %zu, outputs %zu)",
                   layer, batch.tokens, batch.activation_count, batch.expert_id_count, batch.weight_count, output_count);
        return SEABRAID_INVALID;
    }
    if (const char* fault = preflight_fault()) return retire(layer, fault);
    // Every assigned pick must be resident. The engine uses zero-weight sentinels
    // for contributions assigned to another provider.
    bool has_work = false;
    if (!checked_routes(layer, batch, has_work)) return SEABRAID_PROVIDER;
    if (!has_work) {
        std::fill_n(output, output_count, 0.0f);
        return SEABRAID_OK;
    }
    publish(layer, batch);
    return collect(layer, batch.tokens, output, output_count);
}

int32_t StaticQwenProvider::collect(uint32_t layer, uint32_t tokens, float* output,
                                    size_t output_count) noexcept {
    if (const char* fault = accept_reply(tokens)) return retire(layer, fault);
    // Copy first, then judge our own copy: the helper could still rewrite the
    // shared rows between a check on them and the copy.
    std::memcpy(output, channel_->out(), output_count * sizeof(float));
    if (::seabraid::all_finite(output, output_count)) return SEABRAID_OK;
    // Never hand a partial or poisoned row to the merge.
    std::fill_n(output, output_count, 0.0f);
    return retire(layer, "helper reply contains non-finite values");
}

bool StaticQwenProvider::checked_routes(
    uint32_t layer, const seabraid_batch_v1& batch, bool& has_work) const noexcept {
    const auto* held = channel_->held() + size_t{layer} * kGeometry.n_expert;
    for (size_t i = 0; i < batch.expert_id_count; ++i) {
        const int32_t expert = batch.expert_ids[i];
        if (expert == SEABRAID_SKIP_EXPERT_ID && batch.weights[i] == 0.0f) continue;
        const bool in_range = expert >= 0 && expert < kGeometry.n_expert;
        if (!in_range || held[expert] != 1) {
            diag::logf(diag::kError, SEABRAID_PROVIDER, "static Qwen layer %u pick %zu refused: expert id %d is %s",
                       layer, i, expert, in_range ? "not resident" : "out of range");
            return false;
        }
        has_work = true;
    }
    return true;
}

void StaticQwenProvider::publish(uint32_t layer, const seabraid_batch_v1& batch) noexcept {
    auto* header = channel_->header();
    // Own all wire storage before publishing; no helper work touches Rust's
    // borrowed callback buffers, even when the reply arrives after our timeout.
    std::memcpy(channel_->x(), batch.activations, batch.activation_count * sizeof(float));
    std::memcpy(channel_->ids(), batch.expert_ids, batch.expert_id_count * sizeof(int32_t));
    std::memcpy(channel_->weights(), batch.weights, batch.weight_count * sizeof(float));
    header->layer = layer;
    header->n_tok = batch.tokens;
    header->k = kPicks;
    header->ok = 0;
    header->out_mode = seabraid::kOutPerToken;
    header->out_rows = -1;
    header->req_seq.store(++sequence_, std::memory_order_release);
}

bool StaticQwenProvider::replied() const noexcept {
    return channel_->header()->resp_seq.load(std::memory_order_acquire) >= sequence_;
}

// Probes the pidfd on every kLivenessStride-th call and records the exit.
bool StaticQwenProvider::helper_gone(uint32_t& polls) noexcept {
    if (++polls % kLivenessStride != 0 || !helper_exited(watch_, exit_)) return false;
    gone_ = true;
    return true;
}

// True when a reply landed, the lease left "loaded", or the helper died; false at the deadline.
bool StaticQwenProvider::await_reply() noexcept {
    uint32_t polls = 0;
    // WHY liveness in the predicate: a SIGKILLed or OOM-killed helper never
    // changes the channel state, so without it every layer waits the full deadline.
    const auto answered = [&] { return replied() || !loaded(*channel_) || helper_gone(polls); };
    return seabraid::Channel::wait(answered, timeout_ms_ / 1000.0);
}

const char* StaticQwenProvider::accept_reply(uint32_t tokens) noexcept {
    const auto* header = channel_->header();
    if (!await_reply()) return "reply timed out";
    // A reply that landed before the death still counts; the next layer finds the corpse.
    if (!replied() && gone_) return "helper exited";
    // Match the exact response before reading any non-atomic reply fields.
    if (const char* fault = generation_fault()) return fault;
    if (header->resp_seq.load(std::memory_order_acquire) != sequence_) return "reply sequence mismatch";
    if (header->ok != 1) return "helper reported failure";
    if (header->out_mode != seabraid::kOutPerToken || header->out_rows != static_cast<int32_t>(tokens))
        return "malformed reply: output mode or row count";
    return nullptr;
}

const char* StaticQwenProvider::generation_fault() const noexcept {
    if (!loaded(*channel_)) return "helper is no longer loaded";
    if (channel_->epoch() != epoch_) return "helper epoch changed";
    if (!maintenance_idle(*channel_)) return "maintenance traffic in flight";
    if (channel_->header()->req_seq.load(std::memory_order_acquire) != sequence_)
        return "request sequence changed by another publisher";
    return nullptr;
}

const char* StaticQwenProvider::preflight_fault() const noexcept {
    if (sequence_ == std::numeric_limits<uint64_t>::max()) return "request sequence exhausted";
    if (const char* fault = generation_fault()) return fault;
    const auto* header = channel_->header();
    return header->resp_seq.load(std::memory_order_acquire) != sequence_ ? "response sequence out of step" : nullptr;
}

int32_t StaticQwenProvider::retire(uint32_t layer, const char* reason) noexcept {
    // A late reply can still land. Don't publish again or reset the counters in
    // this generation; only the owner of the helper may tear down or replace it.
    // WHY exchange: the transition is logged exactly once, with the reason that caused it.
    if (!retired_.exchange(true, std::memory_order_acq_rel)) {
        char detail[40] = "";
        if (gone_) describe_exit(exit_, detail, sizeof(detail));
        diag::logf(diag::kError, SEABRAID_PROVIDER, "static Qwen provider retired at layer %u: %s%s", layer, reason,
                   detail);
    }
    return SEABRAID_PROVIDER;
}

}  // namespace seabraid::strata_adapter
