// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// Scripted peer on the real private mapping for StaticQwenProvider::tier_swap.
// CHECKER for DESIGN R3/R4: the next batch succeeds after an ACK (epoch re-captured only
// then), REFUSED leaves the epoch, and an uncertain helper op retires the lease.
#include "static_qwen.hpp"

#include "diag_capture.hpp"

#include "seabraid/placement.h"

#include "strata/seabraid/channel.hpp"
#include "strata/core/geometry.hpp"
#include "strata/kernels/window_limits.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

namespace {

using seabraid::strata_adapter::StaticQwenProvider;
namespace capture = seabraid::strata_adapter::capture;
constexpr auto& kGeometry = strata::core::kCompiledGeometry;
constexpr auto kWindow = strata::kernels::kWindowMaxT;
constexpr int64_t kBlobMax = 64;   // v8: room for one scripted blob per slot
constexpr seabraid::ChannelLimits kLimits{kGeometry.n_embd, kWindow * kGeometry.n_expert_used,
                                        kGeometry.n_layers, kGeometry.n_expert, kBlobMax, 0, 0};
constexpr uint32_t kLayer = 1;
constexpr int32_t kOld = 0;                                  // resident at bind
constexpr int32_t kNew = kGeometry.n_expert_used;            // not resident at bind
constexpr uint32_t kTokens = 2;
constexpr uint32_t kDeadlineMs = 1000;
constexpr uint32_t kTierMs = 2000;
constexpr size_t kValues = kTokens * kGeometry.n_embd;
constexpr size_t kPicks = kTokens * kGeometry.n_expert_used;

void check(bool ok, const char* message) {
    if (ok) return;
    std::fprintf(stderr, "%s\n", message);
    std::exit(1);
}

struct Peers {
    seabraid::Channel helper;
    std::unique_ptr<seabraid::Channel> main = std::make_unique<seabraid::Channel>();

    Peers() {
        std::string error;
        const int fd = seabraid::Channel::private_fd(error);
        check(fd >= 0, "create private descriptor");
        check(helper.create_private(fd, kLimits, kWindow, error), "create helper mapping");
        check(main->connect_private(fd, kLimits.hidden, kLimits.cap, kLimits.layers, kLimits.experts, kWindow, error) ==
                  seabraid::Attach::kAttached, "attach main mapping");
        close(fd);
        std::fill_n(held(), kGeometry.n_expert_used, uint8_t{1});
        helper.header()->state.store(seabraid::kLoaded, std::memory_order_release);
    }

    uint8_t* held() { return helper.held() + size_t{kLayer} * kGeometry.n_expert; }

    std::unique_ptr<StaticQwenProvider> bind() {
        std::string error;
        auto lease = StaticQwenProvider::bind(std::move(main), kLayer, kDeadlineMs, error);
        check(bool(lease), "bind prepared Qwen channel");
        return lease;
    }
};

// One decode batch whose picks alternate between expert 1 and `second`.
struct Input {
    std::vector<float> activations = std::vector<float>(kValues, 3.0f);
    std::vector<int32_t> ids = std::vector<int32_t>(kPicks);
    std::vector<float> weights = std::vector<float>(kPicks, 0.25f);
    std::vector<float> output = std::vector<float>(kValues, -19.0f);

    explicit Input(int32_t second) {
        for (size_t i = 0; i < ids.size(); ++i) ids[i] = i % 2 == 0 ? 1 : second;
    }

    int32_t run(StaticQwenProvider& lease) {
        const auto config = lease.config();
        const seabraid_batch_v1 batch{SEABRAID_ABI_VERSION, kTokens, activations.data(), activations.size(),
                                      ids.data(), ids.size(), weights.data(), weights.size()};
        const auto provider = lease.provider();
        return provider.execute(provider.context, &config, &batch, output.data(), output.size());
    }
};

void answer_decode(seabraid::Channel& helper, uint64_t seq) {
    auto* header = helper.header();
    check(seabraid::Channel::wait([&] { return header->req_seq.load(std::memory_order_acquire) == seq; }, 2),
          "provider did not publish the batch");
    std::fill_n(helper.out(), kValues, 12.5f);
    header->out_rows = static_cast<int32_t>(kTokens);
    header->ok = 1;
    header->resp_seq.store(seq, std::memory_order_release);
}

enum class Mode { kAck, kRejected, kFailed, kStale, kJump, kBadCode };

// Per-op result byte the scripted helper reports for `mode`.
uint8_t result_of(Mode mode) {
    switch (mode) {
        case Mode::kRejected: return seabraid::kTierRejected;
        case Mode::kFailed: return seabraid::kTierFailed;
        case Mode::kBadCode: return 9;
        default: return seabraid::kTierApplied;
    }
}

// The helper side of one tier round: validates the op like the real service loop, then
// answers as `mode` scripts. kStale acks without moving held[]; kJump acks a later round.
void serve_tier(Peers& peers, Mode mode, uint8_t kind = seabraid::kTierSwap) {
    auto& ch = peers.helper;
    check(seabraid::Channel::wait([&] { return ch.tier_pending() != 0; }, 2), "provider did not publish a tier op");
    const uint64_t seq = ch.tier_pending();
    const auto op = ch.tier_ops()[0];
    check(ch.header()->tier_n == 1 && op.kind == kind && op.layer_old == int32_t{kLayer} &&
              op.layer_new == int32_t{kLayer} && op.expert_old == kOld && op.expert_new == kNew,
          "expected one same-layer swap op");
    check(op.blob_slot == 0 && seabraid::blob_slot_valid(op), "the lease uses blob slot 0 and nothing else");
    const uint8_t res = result_of(mode);
    ch.header()->tier_res[0] = res;
    if (mode == Mode::kAck) { peers.held()[kOld] = 0; peers.held()[kNew] = 1; }
    if (mode == Mode::kJump) {
        ch.header()->tier_ok = 1;
        __atomic_store_n(&ch.header()->tier_resp, seq + 5, __ATOMIC_RELEASE);
        return;
    }
    ch.tier_ack(seq, res != seabraid::kTierRejected && res != seabraid::kTierFailed);
}

int32_t swap_with(Peers& peers, StaticQwenProvider& lease, Mode mode) {
    std::thread peer([&] { serve_tier(peers, mode); });
    const int32_t rc = lease.tier_swap(kLayer, kOld, kNew, kTierMs);
    peer.join();
    return rc;
}

void ack_adopts_residency_and_epoch() {
    capture::clear();
    Peers peers;
    auto lease = peers.bind();
    const uint64_t before = peers.helper.epoch();
    check(swap_with(peers, *lease, Mode::kAck) == SEABRAID_PLACEMENT_IO_ACK, "ACK for an applied swap");
    check(!lease->holds(kLayer, kOld) && lease->holds(kLayer, kNew), "holds() follows the helper's held[]");
    check(peers.helper.epoch() == before + 1, "the lease advertises the round with one epoch move");
    Input input(kNew);
    std::thread reply([&] { answer_decode(peers.helper, 1); });
    // R3: this fails with "helper epoch changed" if the lease did not re-capture its epoch.
    check(input.run(*lease) == SEABRAID_OK, "the next batch is served after the ACK");
    reply.join();
    check(!lease->retired() && capture::count("") == 0, "an ACKed swap logs nothing and keeps the lease");
}

// v8: the staged bytes travel in blob slot 0 as kTierSwapBlob; a blob that does not fit is refused before any traffic.
void blob_swap_sends_the_engine_bytes() {
    capture::clear();
    Peers peers;
    auto lease = peers.bind();
    std::vector<uint8_t> bytes(kBlobMax);
    for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = static_cast<uint8_t>(i * 7 + 3);
    const seabraid::strata_adapter::TierBlob too_big{bytes.data(), bytes.size() + 1};
    check(lease->tier_swap(kLayer, kOld, kNew, kTierMs, too_big) == SEABRAID_PLACEMENT_IO_REFUSED, "an oversize blob");
    check(peers.helper.header()->tier_seq == 0 && !lease->retired(), "an oversize blob publishes nothing");
    capture::require("the blob does not fit the channel's blob slot", 1);
    std::vector<uint8_t> seen;
    std::thread peer([&] {
        serve_tier(peers, Mode::kAck, seabraid::kTierSwapBlob);
        seen.assign(peers.helper.blob(0), peers.helper.blob(0) + bytes.size());
    });
    const int32_t rc = lease->tier_swap(kLayer, kOld, kNew, kTierMs, {bytes.data(), bytes.size()});
    peer.join();
    check(rc == SEABRAID_PLACEMENT_IO_ACK && seen == bytes, "the helper saw the engine's bytes in blob slot 0");
    check(!lease->holds(kLayer, kOld) && lease->holds(kLayer, kNew), "an ACKed blob swap adopts residency");
}

void rejected_is_refused_without_epoch_change() {
    capture::clear();
    Peers peers;
    auto lease = peers.bind();
    const uint64_t before = peers.helper.epoch();
    check(swap_with(peers, *lease, Mode::kRejected) == SEABRAID_PLACEMENT_IO_REFUSED, "REJECTED maps to REFUSED");
    check(peers.helper.epoch() == before && !lease->retired(), "REFUSED leaves the epoch and the lease alone");
    check(lease->holds(kLayer, kOld) && !lease->holds(kLayer, kNew), "REFUSED leaves residency alone");
    capture::require("the helper rejected the swap", 1);
    check(capture::count("") == 1, "one record for a refusal");
    Input input(2);
    std::thread reply([&] { answer_decode(peers.helper, 1); });
    check(input.run(*lease) == SEABRAID_OK, "the lease still serves after a refusal");
    reply.join();
}

void uncertain_helper_retires(Mode mode, const char* reason) {
    capture::clear();
    Peers peers;
    auto lease = peers.bind();
    const uint64_t before = peers.helper.epoch();
    check(swap_with(peers, *lease, mode) == SEABRAID_PLACEMENT_IO_FAILED, "FAILED for an uncertain round");
    check(lease->retired() && !lease->holds(kLayer, 1), "the lease is retired and claims nothing");
    check(peers.helper.epoch() == before, "a failed round does not move the epoch");
    capture::require(reason, 1, SEABRAID_PROVIDER);
    check(capture::count("") == 1, "one record per failure");
    check(lease->tier_swap(kLayer, kOld, kNew, kTierMs) == SEABRAID_PLACEMENT_IO_FAILED, "a retired lease refuses");
    check(capture::count("") == 1, "the retired lease does not log again");
}

void deadline_retires() {
    capture::clear();
    Peers peers;
    auto lease = peers.bind();
    const auto start = std::chrono::steady_clock::now();
    check(lease->tier_swap(kLayer, kOld, kNew, 150) == SEABRAID_PLACEMENT_IO_TIMEOUT, "TIMEOUT without an ack");
    const auto waited = std::chrono::steady_clock::now() - start;
    check(waited >= std::chrono::milliseconds(150) && waited < std::chrono::seconds(5), "the deadline is honoured");
    check(lease->retired(), "a timed-out round retires the lease");
    capture::require("retired at layer 1: tier round timed out", 1, SEABRAID_PROVIDER);
}

void busy_callback_is_refused() {
    capture::clear();
    Peers peers;
    auto lease = peers.bind();
    Input input(2);
    std::thread decode([&] { check(input.run(*lease) == SEABRAID_OK, "the in-flight batch completes"); });
    check(seabraid::Channel::wait([&] { return peers.helper.header()->req_seq.load(std::memory_order_acquire) == 1; }, 2),
          "decode is in flight");
    check(lease->tier_swap(kLayer, kOld, kNew, kTierMs) == SEABRAID_PLACEMENT_IO_REFUSED, "a running callback refuses");
    check(peers.helper.tier_pending() == 0, "a refused swap publishes no tier op");
    answer_decode(peers.helper, 1);
    decode.join();
    capture::require("a decode callback holds the lease", 1);
    check(!lease->retired(), "a busy refusal keeps the lease");
}

void refuses_bad_requests_without_traffic() {
    capture::clear();
    Peers peers;
    auto lease = peers.bind();
    const int32_t rc = SEABRAID_PLACEMENT_IO_REFUSED;
    check(lease->tier_swap(kGeometry.n_layers, kOld, kNew, kTierMs) == rc, "layer out of range");
    check(lease->tier_swap(kLayer, -1, kNew, kTierMs) == rc, "negative expert");
    check(lease->tier_swap(kLayer, kOld, kGeometry.n_expert, kTierMs) == rc, "expert out of range");
    check(lease->tier_swap(kLayer, kOld, kOld, kTierMs) == rc, "in-place swap");
    check(lease->tier_swap(kLayer, kOld, kNew, 0) == rc, "zero deadline");
    check(lease->tier_swap(kLayer, kOld, kNew, 30001) == rc, "deadline over the bound");
    check(lease->tier_swap(kLayer, kNew, 1, kTierMs) == rc, "displaced expert not resident");
    check(lease->tier_swap(kLayer, kOld, 1, kTierMs) == rc, "incoming expert already resident");
    check(peers.helper.header()->tier_seq == 0 && !lease->retired(), "refusals neither publish nor retire");
    check(capture::count("tier swap") == 8, "one record per refusal");
}

// Defense in depth: a negative blob_max never reaches a cast, and with the per-layer size known only that exact size travels.
void blob_size_must_match_the_layer_exactly() {
    using seabraid::strata_adapter::blob_fits;
    const uint8_t b[8] = {};
    check(!blob_fits({b, 4}, -1, 0) && !blob_fits({b, 4}, INT64_MIN, 0), "a negative blob_max fits nothing");
    check(blob_fits({nullptr, 0}, -1, 0), "no blob needs no slot");
    check(blob_fits({b, 4}, 8, 0) && !blob_fits({b, 9}, 8, 0) && !blob_fits({b, 0}, 8, 0), "unknown size: fits the slot");
    check(blob_fits({b, 4}, 8, 4) && !blob_fits({b, 3}, 8, 4) && !blob_fits({b, 5}, 8, 4), "known size: exactly that size");

    capture::clear();
    Peers peers;
    auto lease = peers.bind();
    lease->set_blob_bytes(std::vector<size_t>(kGeometry.n_layers, kBlobMax - 8));
    std::vector<uint8_t> bytes(kBlobMax);
    check(lease->tier_swap(kLayer, kOld, kNew, kTierMs, {bytes.data(), bytes.size()}) == SEABRAID_PLACEMENT_IO_REFUSED,
          "a blob that fits the slot but not the layer");
    check(lease->tier_swap(kLayer, kOld, kNew, kTierMs, {bytes.data(), bytes.size() - 9}) == SEABRAID_PLACEMENT_IO_REFUSED,
          "a short blob");
    check(peers.helper.header()->tier_seq == 0 && !lease->retired(), "refused blobs publish nothing");
    capture::require("the blob does not fit the channel's blob slot", 2);
}

}  // namespace

int main() {
    capture::start();
    ack_adopts_residency_and_epoch();
    rejected_is_refused_without_epoch_change();
    blob_swap_sends_the_engine_bytes();
    blob_size_must_match_the_layer_exactly();
    uncertain_helper_retires(Mode::kFailed, "retired at layer 1: helper failed the swap");
    uncertain_helper_retires(Mode::kStale, "retired at layer 1: helper residency is stale");
    uncertain_helper_retires(Mode::kJump, "retired at layer 1: tier reply sequence mismatch");
    uncertain_helper_retires(Mode::kBadCode, "retired at layer 1: malformed tier reply: result code");
    deadline_retires();
    busy_callback_is_refused();
    refuses_bad_requests_without_traffic();
    std::puts("Strata v7 lease tier_swap fixture passed");
}
