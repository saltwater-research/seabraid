// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// The scripted peer checks host transport and failure handling, not GPU math.
#include "static_qwen.hpp"

#include "diag_capture.hpp"
#include "static_qwen_peers.hpp"

#include "strata/seabraid/channel.hpp"
#include "strata/core/geometry.hpp"
#include "strata/kernels/window_limits.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <thread>
#include <vector>

namespace {

namespace capture = seabraid::strata_adapter::capture;

struct Input {
    std::vector<float> activations = std::vector<float>(kValues, 3.0f);
    std::vector<int32_t> ids = std::vector<int32_t>(kPicks);
    std::vector<float> weights = std::vector<float>(kPicks, 0.25f);
    std::vector<float> output = std::vector<float>(kValues, -19.0f);

    Input() {
        for (size_t i = 0; i < ids.size(); ++i) ids[i] = i % kGeometry.n_expert_used;
    }

    seabraid_batch_v1 view() const {
        return {SEABRAID_ABI_VERSION, kTokens, activations.data(), activations.size(), ids.data(), ids.size(),
                weights.data(), weights.size()};
    }

    int32_t run(StaticQwenProvider& adapter) {
        const auto config = adapter.config();
        const auto batch = view();
        const auto provider = adapter.provider();
        return provider.execute(provider.context, &config, &batch, output.data(), output.size());
    }

    bool untouched() const {
        return std::all_of(output.begin(), output.end(), [](float value) { return value == -19.0f; });
    }
};

// `poison` replaces the first reply value, standing in for a corrupted helper.
void answer(seabraid::Channel& helper, const Input& input, uint64_t sequence, int32_t rows,
            uint32_t layer = kLayer, float poison = 12.5f) {
    auto* header = helper.header();
    const auto published = [&] { return header->req_seq.load(std::memory_order_acquire) == sequence; };
    check(seabraid::Channel::wait(published, 2), "provider did not publish");
    check(header->layer == layer && header->n_tok == kTokens &&
          header->k == kGeometry.n_expert_used && header->out_mode == seabraid::kOutPerToken,
          "expected the fixed-layer weighted decode request");
    check(std::equal(input.activations.begin(), input.activations.end(), helper.x()), "wrong activation rows");
    check(std::equal(input.ids.begin(), input.ids.end(), helper.ids()), "wrong expert IDs");
    check(std::equal(input.weights.begin(), input.weights.end(), helper.weights()), "wrong router weights");
    std::fill_n(helper.out(), kValues, 12.5f);
    helper.out()[kValues / 2] = poison;
    header->out_rows = rows;
    header->ok = 1;
    header->resp_seq.store(sequence, std::memory_order_release);
}

void serves_weighted_requests() {
    Peers peers;
    auto* header = peers.helper.header();
    header->req_seq.store(7, std::memory_order_relaxed);
    header->resp_seq.store(7, std::memory_order_release);
    auto adapter = peers.bind();
    check(bool(adapter), "bind prepared Qwen channel");
    Input input;
    std::thread reply([&] {
        answer(peers.helper, input, 8, kTokens);
        answer(peers.helper, input, 9, kTokens);
    });
    check(input.run(*adapter) == SEABRAID_OK, "accept exact weighted reply");
    check(input.run(*adapter) == SEABRAID_OK, "reuse a drained request slot");
    reply.join();
    check(std::all_of(input.output.begin(), input.output.end(), [](float value) { return value == 12.5f; }),
          "return all reply rows");
}

void serves_multiple_layers_on_one_lease() {
    constexpr auto second_layer = kLayer + 1;
    Peers peers;
    auto* held = peers.helper.held() + second_layer * kGeometry.n_expert;
    std::fill_n(held, kGeometry.n_expert_used, uint8_t{1});
    auto adapter = peers.bind();
    check(bool(adapter), "bind reusable Qwen channel");
    check(adapter->holds(second_layer, 0) && !adapter->holds(second_layer, -1),
          "query leased residency by layer");
    const auto provider = adapter->multi_provider();
    check(provider.operation_count == kGeometry.n_layers, "one operation per model layer");
    const auto config = adapter->config();
    Input input;
    const auto batch = input.view();
    const auto run = [&](uint32_t layer) {
        return provider.execute(provider.context, layer, &config, &batch,
                                input.output.data(), input.output.size());
    };
    check(run(provider.operation_count) == SEABRAID_INVALID, "out-of-range layer is rejected");
    check(input.untouched() && peers.helper.header()->req_seq.load() == 0,
          "invalid layer leaves the wire and output untouched");
    std::thread reply([&] {
        answer(peers.helper, input, 1, kTokens, second_layer);
        answer(peers.helper, input, 2, kTokens, kLayer);
    });
    check(run(second_layer) == SEABRAID_OK, "selected layer receives the first request");
    check(run(kLayer) == SEABRAID_OK, "reuse one lease for another layer");
    reply.join();
}

void skips_work_assigned_to_the_engine() {
    Peers peers;
    auto adapter = peers.bind();
    check(bool(adapter), "bind a channel for partial routes");
    const auto provider = adapter->multi_provider();
    check(provider.flags & SEABRAID_PROVIDER_PARTIAL_ROUTES, "provider advertises partial routes");
    const auto config = adapter->config();
    Input input;
    const auto batch = input.view();
    const auto run = [&] {
        return provider.execute(provider.context, kLayer, &config, &batch,
                                input.output.data(), input.output.size());
    };
    input.ids[0] = SEABRAID_SKIP_EXPERT_ID;
    check(run() == SEABRAID_PROVIDER && input.untouched(), "nonzero skipped weight is rejected");
    check(peers.helper.header()->req_seq.load() == 0, "invalid skip publishes no work");
    input.weights[0] = 0.0f;
    std::thread reply([&] { answer(peers.helper, input, 1, kTokens); });
    check(run() == SEABRAID_OK, "partial routes continue through the helper");
    reply.join();

    std::fill(input.ids.begin(), input.ids.end(), SEABRAID_SKIP_EXPERT_ID);
    std::fill(input.weights.begin(), input.weights.end(), 0.0f);
    check(run() == SEABRAID_OK && peers.helper.header()->req_seq.load() == 1,
          "an empty assignment needs no helper work");
    check(std::all_of(input.output.begin(), input.output.end(), [](float value) { return value == 0.0f; }),
          "empty assignment contributes zero");
}

void rejects_parallel_callback_on_same_mapping() {
    Peers peers;
    auto adapter = peers.bind();
    check(bool(adapter), "bind for concurrent provider misuse");
    Input first, second;
    int32_t status = SEABRAID_INTERNAL;
    std::thread worker([&] { status = first.run(*adapter); });
    const auto published = [&] { return peers.helper.header()->req_seq.load(std::memory_order_acquire) == 1; };
    check(seabraid::Channel::wait(published, 2), "first callback owns the wire slot");
    check(second.run(*adapter) == SEABRAID_BUSY_OR_STATE && second.untouched(), "reject a second publisher");
    answer(peers.helper, first, 1, kTokens);
    worker.join();
    check(status == SEABRAID_OK, "the admitted callback remains valid");
}

void refuses_wrong_geometry_or_busy_handoff() {
    capture::clear();
    auto mismatched = kLimits;
    mismatched.hidden += 1;
    Peers bad(mismatched);
    check(!bad.bind(), "reject geometry even if both peer mappings agree");
    Peers other_engine;
    other_engine.helper.header()->engine = seabraid::kEngineDsv41;
    check(!other_engine.bind(), "reject another helper engine at identical geometry");
    Peers busy;
    busy.helper.header()->blob_req_seq.store(1, std::memory_order_release);
    check(!busy.bind(), "reject handoff with maintenance in flight");
    // Each of the three refusals wrote one record naming its specific cause.
    capture::require("bind refused", 3);
    capture::require("matching Qwen v7 geometry", 2);
    capture::require("maintenance in flight", 1);
}

void refuses_changed_epoch() {
    capture::clear();
    Peers peers;
    auto adapter = peers.bind();
    check(bool(adapter), "bind for placement change");
    peers.helper.bump_epoch();
    Input input;
    check(input.run(*adapter) == SEABRAID_PROVIDER && input.untouched(), "reject changed placement epoch");
    check(peers.helper.header()->req_seq.load() == 0, "no work admitted on the old placement");
    capture::require("retired at layer 1: helper epoch changed", 1, SEABRAID_PROVIDER);
    check(input.run(*adapter) == SEABRAID_PROVIDER, "a retired lease stays refused");
    capture::require("retired at layer", 1);   // the transition is logged once, not per call
}

void refuses_partial_or_bad_input() {
    capture::clear();
    Peers peers;
    auto adapter = peers.bind();
    check(bool(adapter), "bind for negative inputs");
    Input input;
    const auto provider = adapter->provider();
    auto batch = input.view();
    --batch.weight_count;
    const auto config = adapter->config();
    check(provider.execute(provider.context, &config, &batch, input.output.data(), input.output.size()) ==
          SEABRAID_INVALID, "reject wrong batch shape");
    capture::require("extent mismatch", 1, SEABRAID_INVALID);
    input.ids.back() = kGeometry.n_expert - 1;
    check(input.run(*adapter) == SEABRAID_PROVIDER, "reject a batch with a nonresident expert");
    capture::require("is not resident", 1, SEABRAID_PROVIDER);
    check(peers.helper.header()->req_seq.load() == 0 && input.untouched(), "negative inputs publish no work");
}

void refuses_wrong_reply_and_retires() {
    capture::clear();
    Peers peers;
    auto adapter = peers.bind();
    check(bool(adapter), "bind for bad reply");
    Input input;
    std::thread reply([&] { answer(peers.helper, input, 1, kTokens + 1); });
    check(input.run(*adapter) == SEABRAID_PROVIDER, "reject wrong row count from helper");
    reply.join();
    check(input.untouched(), "bad echo never reaches Seabraid output");
    check(adapter->retired() && !adapter->holds(kLayer, 0), "retired lease advertises no residency");
    check(input.run(*adapter) == SEABRAID_PROVIDER && peers.helper.header()->req_seq.load() == 1,
          "a bad reply retires the generation lease");
    capture::require("retired at layer 1: malformed reply", 1, SEABRAID_PROVIDER);
}

// A structurally valid reply carrying NaN or Inf must never reach the merge.
void refuses_non_finite_reply_rows() {
    for (const float poison : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
                               -std::numeric_limits<float>::infinity()}) {
        capture::clear();
        Peers peers;
        auto adapter = peers.bind();
        check(bool(adapter), "bind for a poisoned reply");
        Input input;
        std::thread reply([&] { answer(peers.helper, input, 1, kTokens, kLayer, poison); });
        check(input.run(*adapter) == SEABRAID_PROVIDER, "reject a reply with non-finite rows");
        reply.join();
        check(std::none_of(input.output.begin(), input.output.end(),
                           [](float value) { return value != value || value == 12.5f; }),
              "no helper row reaches the output");
        check(adapter->retired(), "a poisoned reply retires the lease");
        capture::require("retired at layer 1: helper reply contains non-finite values", 1, SEABRAID_PROVIDER);
    }
}

void late_reply_cannot_reopen_a_retired_lease() {
    capture::clear();
    Peers peers;
    auto adapter = peers.bind(5);
    check(bool(adapter), "bind for no reply");
    Input input;
    check(input.run(*adapter) == SEABRAID_PROVIDER && input.untouched(), "no reply fails without returning data");
    answer(peers.helper, input, 1, kTokens);
    check(input.run(*adapter) == SEABRAID_PROVIDER && peers.helper.header()->req_seq.load() == 1,
          "never republish even after a late reply catches up");
    capture::require("retired at layer 1: reply timed out", 1, SEABRAID_PROVIDER);
}

}  // namespace

int main() {
    capture::start();
    serves_weighted_requests();
    serves_multiple_layers_on_one_lease();
    skips_work_assigned_to_the_engine();
    rejects_parallel_callback_on_same_mapping();
    refuses_wrong_geometry_or_busy_handoff();
    refuses_partial_or_bad_input();
    refuses_changed_epoch();
    refuses_wrong_reply_and_retires();
    refuses_non_finite_reply_rows();
    late_reply_cannot_reopen_a_retired_lease();
    std::puts("Strata v7 fixed-layer weighted provider fixture passed");
}
