// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// The linked-runtime half of the private Qwen fixture: the weighted decode tier, the private helper
// rung and QwenRuntime against the scripted CPU helper this binary execs itself as.
// OWNER: this file. private_qwen_test.cpp keeps the helper lifecycle and admission cases; the helper
//   script, geometry and check_reaped come from private_qwen_script.hpp. WHY two files: the original
//   reached the 500-line limit, and these are the only cases that need the Rust runtime.
#include "private_qwen.hpp"
#include "private_qwen_script.hpp"
#include "private_helper_rung.hpp"
#include "qwen_decode.hpp"
#include "runtime_qwen.hpp"
#include "helper_options.hpp"

#include "diag_capture.hpp"
#include "test_support.hpp"

#include "strata/core/geometry.hpp"
#include "strata/kernels/window_limits.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

using seabraid::strata_adapter::PrivateQwenLaunch;
namespace capture = seabraid::strata_adapter::capture;

using seabraid::strata_adapter::testing::check;
using seabraid::strata_adapter::testing::ScratchFile;

using seabraid::strata_adapter::ExecutorView;
using seabraid::strata_adapter::PrivateHelperRung;
using strata::core::IoStatus;

// Publishes a table that names `rung` for (kLayer, kExpert) and the fallback (2) everywhere else.
void publish_cell(ExecutorView& view, uint64_t epoch, uint8_t rung) {
    std::vector<uint8_t> exec(size_t{kGeometry.n_layers} * kGeometry.n_expert, 2);
    exec[kLayer * kGeometry.n_expert + kExpert] = rung;
    seabraid_placement_snapshot_v1 s{};
    s.abi_version = SEABRAID_ABI_VERSION;
    s.struct_size = sizeof s;
    s.epoch = epoch;
    s.layers = kGeometry.n_layers;
    s.experts = kGeometry.n_expert;
    s.fallback_rung = 2;
    check(view.publish(exec.data(), exec.size(), s), "the fixture's table must publish");
}

// The slot table is per layer: each layer stages its own size, a malformed table never opens, a size the engine
// disagrees with is refused, and a layer outside the table is refused; each refusal is one record.
void per_layer_slots_stage_their_own_size(seabraid::strata_adapter::QwenDecodeTier& tier, const PrivateHelperRung::CopyBlob& fill) {
    std::string error;
    const size_t layers = kGeometry.n_layers;
    for (const auto& bad : {std::vector<size_t>{}, std::vector<size_t>(layers + 1, 8), std::vector<size_t>(layers, 0)}) {
        capture::clear();
        check(!PrivateHelperRung::open(tier, fill, layers, bad, 100, error) && !error.empty(), "a malformed slot table is refused");
        capture::require("private helper rung refused", 1);
    }
    std::vector<size_t> sizes(layers, 8);
    sizes[0] = 40;   // the engine below reports 8 for layer 0: a drifted table must not stage
    auto rung = PrivateHelperRung::open(tier, [&](int64_t l, int64_t e, uint8_t* d, size_t n) { return n == 8 && fill(l, e, d, n); },
                                        layers, sizes, 100, error);
    std::vector<uint8_t> blob;
    check(rung->stage(1, 2, blob, error) == IoStatus::kOk && blob.size() == 8, "a layer stages its own size");
    capture::clear();
    check(rung->stage(0, 2, blob, error) == IoStatus::kRefused && blob.empty(), "a size the engine disagrees with is refused");
    capture::require("private helper stage refused at layer 0 expert 2: copy failed", 1);
    capture::clear();
    check(rung->stage(static_cast<int32_t>(layers), 2, blob, error) == IoStatus::kRefused, "a layer outside the table is refused");
    capture::require("no slot size for the layer", 1);
}

// R1: a claim needs the published table, the physical cell and a live tier, all three.
void demoted_cell_is_not_claimed(seabraid::strata_adapter::QwenDecodeTier& tier, ExecutorView& view,
                                 const std::vector<float>& input, const std::vector<int32_t>& picks,
                                 const std::vector<float>& weights, const std::vector<int32_t>& kind) {
    std::string error;
    const int64_t top_k = kGeometry.n_expert_used;
    check(!tier.holds(kLayer, kExpert) && tier.held(kLayer, kExpert), "R1: nothing is claimed before the first publish");
    publish_cell(view, 1, 0);
    check(tier.begin(kLayer, input.data(), picks.data(), weights.data(), 1, top_k, kind.data(), nullptr, error) && !tier.owns(0),
          "R1: a demoted cell is not claimed");
    check(!tier.holds(kLayer, kExpert) && tier.held(kLayer, kExpert), "R1: demoted, still physically held");
    publish_cell(view, 2, 1);
    check(tier.holds(kLayer, kExpert), "R1: the table naming the helper makes the held cell claimable");
}

void weighted_tier_routes_and_retires(const char* program, bool fail) {
    capture::clear();
    const PrivateQwenLaunch launch{program, fail ? "stuck-request" : "serve", "", 0, 2000, 2000};
    std::string error;
    auto tier = seabraid::strata_adapter::QwenDecodeTier::start(launch, kPlan, fail ? 50 : 1500, error);
    check(bool(tier), error.c_str());
    check(tier->holds(kLayer, kExpert), "tier must advertise its resident expert");
    ExecutorView view;
    tier->attach_view(&view, 1);
    std::string rung_error;
    const auto fill_blob = [](int64_t, int64_t e, uint8_t* dst, size_t n) { std::fill_n(dst, n, uint8_t(e)); return true; };
    std::vector<size_t> sizes(kGeometry.n_layers, 16);
    sizes[kLayer] = 24;   // mixed per-layer quants: this layer's blob is not the others' size
    auto rung_owner = PrivateHelperRung::open(*tier, fill_blob, kGeometry.n_layers, sizes, 100, rung_error);
    check(bool(rung_owner), "a per-layer slot table of the right shape opens");
    auto& rung = *rung_owner;
    per_layer_slots_stage_their_own_size(*tier, fill_blob);

    const auto hidden = kGeometry.n_embd, top_k = kGeometry.n_expert_used;
    const std::vector<float> input(hidden, 4.0f), weights(top_k, 0.25f);
    const std::vector<int32_t> picks(top_k, kExpert);
    std::vector<int32_t> kind(top_k, 0);
    kind[0] = -1;

    demoted_cell_is_not_claimed(*tier, view, input, picks, weights, kind);

    check(tier->begin(kLayer, input.data(), picks.data(), nullptr, 1, top_k, kind.data(), nullptr, error),
          "a per-pick caller should keep its work");
    check(!tier->owns(0), "per-pick callers do not delegate without router weights");
    const float* rows = nullptr;
    check(tier->finish(rows, error) && !rows, "no per-pick submission was published");
    // The silent fallback to local work is logged once per transition, not per call.
    check(tier->begin(kLayer, input.data(), picks.data(), nullptr, 1, top_k, kind.data(), nullptr, error),
          "a second per-pick call also stays local");
    capture::require("no router weights", 1);

    std::vector<int32_t> primary(kGeometry.n_layers * kGeometry.n_expert, -1);
    primary[kLayer * kGeometry.n_expert + kExpert] = 0;
    check(tier->begin(kLayer, input.data(), picks.data(), weights.data(), 1, top_k, kind.data(), primary.data(), error),
          "already resident expert stays with the primary");
    check(!tier->owns(0), "the helper must leave a primary-owned pick local");
    check(tier->finish(rows, error), "a skipped batch finishes without waiting");
    check(!rows, "a skipped batch has no weighted contribution");

    check(tier->begin(kLayer, input.data(), picks.data(), weights.data(), 1, top_k, kind.data(), nullptr, error),
          "weighted serial layer should claim the CPU-served pick");
    check(tier->owns(0) && !tier->owns(1), "weighted helper must claim only its assigned row");
    std::vector<uint8_t> blob;
    check(tier->tier_swap(kLayer, kExpert, 5, 100) == SEABRAID_PLACEMENT_IO_REFUSED, "tier_swap is refused with a pending ticket");
    check(rung.replace(kLayer, kExpert, 5, blob, error) == IoStatus::kRefused, "the rung reports that refusal");
    capture::require("tier swap at layer 1 refused: a layer is pending", 2);
    if (fail) {
        check(!tier->finish(rows, error), "the helper must reject uncertain completion");
        check(error.find("SEABRAID_") != std::string::npos, "the error names the status, not a bare number");
        capture::require("tier retired: collect failed with SEABRAID_", 1);
        check(!rows, "a failed batch must never publish a contribution");
        check(tier->retired(), "uncertain completion must retire the tier");
        // R1: the table still names rung 1 and the cell is still held, yet a retired tier claims nothing.
        check(!tier->holds(kLayer, kExpert) && tier->held(kLayer, kExpert), "retired tier must advertise no experts to the next window");
        tier->disable();
        check_reaped();
        check(tier->begin(kLayer, input.data(), picks.data(), weights.data(), 1, top_k, kind.data(), nullptr, error),
              "next window should proceed locally");
        check(!tier->owns(0), "a retired tier owns no picks");
        check(tier->finish(rows, error), "the next local-only batch has nothing to collect");
        check(!rows, "a retired tier must never return stale output");
        check(tier->begin(kLayer, input.data(), picks.data(), weights.data(), 1, top_k, kind.data(), nullptr, error),
              "a later window also stays local");
        capture::require("work now runs locally", 1);   // first fallback only
        capture::require("disabled by its owner", 0);   // finish() already logged the retirement
    } else {
        check(tier->finish(rows, error) && rows, "valid weighted reply must reach merge");
        check(std::all_of(rows, rows + hidden, [](float value) { return value == 5.0f; }),
              "reported contribution must use the original router weight");
        check(rung.stage(kLayer, kExpert, blob, error) == IoStatus::kOk && blob.size() == 24 && blob[0] == kExpert,
              "stage copies the engine's blob into the staging buffer");
        check(rung.replace(kLayer, kExpert, kExpert, blob, error) == IoStatus::kRefused, "old == new is refused");
        capture::require("old equals new expert", 1);
    }
    tier.reset();
    check_reaped();
}

void runtime_routes_and_reaps(const char* program) {
    const PrivateQwenLaunch launch{program, "serve", "", 0, 2000, 2000};
    std::string error;
    auto runtime = seabraid::strata_adapter::QwenRuntime::start(launch, kPlan, 1500, error);
    check(bool(runtime), error.c_str());
    check(runtime->holds(kLayer, kExpert), "prepared private session advertises the held pick");
    const auto config = runtime->config();
    std::vector<float> input(config.hidden, 4.0f), output(config.hidden, -21.0f);
    std::vector<int32_t> ids(config.top_k, SEABRAID_SKIP_EXPERT_ID);
    std::vector<float> weights(config.top_k, 0.0f);
    ids[0] = kExpert;
    weights[0] = 0.25f;
    const seabraid_batch_v1 batch{SEABRAID_ABI_VERSION, 1, input.data(), input.size(), ids.data(), ids.size(), weights.data(), weights.size()};
    uint64_t ticket = 0;
    check(runtime->submit(kGeometry.n_layers, batch, 0, 0, ticket) == SEABRAID_INVALID,
          "layer outside the prepared operations is rejected");
    check(ticket == 0, "invalid layer leaves the ticket untouched");
    check(runtime->submit(kLayer, batch, SEABRAID_SOURCE_TEXT, 0, ticket) == SEABRAID_OK,
          "submit a partial batch through native Seabraid");
    std::fill(input.begin(), input.end(), -35.0f);
    check(runtime->tier_swap(kLayer, kExpert, 5, 100) == SEABRAID_PLACEMENT_IO_REFUSED, "a session ticket outstanding refuses tier_swap");
    check(runtime->collect(ticket, 3000, output.data(), output.size()) == SEABRAID_OK,
          "collect the exact weighted v7 reply through Seabraid");
    check(std::all_of(output.begin(), output.end(), [](float row) { return row == 5.0f; }),
          "published request persists after the engine reused its input");
    runtime.reset();
    check_reaped();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1) {
        seabraid::QwenHelperOptions args;
        std::string error;
        if (!args.parse(argc, argv, error) || !args.watch_controller(error)) return 1;
        return script(args);
    }
    capture::start();
    setenv("UNLISTED_TEST_SECRET", "must-not-reach-helper", 1);
    setenv("LC_NUMERIC", "C", 1);
    // The launcher requires an absolute, owner-only helper: run a 0755 copy of this fixture.
    ScratchFile trusted;
    trusted.copy_of(argv[0]);
    const std::string program = trusted.path;
    argv[0] = const_cast<char*>(program.c_str());
    runtime_routes_and_reaps(argv[0]);
    weighted_tier_routes_and_retires(argv[0], false);
    weighted_tier_routes_and_retires(argv[0], true);
    std::puts("Strata v7 private Qwen runtime and tier fixture passed");
}
