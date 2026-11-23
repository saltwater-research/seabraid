// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// The decode tier through a real session and the scripted CPU helper: a stale placement epoch
// leaves the layer local (once-logged, tier alive), and the compact_rows launch option opens.
// OWNER: this file; the helper script and geometry come from private_qwen_script.hpp.
#include "private_qwen_script.hpp"

#include "executor_view.hpp"
#include "qwen_decode.hpp"

#include "diag_capture.hpp"
#include "test_support.hpp"

#include <sys/wait.h>

namespace {

using seabraid::strata_adapter::ExecutorView;
using seabraid::strata_adapter::PrivateQwenLaunch;
using seabraid::strata_adapter::QwenDecodeTier;
using seabraid::strata_adapter::QwenRuntime;
namespace capture = seabraid::strata_adapter::capture;
using seabraid::strata_adapter::testing::check;
using seabraid::strata_adapter::testing::ScratchFile;

// A table at `epoch` that names `rung` for (kLayer, kExpert) and rung 2 elsewhere.
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

struct Window {
    std::vector<float> input = std::vector<float>(kGeometry.n_embd, 4.0f);
    std::vector<float> weights = std::vector<float>(kGeometry.n_expert_used, 0.25f);
    std::vector<int32_t> picks = std::vector<int32_t>(kGeometry.n_expert_used, kExpert);
    std::vector<int32_t> kind = std::vector<int32_t>(kGeometry.n_expert_used, 0);
    Window() { kind[0] = -1; }
    bool begin(QwenDecodeTier& tier, std::string& error) {
        return tier.begin(kLayer, input.data(), picks.data(), weights.data(), 1, kGeometry.n_expert_used,
                          kind.data(), nullptr, error);
    }
};

void reaped() {
    check(waitpid(-1, nullptr, WNOHANG) == -1 && errno == ECHILD, "controller must reap its child");
}

void stale_epoch_keeps_the_layer_local(const char* program) {
    capture::clear();
    const PrivateQwenLaunch launch{program, "serve", "", 0, 2000, 2000};
    std::string error;
    auto tier = QwenDecodeTier::start(launch, kPlan, 1500, error);
    check(bool(tier), error.c_str());
    ExecutorView fresh, older;
    publish_cell(fresh, 2, 1);
    publish_cell(older, 1, 1);
    tier->attach_view(&fresh, 1);
    Window w;
    const float* rows = nullptr;
    check(w.begin(*tier, error) && tier->owns(0), "a layer routed at epoch 2 is delegated");
    check(tier->finish(rows, error) && rows && rows[0] == 5.0f, "and its reply is merged");
    tier->attach_view(&older, 1);   // routing now decided against epoch 1, below the accepted 2
    for (int call = 0; call < 2; ++call) {
        check(w.begin(*tier, error), "a stale epoch is not an engine failure");
        check(!tier->owns(0), "the stale layer stays local");
        check(tier->finish(rows, error) && !rows, "nothing is collected for it");
    }
    check(!tier->retired(), "a stale epoch must not retire the tier");
    capture::require("stale placement epoch 1; work stays local", 1);   // two calls, one record
    tier.reset();
    reaped();
}

// WHY the raw runtime and the tier's static classifier: a session refusal that is not provably stale cannot be
// forced through begin() (it refuses a pending ticket itself). The runtime hands back the session's REAL refusals;
// the classifier must tell them apart by the runtime's accepted-epoch mark, which this test also pins (L14): it
// rises only with an accepted submit, whoever made it, because the runtime is the session's only gateway.
void only_a_provably_older_epoch_is_stale(const char* program) {
    capture::clear();
    std::string error;
    auto runtime = QwenRuntime::start(PrivateQwenLaunch{program, "serve-many", "", 0, 2000, 2000}, kPlan, 1500, error);
    check(bool(runtime), error.c_str());
    const size_t hidden = kGeometry.n_embd, top_k = kGeometry.n_expert_used;
    std::vector<float> input(hidden, 4.0f), weights(top_k, 0.25f), output(hidden, 0.0f);
    std::vector<int32_t> ids(top_k, SEABRAID_SKIP_EXPERT_ID);
    ids[0] = kExpert;
    std::fill(weights.begin() + 1, weights.end(), 0.0f);   // the session refuses a skipped pick with a weight
    const seabraid_batch_v1 batch{SEABRAID_ABI_VERSION, 1, input.data(), hidden, ids.data(), top_k, weights.data(), top_k};
    uint64_t ticket = 0, other = 0;
    check(runtime->accepted_epoch() == 0, "a new session has accepted no epoch");
    check(runtime->submit(kLayer, batch, SEABRAID_SOURCE_TEXT, 2, ticket) == SEABRAID_OK, "epoch 2 is accepted");
    check(runtime->accepted_epoch() == 2, "an accepted submit raises the session's mark");
    // A busy-state refusal: a ticket is outstanding, epoch 3 is NEWER than the accepted 2, so it is not staleness.
    const int32_t busy = runtime->submit(kLayer, batch, SEABRAID_SOURCE_TEXT, 3, other);
    check(busy == SEABRAID_BUSY_OR_STATE, "a second submit with a ticket outstanding is refused as a state error");
    check(runtime->accepted_epoch() == 2, "a refused submit does not raise the mark");
    check(!QwenDecodeTier::is_stale_refusal(busy, 3, runtime->accepted_epoch()), "a busy-state refusal at a newer epoch is not stale");
    check(runtime->collect(ticket, 2000, output.data(), output.size()) == SEABRAID_OK, "the epoch-2 layer collects");
    const int32_t stale = runtime->submit(kLayer, batch, SEABRAID_SOURCE_TEXT, 1, other);
    check(stale == SEABRAID_BUSY_OR_STATE, "epoch 1 below the accepted 2 is refused by the session");
    check(runtime->accepted_epoch() == 2, "a stale refusal leaves the mark where the session has it");
    check(QwenDecodeTier::is_stale_refusal(stale, 1, runtime->accepted_epoch()), "and the runtime's mark proves it stale");
    // Statuses that look the same to the tier: not-prepared, busy and closing are BUSY_OR_STATE at any epoch.
    check(!QwenDecodeTier::is_stale_refusal(SEABRAID_BUSY_OR_STATE, 3, 2), "not-prepared at a newer epoch is not stale");
    check(!QwenDecodeTier::is_stale_refusal(SEABRAID_BUSY_OR_STATE, 2, 2), "an epoch equal to the mark is not stale");
    check(!QwenDecodeTier::is_stale_refusal(SEABRAID_BUSY_OR_STATE, 3, 0), "nothing accepted yet: nothing can be older");
    check(!QwenDecodeTier::is_stale_refusal(SEABRAID_BUSY_OR_STATE, 0, 2), "an unversioned batch is the other branch");
    check(!QwenDecodeTier::is_stale_refusal(SEABRAID_INVALID, 1, 2), "INVALID is never stale");
    runtime.reset();
    reaped();
}

void compact_rows_option_opens_a_session(const char* program) {
    capture::clear();
    PrivateQwenLaunch launch{program, "serve", "", 0, 2000, 2000};
    launch.compact_rows = true;
    std::string error;
    auto tier = QwenDecodeTier::start(launch, kPlan, 1500, error);
    check(bool(tier), error.c_str());
    Window w;
    const float* rows = nullptr;
    check(w.begin(*tier, error) && tier->owns(0), "the compacting session accepts the window");
    check(tier->finish(rows, error) && rows && rows[0] == 5.0f, "the one-row batch round-trips unchanged");
    tier.reset();
    reaped();
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
    ScratchFile trusted;
    trusted.copy_of(argv[0]);
    stale_epoch_keeps_the_layer_local(trusted.path.c_str());
    only_a_provably_older_epoch_is_stale(trusted.path.c_str());
    compact_rows_option_opens_a_session(trusted.path.c_str());
    std::puts("Strata stale-epoch and compact-rows tier fixture passed");
}
