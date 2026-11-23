// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// Edge cases of the decode tier and the runtime behind it, against the scripted CPU helper (no GPU, no model):
// a view replaced by "none" after a nonzero epoch, windows of exactly kWindow and kWindow + 1 rows, row
// compaction with one and with no kept row, a NaN router weight, and a helper that dies mid-compacted batch.
// OWNER: this file; the helper script ("serve-many" answers any row count) and geometry come from
//   private_qwen_script.hpp. Case ids (C1..C8) are the rows of the O2 case table in the hardening notes.
#include "private_qwen_script.hpp"

#include "executor_view.hpp"
#include "qwen_decode.hpp"
#include "runtime_qwen.hpp"

#include "diag_capture.hpp"
#include "test_support.hpp"

#include <chrono>
#include <limits>

namespace {

using seabraid::strata_adapter::ExecutorView;
using seabraid::strata_adapter::PrivateQwenLaunch;
using seabraid::strata_adapter::QwenDecodeTier;
using seabraid::strata_adapter::QwenRuntime;
namespace capture = seabraid::strata_adapter::capture;
using seabraid::strata_adapter::testing::check;
using seabraid::strata_adapter::testing::ScratchFile;

constexpr size_t kHidden = kGeometry.n_embd;
constexpr size_t kTopK = kGeometry.n_expert_used;
constexpr float kSentinel = -21.0f;

// A table at `epoch` naming rung 1 for (kLayer, kExpert), rung 2 elsewhere.
void publish_cell(ExecutorView& view, uint64_t epoch) {
    std::vector<uint8_t> exec(size_t{kGeometry.n_layers} * kGeometry.n_expert, 2);
    exec[kLayer * kGeometry.n_expert + kExpert] = 1;
    seabraid_placement_snapshot_v1 s{};
    s.abi_version = SEABRAID_ABI_VERSION;
    s.struct_size = sizeof s;
    s.epoch = epoch;
    s.layers = kGeometry.n_layers;
    s.experts = kGeometry.n_expert;
    s.fallback_rung = 2;
    check(view.publish(exec.data(), exec.size(), s), "the fixture's table must publish");
}

// `tokens` rows; the rows listed in `kept` route their first pick to the helper's resident expert and the
// rest route to expert 0, which the helper does not hold (so the tier masks them). Row t carries the value
// 4 * (t + 1) in every column, and the scripted helper answers (t + 1) + 4 for a row it receives.
struct Window {
    uint32_t tokens;
    std::vector<float> input, weights;
    std::vector<int32_t> picks, kind;
    Window(uint32_t n, const std::vector<uint32_t>& kept)
        : tokens(n), input(size_t{n} * kHidden), weights(size_t{n} * kTopK, 0.25f),
          picks(size_t{n} * kTopK, 0), kind(size_t{n} * kTopK, 0) {
        for (uint32_t t = 0; t < n; ++t) std::fill_n(input.begin() + size_t{t} * kHidden, kHidden, 4.0f * float(t + 1));
        for (uint32_t t : kept) {
            std::fill_n(picks.begin() + size_t{t} * kTopK, kTopK, int32_t(kExpert));
            kind[size_t{t} * kTopK] = -1;
        }
    }
    static Window all(uint32_t n) {
        std::vector<uint32_t> kept(n);
        for (uint32_t t = 0; t < n; ++t) kept[t] = t;
        return Window(n, kept);
    }
    bool begin(QwenDecodeTier& tier, std::string& error) {
        return tier.begin(kLayer, input.data(), picks.data(), weights.data(), tokens, kTopK, kind.data(), nullptr, error);
    }
    static float answer(uint32_t t) { return float(t + 1) + 4.0f; }
};

// True when every row t of `rows` is constant at expected[t] (first and last column checked).
bool rows_are(const float* rows, const std::vector<float>& expected) {
    if (!rows) return false;
    for (size_t t = 0; t < expected.size(); ++t)
        if (rows[t * kHidden] != expected[t] || rows[t * kHidden + kHidden - 1] != expected[t]) return false;
    return true;
}

PrivateQwenLaunch launch_for(const char* program, const char* mode, bool compact) {
    PrivateQwenLaunch launch{program, mode, "", 0, 2000, 2000};
    launch.compact_rows = compact;
    return launch;
}

std::unique_ptr<QwenDecodeTier> start_tier(const char* program, const char* mode, bool compact, uint32_t deadline_ms) {
    capture::clear();
    std::string error;
    auto tier = QwenDecodeTier::start(launch_for(program, mode, compact), kPlan, deadline_ms, error);
    check(bool(tier), error.c_str());
    return tier;
}

// begin + finish for one window: the delegated rows, or nullptr when nothing was delegated. Any engine failure
// ends the fixture.
const float* serve(QwenDecodeTier& tier, Window& w) {
    std::string error;
    const float* rows = nullptr;
    check(w.begin(tier, error), error.c_str());
    check(tier.finish(rows, error), error.c_str());
    return rows;
}

// Raw submit/collect against a QwenRuntime: `pick(t)` hands row t to the helper, every other pick is a skip.
struct RawBatch {
    uint32_t tokens;
    std::vector<float> input, weights, output;
    std::vector<int32_t> ids;
    explicit RawBatch(uint32_t n)
        : tokens(n), input(size_t{n} * kHidden, 4.0f), weights(size_t{n} * kTopK, 0.0f), output(input.size(), kSentinel),
          ids(size_t{n} * kTopK, SEABRAID_SKIP_EXPERT_ID) {}
    void pick(uint32_t t) {
        ids[size_t{t} * kTopK] = kExpert;
        weights[size_t{t} * kTopK] = 0.25f;
    }
    seabraid_batch_v1 batch() const {
        return {SEABRAID_ABI_VERSION, tokens, input.data(), input.size(), ids.data(), ids.size(), weights.data(), weights.size()};
    }
    bool output_is(float value) const {
        return std::all_of(output.begin(), output.end(), [value](float v) { return v == value; });
    }
};

std::unique_ptr<QwenRuntime> start_runtime(const char* program, const char* mode, bool compact, uint32_t deadline_ms) {
    capture::clear();
    std::string error;
    auto runtime = QwenRuntime::start(launch_for(program, mode, compact), kPlan, deadline_ms, error);
    check(bool(runtime), error.c_str());
    return runtime;
}

// C1 helper: while the replacement view claims nothing, `calls` windows stay local without an engine failure.
void stays_local(QwenDecodeTier& tier, Window& w, int calls) {
    std::string error;
    const float* rows = nullptr;
    for (int call = 0; call < calls; ++call) {
        check(w.begin(tier, error), "an unversioned batch after epoch 2 is not an engine failure");
        check(!tier.owns(0), "the layer stays local");
        check(tier.finish(rows, error) && !rows, "nothing is collected for it");
    }
}

// C1: the host replaced the view by "none" after a nonzero epoch was accepted. The session would refuse
// epoch 0 as INVALID; the tier must keep the layer local instead. A replacement view that never published
// claims no cell at all, so it is local for the older, simpler reason (and writes no record of its own).
void unversioned_after_versioned_stays_local(const char* program) {
    auto tier = start_tier(program, "serve-many", false, 1500);
    ExecutorView fresh, never_published;
    publish_cell(fresh, 2);
    tier->attach_view(&fresh, 1);
    Window w(1, {0});
    check(rows_are(serve(*tier, w), {Window::answer(0)}), "a layer routed at epoch 2 is delegated and merged");
    tier->attach_view(nullptr, 1);
    stays_local(*tier, w, 2);
    tier->attach_view(&never_published, 1);
    stays_local(*tier, w, 2);
    check(!tier->retired(), "losing the view must not retire the tier");
    capture::require("unversioned after placement epoch 2", 1, SEABRAID_BUSY_OR_STATE);   // four calls, one record
    capture::require("stale placement epoch", 0);
    tier->attach_view(&fresh, 1);   // the view returns: the helper serves again
    check(rows_are(serve(*tier, w), {Window::answer(0)}), "the restored view delegates again");
    tier.reset();
    check_reaped();
}

// C2: a window of exactly kWindow rows is served end to end; one row more is refused before anything is
// queued, the tier survives, and the next full window still works.
void window_and_one_past(const char* program) {
    auto tier = start_tier(program, "serve-many", false, 1500);
    Window full = Window::all(kWindow);
    std::vector<float> answers(kWindow);
    for (uint32_t t = 0; t < kWindow; ++t) answers[t] = Window::answer(t);
    check(rows_are(serve(*tier, full), answers), "every row of a full window carries its own reply");

    Window over(kWindow + 1, {0});
    std::string error;
    capture::clear();
    check(!over.begin(*tier, error), "kWindow + 1 rows is refused");
    check(error.find("invalid layer or window geometry") != std::string::npos, error.c_str());
    capture::require("invalid layer 1, tokens ", 1, SEABRAID_INVALID);
    check(!tier->retired() && !tier->owns(0), "a refused window queues nothing and leaves the tier alive");
    const float* rows = nullptr;
    check(tier->finish(rows, error) && !rows, "there is nothing to collect");
    check(rows_are(serve(*tier, full), answers), "the next full window is served");
    tier.reset();
    check_reaped();
}

// C3: the runtime itself refuses kWindow + 1 rows and leaves the ticket alone.
void runtime_refuses_one_row_past_the_window(const char* program) {
    auto runtime = start_runtime(program, "serve-many", false, 1500);
    RawBatch raw(kWindow + 1);
    raw.pick(0);
    uint64_t ticket = 99;
    check(runtime->submit(kLayer, raw.batch(), SEABRAID_SOURCE_TEXT, 0, ticket) == SEABRAID_INVALID, "kWindow + 1 rows is refused");
    check(ticket == 99, "a refused submit leaves the ticket untouched");
    runtime.reset();
    check_reaped();
}

// C4a: one kept row of two. The scripted helper answers 4 + x for any row it receives, even at weight zero, so a
// skipped row coming back zero proves it never crossed the wire.
void single_kept_row_scatters(QwenDecodeTier& tier) {
    Window second(2, {1});
    check(rows_are(serve(tier, second), {0.0f, Window::answer(1)}), "kept second row: its reply lands on row 1, row 0 is zero");
    Window first(2, {0});
    check(rows_are(serve(tier, first), {Window::answer(0), 0.0f}), "kept first row scatters to row 0, row 1 is zero");
}

// C4b: kept rows at both ends of a full window with a gap between.
void kept_rows_at_both_ends(QwenDecodeTier& tier) {
    Window ends(kWindow, {0, kWindow - 1});
    std::vector<float> expected(kWindow, 0.0f);
    expected[0] = Window::answer(0);
    expected[kWindow - 1] = Window::answer(kWindow - 1);
    check(rows_are(serve(tier, ends), expected), "both ends carry their own reply and the rows between are zero");
}

// C4c: no pick belongs to the helper: nothing is submitted, the wire sequence does not advance, and the next
// delegated batch is answered.
void empty_window_then_serve(QwenDecodeTier& tier) {
    Window none(kWindow, {});
    check(serve(tier, none) == nullptr && !tier.owns(0), "a window with no pick for the helper is not delegated");
    check(!tier.retired(), "an empty window leaves the tier alive");
    Window one(2, {1});
    check(rows_are(serve(tier, one), {0.0f, Window::answer(1)}), "the next batch is served after an empty window");
}

void compaction_cases(const char* program) {
    auto tier = start_tier(program, "serve-many", true, 1500);
    single_kept_row_scatters(*tier);
    kept_rows_at_both_ends(*tier);
    empty_window_then_serve(*tier);
    tier.reset();
    check_reaped();
}

// C5: a batch whose picks are all skips, submitted straight to a compacting session, never reaches the
// helper and collects as zeros that overwrite the caller's buffer.
void runtime_all_skip_batch_collects_zero(const char* program) {
    auto runtime = start_runtime(program, "serve-many", true, 1500);
    RawBatch raw(2);
    uint64_t ticket = 0;
    check(runtime->submit(kLayer, raw.batch(), SEABRAID_SOURCE_TEXT, 0, ticket) == SEABRAID_OK, "an all-skip batch is accepted");
    check(runtime->collect(ticket, 1000, raw.output.data(), raw.output.size()) == SEABRAID_OK, "and collects");
    check(raw.output_is(0.0f), "as zeros, with no helper round trip");
    runtime.reset();
    check_reaped();
}

// C6: a NaN router weight on a row the helper would serve is the engine's bug: a hard refusal, nothing queued,
// and the tier is still usable for the next finite batch.
void non_finite_router_weight_is_refused(const char* program) {
    auto tier = start_tier(program, "serve-many", true, 1500);
    Window w(2, {1});
    w.weights[kTopK] = std::numeric_limits<float>::quiet_NaN();
    std::string error;
    check(!w.begin(*tier, error), "a NaN weight on a delegated row is refused");
    check(error.find("INVALID") != std::string::npos, error.c_str());
    check(!tier->owns(kTopK) && !tier->retired(), "nothing was queued and the tier is alive");
    const float* rows = nullptr;
    check(tier->finish(rows, error) && !rows, "no ticket is outstanding");
    w.weights[kTopK] = 0.25f;
    check(rows_are(serve(*tier, w), {0.0f, Window::answer(1)}), "the next finite batch is served");
    tier.reset();
    check_reaped();
}

// C7: the helper is SIGKILLed while serving a compacted batch. The layer fails in milliseconds (not at the
// 5 s deadline), the tier retires with the reason, no row is ever published, and later windows run locally.
void helper_death_mid_compacted_batch(const char* program) {
    auto tier = start_tier(program, "die-on-request", true, 5000);
    Window w(2, {1});
    std::string error;
    check(w.begin(*tier, error) && tier->owns(kTopK), "the compacted batch is submitted");
    const auto began = std::chrono::steady_clock::now();
    const float* rows = nullptr;
    check(!tier->finish(rows, error) && !rows, "a dead helper fails the layer and publishes no row");
    check(std::chrono::steady_clock::now() - began < std::chrono::seconds(1), "in under a second, not at the deadline");
    check(error.find("SEABRAID_PROVIDER") != std::string::npos, error.c_str());
    check(tier->retired(), "the tier retires");
    capture::require("retired at layer 1: helper exited (signal 9)", 1, SEABRAID_PROVIDER);
    capture::require("tier retired: collect failed with SEABRAID_PROVIDER", 1);
    check(w.begin(*tier, error) && !tier->owns(kTopK), "later windows run locally");
    check(tier->finish(rows, error) && !rows, "and collect nothing");
    capture::require("work now runs locally", 1);
    tier.reset();
    check_reaped();
}

// C8: the same death seen through the runtime: collect fails and leaves the caller's buffer untouched.
void runtime_death_leaves_the_output_untouched(const char* program) {
    auto runtime = start_runtime(program, "die-on-request", true, 5000);
    RawBatch raw(2);
    raw.pick(1);
    uint64_t ticket = 0;
    check(runtime->submit(kLayer, raw.batch(), SEABRAID_SOURCE_TEXT, 0, ticket) == SEABRAID_OK, "the compacted batch is submitted");
    const auto began = std::chrono::steady_clock::now();
    check(runtime->collect(ticket, 4000, raw.output.data(), raw.output.size()) == SEABRAID_PROVIDER, "a dead helper is a provider failure");
    check(std::chrono::steady_clock::now() - began < std::chrono::seconds(1), "in under a second");
    check(raw.output_is(kSentinel), "no partial or zero-filled row reaches the caller's buffer");
    check(runtime->retired(), "the provider is retired");
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
    ScratchFile trusted;
    trusted.copy_of(argv[0]);
    const char* program = trusted.path.c_str();
    unversioned_after_versioned_stays_local(program);
    window_and_one_past(program);
    runtime_refuses_one_row_past_the_window(program);
    compaction_cases(program);
    runtime_all_skip_batch_collects_zero(program);
    non_finite_router_weight_is_refused(program);
    helper_death_mid_compacted_batch(program);
    runtime_death_leaves_the_output_untouched(program);
    std::puts("Strata decode-tier edge-case fixture passed");
}
