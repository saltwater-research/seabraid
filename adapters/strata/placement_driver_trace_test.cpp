// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// The driver's trace wiring: a scripted 1 layer x 3 expert world (owners {0,1,2}, heat {1,100,10} swaps expert 1 up with
// expert 0) run with the trace on and off. CHECKER for placement_driver_trace.cpp, log_moves/log_failures and the
// timing hooks in placement.cpp: every outcome kind writes its record and its diag line, the existing move line stays
// byte-identical, a trace that cannot be written changes nothing about serving.
#include "placement_driver.hpp"

#include "diag_capture.hpp"

#include "strata/core/ownership_map.hpp"

#include <unistd.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

#define CHECK(expression) do { \
    if (!(expression)) { \
        std::fprintf(stderr, "line %d: %s failed\n%s", __LINE__, #expression, capture::dump().c_str()); \
        std::abort(); \
    } \
} while (false)

namespace capture = seabraid::strata_adapter::capture;
namespace trace = seabraid::strata_adapter::trace;
using seabraid::strata_adapter::ExecutorView;
using seabraid::strata_adapter::MoveContext;
using seabraid::strata_adapter::PlacementDriver;
using seabraid::strata_adapter::PlacementDriverConfig;
using seabraid::strata_adapter::PlacementHost;
using strata::core::IoStatus;

namespace {

constexpr size_t kExperts = 3, kBytes = 4;
const std::vector<float> kHot{1, 100, 10};

std::vector<uint8_t> blob(int32_t expert) { return {uint8_t(expert), uint8_t(expert + 1), uint8_t(expert + 2), 9}; }

class Card : public strata::core::RungIO {
public:
    explicit Card(int expert) { resident[expert] = 1; }
    IoStatus stage(int32_t, int32_t expert, std::vector<uint8_t>& out, std::string&) override {
        out = blob(expert);
        return stage_status;
    }
    IoStatus replace(int32_t, int32_t old_e, int32_t new_e, const std::vector<uint8_t>&, std::string&) override {
        if (replace_status != IoStatus::kOk) return replace_status;
        resident[old_e] = 0;
        resident[new_e] = 1;
        return IoStatus::kOk;
    }
    std::array<uint8_t, kExperts> resident{};
    IoStatus stage_status = IoStatus::kOk, replace_status = IoStatus::kOk;
};

struct Host : PlacementHost {
    Host(Card& lead, Card& helper) : lead_(lead), helper_(helper) {}
    bool drain(std::string&) noexcept override { return true; }
    void lead_cells(std::vector<uint8_t>& out) noexcept override { out.assign(lead_.resident.begin(), lead_.resident.end()); }
    bool helper_cells(uint8_t, std::vector<uint8_t>& out) noexcept override {
        out.assign(helper_.resident.begin(), helper_.resident.end());
        return alive;
    }
    bool helper_alive(uint8_t) noexcept override { return alive; }
    void after_commit(const seabraid_placement_outcome_v1&) noexcept override {}
    Card &lead_, &helper_;
    bool alive = true;
};

seabraid_placement_policy_v2 policy() {
    seabraid_placement_policy_v2 p{};
    p.abi_version = SEABRAID_PLACEMENT_POLICY_V2_VERSION;
    p.struct_size = sizeof p;
    p.max_moves_per_round = 4;
    p.ladder_boundaries = 1;
    p.max_hops_per_round = 1;
    p.promote_margin = 1.0;
    p.max_helper_layer_share = 1.0;
    return p;
}

struct Rig {
    explicit Rig(PlacementDriverConfig config = {}) {
        capture::clear();
        config.every_windows = 1;
        config.policy = policy();
        driver = PlacementDriver::open(1, kExperts, {0, 1, 2}, {&lead, &helper}, nullptr, {kBytes}, config, view, err);
        CHECK(driver && err.empty());
    }
    bool tick() { return driver->tick(kHot.data(), kHot.size(), true, host); }
    ExecutorView view;
    Card lead{0}, helper{1};
    Host host{lead, helper};
    std::unique_ptr<PlacementDriver> driver;
    std::string err;
};

std::string temp_path() {
    char path[] = "/tmp/seabraid_drvtrace_XXXXXX";
    const int fd = mkstemp(path);
    CHECK(fd >= 0);
    close(fd);
    return path;
}

std::string slurp(const std::string& path) {
    std::ifstream in(path);
    std::stringstream text;
    text << in.rdbuf();
    return text.str();
}

bool has(const std::string& text, const char* needle) { return text.find(needle) != std::string::npos; }

size_t count_of(const std::string& text, const char* needle) {
    size_t n = 0;
    for (size_t at = text.find(needle); at != std::string::npos; at = text.find(needle, at + 1)) ++n;
    return n;
}

void committed_round_writes_move_round_and_helper_records() {
    const std::string path = temp_path();
    trace::configure(path.c_str(), 0);
    Rig rig;
    CHECK(rig.tick());
    // The existing move line, byte for byte, then the new evict line for the expert pushed down.
    capture::require("placement move 0/1 1->0 src=card bytes=4 verified=1", 1);
    capture::require("placement evict 0/0 0->1 dst=card", 1);
    const std::string text = slurp(path);
    CHECK(has(text, "\"rec\":\"helper\",\"rung\":1,\"event\":\"up\""));
    CHECK(count_of(text, "\"rec\":\"move\"") == 1);
    for (const char* field : {"\"kind\":\"committed\"", "\"up\":{\"expert\":1,\"from_rung\":1,\"to_rung\":0}",
                              "\"down\":{\"expert\":0,\"from_rung\":0,\"to_rung\":1}", "\"src\":\"card\"", "\"bytes\":4",
                              "\"timed\":true", "\"verified\":1", "\"epoch\":2",
                              "\"rec\":\"round\",\"round\":1,\"epoch\":2,\"planned\":1,\"committed\":1,\"rejected\":0",
                              "\"fallback\":null"})
        CHECK(has(text, field));
    trace::close_trace();
    unlink(path.c_str());
}

void trace_off_changes_no_line_and_writes_no_file() {
    const std::string path = temp_path();
    unlink(path.c_str());
    trace::configure(nullptr, 0);
    Rig rig;
    CHECK(rig.tick());
    capture::require("placement move 0/1 1->0 src=card bytes=4 verified=1", 1);
    capture::require("placement evict 0/0 0->1 dst=card", 1);
    CHECK(access(path.c_str(), F_OK) != 0);
}

void a_failed_stage_is_logged_and_traced_as_rejected() {
    const std::string path = temp_path();
    trace::configure(path.c_str(), 0);
    Rig rig;
    rig.helper.stage_status = IoStatus::kFailed;
    CHECK(rig.tick());
    capture::require("placement fail 0/1 1->0 kind=rejected stage=stage result=failed", 1);
    capture::require("placement move", 0);
    const std::string text = slurp(path);
    CHECK(has(text, "\"kind\":\"rejected\"") && has(text, "\"issue\":{\"stage\":\"stage\",\"result\":\"failed\",\"rung\":1}"));
    CHECK(has(text, "\"verified\":0") && has(text, "\"committed\":0,\"rejected\":1"));
    trace::close_trace();
    unlink(path.c_str());
}

void an_uncertain_write_is_logged_and_traced_as_quarantined() {
    const std::string path = temp_path();
    trace::configure(path.c_str(), 0);
    Rig rig;
    rig.helper.replace_status = IoStatus::kTimeout;
    CHECK(rig.tick());
    capture::require("placement fail 0/1 1->0 kind=quarantined stage=replace result=unacknowledged", 1);
    const std::string text = slurp(path);
    CHECK(has(text, "\"kind\":\"quarantined\"") && has(text, "\"event\":\"closed\"") && has(text, "\"quarantined\":1"));
    trace::close_trace();
    unlink(path.c_str());
}

// RESTORED and SKIPPED cannot be forced from this rig (they need a failed undo and a multi-hop climb), so the
// pure logging functions take hand-built outcomes: the diag line and the record come from the same call.
void restored_and_skipped_outcomes_have_lines_and_records() {
    const std::string path = temp_path();
    trace::configure(path.c_str(), 0);
    capture::start();
    seabraid_placement_outcome_v1 restored{}, skipped{};
    restored.kind = SEABRAID_PLACEMENT_RESTORED;
    restored.layer = 3;
    restored.promote = 5;
    restored.displace = 6;
    restored.upper_rung = 1;
    restored.initial = {SEABRAID_PLACEMENT_REPLACE, SEABRAID_PLACEMENT_FAILED, 2};
    restored.restore = {SEABRAID_PLACEMENT_RESTORE, SEABRAID_PLACEMENT_FAILED, 1};
    skipped.kind = SEABRAID_PLACEMENT_SKIPPED;
    skipped.layer = 3;
    skipped.promote = 5;
    skipped.upper_rung = 0;
    seabraid::strata_adapter::log_failures({restored, skipped}, {16, 16, 16, 16}, 2, MoveContext{4, 8, nullptr});
    capture::require("placement fail 3/5 2->1 kind=restored stage=replace result=failed restore=restore/failed", 1);
    capture::require("placement fail 3/5 1->0 kind=skipped stage=none result=none restore=none/none", 1);
    const std::string text = slurp(path);
    CHECK(count_of(text, "\"rec\":\"move\"") == 2 && has(text, "\"round\":4,\"epoch\":8"));
    CHECK(has(text, "\"restore_issue\":{\"stage\":\"restore\",\"result\":\"failed\",\"rung\":1}"));
    CHECK(has(text, "\"timed\":false"));   // no timing was handed in: untimed, not a made-up zero
    trace::close_trace();
    unlink(path.c_str());
}

// verified=1 only when every write of the hop was read back: the lead always, a helper only with verify_writes, the
// fallback's own copy never.
void verified_tells_the_truth() {
    const std::string path = temp_path();
    trace::configure(path.c_str(), 0);
    capture::start();
    seabraid_placement_outcome_v1 lead_hop{}, helper_hop{}, fallback_hop{};
    for (auto* o : {&lead_hop, &helper_hop, &fallback_hop}) {
        o->kind = SEABRAID_PLACEMENT_COMMITTED;
        o->layer = 0;
    }
    lead_hop.promote = 1;
    lead_hop.upper_rung = 0;       // rung 1 -> 0
    helper_hop.promote = 2;
    helper_hop.upper_rung = 1;     // rung 2 -> 1, fallback is rung 3
    fallback_hop.promote = 3;
    fallback_hop.upper_rung = 1;   // rung 2 -> 1 where rung 2 IS the fallback
    seabraid::strata_adapter::log_moves({lead_hop, helper_hop}, {8}, 3, MoveContext{1, 2, nullptr, true});
    seabraid::strata_adapter::log_moves({fallback_hop}, {8}, 2, MoveContext{1, 2, nullptr, true});
    capture::require("placement move 0/1 1->0 src=card bytes=8 verified=1", 1);
    capture::require("placement move 0/2 2->1 src=card bytes=8 verified=1", 1);
    capture::require("placement move 0/3 2->1 src=fallback bytes=8 verified=0", 1);
    seabraid::strata_adapter::log_moves({lead_hop, helper_hop}, {8}, 3, MoveContext{2, 3, nullptr, false});
    capture::require("placement move 0/1 1->0 src=card bytes=8 verified=0", 1);   // a helper write that was not read back
    capture::require("placement move 0/2 2->1 src=card bytes=8 verified=0", 1);
    const std::string text = slurp(path);
    CHECK(count_of(text, "\"verified\":1") == 2 && count_of(text, "\"verified\":0") == 3);
    trace::close_trace();
    unlink(path.c_str());
}

void fallback_counters_become_round_deltas() {
    const std::string path = temp_path();
    trace::configure(path.c_str(), 0);
    PlacementDriverConfig config;
    int64_t ram = 10, file = 100;
    config.fallback_counters = [&](int64_t& r, int64_t& f) { r = ram; f = file; ram += 3; file += 2; };
    Rig rig(config);
    CHECK(rig.tick());
    CHECK(has(slurp(path), "\"fallback\":{\"ram_hits\":3,\"file_reads\":2}"));
    trace::close_trace();
    unlink(path.c_str());
}

void a_lost_helper_is_traced_with_its_reason_and_the_next_window() {
    const std::string path = temp_path();
    trace::configure(path.c_str(), 0);
    Rig rig;
    CHECK(rig.tick());
    rig.host.alive = false;
    CHECK(rig.tick());
    CHECK(rig.tick());
    const std::string text = slurp(path);
    CHECK(has(text, "\"event\":\"closed\",\"reason\":\"helper is gone\""));
    CHECK(count_of(text, "\"event\":\"served_after_loss\"") == 1);
    trace::close_trace();
    unlink(path.c_str());
}

void an_unwritable_trace_never_changes_serving() {
    capture::start();
    trace::configure("/dev/full", 0);
    Rig rig;
    CHECK(rig.tick());
    CHECK(rig.driver->state() == PlacementDriver::State::kLive && rig.view.epoch() == 2 && rig.view.routes_to(0, 1, 0));
    capture::require("placement move 0/1 1->0 src=card bytes=4 verified=1", 1);
    capture::require("seabraid trace disabled", 1);
    CHECK(!trace::on());
}

}  // namespace

int main() {
    capture::start();
    committed_round_writes_move_round_and_helper_records();
    trace_off_changes_no_line_and_writes_no_file();
    a_failed_stage_is_logged_and_traced_as_rejected();
    an_uncertain_write_is_logged_and_traced_as_quarantined();
    restored_and_skipped_outcomes_have_lines_and_records();
    verified_tells_the_truth();
    fallback_counters_become_round_deltas();
    a_lost_helper_is_traced_with_its_reason_and_the_next_window();
    an_unwritable_trace_never_changes_serving();
    std::puts("placement_driver_trace_test: ok");
    return 0;
}
