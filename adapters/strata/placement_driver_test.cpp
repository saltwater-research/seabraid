// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// PlacementDriver fixture: a scripted PlacementHost and scripted RungIO cards on a 1 layer x 3 expert world.
// Owners start {0,1,2}. Heat {1,100,10} makes the planner swap expert 1 (helper) up with expert 0 (lead).
// CHECKER for R2 (publish only after verify), R4 (an uncertain helper op quarantines), R5 (adopt never moves
// rung 1) and R8 (one record per failure). Every failure test counts the records the branch wrote.
#include "placement_driver.hpp"

#include "diag_capture.hpp"

#include "strata/core/ownership_map.hpp"

#include <dlfcn.h>
#include <unistd.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>

#define CHECK(expression) do { \
    if (!(expression)) { \
        std::fprintf(stderr, "line %d: %s failed\n%s", __LINE__, #expression, capture::dump().c_str()); \
        std::abort(); \
    } \
} while (false)

namespace capture = seabraid::strata_adapter::capture;
using seabraid::strata_adapter::ExecutorView;
using seabraid::strata_adapter::PlacementDriver;
using seabraid::strata_adapter::PlacementDriverConfig;
using seabraid::strata_adapter::PlacementHost;
using strata::core::IoStatus;
using State = PlacementDriver::State;

// WHY an interposed symbol: a poisoned handle needs a Rust panic, which no input produces on purpose. The test
// executable's definition wins at link time, so the driver's call lands here and forwards to the real library.
static bool g_poison_plan = false;
extern "C" int32_t seabraid_placement_plan(seabraid_placement* placement, uint64_t epoch, const double* heat,
                                           size_t count, seabraid_placement_ticket_v1* out) {
    using Fn = int32_t (*)(seabraid_placement*, uint64_t, const double*, size_t, seabraid_placement_ticket_v1*);
    static const Fn real = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "seabraid_placement_plan"));
    return g_poison_plan ? SEABRAID_INTERNAL : real(placement, epoch, heat, count, out);
}

namespace {

constexpr size_t kExperts = 3, kBytes = 4;
const std::vector<float> kHot{1, 100, 10}, kFlat{5, 5, 5};

std::vector<uint8_t> blob(int32_t expert) { return {uint8_t(expert), uint8_t(expert + 1), uint8_t(expert + 2), 9}; }

class Card : public strata::core::RungIO {
public:
    explicit Card(int expert) { resident[expert] = 1; }
    IoStatus stage(int32_t layer, int32_t expert, std::vector<uint8_t>& out, std::string&) override {
        CHECK(layer == 0 && resident[expert]);
        out = blob(expert);
        return IoStatus::kOk;
    }
    IoStatus replace(int32_t layer, int32_t old_e, int32_t new_e, const std::vector<uint8_t>& bytes,
                     std::string&) override {
        CHECK(layer == 0 && resident[old_e] && bytes == blob(new_e));
        if (timeout) return IoStatus::kTimeout;   // completion unknown: residency is left as it was
        resident[old_e] = 0;
        resident[new_e] = 1;
        return IoStatus::kOk;
    }
    std::array<uint8_t, kExperts> resident{};
    bool timeout = false;
};

struct Host : PlacementHost {
    Host(Card& lead, Card& helper, const ExecutorView& view) : lead_(lead), helper_(helper), view_(view) {}
    bool drain(std::string&) noexcept override { ++drains; return drain_ok; }
    void lead_cells(std::vector<uint8_t>& out) noexcept override {
        out.assign(lead_.resident.begin(), lead_.resident.end());
        if (++lead_calls == lead_lie_call) out[2] = !out[2];
    }
    bool helper_cells(uint8_t rung, std::vector<uint8_t>& out) noexcept override {
        CHECK(rung == 1);
        probe_epochs.push_back(view_.epoch());   // what the view shows while the round is still unpublished
        out.assign(helper_.resident.begin(), helper_.resident.end());
        if (static_cast<int>(probe_epochs.size()) == helper_lie_call) out[0] = 0;
        return alive;
    }
    bool helper_alive(uint8_t rung) noexcept override {
        CHECK(rung == 1);
        ++alive_calls;
        return alive;
    }
    void after_commit(const seabraid_placement_outcome_v1& o) noexcept override {
        followed.push_back({o.layer, o.promote, o.displace});
        uppers.push_back(o.upper_rung);
    }
    Card &lead_, &helper_;
    const ExecutorView& view_;
    bool drain_ok = true, alive = true;
    int drains = 0, lead_calls = 0, lead_lie_call = 0, helper_lie_call = 0, alive_calls = 0;
    std::vector<uint64_t> probe_epochs;
    std::vector<std::array<uint32_t, 3>> followed;
    std::vector<uint8_t> uppers;
};

std::string find_line(const char* needle) {
    std::lock_guard<std::mutex> lock(capture::guard());
    for (const auto& record : capture::records())
        if (record.message.find(needle) != std::string::npos) return record.message;
    return {};
}

size_t loud_records() {
    std::lock_guard<std::mutex> lock(capture::guard());
    size_t found = 0;
    for (const auto& record : capture::records()) found += record.level >= seabraid::strata_adapter::diag::kWarn;
    return found;
}

// A v2 policy with v1 meaning: {fallback 0, hops 1}.
seabraid_placement_policy_v2 policy(uint16_t moves, uint8_t ladder) {
    seabraid_placement_policy_v2 p{};
    p.abi_version = SEABRAID_PLACEMENT_POLICY_V2_VERSION;
    p.struct_size = sizeof p;
    p.max_moves_per_round = moves;
    p.ladder_boundaries = ladder;
    p.max_hops_per_round = 1;
    p.promote_margin = 1.0;
    p.max_helper_layer_share = 1.0;
    return p;
}

struct Rig {
    explicit Rig(uint32_t every = 1, std::string qual = {}) {
        capture::clear();
        PlacementDriverConfig config;
        config.every_windows = every;
        config.policy = policy(4, 1);
        config.qual_heat_path = std::move(qual);
        driver = PlacementDriver::open(1, kExperts, {0, 1, 2}, {&lead, &helper}, nullptr, {kBytes}, config, view, err);
        CHECK(driver && err.empty());
        CHECK(view.epoch() == 1 && view.routes_to(0, 1, 1));
    }
    bool tick(const std::vector<float>& heat, bool quiet = true) {
        return driver->tick(heat.data(), heat.size(), quiet, host);
    }
    bool routes(int expert, uint8_t rung) const { return view.routes_to(0, expert, rung); }
    ExecutorView view;
    Card lead{0}, helper{1};
    Host host{lead, helper, view};
    std::unique_ptr<PlacementDriver> driver;
    std::string err;
};

void committed_round_publishes_and_logs_every_field() {
    Rig rig;
    CHECK(rig.tick(kHot));
    CHECK(rig.driver->state() == State::kLive && rig.view.epoch() == 2);
    CHECK(rig.routes(1, 0) && rig.routes(0, 1) && rig.routes(2, 2));
    CHECK(rig.host.drains == 1 && rig.host.followed.size() == 1);
    CHECK((rig.host.followed[0] == std::array<uint32_t, 3>{0, 1, 0}));   // layer, promoted, displaced
    CHECK(rig.host.uppers == std::vector<uint8_t>{0});                    // the whole outcome reaches the host
    capture::require("placement move 0/1 1->0 src=card bytes=4 verified=1", 1);
    CHECK(loud_records() == 0);
    capture::require("placement round=1 ", 1);
    const std::string line = find_line("placement round=1 ");
    for (const char* field : {"epoch=2 ", "planned=1 ", "committed=1 ", "rejected=0 ", "restored=0 ", "qmask=0x0 ",
                              "adopted=0 ", "spares=0 ", "defer=0 ", "helper_share=0.", "table_hash=", "plan_us=",
                              "fence_us=", "stage_us=", "replace_us=", "publish_us=", "follow_us="})
        CHECK(line.find(field) != std::string::npos);
    CHECK(line.size() < 255);   // diag cuts at 255 characters; the last field must have survived
    // Expert 0 (heat 1) now sits on the helper and 100 on the lead, so the helper carries 1 of 111 of the heat.
    CHECK(line.find("helper_share=0.009") != std::string::npos);
}

void zero_move_round_publishes_nothing() {
    Rig rig;
    CHECK(rig.tick(kHot));
    const uint64_t hash = rig.view.table_hash();
    capture::clear();
    CHECK(rig.tick(kHot));   // the planner finds nothing hotter than what the lead holds
    CHECK(rig.view.epoch() == 2 && rig.view.table_hash() == hash);
    CHECK(rig.host.drains == 1 && rig.host.followed.size() == 1);
    capture::require("placement round", 0);
    CHECK(loud_records() == 0);
}

void cadence_waits_then_defers_while_busy() {
    Rig rig(3);
    CHECK(rig.tick(kHot) && rig.tick(kHot));
    CHECK(rig.host.drains == 0 && rig.view.epoch() == 1);   // windows 1 and 2: not due
    CHECK(rig.tick(kHot, false));                           // window 3: due but the engine is busy
    CHECK(rig.host.drains == 0 && rig.view.epoch() == 1);
    CHECK(rig.tick(kHot, true));                            // window 4: first quiet window
    CHECK(rig.host.drains == 1 && rig.view.epoch() == 2);
    CHECK(find_line("placement round=1 ").find("defer=1 ") != std::string::npos);
}

void adopt_moves_the_lead_and_publishes() {
    Rig rig;
    rig.lead.resident = {0, 0, 1};   // Strata's lead<->RAM adapt: the lead now holds expert 2, not expert 0
    CHECK(rig.tick(kFlat));
    CHECK(rig.driver->state() == State::kLive && rig.view.epoch() == 2);
    CHECK(rig.routes(2, 0) && rig.routes(0, 2) && rig.routes(1, 1));
    CHECK(rig.host.drains == 0 && loud_records() == 0);
    CHECK(find_line("placement round=1 ").find("adopted=2 ") != std::string::npos);
}

void r5_adopt_never_moves_rung_one() {
    Rig rig;
    rig.lead.resident = {0, 1, 0};   // the lead grabbed expert 1, which the helper holds
    CHECK(rig.tick(kFlat));
    CHECK(rig.driver->state() == State::kDegraded && !rig.view.rung_open(1));
    CHECK(rig.routes(0, 0) && !rig.routes(1, 1) && rig.routes(1, 2));
    capture::require("placement degraded: adopt would move a helper cell", 1, SEABRAID_BUSY_OR_STATE);
    CHECK(loud_records() == 1);
    const int asked = rig.host.alive_calls;
    CHECK(rig.tick(kHot));   // degraded still plans, but rung 1 is closed so nothing is movable; it is not asked again
    CHECK(rig.host.drains == 0 && loud_records() == 1 && rig.host.alive_calls == asked);
}

void r2_helper_disagreement_never_reaches_the_view() {
    Rig rig;
    rig.host.helper_lie_call = 1;   // liveness asks helper_alive; call 1 of helper_cells is verify, which finds expert 0 missing
    CHECK(rig.tick(kHot));
    CHECK((rig.host.probe_epochs == std::vector<uint64_t>{1}));   // the new table was not live during verify
    CHECK(rig.driver->state() == State::kDegraded && !rig.view.rung_open(1));
    CHECK(rig.routes(0, 2) && rig.routes(1, 0));   // rung 1's cell fell back; the committed lead cell stays
    CHECK(rig.view.epoch() == 3);                  // commit was 2; the quarantine made it 3, and only that was published
    capture::require("placement degraded: helper does not hold a cell the table gives it", 1, SEABRAID_BUSY_OR_STATE);
    CHECK(loud_records() == 1);
    CHECK(rig.host.followed.size() == 1); capture::require("placement move 0/1 1->0 src=card bytes=4 verified=1", 1); capture::require("placement evict 0/0 0->1 dst=card", 1);   // committed before the degrade: still followed and logged
}

void r4_helper_timeout_closes_the_rung() {
    Rig rig;
    rig.helper.timeout = true;
    CHECK(rig.tick(kHot));
    CHECK(rig.driver->state() == State::kDegraded && !rig.view.rung_open(1));
    CHECK(rig.routes(0, 0) && rig.routes(1, 2) && rig.routes(2, 2) && rig.view.epoch() == 2);
    capture::require("placement replace failed", 1);
    capture::require("placement degraded: a helper rung was closed by the round", 1, SEABRAID_BUSY_OR_STATE);
    CHECK(loud_records() == 2);
    CHECK(find_line("placement round=1 ").find("qmask=0x2 ") != std::string::npos);
    CHECK(rig.lead.resident[0] == 1 && rig.lead.resident[1] == 0);   // the lead was restored from its staged blob
}

void lead_disagreement_stops_the_engine() {
    Rig rig;
    rig.host.lead_lie_call = 2;   // call 1 is adopt; call 2 is verify
    CHECK(!rig.tick(kHot));
    CHECK(rig.view.epoch() == 1);   // nothing was published
    capture::require("placement lead disagrees with the table", 1, SEABRAID_INTERNAL);
    CHECK(loud_records() == 1);
}

void abi_error_disables_the_helper() {
    Rig rig;
    g_poison_plan = true;
    CHECK(rig.tick(kHot));
    g_poison_plan = false;
    CHECK(rig.driver->state() == State::kDead && !rig.view.rung_open(1));
    CHECK(rig.routes(0, 0) && rig.routes(1, 2) && rig.routes(2, 2));   // the lead keeps its cell, helper cells fall back
    capture::require("placement driver dead: plan: SEABRAID_INTERNAL", 1, SEABRAID_INTERNAL);
    CHECK(loud_records() == 1);
    const int calls = rig.host.lead_calls;
    CHECK(rig.tick(kHot) && rig.host.lead_calls == calls && loud_records() == 1);   // dead: serving continues, driver idle
}

void refused_drain_skips_and_warns_once_per_sixteen() {
    Rig rig;
    rig.host.drain_ok = false;
    for (int i = 0; i < 17; ++i) CHECK(rig.tick(kHot));
    CHECK(rig.view.epoch() == 1 && rig.driver->state() == State::kLive);
    capture::require("drain refused", 2);   // the 1st and the 17th refusal
    CHECK(loud_records() == 2);
    rig.host.drain_ok = true;
    CHECK(rig.tick(kHot) && rig.view.epoch() == 2);   // the next quiet round recovers
}

void a_lost_helper_is_caught_by_the_liveness_probe() {
    Rig rig;
    rig.host.alive = false;
    CHECK(rig.tick(kHot));
    CHECK(rig.driver->state() == State::kDegraded && !rig.view.rung_open(1) && rig.host.drains == 0);
    CHECK(rig.host.probe_epochs.empty());   // the cheap probe found it; no table was copied
    capture::require("placement degraded: helper is gone", 1, SEABRAID_BUSY_OR_STATE);
    CHECK(loud_records() == 1);
}

void a_bad_shape_is_refused_once() {
    capture::clear();
    ExecutorView view;
    Card lead(0), helper(1);
    PlacementDriverConfig config;
    std::string err;
    CHECK(!PlacementDriver::open(1, kExperts, {0, 1}, {&lead, &helper}, nullptr, {kBytes}, config, view, err) && !err.empty());
    capture::require("placement open refused: bad layout", 1);
    err.clear();
    config.every_windows = 0;
    CHECK(!PlacementDriver::open(1, kExperts, {0, 1, 2}, {&lead, &helper}, nullptr, {kBytes}, config, view, err));
    capture::require("placement open refused", 2);
    config.every_windows = 1;
    config.policy = policy(4, 1);
    config.policy.fallback_boundary = 1;   // a fallback boundary with no fallback rung bound
    CHECK(!PlacementDriver::open(1, kExperts, {0, 1, 2}, {&lead, &helper}, nullptr, {kBytes}, config, view, err));
    capture::require("placement open refused: fallback_boundary needs a fallback rung", 1);
    CHECK(loud_records() == 3 && view.epoch() == 0);
}

// The operator's view, end to end: the real forwarder, the real Seabraid threshold and the real fd 2. WHY a temp file
// on fd 2: the default sink writes there, and the capturing sink the other cases use would hide a threshold bug.
void the_round_record_reaches_stderr_exactly_once() {
    namespace diag = seabraid::strata_adapter::diag;
    CHECK(seabraid_set_log_sink(nullptr, nullptr, SEABRAID_LOG_WARN) == SEABRAID_OK);   // the library default
    diag::set_sink(nullptr);   // unset, so open() installs the forwarder into seabraid_log
    diag::set_min_level(diag::kInfo);
    std::FILE* file = std::tmpfile();
    CHECK(file != nullptr);
    std::fflush(stderr);
    const int saved = dup(2);
    CHECK(saved >= 0 && dup2(fileno(file), 2) >= 0);
    seabraid_log(SEABRAID_LOG_INFO, 0, "probe-before-open");   // INFO noise with placement off stays hidden
    {
        Rig rig;
        rig.tick(kHot);
        seabraid_log(SEABRAID_LOG_WARN, 0, "probe-warn");   // WARN is not lost by the lower threshold
    }
    dup2(saved, 2);
    close(saved);
    std::string text;
    char chunk[512];
    std::rewind(file);
    for (size_t n; (n = std::fread(chunk, 1, sizeof chunk, file)) > 0;) text.append(chunk, n);
    std::fclose(file);
    const auto times = [&](const char* needle) {
        size_t found = 0;
        for (size_t at = text.find(needle); at != std::string::npos; at = text.find(needle, at + 1)) ++found;
        return found;
    };
    capture::start();   // back to the capturing sink for the other cases
    CHECK(times("seabraid: INFO: placement round=1 ") == 1 && times("placement round=") == 1);
    CHECK(times("seabraid: WARN: probe-warn") == 1 && times("probe-before-open") == 0);
}

// SEABRAID_PLACEMENT_QUAL: the bias reaches the planner (flat real heat, the walk file makes expert 1 hot) and the round
// says so; a malformed file refuses the open.
void qual_heat_steers_placement_only() {
    char walk[] = "/tmp/seabraid_qual_XXXXXX";
    const int fd = mkstemp(walk);
    const char text[] = "# walk expert 1 up from round 1\n1 0 1 1000\n";
    CHECK(fd >= 0 && write(fd, text, sizeof text - 1) == static_cast<ssize_t>(sizeof text - 1) && close(fd) == 0);
    {
        Rig rig(1, walk);
        capture::require("QUALIFICATION heat bias active (1 entries)", 1);
        CHECK(rig.tick(kFlat));
        CHECK(rig.routes(1, 0) && rig.routes(0, 1));   // flat usage, but the bias moved expert 1 to the lead
        capture::require("placement round=1 skipped=0 qual=1", 1);
    }
    unlink(walk);
    capture::clear();
    ExecutorView view;
    Card lead(0), helper(1);
    PlacementDriverConfig config;
    config.qual_heat_path = "/nonexistent/qual-walk";   // grammar cases: placement_qual_test
    std::string err;
    CHECK(!PlacementDriver::open(1, kExperts, {0, 1, 2}, {&lead, &helper}, nullptr, {kBytes}, config, view, err));
    capture::require("placement open refused: the qualification heat file is unreadable or malformed", 1);
}

void ram_follows_only_the_lead_boundary() {
    using seabraid::strata_adapter::ram_follows;
    seabraid_placement_outcome_v1 o{};
    o.kind = SEABRAID_PLACEMENT_COMMITTED;
    CHECK(ram_follows(o));
    o.upper_rung = 1;   // helper 1 <-> helper 2 leaves the lead's complement alone
    CHECK(!ram_follows(o));
}

}  // namespace

int main() {
    capture::start();
    the_round_record_reaches_stderr_exactly_once();
    committed_round_publishes_and_logs_every_field();
    zero_move_round_publishes_nothing();
    cadence_waits_then_defers_while_busy();
    adopt_moves_the_lead_and_publishes();
    r5_adopt_never_moves_rung_one();
    r2_helper_disagreement_never_reaches_the_view();
    r4_helper_timeout_closes_the_rung();
    lead_disagreement_stops_the_engine();
    abi_error_disables_the_helper();
    refused_drain_skips_and_warns_once_per_sixteen();
    a_lost_helper_is_caught_by_the_liveness_probe();
    a_bad_shape_is_refused_once();
    qual_heat_steers_placement_only();
    ram_follows_only_the_lead_boundary();
    std::puts("Seabraid Strata placement driver fixture passed");
}
