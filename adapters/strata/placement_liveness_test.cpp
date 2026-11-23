// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// F11 fixture (audit 2026-11-19): three helpers, two successive losses.  1 layer x 6 experts; the lead and helpers
// 1..3 hold experts 0..3, experts 4 and 5 are fallback.  After helper 1 is lost, rounds still plan among the survivors
// (helper 3's hot expert climbs into helper 2), helper 3 is still liveness-checked on every tick, and its later loss is
// caught on the very next tick, not never.  Before F11 the driver stopped all checks and all planning at the first loss.
#include "placement_driver.hpp"

#include "diag_capture.hpp"

#include "strata/core/ownership_map.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
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

namespace {

constexpr size_t kExperts = 6, kBytes = 4, kCards = 4;   // lead + three helpers; rung 4 is the fallback
const std::vector<float> kFlat{5, 5, 5, 5, 5, 5};
const std::vector<float> kHelper3Hot{5, 5, 1, 100, 5, 5};   // expert 3 (helper 3) far hotter than expert 2 (helper 2)

std::vector<uint8_t> blob(int32_t expert) { return {uint8_t(expert), uint8_t(expert + 1), uint8_t(expert + 2), 7}; }

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
        resident[old_e] = 0;
        resident[new_e] = 1;
        return IoStatus::kOk;
    }
    std::array<uint8_t, kExperts> resident{};
    bool alive = true;
    int alive_calls = 0;
};

struct Host : PlacementHost {
    explicit Host(std::array<Card*, kCards> cards) : cards_(cards) {}
    bool drain(std::string&) noexcept override { ++drains; return true; }
    void lead_cells(std::vector<uint8_t>& out) noexcept override {
        out.assign(cards_[0]->resident.begin(), cards_[0]->resident.end());
    }
    bool helper_cells(uint8_t rung, std::vector<uint8_t>& out) noexcept override {
        CHECK(rung >= 1 && rung < kCards);
        out.assign(cards_[rung]->resident.begin(), cards_[rung]->resident.end());
        return cards_[rung]->alive;
    }
    bool helper_alive(uint8_t rung) noexcept override {
        CHECK(rung >= 1 && rung < kCards);
        ++cards_[rung]->alive_calls;
        return cards_[rung]->alive;
    }
    void after_commit(const seabraid_placement_outcome_v1&) noexcept override { ++commits; }
    std::array<Card*, kCards> cards_;
    int drains = 0, commits = 0;
};

struct Rig {
    Rig() {
        capture::clear();
        PlacementDriverConfig config;
        config.every_windows = 1;
        config.policy = seabraid_placement_policy_v2{};   // every card boundary, no cooldown, no share cap, v1 meaning
        config.policy.abi_version = SEABRAID_PLACEMENT_POLICY_V2_VERSION;
        config.policy.struct_size = sizeof config.policy;
        config.policy.max_moves_per_round = 8;
        config.policy.max_hops_per_round = 1;
        config.policy.promote_margin = 1.0;
        config.policy.max_helper_layer_share = 1.0;
        driver = PlacementDriver::open(1, kExperts, {0, 1, 2, 3, 4, 4}, {&lead, &h1, &h2, &h3}, nullptr, {kBytes}, config,
                                       view, err);
        CHECK(driver && err.empty());
    }
    bool tick(const std::vector<float>& heat) { return driver->tick(heat.data(), heat.size(), true, host); }
    bool routes(int expert, uint8_t rung) const { return view.routes_to(0, expert, rung); }
    ExecutorView view;
    Card lead{0}, h1{1}, h2{2}, h3{3};
    Host host{{&lead, &h1, &h2, &h3}};
    std::unique_ptr<PlacementDriver> driver;
    std::string err;
};

void loss_one_closes_only_helper_one(Rig& rig) {
    CHECK(rig.tick(kFlat) && rig.driver->state() == State::kLive && rig.host.commits == 0);
    CHECK(rig.h1.alive_calls == 1 && rig.h2.alive_calls == 1);   // every helper, every tick
    CHECK(rig.h3.alive_calls == 1);
    rig.h1.alive = false;
    CHECK(rig.tick(kFlat));
    CHECK(rig.driver->state() == State::kDegraded && !rig.view.rung_open(1) && rig.routes(1, 4));
    capture::require("placement degraded: helper is gone", 1, SEABRAID_BUSY_OR_STATE);
}

void survivors_keep_planning_and_being_checked(Rig& rig) {
    CHECK(rig.tick(kHelper3Hot));   // planning continues among the survivors: helper 3's hot expert climbs to helper 2
    CHECK(rig.host.commits == 1 && rig.routes(3, 2) && rig.routes(2, 3));
    CHECK(rig.h2.resident[3] && rig.h3.resident[2]);
    CHECK(rig.driver->state() == State::kDegraded && rig.view.rung_open(2) && rig.view.rung_open(3));
    CHECK(rig.h1.alive_calls == 2);   // a closed rung is not asked again
    const int before = rig.h3.alive_calls;
    CHECK(rig.tick(kFlat) && rig.h3.alive_calls == before + 1);   // helper 3 is still checked after loss 1
}

void loss_two_is_caught_on_the_next_tick(Rig& rig) {
    rig.h3.alive = false;
    CHECK(rig.tick(kFlat));
    CHECK(!rig.view.rung_open(3) && rig.view.rung_open(2) && rig.routes(2, 4) && rig.routes(3, 2));
    CHECK(rig.driver->state() == State::kDegraded);
    capture::require("placement degraded: helper is gone", 2, SEABRAID_BUSY_OR_STATE);
    capture::require("placement degraded", 2);   // one record per loss, never one per round
    CHECK(rig.tick(kFlat) && rig.tick(kFlat));
    capture::require("placement degraded", 2);
    CHECK(rig.routes(0, 0));   // the lead keeps serving
}

void two_successive_helper_losses() {
    Rig rig;
    loss_one_closes_only_helper_one(rig);
    survivors_keep_planning_and_being_checked(rig);
    loss_two_is_caught_on_the_next_tick(rig);
}

// A loss between rounds is caught by liveness even when no round is due, and the lead is never asked to stop.
void a_loss_between_rounds_is_caught_without_a_round() {
    Rig rig;
    rig.h2.alive = false;
    rig.h3.alive = false;   // two losses in one tick: both closed, one record each
    CHECK(rig.tick(kFlat));
    CHECK(!rig.view.rung_open(2) && !rig.view.rung_open(3) && rig.view.rung_open(1));
    capture::require("placement degraded: helper is gone", 2, SEABRAID_BUSY_OR_STATE);
    CHECK(rig.driver->state() == State::kDegraded && rig.routes(0, 0) && rig.routes(1, 1));
}

}  // namespace

int main() {
    capture::start();
    two_successive_helper_losses();
    a_loss_between_rounds_is_caught_without_a_round();
    std::puts("Seabraid Strata placement liveness fixture passed");
}
