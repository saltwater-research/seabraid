// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
#include "placement.hpp"

#include "diag_capture.hpp"

#include "strata/core/ownership_map.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <stdexcept>

#define CHECK(expression) do { \
    if (!(expression)) { \
        std::fprintf(stderr, "line %d: %s failed\n", __LINE__, #expression); \
        std::abort(); \
    } \
} while (false)

namespace {

using seabraid::strata_adapter::PlacementRound;
using seabraid::strata_adapter::PlacementRungs;
using seabraid::strata_adapter::apply_placement_at_fence;
using seabraid::strata_adapter::checked_rung;
namespace capture = seabraid::strata_adapter::capture;
using strata::core::IoStatus;

enum class Failure { none, lower_refusal, lower_timeout, short_stage, read_exception };

const std::array<std::vector<uint8_t>, 2> source{{{10, 20, 30}, {40, 50, 60}}};

struct Engine {
    bool fenced = false;
    bool drained = false;
    uint64_t route_epoch = 1;
    std::vector<uint8_t> executors{0, 1, 2};
    std::vector<int> trace;

    void check_drained() const {
        CHECK(fenced && drained);
        CHECK(route_epoch == 1);
        CHECK(executors[0] == 0 && executors[1] == 1);
    }

    void publish(const PlacementRound& round) {
        check_drained();
        executors = round.executors;
        route_epoch = round.snapshot.epoch;
        fenced = false;
    }
};

class Card : public strata::core::RungIO {
public:
    Card(Engine& engine, int32_t rank) : resident(rank), bytes(source[rank]), engine_(engine), rank_(rank) {}

    IoStatus stage(int32_t layer, int32_t expert, std::vector<uint8_t>& out, std::string&) override {
        engine_.check_drained();
        CHECK(layer == 0 && expert == resident);
        engine_.trace.push_back(rank_);
        if (failure == Failure::read_exception) throw std::runtime_error("unknown provider completion");
        out = bytes;
        if (failure == Failure::short_stage) out.pop_back();
        return IoStatus::kOk;
    }

    IoStatus replace(int32_t layer, int32_t old_expert, int32_t new_expert,
                     const std::vector<uint8_t>& input, std::string& error) override {
        engine_.check_drained();
        CHECK(layer == 0 && old_expert == resident);
        CHECK(input == source[new_expert]);
        engine_.trace.push_back(rank_ + 2);
        if (failure == Failure::lower_refusal) {
            error = "helper refused before writing";
            return IoStatus::kRefused;
        }
        if (failure == Failure::lower_timeout) {
            bytes[0] = input[0];
            error = "helper did not acknowledge the slot write";
            return IoStatus::kTimeout;
        }
        resident = new_expert;
        bytes = input;
        return IoStatus::kOk;
    }

    Failure failure = Failure::none;
    int32_t resident;
    std::vector<uint8_t> bytes;

private:
    Engine& engine_;
    int32_t rank_;
};

void check_route(const Engine& engine, const Card& primary, const Card& helper, uint64_t epoch,
                 uint8_t first, uint8_t second) {
    CHECK(engine.route_epoch == epoch && !engine.fenced);
    CHECK((engine.executors == std::vector<uint8_t>{first, second, 2}));
    for (unsigned expert = 0; expert < source.size(); ++expert) {
        const auto rank = engine.executors[expert];
        if (rank == 2) continue;
        const auto& card = rank == 0 ? primary : helper;
        CHECK(card.resident == static_cast<int32_t>(expert) && card.bytes == source[expert]);
    }
}

void run(Failure failure, uint32_t outcome, uint32_t error, uint64_t epoch, uint64_t closed,
         const std::vector<int>& trace, uint8_t first, uint8_t second, const char* logged = nullptr) {
    capture::clear();
    Engine engine;
    Card primary(engine, 0), helper(engine, 1);
    helper.failure = failure;
    const std::vector<strata::core::RungIO*> cards{&primary, &helper};
    const std::vector<size_t> sizes{source[0].size()};
    const PlacementRungs rungs{cards, sizes};
    const seabraid_placement_layout_v1 layout{SEABRAID_ABI_VERSION, 1, 3, 2, engine.executors.data(), engine.executors.size()};
    seabraid_placement* placement = nullptr;
    CHECK(seabraid_placement_open(&layout, nullptr, &placement) == SEABRAID_OK);
    const double heat[] = {1.0, 100.0, 10.0};
    seabraid_placement_ticket_v1 ticket{SEABRAID_ABI_VERSION, sizeof ticket, 0, 0, 0};
    CHECK(seabraid_placement_plan(placement, 1, heat, std::size(heat), &ticket) == SEABRAID_OK);
    CHECK(ticket.promotion_count == 1);

    engine.fenced = true;
    engine.drained = true;
    PlacementRound round;
    round.executors.resize(2);
    CHECK(apply_placement_at_fence(placement, ticket, rungs, round) == SEABRAID_INVALID);
    CHECK(engine.trace.empty());
    capture::require("placement snapshot failed: SEABRAID_INVALID", 1, SEABRAID_INVALID);
    capture::clear();
    round.executors.resize(engine.executors.size());
    CHECK(apply_placement_at_fence(placement, ticket, rungs, round) == SEABRAID_OK);
    // A failing provider writes exactly one record naming the failing call;
    // the clean run writes none.
    if (logged) capture::require(logged, 1);
    else capture::require("placement", 0);
    CHECK(engine.trace == trace);
    CHECK(round.outcomes.size() == 1 && round.outcomes[0].kind == outcome);
    CHECK(round.outcomes[0].initial.error == error);
    CHECK(round.snapshot.quarantined_rungs == closed);
    engine.publish(round);
    check_route(engine, primary, helper, epoch, first, second);
    CHECK(seabraid_placement_close(placement) == SEABRAID_OK);
}

}  // namespace

// The runtime chooses the rung; a value past the bound cards must fail closed
// (null card, one logged error) instead of indexing the table.
void out_of_range_rung_fails_closed() {
    Engine engine;
    Card primary(engine, 0), helper(engine, 1);
    const std::vector<strata::core::RungIO*> cards{&primary, &helper};
    const std::vector<size_t> sizes{source[0].size()};
    const PlacementRungs rungs{cards, sizes};
    capture::clear();
    CHECK(checked_rung(rungs, 0) == &primary && checked_rung(rungs, 1) == &helper);
    capture::require("placement refused", 0);
    CHECK(checked_rung(rungs, 2) == nullptr);   // exactly one past the end
    CHECK(checked_rung(rungs, 255) == nullptr);
    capture::require("placement refused: rung 2 is outside the 2 bound cards", 1, SEABRAID_INVALID);
    capture::require("placement refused: rung 255", 1, SEABRAID_INVALID);
    const std::vector<strata::core::RungIO*> none;
    const PlacementRungs empty{none, sizes};
    CHECK(checked_rung(empty, 0) == nullptr);
    // A bound fallback serves exactly rung == cards.size(), nothing past it.
    Card fallback(engine, 0);
    const PlacementRungs with_fallback{cards, sizes, &fallback};
    capture::clear();
    CHECK(checked_rung(with_fallback, 1) == &helper && checked_rung(with_fallback, 2) == &fallback);
    capture::require("placement refused", 0);
    CHECK(checked_rung(with_fallback, 3) == nullptr);
    capture::require("placement refused: rung 3 is outside the 2 bound cards and the fallback", 1, SEABRAID_INVALID);
}

// A multi-hop climb of one expert is one promotion per hop, so a tiny layout (3 cells here) can carry more promotions than
// half its cells.  The count bound must be the cell count: 2 promotions on 3 cells pass it (the library then judges the
// ticket), 4 on 3 cells are refused by it.  The ticket count is edited because the 1-layer planner cannot yet emit a climb.
void promotion_count_bound_is_the_cell_count() {
    Engine engine;
    Card primary(engine, 0), helper(engine, 1);
    const std::vector<strata::core::RungIO*> cards{&primary, &helper};
    const std::vector<size_t> sizes{source[0].size()};
    const PlacementRungs rungs{cards, sizes};
    const seabraid_placement_layout_v1 layout{SEABRAID_ABI_VERSION, 1, 3, 2, engine.executors.data(), engine.executors.size()};
    seabraid_placement* placement = nullptr;
    CHECK(seabraid_placement_open(&layout, nullptr, &placement) == SEABRAID_OK);
    const double heat[] = {1.0, 100.0, 10.0};
    seabraid_placement_ticket_v1 ticket{SEABRAID_ABI_VERSION, sizeof ticket, 0, 0, 0};
    CHECK(seabraid_placement_plan(placement, 1, heat, std::size(heat), &ticket) == SEABRAID_OK);
    PlacementRound round;
    round.executors.resize(engine.executors.size());
    engine.fenced = engine.drained = true;
    capture::clear();
    ticket.promotion_count = 2;                                  // 2 > 3 / 2: the old half-the-cells bound refused this
    const int32_t first = apply_placement_at_fence(placement, ticket, rungs, round);
    CHECK(first == SEABRAID_OK);                                 // the plan itself was valid; only its count was edited
    capture::require("promotions exceed", 0);
    capture::require("exceed half", 0);
    ticket.promotion_count = 4;                                  // more promotions than cells: corrupt
    CHECK(apply_placement_at_fence(placement, ticket, rungs, round) == SEABRAID_INVALID);
    capture::require("placement refused: 4 promotions exceed the 3 executors", 1, SEABRAID_INVALID);
    CHECK(seabraid_placement_close(placement) == SEABRAID_OK);
}

int main() {
    capture::start();
    run(Failure::none, SEABRAID_PLACEMENT_COMMITTED, 0, 2, 0, {1, 0, 2, 3}, 1, 0);
    run(Failure::lower_refusal, SEABRAID_PLACEMENT_RESTORED, SEABRAID_PLACEMENT_REFUSED, 1, 0, {1, 0, 2, 3, 2}, 0, 1,
        "placement replace failed");
    run(Failure::lower_timeout, SEABRAID_PLACEMENT_QUARANTINED, SEABRAID_PLACEMENT_UNACKNOWLEDGED, 2, 2, {1, 0, 2, 3, 2}, 0, 2,
        "placement replace failed");
    run(Failure::short_stage, SEABRAID_PLACEMENT_REJECTED, SEABRAID_PLACEMENT_MALFORMED_BLOB, 1, 0, {1}, 0, 1,
        "placement stage returned");
    run(Failure::read_exception, SEABRAID_PLACEMENT_QUARANTINED, SEABRAID_PLACEMENT_UNACKNOWLEDGED, 2, 2, {1}, 0, 2,
        "provider threw");
    out_of_range_rung_fails_closed();
    promotion_count_bound_is_the_cell_count();
    std::puts("Seabraid Strata placement fixture passed");
}
