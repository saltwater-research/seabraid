// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// The cadence and the one fenced round of live placement.
// OWNER: when to plan, what to adopt, the order apply -> verify -> publish, and the fallback after a loss.
// CONTRACT: drives only the existing ABI (seabraid/placement.h) and apply_placement_at_fence; never includes Strata
//   engine internals. The engine hands in PlacementHost and the RungIO cards. Engine thread only.
// SIZE BUDGET: header 110, placement_driver.cpp 250, placement_driver_open.cpp 135 (open_v2, the fallback rung, the qual file), placement_round_log.cpp 200,
//   placement_liveness.cpp 120.
// FAILURE CONTRACT: a refused round is skipped and logged; a lost helper closes its rung (liveness is checked on every
//   tick, F11) and its cells fall back while planning continues among the open rungs; a lost or disagreeing lead makes
//   tick() return false; an ABI error sets kDead. One diag record per failure branch.
// CHECKER: placement_driver_test (R2 publish after verify, R4 timeout quarantines, R5 adopt never moves rung 1, R8) and
//   placement_liveness_test (three helpers, two successive losses).
#pragma once

#include "executor_view.hpp"
#include "placement.hpp"
#include "placement_qual.hpp"
#include "placement_round.hpp"

#include <functional>
#include <memory>
#include <string>

namespace seabraid::strata_adapter {

// What only the engine knows. Called on the engine thread at a quiet window boundary.
struct PlacementHost {
    virtual ~PlacementHost() = default;
    virtual bool drain(std::string& err) noexcept = 0;                       // false: skip round (logged)
    virtual void lead_cells(std::vector<uint8_t>& is_lead) noexcept = 0;
    virtual bool helper_cells(uint8_t rung, std::vector<uint8_t>& held) noexcept = 0;  // false: helper gone
    // Cheap, every tick: false once helper `rung` is closed or retired.  No table copy.  The real EngineHost deliberately does NO pid
    // probe (a recycled pid would read as alive, a stale one as dead): a dead named helper is caught when its next layer request or
    // round op fails and closes the card.
    virtual bool helper_alive(uint8_t rung) noexcept = 0;
    // One COMMITTED hop, after its table is published.  The host decides what follows (ram_follows() is the rule for
    // the engine's RAM copy: only the lead <-> helper 1 boundary changes the lead's complement).
    virtual void after_commit(const seabraid_placement_outcome_v1& outcome) noexcept = 0;
    // After a round that worked (its round line is written): the host's own per-round lines, e.g. the helper file-io counters.
    // Not pure: a host with nothing to add leaves it.
    virtual void after_round() noexcept {}
};

// WHY helper_op_timeout_ms and verify_writes are here: the host builds the helper RungIOs from them; the driver never
// waits and never reads a slot.
struct PlacementDriverConfig {
    uint32_t every_windows = 128;
    // Opened with seabraid_placement_open_v2.  abi_version 0 means "the Rust defaults" (= v1 defaults, no fallback
    // boundary, adjacent-only); fallback_boundary 1 needs a fallback RungIO at open.
    seabraid_placement_policy_v2 policy{};
    uint32_t helper_op_timeout_ms = 2000;
    bool verify_writes = true;     // helper RungIOs send the bytes (kTierSwapBlob) and the helper reads its slot back
    std::string qual_heat_path;    // SEABRAID_PLACEMENT_QUAL: scripted heat bias for qualification; empty = none
    // Trace only: reads the engine's FallbackRung counters (cumulative ram_hits(), file_reads()) so a round record can say
    // how many fallback stages came from RAM and how many from NVMe. Empty = the host does not give them (record: null).
    std::function<void(int64_t& ram_hits, int64_t& file_reads)> fallback_counters;
};

class PlacementDriver {
public:
    // kDegraded = at least one helper rung is closed; the driver still plans among the open ones.
    enum class State { kLive, kDegraded, kDead };

    // Publishes the initial table through `view`. nullptr with `err` set on a bad shape or policy.
    // `fallback` serves rung == cards.size() (nullptr: none; then policy.fallback_boundary must be 0).
    static std::unique_ptr<PlacementDriver> open(uint32_t layers, uint32_t experts, std::vector<uint8_t> owners,
        std::vector<strata::core::RungIO*> cards, strata::core::RungIO* fallback, std::vector<size_t> slot_bytes,
        const PlacementDriverConfig&, ExecutorView& view, std::string& err);
    ~PlacementDriver();
    PlacementDriver(const PlacementDriver&) = delete;  PlacementDriver& operator=(const PlacementDriver&) = delete;

    // `heat` is layers * experts floats. false: the engine must stop (lead rung lost or disagreeing).
    bool tick(const float* heat, size_t n, bool quiet, PlacementHost& host) noexcept;
    State state() const noexcept { return state_; }

private:
    enum class Flow { kGo, kDone, kStop };   // continue the round, round over, engine must stop
    enum Skip { kPlanSkip, kDrainSkip, kSkipKinds };
    PlacementDriver(seabraid_placement* handle, std::vector<uint8_t> owners, std::vector<strata::core::RungIO*> cards,
                    strata::core::RungIO* fallback, std::vector<size_t> slot_bytes, const PlacementDriverConfig& config,
                    ExecutorView& view, QualHeat qual);
    bool init_buffers() noexcept, publish_initial() noexcept;   // in placement_driver_open.cpp
    bool due(bool quiet) noexcept;
    Flow run_round(const float* heat, size_t n, PlacementHost& host) noexcept;
    void check_liveness(PlacementHost& host) noexcept;          // placement_liveness.cpp, with degrade and die
    bool rung_closed(uint8_t rung) const noexcept;
    Flow adopt(PlacementHost& host) noexcept, apply(PlacementHost& host) noexcept;
    Flow verify(PlacementHost& host) noexcept, publish_round() noexcept;
    void follow(PlacementHost& host) noexcept;      // after_commit per committed hop, then the move lines
    Flow plan(const float* heat, size_t n) noexcept;
    bool publish_staged() noexcept;                 // false: the view refused it and the driver is dead
    Flow degrade(uint8_t rung, const char* why) noexcept, die(int32_t status, const char* what) noexcept;
    Flow skip(Skip kind, int32_t status, const char* what) noexcept;
    void note_degraded(const char* why) noexcept;
    void note_loss(uint8_t rung, const char* why) noexcept;      // trace: a helper rung closed (placement_driver.cpp)
    void trace_tick() noexcept;                                  // trace: helper "up" once, "served_after_loss" after a loss
    void trace_round(bool counted, int64_t ram0, int64_t file0) noexcept;
    void time_outcomes() noexcept;                               // trace: stage/write time per outcome of the round
    MoveContext move_context() const noexcept { return {rounds_, epoch_, &timings_, config_.verify_writes}; }
    bool read_fallback(int64_t& ram_hits, int64_t& file_reads) const noexcept;
    int32_t snapshot_staging() noexcept;
    uint64_t closed_mask() const noexcept { return round_.snapshot.quarantined_rungs; }
    uint8_t fallback() const noexcept { return static_cast<uint8_t>(cards_.size()); }

    seabraid_placement* handle_;
    std::vector<strata::core::RungIO*> cards_;
    std::vector<size_t> slot_bytes_;
    PlacementRungs rungs_;
    PlacementDriverConfig config_;
    ExecutorView& view_;
    seabraid_placement_snapshot_v1 last_good_{};
    PlacementRound round_;   // the staging table: never the published one
    seabraid_placement_ticket_v1 ticket_{};
    std::vector<uint8_t> published_, lead_, held_, next_;
    std::vector<double> heat_;
    RoundStats stats_;
    QualHeat qual_;
    State state_ = State::kLive;
    uint64_t epoch_ = 1, published_epoch_ = 0, window_ = 0, rounds_ = 0, deferred_ = 0;
    uint32_t skips_[kSkipKinds] = {};
    std::vector<trace::PairTiming> timings_;   // per outcome of the round, filled only while the trace is on
    uint64_t lost_at_us_ = 0;                  // trace: when the latest helper loss was seen; 0 = none waiting
    uint8_t lost_rung_ = 0;
    bool announced_ = false;                   // trace: the helper "up" events were written
};

}  // namespace seabraid::strata_adapter
