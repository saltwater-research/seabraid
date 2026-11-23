// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// PlacementDriver lifecycle: validation, construction, the first publish and teardown. OWNER: the startup half of
// the driver (placement_driver.cpp owns the round). FAILURE CONTRACT: a bad shape or policy returns nullptr with `err`
// set and one diag record (the engine exits 2); nothing here throws.
#include "placement_driver.hpp"

#include "diag.hpp"
#include "diag_seabraid.hpp"

#include <algorithm>
#include <new>
#include <utility>

namespace seabraid::strata_adapter {
using diag::kError;

PlacementDriver::PlacementDriver(seabraid_placement* handle, std::vector<uint8_t> owners,
                                 std::vector<strata::core::RungIO*> cards, strata::core::RungIO* fallback,
                                 std::vector<size_t> slot_bytes, const PlacementDriverConfig& config,
                                 ExecutorView& view, QualHeat qual)
    : handle_(handle), cards_(std::move(cards)),
      slot_bytes_(std::move(slot_bytes)), rungs_{cards_, slot_bytes_, fallback}, config_(config), view_(view),
      published_(std::move(owners)), qual_(std::move(qual)) {}

PlacementDriver::~PlacementDriver() {
    const int32_t status = seabraid_placement_close(handle_);
    if (status != SEABRAID_OK) diag::logf(kError, status, "placement close failed: %s", diag::status_name(status));
}

bool PlacementDriver::init_buffers() noexcept {
    try {
        const size_t cells = published_.size();
        round_.executors.assign(cells, 0);
        lead_.assign(cells, 0);
        held_.assign(cells, 0);
        next_.assign(cells, 0);
        heat_.assign(cells, 0.0);
        return true;
    } catch (const std::bad_alloc&) {
        diag::logf(kError, SEABRAID_INTERNAL, "placement open: cannot allocate the staging tables for %zu cells",
                   published_.size());
        return false;
    }
}

namespace {

// WHY refuse a fallback boundary without a fallback rung here: the core would plan swaps whose first hop hits it.
const char* config_fault(const PlacementDriverConfig& config, const strata::core::RungIO* fallback) noexcept {
    if (config.policy.abi_version != 0 && config.policy.fallback_boundary != 0 && fallback == nullptr)
        return "fallback_boundary needs a fallback rung";
    return config.every_windows == 0 ? "every_windows is zero" : nullptr;
}

// Names the first thing wrong with the startup arguments, or nullptr. WHY a table of checks and not asserts: the
// engine passes these from its own arrays, and a mismatch must exit 2 with a code, never index out of bounds.
const char* open_fault(uint32_t layers, uint32_t experts, size_t owners, const std::vector<strata::core::RungIO*>& cards,
                       const strata::core::RungIO* fallback, size_t slot_tables, const PlacementDriverConfig& config) noexcept {
    const uint64_t cells = static_cast<uint64_t>(layers) * experts;
    if (cells == 0 || cells > SEABRAID_PLACEMENT_MAX_CELLS || owners != cells) return "bad layout";
    if (cards.empty() || cards.size() > SEABRAID_PLACEMENT_MAX_CARDS) return "bad card table";
    if (std::find(cards.begin(), cards.end(), nullptr) != cards.end()) return "bad card table";
    if (slot_tables != layers) return "slot-size table does not match the layer count";
    return config_fault(config, fallback);
}

}  // namespace

std::unique_ptr<PlacementDriver> PlacementDriver::open(uint32_t layers, uint32_t experts,
        std::vector<uint8_t> owners, std::vector<strata::core::RungIO*> cards, strata::core::RungIO* fallback,
        std::vector<size_t> slot_bytes, const PlacementDriverConfig& config, ExecutorView& view, std::string& err) {
    install_seabraid_log_sink();
    // WHY here: the driver owns the INFO round record, so it asks for it to be visible; and only once it is in use,
    // so a run with --retier-promote off never sees the Seabraid INFO stream.
    show_info_records();
    QualHeat qual;
    const char* fault = open_fault(layers, experts, owners.size(), cards, fallback, slot_bytes.size(), config);
    if (!fault && !config.qual_heat_path.empty() && !qual.load(config.qual_heat_path, layers, experts))
        fault = "the qualification heat file is unreadable or malformed";
    if (fault) {
        err = fault;
        diag::logf(kError, SEABRAID_INVALID, "placement open refused: %s", fault);
        return nullptr;
    }
    const seabraid_placement_layout_v1 layout{SEABRAID_ABI_VERSION, layers, experts,
                                              static_cast<uint8_t>(cards.size()), owners.data(), owners.size()};
    // WHY abi_version 0 means defaults: a zero-initialized config must not reach Rust as a malformed policy.
    // WHY always open_v2: v2 with {fallback 0, hops 1} is exactly v1 (placement.h), so one entry point serves both.
    const auto* policy = config.policy.abi_version != 0 ? &config.policy : nullptr;
    seabraid_placement* handle = nullptr;
    const int32_t status = seabraid_placement_open_v2(&layout, policy, &handle);
    if (status != SEABRAID_OK) {
        err = diag::status_name(status);
        diag::logf(kError, status, "placement open refused by the core: %s", err.c_str());
        return nullptr;
    }
    std::unique_ptr<PlacementDriver> driver(new (std::nothrow) PlacementDriver(
        handle, std::move(owners), std::move(cards), fallback, std::move(slot_bytes), config, view, std::move(qual)));
    if (!driver) {
        seabraid_placement_close(handle);
        err = "out of memory";
        diag::logf(kError, SEABRAID_INTERNAL, "placement open: cannot allocate the driver");
        return nullptr;
    }
    if (!driver->init_buffers() || !driver->publish_initial()) {
        err = "cannot publish the initial table";
        return nullptr;
    }
    return driver;
}

bool PlacementDriver::publish_initial() noexcept {
    const int32_t status = snapshot_staging();
    if (status != SEABRAID_OK) {
        diag::logf(kError, status, "placement open: first snapshot failed: %s", diag::status_name(status));
        return false;
    }
    if (!view_.publish(round_.executors.data(), round_.executors.size(), round_.snapshot)) {
        diag::logf(kError, SEABRAID_INVALID, "placement open: the view refused the initial table");
        return false;
    }
    published_ = round_.executors;
    published_epoch_ = round_.snapshot.epoch;
    return true;
}

}  // namespace seabraid::strata_adapter
