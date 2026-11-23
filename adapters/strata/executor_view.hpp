// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// The one published routing table: which rung executes each (layer, expert).
// OWNER: this header. The Rust core owns the placement map; this is a verified
//   COPY of its executor snapshot that tier predicates read. It changes only by
//   publish(), called by the placement driver at the engine fence.
// CONTRACT: publish() from one thread (the engine thread, fence held); every
//   other member from any thread, lock-free for the reader. Depends on
//   seabraid/placement.h and diag.hpp only, never on Strata or a tier.
// FAILURE: publish() returns false and logs once, leaving the old table live;
//   readers answer false (route nowhere) before the first publish, so a tier
//   can claim a cell only after a table was verified and published.
// CHECKER: executor_view_test.cpp (incl. a concurrent reader; run under TSan).
#pragma once

#include "seabraid/placement.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace seabraid::strata_adapter {

class ExecutorView {
public:
    // Replaces the table when `exec` has layers*experts entries, every rung is
    // below 64, none is quarantined, the shape matches the first publish and
    // the epoch does not go backwards. `snapshot` is Rust's seabraid_placement_snapshot.
    bool publish(const uint8_t* exec, size_t n, const seabraid_placement_snapshot_v1& snapshot) noexcept;

    // True when the published table names `rung` for the cell. False before the
    // first publish, for out-of-range ids and for a quarantined rung.
    bool routes_to(int64_t layer, int64_t expert, uint8_t rung) const noexcept;
    bool rung_open(uint8_t rung) const noexcept;  // published and not quarantined

    uint64_t epoch() const noexcept { return epoch_.load(std::memory_order_acquire); }
    // FNV-1a over shape and executors (epoch excluded), so two arms that route
    // identically report the same hash. Zero before the first publish.
    uint64_t table_hash() const noexcept { return hash_.load(std::memory_order_acquire); }

private:
    struct Table {
        std::vector<uint8_t> exec;
        uint64_t quarantined = 0;
        uint32_t layers = 0, experts = 0;
        mutable std::atomic<uint32_t> readers{0};
    };
    // Returns the table with a reader reference held, or nullptr (before first publish).
    const Table* enter() const noexcept;
    static void leave(const Table* t) noexcept { t->readers.fetch_sub(1, std::memory_order_release); }
    bool accepts(const uint8_t* exec, size_t n, const seabraid_placement_snapshot_v1& s) const noexcept;

    Table tables_[2];
    std::atomic<const Table*> current_{nullptr};
    std::atomic<uint64_t> epoch_{0};
    std::atomic<uint64_t> hash_{0};
    std::atomic<bool> publishing_{false};
};

}  // namespace seabraid::strata_adapter
