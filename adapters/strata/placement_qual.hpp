// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// The qualification heat hook (SEABRAID_PLACEMENT_QUAL=<file>): a scripted per-cell heat bias added before plan().
// OWNER: the walk-file grammar and how a bias meets the planner's heat.  CONTRACT: load() once at open, apply() before
//   every plan; it changes only the heat handed to the planner, never routing, never the engine's usage counters.
// WHY it exists: real routing cannot be steered to walk one named expert through every rung on demand; a heat bias
//   changes only where an expert lives, which is exactly what qualification tests.
// File: one entry per line, `<from_round> <layer> <expert> <bias>` ('#' starts a comment).  From plan round
//   `from_round` on, `bias` is added to that cell's heat (the result is clamped at 0); a later line for the same cell
//   replaces an earlier one once its round is reached, so a walk up then down is two lines.
// SIZE BUDGET: header 50, source 100.  FAILURE CONTRACT: load() refuses (false, nothing kept) an unreadable file, a
//   malformed line, a cell outside the layout, a non-finite bias, no entries or more than kMaxEntries; open then exits 2.
// CHECKER: placement_driver_test (parse cases, refusal, bias reaching the planner, the qual=1 record).
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace seabraid::strata_adapter {

class QualHeat {
public:
    static constexpr size_t kMaxEntries = 4096;
    bool load(const std::string& path, uint32_t layers, uint32_t experts) noexcept;
    // Adds the bias active at plan round `round` to `heat` (layers * experts). True when any entry was active.
    bool apply(std::vector<double>& heat, uint64_t round) const noexcept;
    size_t entries() const noexcept { return entries_.size(); }

private:
    struct Entry {
        uint64_t from_round;
        size_t cell;
        double bias;
    };
    std::vector<Entry> entries_;
};

}  // namespace seabraid::strata_adapter
