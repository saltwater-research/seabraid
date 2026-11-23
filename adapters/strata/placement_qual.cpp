// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
#include "placement_qual.hpp"

#include "diag.hpp"

#include "seabraid/seabraid.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <utility>

namespace seabraid::strata_adapter {
namespace {

// One non-comment line -> an entry; false on anything malformed or outside the layout.
bool parse_line(const std::string& line, uint32_t layers, uint32_t experts, uint64_t& round, size_t& cell,
                double& bias) {
    std::istringstream in(line);
    long long from = -1, layer = -1, expert = -1;
    std::string rest;
    if (!(in >> from >> layer >> expert >> bias) || (in >> rest)) return false;
    if (from < 0 || layer < 0 || expert < 0 || layer >= layers || expert >= experts || !std::isfinite(bias)) return false;
    round = static_cast<uint64_t>(from);
    cell = static_cast<size_t>(layer) * experts + static_cast<size_t>(expert);
    return true;
}

bool blank_or_comment(const std::string& line) {
    const size_t first = line.find_first_not_of(" \t\r");
    return first == std::string::npos || line[first] == '#';
}

}  // namespace

bool QualHeat::load(const std::string& path, uint32_t layers, uint32_t experts) noexcept {
    try {
        std::ifstream file(path);
        std::vector<Entry> parsed;
        std::string line;
        size_t number = 0;
        while (file && std::getline(file, line)) {
            ++number;
            if (blank_or_comment(line)) continue;
            Entry entry{};
            if (parsed.size() == kMaxEntries || !parse_line(line, layers, experts, entry.from_round, entry.cell, entry.bias)) {
                diag::logf(diag::kError, SEABRAID_INVALID, "placement qualification heat: line %zu refused", number);
                return false;
            }
            parsed.push_back(entry);
        }
        if (parsed.empty()) return false;   // unreadable or empty: the caller's refusal record names it
        entries_ = std::move(parsed);
    } catch (...) {  // allow-error-path: the caller refuses the open and logs once
        return false;
    }
    // WHY WARN and not INFO: a steered placement must never be mistaken for a production run in a log.
    diag::logf(diag::kWarn, SEABRAID_OK, "placement: QUALIFICATION heat bias active (%zu entries): placement is steered, "
               "routing is not; round lines carry qual=1", entries_.size());
    return true;
}

bool QualHeat::apply(std::vector<double>& heat, uint64_t round) const noexcept {
    bool active = false;
    // WHY in file order with "a later line replaces": a walk is written top to bottom, up then down.
    std::vector<std::pair<size_t, double>> bias;
    try {
        for (const Entry& entry : entries_) {
            if (entry.from_round > round || entry.cell >= heat.size()) continue;
            bool replaced = false;
            for (auto& [cell, value] : bias)
                if (cell == entry.cell) { value = entry.bias; replaced = true; }
            if (!replaced) bias.emplace_back(entry.cell, entry.bias);
        }
    } catch (...) {  // allow-error-path: no bias this round is the safe answer (placement stays unsteered)
        return false;
    }
    for (const auto& [cell, value] : bias) {
        heat[cell] = std::max(0.0, heat[cell] + value);
        active = true;
    }
    return active;
}

}  // namespace seabraid::strata_adapter
