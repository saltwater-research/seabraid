// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// Test-only: a capturing diag sink so fixtures can assert that a failure wrote
// exactly the record an operator would need. OWNER: the three Strata
// adapter fixtures; nothing outside the tests includes it.
#pragma once

#include "diag.hpp"

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

namespace seabraid::strata_adapter::capture {

struct Record {
    uint32_t level;
    int32_t status;
    std::string message;
};

inline std::mutex& guard() { static std::mutex m; return m; }
inline std::vector<Record>& records() { static std::vector<Record> r; return r; }

inline void sink(uint32_t level, int32_t status, const char* message) noexcept {
    try {
        std::lock_guard<std::mutex> lock(guard());
        records().push_back({level, status, message});
    } catch (...) {
        std::abort();   // a fixture that cannot record must not pretend it passed
    }
}

inline void start() {
    std::lock_guard<std::mutex> lock(guard());
    records().clear();
    diag::set_min_level(diag::kDebug);
    diag::set_sink(sink);
}

inline void clear() { std::lock_guard<std::mutex> lock(guard()); records().clear(); }

// Number of records whose text contains `needle` (and, when status >= 0, that status).
inline size_t count(const char* needle, int32_t status = -1) {
    std::lock_guard<std::mutex> lock(guard());
    size_t found = 0;
    for (const auto& r : records())
        if (r.message.find(needle) != std::string::npos && (status < 0 || r.status == status)) ++found;
    return found;
}

// Every captured line, for a failure message.
inline std::string dump() {
    std::lock_guard<std::mutex> lock(guard());
    std::string out;
    for (const auto& r : records()) out += "  [" + std::to_string(r.level) + "] " + r.message + "\n";
    return out;
}

// Fails the fixture unless exactly `times` records match; prints what was captured.
inline void require(const char* needle, size_t times = 1, int32_t status = -1) {
    if (count(needle, status) == times) return;
    std::fprintf(stderr, "expected %zu log record(s) containing \"%s\", captured:\n%s", times, needle,
                 dump().c_str());
    std::exit(1);
}

}  // namespace seabraid::strata_adapter::capture
