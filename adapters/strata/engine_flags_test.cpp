// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// CHECKER for engine_flags.hpp: --seabraid-compact-rows defaults off (the library field's default), takes exactly 0 or 1, names
// itself in a bad-value error, and a flag that is not ours is left for the engine's other parsers.
#include "engine_flags.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

#define CHECK(expression) do { \
    if (!(expression)) { \
        std::fprintf(stderr, "line %d: %s failed\n", __LINE__, #expression); \
        std::abort(); \
    } \
} while (false)

using seabraid::strata_adapter::EngineFlags;
using seabraid::strata_adapter::parse_engine_flag;

int main() {
    EngineFlags flags;
    std::string err;
    const char* value = nullptr;
    const auto next = [&](const char*) { return value; };

    CHECK(!flags.compact_rows);                                                       // DEFAULT: off until measured
    value = "1";
    CHECK(parse_engine_flag("--seabraid-compact-rows", next, flags, err) == 1 && flags.compact_rows);
    value = "0";
    CHECK(parse_engine_flag("--seabraid-compact-rows", next, flags, err) == 1 && !flags.compact_rows);
    for (const char* bad : {"2", "on", "", "-1", "10"}) {
        flags.compact_rows = true;
        value = bad;
        err.clear();
        CHECK(parse_engine_flag("--seabraid-compact-rows", next, flags, err) == 2);
        CHECK(err.find("--seabraid-compact-rows") != std::string::npos && flags.compact_rows);   // the value is left as it was
    }
    value = nullptr;                                                                   // no value at all
    CHECK(parse_engine_flag("--seabraid-compact-rows", next, flags, err) == 2);
    value = "1";
    flags.compact_rows = false;
    CHECK(parse_engine_flag("--retier-file-moves", next, flags, err) == 0 && !flags.compact_rows);   // not ours: untouched
    CHECK(parse_engine_flag("--seabraid-compact-rowsX", next, flags, err) == 0);
    CHECK(std::string(seabraid::strata_adapter::engine_flags_help()).find("--seabraid-compact-rows 0|1") != std::string::npos);
    std::puts("Seabraid Strata engine flags fixture passed");
}
