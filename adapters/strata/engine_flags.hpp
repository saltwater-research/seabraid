// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
#pragma once
// The engine command-line flags the Seabraid adapter owns (the ones that are not --retier-*: those are the compat add-on's).
// OWNER: the spelling, the values and the default of each flag; the engine's argument loop only forwards to parse_engine_flag.
// CONTRACT: pure (no I/O, no engine include).  SIZE BUDGET: header 50, engine_flags.cpp 70.
// FAILURE CONTRACT: a flag with a bad value is an error with the flag named (the engine exits 2), never a silent default.
// CHECKER: engine_flags_test (default off, each value, a bad value, a foreign flag is not consumed).
#include <functional>
#include <string>

namespace seabraid::strata_adapter {

struct EngineFlags {
    // --seabraid-compact-rows 0|1: opens the private helper's session with SEABRAID_PROVIDER_COMPACT_ROWS (the helper then gets only
    // rows with a real pick).  Default OFF until measured; this flag is the one source of the option (no environment variable).
    bool compact_rows = false;
};

using NextFn = std::function<const char*(const char* flag)>;   // reads the value that follows `flag`; nullptr when there is none

// 0: not one of ours; 1: consumed; 2: the value is bad (`err` says which flag and what was expected).
int parse_engine_flag(const std::string& arg, const NextFn& next, EngineFlags& flags, std::string& err);

const char* engine_flags_help();

}  // namespace seabraid::strata_adapter
