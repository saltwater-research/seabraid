// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
#include "engine_flags.hpp"

#include <cstring>

namespace seabraid::strata_adapter {

int parse_engine_flag(const std::string& arg, const NextFn& next, EngineFlags& flags, std::string& err) {
    if (arg != "--seabraid-compact-rows") return 0;
    const char* value = next ? next("--seabraid-compact-rows") : nullptr;
    if (value != nullptr && std::strcmp(value, "0") == 0) flags.compact_rows = false;
    else if (value != nullptr && std::strcmp(value, "1") == 0) flags.compact_rows = true;
    else {
        err = "--seabraid-compact-rows: expected 0 or 1";
        return 2;
    }
    return 1;
}

const char* engine_flags_help() {
    return
        "  --seabraid-compact-rows 0|1  with the private Seabraid Qwen helper (SEABRAID_QWEN_HELPER): hand the helper only the rows\n"
        "                       that have a real pick instead of every row of the batch.  Default 0.\n";
}

}  // namespace seabraid::strata_adapter
