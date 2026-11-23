// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
#include "diag_seabraid.hpp"

#include "diag.hpp"

#include "seabraid/seabraid.h"

#include <cstdio>

namespace seabraid::strata_adapter {
namespace {

void forward(uint32_t level, int32_t status, const char* message) noexcept {
    if (seabraid_log(level, status, message) == SEABRAID_OK) return;
    // WHY stderr here: a sink that can drop records silently would break the
    // "every error is logged" rule exactly when logging is misconfigured.
    std::fprintf(stderr, "seabraid-strata: level %u: %s\n", level, message);
}

}  // namespace

void install_seabraid_log_sink() noexcept { diag::set_sink_if_unset(forward); }

void show_info_records() noexcept {
    // WHY stderr on failure: the call that makes records visible cannot report through the sink it failed to set.
    if (seabraid_set_log_sink(nullptr, nullptr, SEABRAID_LOG_INFO) != SEABRAID_OK)
        std::fprintf(stderr, "seabraid-strata: the log threshold could not be lowered; round records stay hidden\n");
}

}  // namespace seabraid::strata_adapter
