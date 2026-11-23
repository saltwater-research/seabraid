// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
#include "diag.hpp"

#include "seabraid/seabraid.h"

#include <atomic>
#include <cstdarg>
#include <cstdio>

namespace seabraid::strata_adapter::diag {
namespace {

std::atomic<Sink> g_sink{nullptr};
// Info and above by default; debug is for developers who opt in.
std::atomic<uint32_t> g_min_level{kInfo};

const char* level_name(uint32_t level) noexcept {
    switch (level) {
    case kDebug: return "DEBUG";
    case kInfo: return "INFO";
    case kWarn: return "WARN";
    default: return "ERROR";
    }
}

void write_stderr(uint32_t level, int32_t, const char* message) noexcept {
    std::fprintf(stderr, "seabraid-strata: %s: %s\n", level_name(level), message);
}

}  // namespace

void set_sink(Sink sink) noexcept { g_sink.store(sink, std::memory_order_release); }

void set_sink_if_unset(Sink sink) noexcept {
    Sink expected = nullptr;
    g_sink.compare_exchange_strong(expected, sink, std::memory_order_acq_rel);
}

void set_min_level(Level level) noexcept { g_min_level.store(level, std::memory_order_release); }

void logf(Level level, int32_t status, const char* format, ...) noexcept {
    if (level < g_min_level.load(std::memory_order_acquire)) return;
    char buffer[256];
    va_list args;
    va_start(args, format);
    const int written = std::vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    if (written < 0) std::snprintf(buffer, sizeof(buffer), "log message could not be formatted");
    const Sink sink = g_sink.load(std::memory_order_acquire);
    (sink ? sink : write_stderr)(level, status, buffer);
}

const char* status_name(int32_t status) noexcept {
    switch (status) {
    case SEABRAID_OK: return "SEABRAID_OK";
    case SEABRAID_INVALID: return "SEABRAID_INVALID";
    case SEABRAID_BUSY_OR_STATE: return "SEABRAID_BUSY_OR_STATE";
    case SEABRAID_TIMEOUT: return "SEABRAID_TIMEOUT";
    case SEABRAID_PROVIDER: return "SEABRAID_PROVIDER";
    case SEABRAID_INTERNAL: return "SEABRAID_INTERNAL";
    default: return "SEABRAID_UNKNOWN";
    }
}

}  // namespace seabraid::strata_adapter::diag
