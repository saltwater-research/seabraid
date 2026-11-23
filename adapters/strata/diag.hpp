// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// Adapter diagnostics seam.
// OWNER: this header is the only place the Strata adapter decides where a log
//   record goes. CONTRACT: callers use diag::logf; the runtime-linked layer
//   calls diag::set_sink once to forward into seabraid_log; the standalone v7
//   fixtures (built without the Rust runtime) keep the stderr default.
// SIZE BUDGET: ~60 lines header, ~80 lines source. FAILURE CONTRACT: logging
//   never throws, never allocates, never fails the caller; a record that cannot
//   be formatted is replaced by a fixed notice instead of being dropped.
// CHECKER: private_qwen_test and static_qwen_test install a capturing sink and
//   assert records. Log text carries codes, counts and ids only - never paths,
//   environment values, hostnames or tensor data.
#pragma once

#include <cstdint>

namespace seabraid::strata_adapter::diag {

// Values equal seabraid_log_level so the forwarding sink needs no mapping table.
enum Level : uint32_t { kDebug = 0, kInfo = 1, kWarn = 2, kError = 3, kOff = 4 };

using Sink = void (*)(uint32_t level, int32_t status, const char* message) noexcept;

// WHY an atomic function pointer and not a mutex: logging happens on engine
// decode threads and in a noexcept destructor; a lock-free load cannot block
// or throw there. nullptr restores the stderr default.
void set_sink(Sink sink) noexcept;
// Installs `sink` only when none is set, so an embedding host or a test that
// chose its own sink is never overridden by a library default.
void set_sink_if_unset(Sink sink) noexcept;
void set_min_level(Level level) noexcept;

// Fixed 256-byte buffer: no allocation, so it is safe in noexcept paths.
void logf(Level level, int32_t status, const char* format, ...) noexcept
    __attribute__((format(printf, 3, 4)));

// Named SEABRAID_* status for log text; "SEABRAID_UNKNOWN" for anything else.
const char* status_name(int32_t status) noexcept;

}  // namespace seabraid::strata_adapter::diag
