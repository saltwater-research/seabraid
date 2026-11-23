// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
// Runtime-linked half of the diagnostics seam: forwards adapter records into
// seabraid_log. OWNER: the only adapter file that calls seabraid_log. Kept out of
// diag.cpp so the standalone v7 fixtures link without the Rust runtime.
// FAILURE CONTRACT: if seabraid_log itself refuses a record, the record is written
// to stderr so it is never lost. CHECKER: strata_runtime_qwen / strata_placement.
#pragma once

namespace seabraid::strata_adapter {

// Idempotent and non-overriding: an already-installed sink (host or test) wins.
void install_seabraid_log_sink() noexcept;

// Makes INFO records reach the process stderr (WARN and ERROR still do). WHY its own call: the Seabraid default sink
// threshold is WARN, and the per-round placement record is INFO, so it was dropped; only the placement driver opts in,
// so a run without placement keeps the quiet default. Replaces the process-wide sink with the default one at INFO:
// Strata installs no sink of its own.
void show_info_records() noexcept;

}  // namespace seabraid::strata_adapter
