// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
/* Run results: latency statistics, the installed log sink, status reporting and the one-line JSON output. */
#ifndef SEABRAID_SIM_REPORT_H
#define SEABRAID_SIM_REPORT_H

#include "seabraid/seabraid.h"
#include "sim_cli.h"

#include <stdint.h>

typedef struct sim_result {
    double p50_us, p95_us, p99_us, max_us;
    double layers_per_s, rows_per_s;
    double max_abs_err;    /* worst |sum - reference| over verified layers: informational */
    double max_scaled_err; /* worst error in units of its float32 rounding bound (sim_error_bound); passes when sim_error_ok */
    uint64_t verified, injected, failures_observed, failures_status;
    uint64_t log_records, log_provider_records;
    uint64_t swaps;
    uint64_t rows_submitted, rows_shipped; /* summed over every session's status: the compaction evidence */
    const char *placement;
    int ok;
} sim_result;

/* Prints the failing call, its status and seabraid_last_error to stderr. Returns 0 for SEABRAID_OK, else 1. */
int sim_check(int32_t status, const char *call);

/* Installs the counting log sink (static storage: the ABI requires the context to outlive the process's use). */
int sim_log_install(void);
void sim_log_totals(uint64_t *records, uint64_t *provider_records);

/* Sorts `ns` in place and fills p50..max and layers_per_s (samples per summed second); the caller derives rows_per_s. */
void sim_summarize_latency(uint64_t *ns, size_t count, sim_result *out);

/* Exactly one JSON line on stdout. */
void sim_print_json(const sim_config *cfg, const sim_result *r);

#endif
