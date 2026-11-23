// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
/* Command-line configuration of one simulator run. See tools/sim/README comment in sim_main.c. */
#ifndef SEABRAID_SIM_CLI_H
#define SEABRAID_SIM_CLI_H

#include <stdint.h>
#include <stdio.h>

typedef struct sim_config {
    uint32_t hidden, experts, top_k, tokens, devices, layers, steps, warmup, placement_rounds;
    uint32_t compact_rows;    /* 1: devices open with SEABRAID_PROVIDER_COMPACT_ROWS */
    uint32_t validate_only;   /* 1: check the flags and exit 0 without running (matrix.sh pre-validates its grid with it) */
    uint64_t seed;
    uint64_t placement_epoch; /* stamped on every batch; 0 = unversioned */
    double fail_rate;
    char kernel[16];
} sim_config;

#define SIM_CLI_HELP 2 /* sim_parse_args result: --help or -h was given; the caller prints usage and exits 0 */
#define SIM_CLI_CAPS 3 /* sim_parse_args result: --print-caps was given; the caller calls sim_print_caps and exits 0 */

/* Fills `cfg` from defaults and argv, then validates it. Returns 0, 1 with a one-line reason in `err`, or SIM_CLI_HELP. */
int sim_parse_args(int argc, char **argv, sim_config *cfg, char *err, unsigned err_size);

/* Writes every per-flag cap, one "name value" line each, plus the combined limits as "bytes N" and "work N". */
void sim_print_caps(FILE *out);

/* Writes the flag table (names, meaning, defaults) to `out`; generated from the parser's own table. */
void sim_print_usage(FILE *out);

#endif
