// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
/* Placement exercise: fill, then rounds of plan/apply against drifting synthetic heat. */
#ifndef SEABRAID_SIM_PLACEMENT_H
#define SEABRAID_SIM_PLACEMENT_H

#include "sim_cli.h"

#include <stdint.h>

typedef struct sim_placement_result {
    uint64_t swaps;      /* committed outcomes across all rounds */
    uint64_t plans;      /* rounds whose plan had at least one promotion */
    const char *note;    /* "off", "ok" or "skipped: <reason>" (static string) */
} sim_placement_result;

/* Why the placement phase will not run for `cfg` ("off" or "skipped: <reason>", static strings), or NULL when it will:
 * sim_cli.c asks it so the memory and work estimates count placement only when it really runs. */
const char *sim_placement_skip_reason(const sim_config *cfg);

/* Returns 0 on success (including a skip, which is explained in `note`) and 1 on an unexpected ABI status or
 * an outcome that is not a commit (the in-process provider always acknowledges, so anything else is a bug). */
int sim_placement_run(const sim_config *cfg, sim_placement_result *out);

#endif
