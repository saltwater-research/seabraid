// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
#ifndef SEABRAID_GGML_ROUTES_INTERNAL_H
#define SEABRAID_GGML_ROUTES_INTERNAL_H

#include <stdbool.h>

#include "seabraid/seabraid.h"

/* The two widths this adapter shares with the rest of Seabraid; src/picks/golden.txt records both (`limit` rows)
 * and test_routes.c holds these to them.
 * NARROW_ROW: rows wider than this are checked for repeats by sorting, narrower ones by scanning; the same width as
 * NARROW_ROW in src/picks.rs, so both layers change strategy together.
 * MAX_PICKS: the widest top-k the routed graph accepts (its per-token views are a fixed array). */
enum { SEABRAID_GGML_NARROW_ROW = 64, SEABRAID_GGML_MAX_PICKS = 64 };

/* NULL when every route in the batch is acceptable, else a static reason. Real
 * ids must be in range and unique per token with a finite weight; SEABRAID_SKIP_EXPERT_ID is
 * accepted only with weight exactly 0.0. Counts must already be validated. */
const char *seabraid_ggml_routes_fault(const seabraid_config_v1 *config, const seabraid_batch_v1 *batch);

/* Ids safe to upload to MUL_MAT_ID: the batch's own array, or a heap copy with
 * each skipped pick mapped to a distinct unused expert (freed by the caller
 * through *owned). NULL only on allocation failure. */
const int32_t *seabraid_ggml_graph_ids(const seabraid_config_v1 *config, const seabraid_batch_v1 *batch,
                                       int32_t **owned);

#endif
