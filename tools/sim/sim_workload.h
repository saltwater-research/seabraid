// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
/* Synthetic engine-side data: routed rows, per-device masks, device outputs and the reference result. */
#ifndef SEABRAID_SIM_WORKLOAD_H
#define SEABRAID_SIM_WORKLOAD_H

#include "seabraid/seabraid.h"
#include "sim_cli.h"
#include "sim_util.h"

typedef struct sim_workload {
    sim_config cfg;
    sim_rng rng;
    float *activations;     /* tokens * hidden */
    int32_t *ids;           /* tokens * top_k, unmasked */
    float *weights;         /* tokens * top_k, each row sums to 1 */
    int32_t *masked_ids;    /* devices * tokens * top_k */
    float *masked_weights;  /* devices * tokens * top_k */
    float *outputs;         /* devices * tokens * hidden */
    double *reference;      /* tokens * hidden */
    double *magnitude;      /* tokens * hidden: sum |term| of each reference element, the scale of its rounding bound */
    uint32_t *perm;         /* experts, scratch for distinct picks */
} sim_workload;

sim_workload *sim_workload_new(const sim_config *cfg);
void sim_workload_free(sim_workload *w);

/* Draws fresh rows, builds every device's masked batch and the reference. Not part of the timed region. */
void sim_workload_generate(sim_workload *w);

/* Device d's batch. Device d owns the experts e with e * devices / experts == d (a disjoint, contiguous slice). */
void sim_workload_batch(const sim_workload *w, uint32_t device, seabraid_batch_v2 *out);
float *sim_workload_output(const sim_workload *w, uint32_t device);
size_t sim_workload_output_count(const sim_workload *w);

/* Sums the device outputs in device order (the engine's merge). Returns the worst sim_scaled_error against the
 * reference (judged by sim_error_ok; +inf for any non-finite output) and stores the worst absolute |sum - reference|
 * in *max_abs (also +inf for a non-finite output). */
double sim_workload_max_error(const sim_workload *w, double *max_abs);

#endif
