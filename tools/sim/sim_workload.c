// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
#include "sim_workload.h"

#include "sim_providers.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* The checking side's buffers (reference, its rounding-bound scale, the pick scratch): split out of alloc_buffers to keep
 * each function under the repository's complexity limit. */
static int alloc_reference(sim_workload *w, size_t cells)
{
    w->reference = (double *)sim_calloc2(cells, 1, sizeof(double));
    w->magnitude = (double *)sim_calloc2(cells, 1, sizeof(double));
    w->perm = (uint32_t *)sim_calloc2(w->cfg.experts, 1, sizeof(uint32_t));
    return w->reference && w->magnitude && w->perm;
}

/* Returns 0 when any buffer cannot be had, including a size product that overflows size_t. WHY every product goes
 * through sim_calloc2: --tokens, --top-k and --hidden are user-sized uint32 values, so tokens * top_k * devices can
 * pass 2^64 (or 2^32 on a 32-bit host); an unchecked product wraps to a small block that the workload then writes past. */
static int alloc_buffers(sim_workload *w)
{
    const sim_config *cfg = &w->cfg;
    size_t picks, cells;
    if (!sim_mul_size(cfg->tokens, cfg->top_k, &picks) || !sim_mul_size(cfg->tokens, cfg->hidden, &cells)) {
        return 0;
    }
    w->activations = (float *)sim_calloc2(cells, 1, sizeof(float));
    w->ids = (int32_t *)sim_calloc2(picks, 1, sizeof(int32_t));
    w->weights = (float *)sim_calloc2(picks, 1, sizeof(float));
    w->masked_ids = (int32_t *)sim_calloc2(picks, cfg->devices, sizeof(int32_t));
    w->masked_weights = (float *)sim_calloc2(picks, cfg->devices, sizeof(float));
    w->outputs = (float *)sim_calloc2(cells, cfg->devices, sizeof(float));
    return w->activations && w->ids && w->weights && w->masked_ids && w->masked_weights && w->outputs &&
           alloc_reference(w, cells);
}

void sim_workload_free(sim_workload *w)
{
    if (w == NULL) {
        return;
    }
    free(w->activations);
    free(w->ids);
    free(w->weights);
    free(w->masked_ids);
    free(w->masked_weights);
    free(w->outputs);
    free(w->reference);
    free(w->magnitude);
    free(w->perm);
    free(w);
}

sim_workload *sim_workload_new(const sim_config *cfg)
{
    sim_workload *w = (sim_workload *)calloc(1, sizeof *w);
    if (w == NULL) {
        return NULL;
    }
    w->cfg = *cfg;
    sim_rng_seed(&w->rng, cfg->seed, 0);
    if (!alloc_buffers(w)) {
        sim_workload_free(w);
        return NULL;
    }
    for (uint32_t e = 0; e < cfg->experts; e++) {
        w->perm[e] = e;
    }
    return w;
}

static uint32_t owner_of(const sim_config *cfg, int32_t expert)
{
    return (uint32_t)(((uint64_t)(uint32_t)expert * cfg->devices) / cfg->experts);
}

/* top_k distinct experts per row via a partial Fisher-Yates over a persistent permutation (any permutation
 * of the experts is a valid start), with weights normalised to sum to one like a router's softmax. */
static void draw_row(sim_workload *w, size_t token)
{
    const sim_config *cfg = &w->cfg;
    float total = 0.0f;
    for (uint32_t k = 0; k < cfg->top_k; k++) {
        uint32_t j = k + sim_rng_below(&w->rng, cfg->experts - k);
        uint32_t swap = w->perm[k];
        w->perm[k] = w->perm[j];
        w->perm[j] = swap;
        w->ids[token * cfg->top_k + k] = (int32_t)w->perm[k];
        w->weights[token * cfg->top_k + k] = 0.1f + (float)sim_rng_unit(&w->rng);
        total += w->weights[token * cfg->top_k + k];
    }
    for (uint32_t k = 0; k < cfg->top_k; k++) {
        w->weights[token * cfg->top_k + k] /= total;
    }
}

/* WHY mask instead of compacting rows: docs/ABI.md ("Partial routes") says the engine sends each device the
 * SAME rows with the picks it does not own masked (SKIP id, weight zero), then sums the partial outputs. */
static void build_masks(sim_workload *w)
{
    const sim_config *cfg = &w->cfg;
    size_t picks = (size_t)cfg->tokens * cfg->top_k;
    for (uint32_t d = 0; d < cfg->devices; d++) {
        for (size_t i = 0; i < picks; i++) {
            int mine = owner_of(cfg, w->ids[i]) == d;
            w->masked_ids[d * picks + i] = mine ? w->ids[i] : SEABRAID_SKIP_EXPERT_ID;
            w->masked_weights[d * picks + i] = mine ? w->weights[i] : 0.0f;
        }
    }
}

void sim_workload_batch(const sim_workload *w, uint32_t device, seabraid_batch_v2 *out)
{
    size_t picks = (size_t)w->cfg.tokens * w->cfg.top_k;
    memset(out, 0, sizeof *out);
    out->abi_version = SEABRAID_BATCH_V2_VERSION;
    out->struct_size = sizeof *out;
    out->source_modalities = SEABRAID_SOURCE_TEXT;
    out->placement_epoch = w->cfg.placement_epoch;
    out->tokens = w->cfg.tokens;
    out->activation_format = SEABRAID_FORMAT_ROUTED_F32;
    out->activations = w->activations;
    out->activation_count = (size_t)w->cfg.tokens * w->cfg.hidden;
    out->expert_ids = w->masked_ids + device * picks;
    out->expert_id_count = picks;
    out->weights = w->masked_weights + device * picks;
    out->weight_count = picks;
}

size_t sim_workload_output_count(const sim_workload *w)
{
    return (size_t)w->cfg.tokens * w->cfg.hidden;
}

float *sim_workload_output(const sim_workload *w, uint32_t device)
{
    return w->outputs + device * sim_workload_output_count(w);
}

void sim_workload_generate(sim_workload *w)
{
    seabraid_config_v1 config = {SEABRAID_ABI_VERSION, w->cfg.hidden, w->cfg.experts, w->cfg.tokens, w->cfg.top_k};
    seabraid_batch_v2 full;
    for (size_t i = 0; i < sim_workload_output_count(w); i++) {
        w->activations[i] = (float)(2.0 * sim_rng_unit(&w->rng) - 1.0);
    }
    for (size_t t = 0; t < w->cfg.tokens; t++) {
        draw_row(w, t);
    }
    build_masks(w);
    sim_workload_batch(w, 0, &full);
    full.expert_ids = w->ids;
    full.weights = w->weights;
    sim_reference(&config, &full, w->reference, w->magnitude);
}

double sim_workload_max_error(const sim_workload *w, double *max_abs)
{
    double worst = 0.0;
    *max_abs = 0.0;
    for (size_t i = 0; i < sim_workload_output_count(w); i++) {
        float sum = 0.0f;
        double err, scaled;
        for (uint32_t d = 0; d < w->cfg.devices; d++) {
            sum += sim_workload_output(w, d)[i];
        }
        err = fabs((double)sum - w->reference[i]);
        if (!isfinite(err)) {
            err = INFINITY; /* WHY: a NaN would lose the `err > *max_abs` below and vanish from the report */
        }
        scaled = sim_scaled_error((double)sum, w->reference[i],
                                  sim_error_bound(w->magnitude[i], w->cfg.top_k, w->cfg.devices));
        *max_abs = err > *max_abs ? err : *max_abs;
        worst = scaled > worst ? scaled : worst; /* sim_scaled_error never returns NaN, so this keeps the maximum */
    }
    return worst;
}
