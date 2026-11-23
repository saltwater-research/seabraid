// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
#include "sim_providers.h"

#include <math.h>
#include <string.h>

/* The operation every kernel computes is the one the ABI documents: docs/ABI.md ("Partial routes") and the
 * reference provider in examples/minimal_provider.c define a row's output as
 *     sum over picks of weight * activation * (expert_id + 1)
 * and a SEABRAID_SKIP_EXPERT_ID pick (weight zero) contributes nothing. This file mirrors that contract and
 * invents no second definition: the two kernels differ only in loop order, never in arithmetic order. */

/* WHY a bias parameter: the "off_by_one" self-test kernel below must differ from the real one in exactly the
 * multiplier, so both share this loop and only `bias` (1 for the contract, 2 for the mutation) changes. */
static void scalar_loop(const seabraid_config_v1 *config, const seabraid_batch_v2 *batch, float *output, int bias)
{
    const float *activations = (const float *)batch->activations; /* the sim submits ROUTED_F32 only */
    size_t token, column, pick;
    for (token = 0; token < batch->tokens; token++) {
        for (column = 0; column < config->hidden; column++) {
            float sum = 0.0f;
            for (pick = token * config->top_k; pick < (token + 1) * config->top_k; pick++) {
                if (batch->expert_ids[pick] == SEABRAID_SKIP_EXPERT_ID) {
                    continue;
                }
                sum += batch->weights[pick] * activations[token * config->hidden + column] *
                       (float)(batch->expert_ids[pick] + bias);
            }
            output[token * config->hidden + column] = sum;
        }
    }
}

static void kernel_scalar(const seabraid_config_v1 *config, const seabraid_batch_v2 *batch, float *output)
{
    scalar_loop(config, batch, output, 1);
}

/* Self-test kernels: each is wrong on purpose and the verifier must fail the run (exit 2). WHY they live here and
 * not in a test file: the check is end to end (kernel -> session -> merge -> verdict -> exit code), and the CLI
 * test (check_cli.sh) can only drive the shipped binary. They are named in --help so nobody mistakes them for real kernels. */
static void kernel_off_by_one(const seabraid_config_v1 *config, const seabraid_batch_v2 *batch, float *output)
{
    scalar_loop(config, batch, output, 2);
}

static void kernel_nan(const seabraid_config_v1 *config, const seabraid_batch_v2 *batch, float *output)
{
    scalar_loop(config, batch, output, 1);
    output[0] = NAN;
}

static void kernel_inf(const seabraid_config_v1 *config, const seabraid_batch_v2 *batch, float *output)
{
    scalar_loop(config, batch, output, 1);
    output[0] = INFINITY;
}

/* Pick-outer, column-inner, unrolled by four: the layout a vectorizing provider would use. Each output
 * element still accumulates its picks in order, so results are bit-identical to kernel_scalar. */
static void add_pick(float *out, const float *act, size_t hidden, float weight, float scale)
{
    size_t column = 0;
    for (; column + 4 <= hidden; column += 4) {
        out[column] += weight * act[column] * scale;
        out[column + 1] += weight * act[column + 1] * scale;
        out[column + 2] += weight * act[column + 2] * scale;
        out[column + 3] += weight * act[column + 3] * scale;
    }
    for (; column < hidden; column++) {
        out[column] += weight * act[column] * scale;
    }
}

static void kernel_unrolled(const seabraid_config_v1 *config, const seabraid_batch_v2 *batch, float *output)
{
    const float *activations = (const float *)batch->activations;
    size_t token, pick;
    for (token = 0; token < batch->tokens; token++) {
        float *row = output + token * config->hidden;
        memset(row, 0, config->hidden * sizeof *row);
        for (pick = token * config->top_k; pick < (token + 1) * config->top_k; pick++) {
            if (batch->expert_ids[pick] != SEABRAID_SKIP_EXPERT_ID) {
                add_pick(row, activations + token * config->hidden, config->hidden,
                         batch->weights[pick], (float)(batch->expert_ids[pick] + 1));
            }
        }
    }
}

/* THE OWNER of the kernel list: sim_cli.c validates --kernel and prints --help from this table through sim_kernel_find
 * and sim_kernel_at, and sim_device_init selects from it, so a kernel exists in exactly one place. */
static const sim_kernel_info KERNELS[] = {
    {"scalar", "the reference: column-outer loop (the default)", 0, kernel_scalar},
    {"unrolled", "pick-outer, column loop unrolled by four; bit-identical results", 0, kernel_unrolled},
    {"nan", "one NaN output (the run must fail with exit 2)", 1, kernel_nan},
    {"inf", "one infinite output (the run must fail with exit 2)", 1, kernel_inf},
    {"off_by_one", "expert multiplier id + 2 (the run must fail with exit 2)", 1, kernel_off_by_one},
};

const sim_kernel_info *sim_kernel_at(size_t index)
{
    return index < sizeof KERNELS / sizeof KERNELS[0] ? &KERNELS[index] : NULL;
}

const sim_kernel_info *sim_kernel_find(const char *name)
{
    size_t i;
    for (i = 0; i < sizeof KERNELS / sizeof KERNELS[0]; i++) {
        if (strcmp(name, KERNELS[i].name) == 0) {
            return &KERNELS[i];
        }
    }
    return NULL;
}

int sim_device_init(sim_device *device, const sim_config *cfg, uint32_t index)
{
    const sim_kernel_info *kernel = sim_kernel_find(cfg->kernel);
    memset(device, 0, sizeof *device);
    if (kernel == NULL) {
        return 1;
    }
    device->kernel = kernel->fn;
    device->fail_rate = cfg->fail_rate;
    sim_rng_seed(&device->rng, cfg->seed, (uint64_t)index + 1);
    atomic_init(&device->injected, 0);
    return 0;
}

int32_t sim_device_execute(void *context, const seabraid_config_v1 *config, const seabraid_batch_v2 *batch,
                           float *output, size_t output_count)
{
    sim_device *device = (sim_device *)context;
    if (output_count != (size_t)batch->tokens * config->hidden) {
        return SEABRAID_INVALID;
    }
    /* WHY draw only when enabled: a zero rate must not consume randomness, so a failure-free run is
     * reproducible regardless of the injection code. */
    if (device->fail_rate > 0.0 && sim_rng_unit(&device->rng) < device->fail_rate) {
        atomic_fetch_add(&device->injected, 1);
        return SEABRAID_PROVIDER;
    }
    device->kernel(config, batch, output);
    return SEABRAID_OK;
}

void sim_reference(const seabraid_config_v1 *config, const seabraid_batch_v2 *batch, double *output,
                   double *magnitude)
{
    const float *activations = (const float *)batch->activations;
    size_t token, column, pick;
    for (token = 0; token < batch->tokens; token++) {
        for (column = 0; column < config->hidden; column++) {
            double sum = 0.0, mag = 0.0;
            for (pick = token * config->top_k; pick < (token + 1) * config->top_k; pick++) {
                double term = (double)batch->weights[pick] * (double)activations[token * config->hidden + column] *
                              (double)(batch->expert_ids[pick] + 1);
                sum += term;
                mag += fabs(term);
            }
            output[token * config->hidden + column] = sum;
            magnitude[token * config->hidden + column] = mag;
        }
    }
}
