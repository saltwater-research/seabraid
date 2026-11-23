// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
/* One simulated device: a v2 provider callback with a selectable kernel and seeded failure injection. */
#ifndef SEABRAID_SIM_PROVIDERS_H
#define SEABRAID_SIM_PROVIDERS_H

#include "seabraid/seabraid.h"
#include "sim_cli.h"
#include "sim_util.h"

#include <stdatomic.h>

typedef void (*sim_kernel_fn)(const seabraid_config_v1 *config, const seabraid_batch_v2 *batch, float *output);

/* One row of the kernel table (sim_providers.c): `self_test` marks a kernel that is wrong on purpose;
 * --help prints "SELF-TEST, wrong on purpose:" before the help of every such row, so the rows do not repeat it. */
typedef struct sim_kernel_info {
    const char *name;
    const char *help;
    int self_test;
    sim_kernel_fn fn;
} sim_kernel_info;

/* The kernel called `name`, or NULL. */
const sim_kernel_info *sim_kernel_find(const char *name);

/* The index-th kernel in table order, or NULL past the end: lets a caller list them without knowing the count. */
const sim_kernel_info *sim_kernel_at(size_t index);

typedef struct sim_device {
    sim_kernel_fn kernel;
    double fail_rate;
    sim_rng rng;                    /* touched only by this device's worker thread */
    atomic_uint_fast64_t injected;  /* read by the main thread after collect or close */
} sim_device;

/* Returns 0, or 1 when cfg->kernel is not a known kernel. */
int sim_device_init(sim_device *device, const sim_config *cfg, uint32_t index);

/* seabraid_provider_v3.execute for a device. The operation id is the layer; the math does not depend on it. With
 * COMPACT_ROWS the batch holds only the rows that kept a pick, so every size here comes from batch->tokens. */
int32_t sim_device_execute(void *context, const seabraid_config_v1 *config, const seabraid_batch_v2 *batch,
                           float *output, size_t output_count);

/* Single-device reference over the UNMASKED batch, in double precision. `magnitude` receives sum |term| per element:
 * the scale the float32 rounding bound is built on (sim_error_bound). */
void sim_reference(const seabraid_config_v1 *config, const seabraid_batch_v2 *batch, double *output,
                   double *magnitude);

#endif
