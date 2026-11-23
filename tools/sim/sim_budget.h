// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
/* The simulator's combined budget: how much memory and time one configuration may cost. OWNER of both limits and of the
 * estimates they are compared with; sim_cli.c refuses a configuration past either, and the JSON line prints the memory
 * estimate. It reads the config and asks sim_placement whether the placement phase runs; it never imports the parser. */
#ifndef SEABRAID_SIM_BUDGET_H
#define SEABRAID_SIM_BUDGET_H

#include "sim_cli.h"

/* WHY a combined byte cap: the per-flag caps alone allow tokens 2^20 x hidden 4096 (16 GiB of activations alone) or
 * layers 65536 x steps 10^6 (500 GB of latency samples); each flag is within its cap and the product is not. 4 GiB
 * is far above any real run and still allocatable on the machines this runs on. */
#define SIM_MAX_BYTES 4294967296.0

/* WHY a work cap of its own: SIM_MAX_BYTES bounds memory only. Time is the other product the per-flag caps leave open:
 * (warmup + steps) * layers layer calls, each costing per-element work and a fixed call cost per device. The unit is
 * one nanosecond as measured on the reference box (see sim_estimate_work), so 6e11 is about ten minutes at most, and
 * the estimate is at or above the measured time; anything past it is refused up front, where --validate-only and
 * matrix.sh see it, instead of running for hours. The time is that of an optimised build (-O2, which the tool's CMake
 * adds when no build type is given); a Debug build ran up to 3.6 times over it when re-measured (19 in review). */
#define SIM_MAX_WORK 600000000000.0

/* Upper bound on the peak resident bytes of a run with `cfg` (the measurements behind its constants: sim_budget.c). */
double sim_estimate_bytes(const sim_config *cfg);

/* Estimated running time of a run with `cfg`, in nanoseconds on the reference box. */
double sim_estimate_work(const sim_config *cfg);

#endif
