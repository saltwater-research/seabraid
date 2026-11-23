/* Copyright 2026 Saltwater Research LLC
 * SPDX-License-Identifier: Apache-2.0 */
/* The public headers must compile as strict ANSI C (-std=c89 -pedantic): no // comments, no mixed
 * declarations, no C99 syntax of their own. WHY: embedders of older C toolchains include them directly.
 * Only <stdint.h> and <stddef.h> are required from the platform. This file is itself C89. */
#include "seabraid/placement.h"
#include "seabraid/seabraid.h"

int main(void)
{
    seabraid_batch_v1 batch;
    seabraid_placement_outcome_v1 outcome;
    seabraid_batch_v2 batch2 = {0};
    seabraid_provider_v3 provider3 = {0};
    batch.abi_version = SEABRAID_ABI_VERSION;
    outcome.abi_version = SEABRAID_ABI_VERSION;
    batch2.abi_version = SEABRAID_BATCH_V2_VERSION;
    provider3.abi_version = SEABRAID_PROVIDER_V3_VERSION;
    return (sizeof batch + sizeof outcome + sizeof batch2 + sizeof provider3 == 0 || batch2.tokens != 0 || provider3.flags != 0) ? 1 : (int)(outcome.abi_version - batch.abi_version);
}
