/* Copyright 2026 Saltwater Research LLC
 * SPDX-License-Identifier: Apache-2.0 */
/* Three numbers must agree for an installed SDK: the epoch the package config republishes (CONFIG_EPOCH, set by
 * CMake from Seabraid_ABI_EPOCH), the epoch in the installed header, and the one the loaded library reports. */
#include <stdio.h>

#include "seabraid/seabraid.h"

int main(void)
{
    if ((unsigned)CONFIG_EPOCH != (unsigned)SEABRAID_ABI_EPOCH) {
        fprintf(stderr, "package config epoch %u != header epoch %u\n", (unsigned)CONFIG_EPOCH, (unsigned)SEABRAID_ABI_EPOCH);
        return 1;
    }
    if (seabraid_abi_epoch() != SEABRAID_ABI_EPOCH) {
        fprintf(stderr, "library epoch %u != header epoch %u\n", (unsigned)seabraid_abi_epoch(), (unsigned)SEABRAID_ABI_EPOCH);
        return 1;
    }
    return 0;
}
