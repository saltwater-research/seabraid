// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
/* One call that reaches both runtime symbols the adapter needs: a refused cache_create logs through seabraid_log,
 * and the same function reads seabraid_abi_epoch on its accepting path. If the link had no runtime, this program
 * would not link at all; if it linked the wrong one, the refusal below would not come back as NULL. */
#include "seabraid_ggml.h"

#include <stdio.h>

int main(void) {
    seabraid_ggml_cache *cache = seabraid_ggml_cache_create_slots(NULL, 1);
    if (cache) {
        seabraid_ggml_cache_destroy(cache);
        fprintf(stderr, "a null backend must be refused\n");
        return 1;
    }
    seabraid_ggml_transfer_stats stats = {SEABRAID_ABI_VERSION, sizeof stats};
    if (seabraid_ggml_get_transfer_stats(&stats) != SEABRAID_OK || stats.batches != 0) {
        fprintf(stderr, "a fresh process must report zero batches\n");
        return 1;
    }
    puts("seabraid_ggml linked against the installed SDK");
    return 0;
}
