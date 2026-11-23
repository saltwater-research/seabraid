// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
/* The transfer counter must equal what the shapes say moved: per batch the activation (tokens x hidden F32),
 * the ids (tokens x top_k I32) and the weights (tokens x top_k F32) go up in three uploads and tokens x hidden
 * F32 comes back in one download. The expert weights are all zero here; only the byte counts are under test. */
#include "seabraid_ggml.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "ggml-alloc.h"

enum { HIDDEN = 32, EXPERTS = 2, TOP_K = 2, MAX_TOKENS = 4 };

/* The fixture links no runtime, so it supplies the log sink (and keeps the last record) and the epoch. */
static char last_log[200];

int32_t seabraid_log(uint32_t level, int32_t status, const char *message) {
    (void) level;
    (void) status;
    snprintf(last_log, sizeof(last_log), "%s", message);
    return SEABRAID_OK;
}

uint32_t seabraid_abi_epoch(void) {
    return SEABRAID_ABI_EPOCH;
}

static bool zero_weights(struct ggml_tensor *tensor) {
    float rows[EXPERTS * HIDDEN * HIDDEN] = {0};
    unsigned char quantized[EXPERTS * HIDDEN * 18]; /* Q4_0: 18 bytes per 32 values */
    const size_t bytes = ggml_quantize_chunk(GGML_TYPE_Q4_0, rows, quantized, 0, EXPERTS * HIDDEN, HIDDEN, NULL);
    if (bytes != sizeof(quantized) || bytes != ggml_nbytes(tensor)) {
        return false;
    }
    ggml_backend_tensor_set(tensor, quantized, 0, bytes);
    return true;
}

static int32_t run_batch(seabraid_ggml_binding *binding, uint32_t tokens) {
    static const float activation[MAX_TOKENS * HIDDEN] = {[3] = 1.0f};
    static const int32_t ids[MAX_TOKENS * TOP_K] = {0, 1, 0, 1, 0, 1, 0, 1};
    static const float weights[MAX_TOKENS * TOP_K] = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
    float output[MAX_TOKENS * HIDDEN];
    const seabraid_config_v1 config = {SEABRAID_ABI_VERSION, HIDDEN, EXPERTS, MAX_TOKENS, TOP_K};
    const seabraid_batch_v1 batch = {SEABRAID_ABI_VERSION, tokens, activation, (size_t) tokens * HIDDEN,
                                     ids, (size_t) tokens * TOP_K, weights, (size_t) tokens * TOP_K};
    return seabraid_ggml_execute(binding->cache, &binding->layer, &config, &batch, output, (size_t) tokens * HIDDEN);
}

static bool read_stats(seabraid_ggml_transfer_stats *stats) {
    *stats = (seabraid_ggml_transfer_stats) {SEABRAID_ABI_VERSION, sizeof(*stats)};
    return seabraid_ggml_get_transfer_stats(stats) == SEABRAID_OK;
}

static bool all_zero(const seabraid_ggml_transfer_stats *stats) {
    return !stats->sent_bytes && !stats->sent_transfers && !stats->returned_bytes &&
           !stats->returned_transfers && !stats->batches && !stats->weight_bytes_staged;
}

static bool refusal_checks(void) {
    seabraid_ggml_transfer_stats bad_version = {SEABRAID_ABI_VERSION + 1, sizeof(bad_version)};
    seabraid_ggml_transfer_stats short_struct = {SEABRAID_ABI_VERSION, 4};
    /* One byte below the v1 size: the largest size the old code accepted and then overran. */
    seabraid_ggml_transfer_stats just_short = {SEABRAID_ABI_VERSION, sizeof(just_short) - 1};
    return seabraid_ggml_get_transfer_stats(NULL) == SEABRAID_INVALID &&
           seabraid_ggml_get_transfer_stats(&bad_version) == SEABRAID_INVALID &&
           seabraid_ggml_get_transfer_stats(&short_struct) == SEABRAID_INVALID &&
           seabraid_ggml_get_transfer_stats(&just_short) == SEABRAID_INVALID &&
           just_short.struct_size == sizeof(just_short) - 1 && short_struct.struct_size == 4;
}

/* A newer caller with a larger struct gets the v1 bytes, a written struct_size of the v1 size (not its own),
 * and an untouched tail. */
static bool larger_caller_checks(void) {
    struct { seabraid_ggml_transfer_stats v1; uint64_t tail; } bigger;
    memset(&bigger, 0xAB, sizeof(bigger));
    bigger.v1.abi_version = SEABRAID_ABI_VERSION;
    bigger.v1.struct_size = sizeof(bigger);
    if (seabraid_ggml_get_transfer_stats(&bigger.v1) != SEABRAID_OK) {
        return false;
    }
    return bigger.v1.struct_size == sizeof(seabraid_ggml_transfer_stats) && bigger.tail == 0xABABABABABABABABull;
}

static bool header_checks(void) {
    return refusal_checks() && larger_caller_checks();
}

/* Sizes follow from the shapes, not from the adapter: 3 tokens up 432, down 384; 1 token up 144, down 128. */
static bool counts_exact(const seabraid_ggml_transfer_stats *stats) {
    const uint64_t f = sizeof(float);
    const uint64_t up3 = 3 * HIDDEN * f + 3 * TOP_K * sizeof(int32_t) + 3 * TOP_K * f;
    const uint64_t up1 = 1 * HIDDEN * f + 1 * TOP_K * sizeof(int32_t) + 1 * TOP_K * f;
    const uint64_t down3 = 3 * HIDDEN * f;
    const uint64_t down1 = 1 * HIDDEN * f;
    return stats->struct_size == sizeof(*stats) && stats->sent_bytes == 2 * up3 + up1 &&
           stats->sent_transfers == 9 && stats->returned_bytes == 2 * down3 + down1 &&
           stats->returned_transfers == 3 && stats->batches == 3 && stats->weight_bytes_staged == 0;
}

static bool run_known_batches(seabraid_ggml_binding *binding) {
    return run_batch(binding, 3) == SEABRAID_OK && run_batch(binding, 3) == SEABRAID_OK &&
           run_batch(binding, 1) == SEABRAID_OK;
}

static int check_counts(seabraid_ggml_binding *binding) {
    seabraid_ggml_transfer_stats stats;
    seabraid_ggml_reset_transfer_stats();
    if (!header_checks() || !read_stats(&stats) || !all_zero(&stats)) {
        return 2;
    }
    if (!run_known_batches(binding)) {
        return 3;
    }
    /* A refused batch (more tokens than max_tokens) moves nothing. */
    if (run_batch(binding, MAX_TOKENS + 1) == SEABRAID_OK) {
        return 4;
    }
    if (!read_stats(&stats) || !counts_exact(&stats)) {
        return 5;
    }
    seabraid_ggml_reset_transfer_stats();
    return read_stats(&stats) && all_zero(&stats) ? 0 : 6;
}

static int check_summary(seabraid_ggml_binding *binding) {
    seabraid_ggml_reset_transfer_stats();
    if (run_batch(binding, 2) != SEABRAID_OK) {
        return 7;
    }
    last_log[0] = '\0';
    seabraid_ggml_cache_destroy(binding->cache); /* teardown logs the summary once */
    binding->cache = NULL;
    /* 2 tokens: up 2*32*4 + 2*2*4 + 2*2*4 = 288, down 256; three uploads, one download. */
    return strcmp(last_log, "transfers: 1 batches, 288 bytes sent in 3 uploads, 256 bytes returned in 1 downloads") == 0
               ? 0 : 8;
}

int main(void) {
    ggml_backend_load_all();
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    struct ggml_context *ctx = backend ? ggml_init((struct ggml_init_params) {3 * ggml_tensor_overhead(), NULL, true}) : NULL;
    if (!ctx) {
        return 1;
    }
    seabraid_ggml_binding binding = {seabraid_ggml_cache_create(backend), {
        ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_0, HIDDEN, HIDDEN, EXPERTS),
        ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_0, HIDDEN, HIDDEN, EXPERTS),
        ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_0, HIDDEN, HIDDEN, EXPERTS)}};
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    int result = 1;
    if (binding.cache && buffer && zero_weights(binding.layer.gate) && zero_weights(binding.layer.up) &&
        zero_weights(binding.layer.down)) {
        result = check_counts(&binding);
        if (result == 0) {
            result = check_summary(&binding);
        }
    }
    seabraid_ggml_cache_destroy(binding.cache);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return result;
}
