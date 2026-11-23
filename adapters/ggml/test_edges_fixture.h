// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
/* Shared fixture of the two edge executables (test_edges.c, test_edges_cache.c); include once per translation unit.
 * OWNER: this header. WHY a header and two executables: one file passed the 500-line gate, and the route/bounds
 * cases and the cache/counter cases are two concepts that share only this setup.
 * Each expert is the identity (one exactly representable Q4_0 coefficient), so a token with a one-hot activation
 * of 1.0 returns sum(weight) * silu(1) at that column and a refusal is distinguishable from a silent drop. The
 * fixtures link no runtime: this header supplies the log sink (it counts records) and the epoch. */
#ifndef SEABRAID_GGML_TEST_EDGES_FIXTURE_H
#define SEABRAID_GGML_TEST_EDGES_FIXTURE_H

#include "_seabraid_ggml_cache.h"
#include "_seabraid_ggml_stats.h"

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#ifndef _WIN32
#include <pthread.h>
#include <stdatomic.h>
#endif

#include "ggml-alloc.h"

enum { W = 32, E = 2, MAX_TOKENS = 4, MAX_PICKS = 4, SENTINEL_COUNT = MAX_TOKENS * W };

static const float SILU_ONE = 0.7310586f; /* silu(1) = 1 / (1 + e^-1); the identity experts return silu(1) * 1 */
static const float SENTINEL = -7.0f;

enum { MAX_RECORDS = 64 };
static char records[MAX_RECORDS][200];
static int record_count;

int32_t seabraid_log(uint32_t level, int32_t status, const char *message) {
    (void) level;
    (void) status;
    if (record_count < MAX_RECORDS) {
        snprintf(records[record_count], sizeof(records[0]), "%s", message);
    }
    ++record_count;
    return SEABRAID_OK;
}

uint32_t seabraid_abi_epoch(void) {
    return SEABRAID_ABI_EPOCH;
}

static int failures;
#define CHECK(cond)                                                                \
    do {                                                                           \
        if (!(cond)) {                                                             \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);        \
            ++failures;                                                            \
        }                                                                          \
    } while (0)

static inline int logged(const char *needle) {
    int found = 0;
    for (int i = 0; i < record_count && i < MAX_RECORDS; ++i) {
        found += strstr(records[i], needle) != NULL;
    }
    return found;
}

static inline void log_clear(void) {
    record_count = 0;
}

struct env {
    ggml_backend_t backend;
    struct ggml_context *ctx;
    ggml_backend_buffer_t buffer;
    seabraid_ggml_layer layer;
};

static inline bool upload_identity(struct ggml_tensor *tensor) {
    float rows[E][W][W] = {{{0}}};
    unsigned char quantized[E * W * 18]; /* 32 values per Q4_0 block: 16 packed nibble bytes + one fp16 scale */
    for (int e = 0; e < E; ++e) {
        for (int row = 0; row < W; ++row) {
            rows[e][row][row] = 1.0f;
        }
    }
    const size_t bytes = ggml_quantize_chunk(GGML_TYPE_Q4_0, &rows[0][0][0], quantized, 0, E * W, W, NULL);
    if (bytes != sizeof(quantized) || bytes != ggml_nbytes(tensor)) {
        return false;
    }
    ggml_backend_tensor_set(tensor, quantized, 0, bytes);
    return true;
}

static inline bool env_open(struct env *e) {
    ggml_backend_load_all();
    e->backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    e->ctx = e->backend ? ggml_init((struct ggml_init_params) {3 * ggml_tensor_overhead(), NULL, true}) : NULL;
    if (!e->ctx) {
        return false;
    }
    e->layer.gate = ggml_new_tensor_3d(e->ctx, GGML_TYPE_Q4_0, W, W, E);
    e->layer.up = ggml_new_tensor_3d(e->ctx, GGML_TYPE_Q4_0, W, W, E);
    e->layer.down = ggml_new_tensor_3d(e->ctx, GGML_TYPE_Q4_0, W, W, E);
    e->buffer = ggml_backend_alloc_ctx_tensors(e->ctx, e->backend);
    return e->buffer && upload_identity(e->layer.gate) && upload_identity(e->layer.up) &&
           upload_identity(e->layer.down);
}

static inline void env_close(struct env *e) {
    ggml_backend_buffer_free(e->buffer);
    ggml_free(e->ctx);
    ggml_backend_free(e->backend);
}

static float g_out[SENTINEL_COUNT];

/* One batch of `tokens` rows with a one-hot 1.0 at column 3, `top_k` picks per row, through the direct path.
 * `output_count` and the three counts are passed through so a test can lie about them. */
static inline int32_t try_batch(seabraid_ggml_cache *cache, const struct env *e, uint32_t tokens, uint32_t top_k,
                         const int32_t *ids, const float *weights) {
    static float activation[MAX_TOKENS * W];
    memset(activation, 0, sizeof(activation));
    for (uint32_t t = 0; t < tokens && t < MAX_TOKENS; ++t) {
        activation[t * W + 3] = 1.0f;
    }
    for (int i = 0; i < SENTINEL_COUNT; ++i) {
        g_out[i] = SENTINEL;
    }
    const seabraid_config_v1 config = {SEABRAID_ABI_VERSION, W, E, MAX_TOKENS, top_k};
    const seabraid_batch_v1 batch = {SEABRAID_ABI_VERSION, tokens, activation, (size_t) tokens * W,
                                     ids, (size_t) tokens * top_k, weights, (size_t) tokens * top_k};
    return seabraid_ggml_execute(cache, &e->layer, &config, &batch, g_out, (size_t) tokens * W);
}

static inline bool output_untouched(void) {
    for (int i = 0; i < SENTINEL_COUNT; ++i) {
        if (g_out[i] != SENTINEL) {
            return false;
        }
    }
    return true;
}

/* Row `token` holds sum_w * silu(1) at column 3 and zero elsewhere. */
static inline bool row_is(uint32_t token, float sum_w) {
    for (int c = 0; c < W; ++c) {
        const float expected = c == 3 ? sum_w * SILU_ONE : 0.0f;
        if (!(fabsf(g_out[token * W + c] - expected) <= 0.002f)) {
            return false;
        }
    }
    return true;
}


#endif
