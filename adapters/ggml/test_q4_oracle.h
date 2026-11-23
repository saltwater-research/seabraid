// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
/* The Q4_0 oracle shared by the fixture halves: the exactly representable projections, the scalar reference
 * (oracle_add) and the comparison. WHY a header and not a third object: test_q4.c may be embedded in a host
 * binary (seabraid_ggml_fixture), and a header keeps every helper private to the file that includes it. */
#ifndef SEABRAID_GGML_TEST_Q4_ORACLE_H
#define SEABRAID_GGML_TEST_Q4_ORACLE_H

#include "seabraid_ggml.h"

#include <math.h>

#include "ggml-alloc.h"

enum { WIDTH = 32, EXPERTS = 2, TOKENS = 2 };

struct projection {
    int shift[EXPERTS];
    float scale[EXPERTS];
};

static const struct projection gate = {{1, 3}, {1.0f, -2.0f}};
static const struct projection up   = {{1, 3}, {2.0f,  1.0f}};
static const struct projection down = {{2, 4}, {1.0f,  3.0f}};

static const int inputs[TOKENS] = {7, 2};
static const float activation[TOKENS * WIDTH] = {[7] = 1.5f, [WIDTH + 2] = -0.75f};
static const int32_t experts[TOKENS * EXPERTS] = {0, 1, 1, 0};
static const float weights[TOKENS * EXPERTS] = {0.25f, 0.75f, 0.6f, 0.4f};

static inline bool upload(struct ggml_tensor *tensor, const struct projection *projection) {
    float rows[EXPERTS][WIDTH][WIDTH] = {0};
    /* 32 values per Q4_0 block: 16 packed nibble bytes + one fp16 scale. */
    unsigned char quantized[EXPERTS * WIDTH * 18];
    for (int e = 0; e < EXPERTS; ++e) {
        for (int row = 0; row < WIDTH; ++row) {
            rows[e][row][(row + projection->shift[e]) % WIDTH] = projection->scale[e];
        }
    }
    const size_t bytes = ggml_quantize_chunk(GGML_TYPE_Q4_0, &rows[0][0][0], quantized,
                                             0, EXPERTS * WIDTH, WIDTH, NULL);
    if (bytes != sizeof(quantized) || bytes != ggml_nbytes(tensor)) {
        return false;
    }
    ggml_backend_tensor_set(tensor, quantized, 0, bytes);
    return true;
}

static inline void oracle_add(float *output, const seabraid_batch_v1 *batch, size_t stride, size_t reduce_picks,
                       const struct projection *gate_projection, const struct projection *up_projection) {
    for (size_t token = 0; token < batch->tokens; ++token) {
        const float value = batch->activations[token * WIDTH + (size_t) inputs[token]];
        for (size_t pick = 0; pick < reduce_picks; ++pick) {
            const size_t route = token * stride + pick;
            const int e = batch->expert_ids[route];
            if (e < 0) {
                continue; /* skipped: owned by another device, contributes nothing */
            }
            const int row = (inputs[token] - gate_projection->shift[e] - down.shift[e] + WIDTH) % WIDTH;
            const float z = value * gate_projection->scale[e];
            const float swiglu = (z / (1.0f + expf(-z))) * (value * up_projection->scale[e]);
            output[token * WIDTH + (size_t) row] += batch->weights[route] * swiglu * down.scale[e];
        }
    }
}

static inline bool rows_match(const float *output, const float *expected, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        const float tolerance = 0.003f * (1.0f + fabsf(expected[i]));
        if (!(fabsf(expected[i] - output[i]) <= tolerance)) {
            return false;
        }
    }
    return true;
}

static inline bool matches_expected(const float *output, const seabraid_batch_v1 *batch, size_t top_k,
                             const struct projection *gate_projection,
                             const struct projection *up_projection) {
    float expected[TOKENS * WIDTH] = {0};
    oracle_add(expected, batch, top_k, top_k, gate_projection, up_projection);
    return rows_match(output, expected, batch->tokens * WIDTH);
}

/* test_q4_scheduler.c: the host-scheduler continuation fixture (0 on success, 20..22 on a fault). */
int seabraid_ggml_q4_run_scheduler(ggml_backend_t backend, seabraid_ggml_layer layer);

#endif
