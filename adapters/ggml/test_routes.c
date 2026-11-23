// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
/* Route-check order and cost, on seabraid_ggml_routes_fault alone (no ggml, no backend). OWNER: this file for the
 * rules "the lowest-index defect is reported, and at one pick range, then finite, then repeat" (the order of
 * src/picks.rs), "a skip's zero weight is a bit test" and "a 64,000-pick row is checked in O(top_k log top_k)".
 * The same source builds twice: seabraid_ggml_routes (the library's routes) and seabraid_ggml_routes_fastmath
 * (this file plus _seabraid_ggml_routes.c compiled with -ffast-math, which may fold a float compare of NaN).
 * NaN and Inf are made from bits so this file never leans on a float compare itself.
 * Its table half lives in test_routes_golden.h: the adapter to src/picks/golden.txt (the pick rule's table, OWNER src/picks.rs) and to the widths
 * `NARROW_ROW` and `MAX_PICKS` recorded there. FAILURE: fail-fast; a failed check prints its row and exits 1. */
#include "_seabraid_ggml_routes.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int failures;
#define CHECK(cond)                                                         \
    do {                                                                    \
        if (!(cond)) {                                                      \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++failures;                                                     \
        }                                                                   \
    } while (0)

static float from_bits(uint32_t bits) {
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

#define NAN_W from_bits(0x7fc00000u)
#define INF_W from_bits(0x7f800000u)
#define NEG_ZERO from_bits(0x80000000u)
#define DENORMAL from_bits(0x00000001u)

/* The refusal texts of the adapter, spelled once. */
static const char REPEAT[] = "routes hold a repeated expert id";
static const char RANGE[] = "routes hold an out-of-range expert id";
static const char FINITE[] = "routes hold a non-finite weight";
static const char SKIPPED[] = "a skipped pick has a nonzero weight";

#define MAX_ROW 128

/* One token of `top_k` picks over `experts` experts: ids 0..top_k-1 and weight 0.5, ready to be spoiled. */
struct row {
    int32_t ids[MAX_ROW];
    float weights[MAX_ROW];
    uint32_t top_k;
    uint32_t experts;
};

static void row_init(struct row *row, uint32_t top_k) {
    row->top_k = top_k;
    row->experts = top_k + 100;
    for (uint32_t i = 0; i < top_k; ++i) {
        row->ids[i] = (int32_t) i;
        row->weights[i] = 0.5f;
    }
}

static const char *row_fault(const struct row *row) {
    const seabraid_config_v1 config = {SEABRAID_ABI_VERSION, 32, row->experts, 1, row->top_k};
    const seabraid_batch_v1 batch = {SEABRAID_ABI_VERSION, 1, NULL, 0, row->ids, row->top_k, row->weights, row->top_k};
    return seabraid_ggml_routes_fault(&config, &batch);
}

static void check_text(const struct row *row, const char *expected, int line) {
    const char *fault = row_fault(row);
    if (expected ? (!fault || strcmp(fault, expected) != 0) : fault != NULL) {
        fprintf(stderr, "FAIL line %d (top_k %u): expected \"%s\", got \"%s\"\n", line, row->top_k,
                expected ? expected : "(accepted)", fault ? fault : "(accepted)");
        ++failures;
    }
}
#define EXPECT(row, text) check_text(&(row), (text), __LINE__)

/* The same defects in a narrow row (top_k 8) and in a wide one (top_k 100): both strategies must agree. */
static void defects_by_index(uint32_t top_k) {
    const uint32_t early = top_k / 4, late = top_k - 3; /* early < late in both widths */
    struct row row;

    row_init(&row, top_k); /* of two defects at different picks the lower index is reported */
    row.weights[early] = NAN_W;
    row.ids[late] = (int32_t) row.experts; /* out of range */
    EXPECT(row, FINITE);
    row_init(&row, top_k);
    row.ids[early] = (int32_t) row.experts;
    row.weights[late] = NAN_W;
    EXPECT(row, RANGE);

    row_init(&row, top_k); /* a repeat before a bad weight, and the reverse */
    row.ids[early] = 0;
    row.weights[late] = INF_W;
    EXPECT(row, REPEAT);
    row_init(&row, top_k);
    row.weights[early] = INF_W;
    row.ids[late] = 0;
    EXPECT(row, FINITE);

    row_init(&row, top_k); /* one pick with two defects: range, then finite, then repeat */
    row.ids[late] = 0;
    row.weights[late] = NAN_W;
    EXPECT(row, FINITE);
    row_init(&row, top_k);
    row.ids[late] = (int32_t) row.experts;
    row.weights[late] = NAN_W;
    EXPECT(row, RANGE);
    row_init(&row, top_k);
    row.ids[late] = -2;
    row.weights[late] = NAN_W;
    EXPECT(row, RANGE);
    row_init(&row, top_k);
    row.ids[late] = 0;
    EXPECT(row, REPEAT);

    row_init(&row, top_k); /* a skip is never a repeat, and its own fault is the first of its pick */
    row.ids[early] = -1;
    row.weights[early] = 0.0f;
    row.ids[late] = -1;
    row.weights[late] = 0.0f;
    EXPECT(row, NULL);
    row.weights[late] = 1.0f;
    EXPECT(row, SKIPPED);
}

/* Two tokens: a defect in the second row never hides the first row's, whichever strategy checks it. */
static void first_row_wins(void) {
    int32_t ids[2 * 100];
    float weights[2 * 100];
    for (int i = 0; i < 200; ++i) {
        ids[i] = i % 100;
        weights[i] = 0.5f;
    }
    ids[99] = 0;              /* row 0: a repeat at its last pick */
    weights[100 + 4] = NAN_W; /* row 1: a bad weight much earlier in its own row */
    const seabraid_config_v1 config = {SEABRAID_ABI_VERSION, 32, 200, 2, 100};
    const seabraid_batch_v1 batch = {SEABRAID_ABI_VERSION, 2, NULL, 0, ids, 200, weights, 200};
    const char *fault = seabraid_ggml_routes_fault(&config, &batch);
    CHECK(fault && strcmp(fault, REPEAT) == 0);
}

/* A skip carries zero weight of either sign; NaN, Inf, a denormal and any nonzero value are refused. */
static void skip_weights(void) {
    struct row row;
    row_init(&row, 4);
    row.ids[2] = -1;
    row.weights[2] = 0.0f;
    EXPECT(row, NULL);
    row.weights[2] = NEG_ZERO;
    EXPECT(row, NULL);
    row.weights[2] = NAN_W;
    EXPECT(row, SKIPPED);
    row.weights[2] = from_bits(0xffc00001u); /* a negative NaN with a payload */
    EXPECT(row, SKIPPED);
    row.weights[2] = INF_W;
    EXPECT(row, SKIPPED);
    row.weights[2] = DENORMAL;
    EXPECT(row, SKIPPED);
    row.weights[2] = from_bits(0x80000001u);
    EXPECT(row, SKIPPED);
    row.weights[2] = -0.5f;
    EXPECT(row, SKIPPED);
}

#ifndef SEABRAID_ROUTES_FASTMATH
/* One 64,000-pick row, the width src/picks.rs measured at 12.7 s with the quadratic scan: CPU time under a
 * quarter second (the sort takes a few milliseconds), same refusal text, and a clean row is accepted. */
static void wide_row_is_fast(void) {
    enum { WIDTH = 64000 };
    int32_t *ids = malloc(WIDTH * sizeof(*ids));
    float *weights = malloc(WIDTH * sizeof(*weights));
    CHECK(ids && weights);
    if (!ids || !weights) {
        free(ids);
        free(weights);
        return;
    }
    for (int i = 0; i < WIDTH; ++i) {
        ids[i] = i;
        weights[i] = 0.5f;
    }
    const seabraid_config_v1 config = {SEABRAID_ABI_VERSION, 32, WIDTH, 1, WIDTH};
    const seabraid_batch_v1 batch = {SEABRAID_ABI_VERSION, 1, NULL, 0, ids, WIDTH, weights, WIDTH};
    const clock_t start = clock();
    CHECK(seabraid_ggml_routes_fault(&config, &batch) == NULL);
    ids[WIDTH - 1] = 0;
    const char *fault = seabraid_ggml_routes_fault(&config, &batch);
    const double seconds = (double) (clock() - start) / CLOCKS_PER_SEC;
    CHECK(fault && strcmp(fault, REPEAT) == 0);
    if (seconds >= 0.25) {
        fprintf(stderr, "FAIL: two 64,000-pick rows took %.3f s of CPU (limit 0.25)\n", seconds);
        ++failures;
    }
    printf("two 64,000-pick rows checked in %.4f s\n", seconds);
    free(ids);
    free(weights);
}
#endif

#include "test_routes_golden.h"

int main(void) {
    golden_table();
    defects_by_index(8);
    defects_by_index(100);
    first_row_wins();
    skip_weights();
#ifndef SEABRAID_ROUTES_FASTMATH
    wide_row_is_fast();
#endif
    if (failures) {
        fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    puts("routes order, skip bits and wide rows ok");
    return 0;
}
