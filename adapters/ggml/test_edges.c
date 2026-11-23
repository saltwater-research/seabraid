// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
/* Edge cases of the direct execute path on the CPU backend: route faults (range, repeat, non-finite weight, skips),
 * token and count bounds (0, 1, max, past max, UINT32_MAX), top_k == experts and top_k past it. OWNER: this file;
 * test_q4.c keeps the oracle and scheduler fixtures, test_edges_cache.c the cache, counter and append cases. */
#include "test_edges_fixture.h"

/* E1: every route fault is refused with its own reason, leaves the output and the workspace untouched, and the
 * cache still serves a valid batch afterwards. Two tokens, top_k 2; the fault sits in the second token. */
struct route_case {
    const char *name;
    int32_t ids[4];
    float weights[4];
    const char *needle; /* NULL: accepted */
};

static const struct route_case ROUTE_CASES[] = {
    {"id == experts", {0, 1, 0, 2}, {0.25f, 0.75f, 0.5f, 0.5f}, "out-of-range expert id"},
    {"id -2", {0, 1, -2, 1}, {0.25f, 0.75f, 0.5f, 0.5f}, "out-of-range expert id"},
    {"id INT32_MIN", {0, 1, INT32_MIN, 1}, {0.25f, 0.75f, 0.5f, 0.5f}, "out-of-range expert id"},
    {"id INT32_MAX", {0, 1, 0, INT32_MAX}, {0.25f, 0.75f, 0.5f, 0.5f}, "out-of-range expert id"},
    {"repeated id", {0, 1, 1, 1}, {0.25f, 0.75f, 0.5f, 0.5f}, "repeated expert id"},
    {"NaN weight", {0, 1, 0, 1}, {0.25f, 0.75f, 0.5f, NAN}, "non-finite weight"},
    {"Inf weight", {0, 1, 0, 1}, {0.25f, 0.75f, 0.5f, INFINITY}, "non-finite weight"},
    {"-Inf weight", {0, 1, 0, 1}, {0.25f, 0.75f, 0.5f, -INFINITY}, "non-finite weight"},
    {"skip, nonzero weight", {0, 1, 0, -1}, {0.25f, 0.75f, 0.5f, 0.5f}, "skipped pick has a nonzero weight"},
    {"skip, NaN weight", {0, 1, 0, -1}, {0.25f, 0.75f, 0.5f, NAN}, "skipped pick has a nonzero weight"},
    {"skip, negative zero", {0, 1, 0, -1}, {0.25f, 0.75f, 0.5f, -0.0f}, NULL},
    {"negative weight", {0, 1, 0, 1}, {0.25f, 0.75f, -0.5f, 0.5f}, NULL},
};

/* One case: a refusal is INVALID with exactly one record naming its reason, no output byte and no preparation. */
static bool route_case_holds(seabraid_ggml_cache *cache, const struct env *e, const struct route_case *c) {
    const uint64_t before = cache->preparations;
    log_clear();
    const int32_t status = try_batch(cache, e, 2, 2, c->ids, c->weights);
    if (!c->needle) {
        return status == SEABRAID_OK;
    }
    return status == SEABRAID_INVALID && output_untouched() && cache->preparations == before && logged(c->needle) == 1;
}

static void routes_faults(seabraid_ggml_cache *cache, const struct env *e) {
    static const float good[4] = {0.25f, 0.75f, 0.5f, 0.5f};
    for (size_t i = 0; i < sizeof(ROUTE_CASES) / sizeof(ROUTE_CASES[0]); ++i) {
        if (!route_case_holds(cache, e, &ROUTE_CASES[i])) {
            fprintf(stderr, "route case \"%s\" failed (records %d)\n", ROUTE_CASES[i].name, record_count);
            ++failures;
        }
    }
    CHECK(try_batch(cache, e, 2, 2, (const int32_t[]) {0, 1, 1, 0}, good) == SEABRAID_OK);
    CHECK(row_is(0, 1.0f) && row_is(1, 1.0f)); /* the refusals above left the workspace usable */
}

/* E2: token and count bounds. Tokens 0, 1, max and past max (including UINT32_MAX) and every count that
 * disagrees with the shapes; the bound faults name themselves and write nothing. */
static void token_and_count_bounds(seabraid_ggml_cache *cache, const struct env *e) {
    static const int32_t ids[MAX_TOKENS * 2] = {0, 1, 1, 0, 0, 1, 1, 0};
    static const float weights[MAX_TOKENS * 2] = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
    log_clear();
    CHECK(try_batch(cache, e, 0, 2, ids, weights) == SEABRAID_INVALID && output_untouched());
    CHECK(logged("outside the supported bounds") == 1);
    CHECK(try_batch(cache, e, 1, 2, ids, weights) == SEABRAID_OK && row_is(0, 1.0f));
    CHECK(try_batch(cache, e, MAX_TOKENS, 2, ids, weights) == SEABRAID_OK);
    for (uint32_t t = 0; t < MAX_TOKENS; ++t) {
        CHECK(row_is(t, 1.0f));
    }
    log_clear();
    CHECK(try_batch(cache, e, MAX_TOKENS + 1, 2, ids, weights) == SEABRAID_INVALID && output_untouched());
    CHECK(try_batch(cache, e, UINT32_MAX, 2, ids, weights) == SEABRAID_INVALID && output_untouched());
    CHECK(logged("outside the supported bounds") == 2);
    log_clear();
    CHECK(try_batch(cache, e, 2, 0, ids, weights) == SEABRAID_INVALID);  /* top_k 0 */
    CHECK(try_batch(cache, e, 1, 65, ids, weights) == SEABRAID_INVALID); /* top_k past the 64-pick bound */
    CHECK(logged("outside the supported bounds") == 2);

    const seabraid_config_v1 config = {SEABRAID_ABI_VERSION, W, E, MAX_TOKENS, 2};
    float activation[2 * W] = {0};
    seabraid_batch_v1 batch = {SEABRAID_ABI_VERSION, 2, activation, 2 * W, ids, 4, weights, 4};
    log_clear();
    CHECK(seabraid_ggml_execute(cache, &e->layer, &config, &batch, g_out, 2 * W - 1) == SEABRAID_INVALID);
    CHECK(seabraid_ggml_execute(cache, &e->layer, &config, &batch, g_out, 2 * W + 1) == SEABRAID_INVALID);
    batch.activation_count = 2 * W - 1;
    CHECK(seabraid_ggml_execute(cache, &e->layer, &config, &batch, g_out, 2 * W) == SEABRAID_INVALID);
    batch.activation_count = 2 * W;
    batch.expert_id_count = 3;
    CHECK(seabraid_ggml_execute(cache, &e->layer, &config, &batch, g_out, 2 * W) == SEABRAID_INVALID);
    batch.expert_id_count = 4;
    batch.weight_count = 5;
    CHECK(seabraid_ggml_execute(cache, &e->layer, &config, &batch, g_out, 2 * W) == SEABRAID_INVALID);
    CHECK(logged("element counts do not match") == 5);
    batch.weight_count = 4;

    log_clear();
    batch.activations = NULL;
    CHECK(seabraid_ggml_execute(cache, &e->layer, &config, &batch, g_out, 2 * W) == SEABRAID_INVALID);
    batch.activations = activation;
    batch.expert_ids = NULL;
    CHECK(seabraid_ggml_execute(cache, &e->layer, &config, &batch, g_out, 2 * W) == SEABRAID_INVALID);
    batch.expert_ids = ids;
    batch.weights = NULL;
    CHECK(seabraid_ggml_execute(cache, &e->layer, &config, &batch, g_out, 2 * W) == SEABRAID_INVALID);
    batch.weights = weights;
    CHECK(logged("a batch buffer is null") == 3);

    log_clear();
    CHECK(seabraid_ggml_execute(NULL, &e->layer, &config, &batch, g_out, 2 * W) == SEABRAID_INVALID);
    CHECK(seabraid_ggml_execute(cache, NULL, &config, &batch, g_out, 2 * W) == SEABRAID_INVALID);
    CHECK(seabraid_ggml_execute(cache, &e->layer, NULL, &batch, g_out, 2 * W) == SEABRAID_INVALID);
    CHECK(seabraid_ggml_execute(cache, &e->layer, &config, NULL, g_out, 2 * W) == SEABRAID_INVALID);
    CHECK(seabraid_ggml_execute(cache, &e->layer, &config, &batch, NULL, 2 * W) == SEABRAID_INVALID);
    CHECK(logged("a required argument is null") == 5);
    seabraid_config_v1 bad_config = config;
    bad_config.abi_version = SEABRAID_ABI_VERSION + 1;
    seabraid_batch_v1 bad_batch = batch;
    bad_batch.abi_version = SEABRAID_ABI_VERSION + 1;
    log_clear();
    CHECK(seabraid_ggml_execute(cache, &e->layer, &bad_config, &batch, g_out, 2 * W) == SEABRAID_INVALID);
    CHECK(seabraid_ggml_execute(cache, &e->layer, &config, &bad_batch, g_out, 2 * W) == SEABRAID_INVALID);
    CHECK(logged("ABI version") == 2);
}

/* E3: top_k == experts, skips that fill every pick, and top_k past the expert count. */
static void skips_and_top_k(seabraid_ggml_cache *cache, const struct env *e) {
    static const float zero[MAX_TOKENS * 3] = {0};
    static const int32_t all_skip[MAX_TOKENS * 3] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
    CHECK(try_batch(cache, e, MAX_TOKENS, 2, all_skip, zero) == SEABRAID_OK); /* top_k == experts, every pick skipped */
    for (uint32_t t = 0; t < MAX_TOKENS; ++t) {
        CHECK(row_is(t, 0.0f));
    }
    static const int32_t one_real[4] = {1, -1, -1, 0};
    static const float one_real_w[4] = {0.5f, 0.0f, 0.0f, 0.25f};
    CHECK(try_batch(cache, e, 2, 2, one_real, one_real_w) == SEABRAID_OK); /* a real pick beside a skip, both orders */
    CHECK(row_is(0, 0.5f) && row_is(1, 0.25f));

    log_clear();
    static const int32_t three[6] = {0, 1, -1, 0, 1, -1};
    static const float three_w[6] = {0.5f, 0.5f, 0.0f, 0.5f, 0.5f, 0.0f};
    CHECK(try_batch(cache, e, 2, 3, three, three_w) == SEABRAID_INVALID && output_untouched());
    CHECK(logged("skipped picks need top-k no larger than the expert count") == 1);
    log_clear();
    static const int32_t no_skip_three[6] = {0, 1, 0, 0, 1, 0}; /* top_k 3 of 2 experts, no skip: a repeat is forced */
    CHECK(try_batch(cache, e, 2, 3, no_skip_three, three_w) == SEABRAID_INVALID && output_untouched());
    CHECK(logged("repeated expert id") == 1);
}


int main(void) {
    struct env e;
    if (!env_open(&e)) {
        return 1;
    }
    seabraid_ggml_cache *cache = seabraid_ggml_cache_create_slots(e.backend, 2);
    if (!cache) {
        return 1;
    }
    routes_faults(cache, &e);
    token_and_count_bounds(cache, &e);
    skips_and_top_k(cache, &e);
    seabraid_ggml_cache_destroy(cache);
    env_close(&e);
    if (failures) {
        fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    puts("ggml adapter route and bounds edge fixture passed");
    return 0;
}
