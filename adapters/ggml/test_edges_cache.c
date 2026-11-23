// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
/* Edge cases of the cache lifecycle, provider misuse, the process-wide transfer counter under threads, and
 * append_swiglu refusals, on the CPU backend. OWNER: this file; the shared setup is test_edges_fixture.h and the
 * route and bounds cases are test_edges.c. */
#include "test_edges_fixture.h"

static seabraid_ggml_cache *must_cache(const struct env *e, uint32_t slots) {
    seabraid_ggml_cache *cache = seabraid_ggml_cache_create_slots(e->backend, slots);
    CHECK(cache != NULL);
    return cache;
}

/* E4: slot-count bounds, destroy of NULL and of an untouched cache, and the teardown summary: one record when
 * a batch ran, none when no batch ran or the counters were reset. */
static void cache_lifecycle(const struct env *e) {
    log_clear();
    CHECK(seabraid_ggml_cache_create_slots(NULL, 1) == NULL);
    CHECK(seabraid_ggml_cache_create_slots(e->backend, 0) == NULL);
    CHECK(seabraid_ggml_cache_create_slots(e->backend, 1025) == NULL);
    CHECK(logged("cache_create refused") == 3);
    seabraid_ggml_cache_destroy(NULL);
    seabraid_ggml_cache *widest = must_cache(e, 1024);
    seabraid_ggml_cache_destroy(widest); /* 1024 empty slots */

    seabraid_ggml_reset_transfer_stats();
    log_clear();
    seabraid_ggml_cache *idle = must_cache(e, 3);
    seabraid_ggml_cache_destroy(idle);
    CHECK(logged("transfers:") == 0); /* no batch ran: nothing to summarize */

    seabraid_ggml_cache *used = must_cache(e, 2);
    const seabraid_ggml_layer swapped = {e->layer.up, e->layer.gate, e->layer.down};
    const seabraid_config_v1 config = {SEABRAID_ABI_VERSION, W, E, MAX_TOKENS, 2};
    float activation[W] = {0};
    const int32_t ids[2] = {0, 1};
    const float weights[2] = {0.5f, 0.5f};
    const seabraid_batch_v1 batch = {SEABRAID_ABI_VERSION, 1, activation, W, ids, 2, weights, 2};
    CHECK(seabraid_ggml_execute(used, &e->layer, &config, &batch, g_out, W) == SEABRAID_OK);
    CHECK(seabraid_ggml_execute(used, &swapped, &config, &batch, g_out, W) == SEABRAID_OK); /* both slots occupied */
    log_clear();
    seabraid_ggml_cache_destroy(used);
    CHECK(logged("transfers: 2 batches") == 1);

    seabraid_ggml_cache *after_reset = must_cache(e, 1);
    CHECK(seabraid_ggml_execute(after_reset, &e->layer, &config, &batch, g_out, W) == SEABRAID_OK);
    seabraid_ggml_reset_transfer_stats();
    log_clear();
    seabraid_ggml_cache_destroy(after_reset);
    CHECK(logged("transfers:") == 0); /* the batch ran, but the counters were zeroed before teardown */

    char tiny[8];
    char fits[200];
    seabraid_ggml_cache *again = must_cache(e, 1);
    CHECK(seabraid_ggml_execute(again, &e->layer, &config, &batch, g_out, W) == SEABRAID_OK);
    CHECK(seabraid_ggml_stats_summary(NULL, sizeof(fits)) == 0);
    CHECK(seabraid_ggml_stats_summary(fits, 0) == 0);
    CHECK(seabraid_ggml_stats_summary(tiny, sizeof(tiny)) == 0);
    CHECK(seabraid_ggml_stats_summary(fits, sizeof(fits)) > 0);
    seabraid_ggml_cache_destroy(again);
    seabraid_ggml_reset_transfer_stats();
}

/* E5: providers handed nothing, a table with no layers, a cache-less table and an operation id of UINT32_MAX. */
static void provider_misuse(const struct env *e) {
    const seabraid_config_v1 config = {SEABRAID_ABI_VERSION, W, E, MAX_TOKENS, 2};
    float activation[W] = {0};
    const int32_t ids[2] = {0, 1};
    const float weights[2] = {0.5f, 0.5f};
    const seabraid_batch_v1 batch = {SEABRAID_ABI_VERSION, 1, activation, W, ids, 2, weights, 2};
    float out[W];
    log_clear();
    const seabraid_provider_v1 single = seabraid_ggml_provider(NULL);
    CHECK(single.execute(single.context, &config, &batch, out, W) == SEABRAID_INVALID);
    const seabraid_provider_v2 none = seabraid_ggml_multi_provider(NULL);
    CHECK(none.operation_count == 0 && (none.flags & SEABRAID_PROVIDER_PARTIAL_ROUTES));
    CHECK(none.execute(none.context, 0, &config, &batch, out, W) == SEABRAID_INVALID);
    CHECK(logged("null binding") == 1 && logged("null or empty layer table") == 1);

    seabraid_ggml_cache *cache = must_cache(e, 1);
    seabraid_ggml_layer_table empty = {cache, NULL, 1};
    CHECK(seabraid_ggml_multi_provider(&empty).execute(&empty, 0, &config, &batch, out, W) == SEABRAID_INVALID);
    seabraid_ggml_layer_table no_cache = {NULL, &e->layer, 1};
    CHECK(seabraid_ggml_multi_provider(&no_cache).execute(&no_cache, 0, &config, &batch, out, W) == SEABRAID_INVALID);
    seabraid_ggml_layer_table table = {cache, &e->layer, 1};
    const seabraid_provider_v2 provider = seabraid_ggml_multi_provider(&table);
    log_clear();
    CHECK(provider.execute(provider.context, UINT32_MAX, &config, &batch, out, W) == SEABRAID_INVALID);
    CHECK(provider.execute(provider.context, 1, &config, &batch, out, W) == SEABRAID_INVALID);
    CHECK(logged("outside 1 layers") == 2);
    CHECK(provider.execute(provider.context, 0, &config, &batch, out, W) == SEABRAID_OK);
    seabraid_ggml_cache_destroy(cache);
    seabraid_ggml_reset_transfer_stats();
}

#ifndef _WIN32
enum { THREADS = 4, ROUNDS = 20000, SEND_BYTES = 7, SEND_TRANSFERS = 3, RETURN_BYTES = 5 };

static void *reset_once(void *unused) {
    (void) unused;
    seabraid_ggml_reset_transfer_stats();
    return NULL;
}

static void *hammer(void *unused) {
    (void) unused;
    for (int i = 0; i < ROUNDS; ++i) {
        seabraid_ggml_stats_sent(SEND_BYTES, SEND_TRANSFERS);
        seabraid_ggml_stats_returned(RETURN_BYTES);
        seabraid_ggml_stats_batch_done();
    }
    return NULL;
}

static atomic_int reading = 1;
static atomic_int reader_saw_decrease;

static void *watch(void *unused) {
    (void) unused;
    uint64_t last_batches = 0, last_sent = 0;
    while (atomic_load(&reading)) {
        seabraid_ggml_transfer_stats now = {SEABRAID_ABI_VERSION, sizeof(now)};
        if (seabraid_ggml_get_transfer_stats(&now) != SEABRAID_OK || now.batches < last_batches || now.sent_bytes < last_sent) {
            atomic_store(&reader_saw_decrease, 1);
        }
        last_batches = now.batches;
        last_sent = now.sent_bytes;
    }
    return NULL;
}

/* E6: the relaxed atomics lose no update across threads, a concurrent reader never sees a total go backwards
 * (no reset runs during that phase), and a reset racing the adders neither crashes nor leaves a torn value. */
static void counter_races(void) {
    pthread_t adders[THREADS], reader, resetter;
    seabraid_ggml_reset_transfer_stats();
    atomic_store(&reading, 1);
    pthread_create(&reader, NULL, watch, NULL);
    for (int i = 0; i < THREADS; ++i) {
        pthread_create(&adders[i], NULL, hammer, NULL);
    }
    for (int i = 0; i < THREADS; ++i) {
        pthread_join(adders[i], NULL);
    }
    atomic_store(&reading, 0);
    pthread_join(reader, NULL);
    seabraid_ggml_transfer_stats total = {SEABRAID_ABI_VERSION, sizeof(total)};
    CHECK(seabraid_ggml_get_transfer_stats(&total) == SEABRAID_OK);
    const uint64_t n = (uint64_t) THREADS * ROUNDS;
    CHECK(total.batches == n && total.sent_bytes == n * SEND_BYTES && total.sent_transfers == n * SEND_TRANSFERS);
    CHECK(total.returned_bytes == n * RETURN_BYTES && total.returned_transfers == n);
    CHECK(!atomic_load(&reader_saw_decrease));

    atomic_store(&reading, 1);
    for (int i = 0; i < THREADS; ++i) {
        pthread_create(&adders[i], NULL, hammer, NULL);
    }
    pthread_create(&resetter, NULL, reset_once, NULL);
    for (int i = 0; i < THREADS; ++i) {
        pthread_join(adders[i], NULL);
    }
    pthread_join(resetter, NULL);
    seabraid_ggml_reset_transfer_stats();
    seabraid_ggml_transfer_stats zero = {SEABRAID_ABI_VERSION, sizeof(zero)};
    CHECK(seabraid_ggml_get_transfer_stats(&zero) == SEABRAID_OK && zero.batches == 0 && zero.sent_bytes == 0);
}
#endif

/* E7: append_swiglu refuses zero reduce_picks and a missing graph without touching the graph, and reports a
 * given reason once per thread until the reason changes or a call succeeds. */
static void append_refusals(const struct env *e) {
    enum { NODES = 64 };
    const size_t arena = NODES * ggml_tensor_overhead() + ggml_graph_overhead_custom(NODES, false);
    struct ggml_context *ctx = ggml_init((struct ggml_init_params) {arena, NULL, true});
    CHECK(ctx != NULL);
    if (!ctx) {
        return;
    }
    struct ggml_cgraph *graph = ggml_new_graph_custom(ctx, NODES, false);
    struct ggml_tensor *input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, W, 1, 2);
    struct ggml_tensor *ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 2);
    struct ggml_tensor *weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 2, 2);
    const int empty = ggml_graph_n_nodes(graph);

    log_clear();
    CHECK(seabraid_ggml_append_swiglu(ctx, graph, e->layer, input, ids, weights, 2).output != NULL); /* resets the dedupe */
    CHECK(logged("append_swiglu refused") == 0);
    const int built = ggml_graph_n_nodes(graph);
    CHECK(seabraid_ggml_append_swiglu(ctx, graph, e->layer, input, ids, weights, 0).output == NULL);
    CHECK(seabraid_ggml_append_swiglu(ctx, graph, e->layer, input, ids, weights, 0).output == NULL);
    CHECK(logged("append_swiglu refused") == 1); /* the same reason, once */
    CHECK(seabraid_ggml_append_swiglu(ctx, NULL, e->layer, input, ids, weights, 2).output == NULL);
    CHECK(logged("null context or graph") == 1 && logged("append_swiglu refused") == 2); /* a new reason reports */
    CHECK(seabraid_ggml_append_swiglu(NULL, graph, e->layer, input, ids, weights, 2).output == NULL);
    CHECK(logged("append_swiglu refused") == 2);
    CHECK(seabraid_ggml_append_swiglu(ctx, graph, e->layer, input, ids, weights, 3).output == NULL); /* past top_k */
    CHECK(logged("append_swiglu refused") == 3);
    CHECK(ggml_graph_n_nodes(graph) == built && built > empty); /* refusals added no node */
    CHECK(seabraid_ggml_append_swiglu(ctx, graph, e->layer, input, ids, weights, 1).output != NULL); /* reduce 1..top_k ok */
    CHECK(seabraid_ggml_append_swiglu(ctx, graph, e->layer, input, ids, weights, 0).output == NULL);
    CHECK(logged("append_swiglu refused") == 4); /* a success in between re-arms the report */
    ggml_free(ctx);
}


int main(void) {
    struct env e;
    if (!env_open(&e)) {
        return 1;
    }
    cache_lifecycle(&e);
    provider_misuse(&e);
#ifndef _WIN32
    counter_races();
#endif
    append_refusals(&e);
    env_close(&e);
    if (failures) {
        fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    puts("ggml adapter cache and counter edge fixture passed");
    return 0;
}
