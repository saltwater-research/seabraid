// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
/* The execution half of the adapter: the plan cache, batch refusal order and the provider entry points. ggml owns
 * matmul/quantization; this adapter owns graph metadata and device workspace. WHY split from
 * _seabraid_ggml_graph.c: this file owns all mutable state (cache slots, counters), that one owns none.
 * FAILURE: fail-fast. A refused batch or layer is reported once through `report` and returned as a status with the
 * workspace untouched; an allocation failure leaves an empty slot, never a plan that was not built. */
#include "_seabraid_ggml_cache.h"
#include "_seabraid_ggml_graph.h"
#include "_seabraid_ggml_routes.h"
#include "_seabraid_ggml_stats.h"

#include <stdlib.h>

#include "ggml-alloc.h"

static int32_t run_graph(ggml_backend_t backend, const struct batch_graph *job,
                         const seabraid_batch_v1 *batch, const int32_t *ids, float *output) {
    ggml_backend_tensor_set(job->activation, batch->activations, 0, ggml_nbytes(job->activation));
    ggml_backend_tensor_set(job->ids, ids, 0, ggml_nbytes(job->ids));
    ggml_backend_tensor_set(job->weights, batch->weights, 0, ggml_nbytes(job->weights));
    seabraid_ggml_stats_sent(ggml_nbytes(job->activation) + ggml_nbytes(job->ids) + ggml_nbytes(job->weights), 3);
    /* The synchronous form fences device work before the caller releases its lease. */
    const enum ggml_status computed = ggml_backend_graph_compute(backend, job->graph);
    if (computed != GGML_STATUS_SUCCESS) {
        return report(SEABRAID_LOG_ERROR, SEABRAID_PROVIDER, "graph compute failed with ggml status %d", (int) computed);
    }
    ggml_backend_tensor_get(job->output, output, 0, ggml_nbytes(job->output));
    seabraid_ggml_stats_returned(ggml_nbytes(job->output));
    seabraid_ggml_stats_batch_done();
    return SEABRAID_OK;
}

static void clear_plan(struct batch_plan *plan) {
    ggml_backend_buffer_free(plan->buffer);
    ggml_free(plan->metadata);
    *plan = (struct batch_plan) {0};
}

/* WHY a slot cap: the slot array is the memory bound (slots x one workspace),
 * so an absurd request is refused rather than silently reserving it. */
enum { MAX_SLOTS = 1024 };

seabraid_ggml_cache *seabraid_ggml_cache_create_slots(ggml_backend_t backend, uint32_t slots) {
    if (!backend || !slots || slots > MAX_SLOTS) {
        report(SEABRAID_LOG_ERROR, SEABRAID_INVALID, "cache_create refused: null backend or slot count %u outside 1..%d",
               (unsigned) slots, (int) MAX_SLOTS);
        return NULL;
    }
    /* WHY here: the cache is the first thing every consumer of this adapter creates, and a libseabraid built for
     * another struct layout would misread every batch this adapter validates. Refuse before any state exists. */
    if (seabraid_abi_epoch() != SEABRAID_ABI_EPOCH) {
        report(SEABRAID_LOG_ERROR, SEABRAID_INVALID, "cache_create refused: library ABI epoch %u, adapter built for %u",
               (unsigned) seabraid_abi_epoch(), (unsigned) SEABRAID_ABI_EPOCH);
        return NULL;
    }
    seabraid_ggml_cache *cache = malloc(sizeof(*cache));
    struct batch_plan *plans = calloc(slots, sizeof(*plans));
    if (!cache || !plans) {
        free(cache);
        free(plans);
        report(SEABRAID_LOG_ERROR, SEABRAID_INTERNAL, "cache_create failed: out of memory");
        return NULL;
    }
    *cache = (seabraid_ggml_cache) {.backend = backend, .plans = plans, .slots = slots};
    return cache;
}

seabraid_ggml_cache *seabraid_ggml_cache_create(ggml_backend_t backend) {
    /* One slot keeps the original bound of a single resident workspace. */
    return seabraid_ggml_cache_create_slots(backend, 1);
}

void seabraid_ggml_cache_destroy(seabraid_ggml_cache *cache) {
    if (!cache) {
        return;
    }
    /* The teardown point of a run: one summary of the process totals (nothing when no batch ran). */
    char summary[160];
    if (seabraid_ggml_stats_summary(summary, sizeof(summary))) {
        report(SEABRAID_LOG_INFO, SEABRAID_OK, "%s", summary);
    }
    for (uint32_t slot = 0; slot < cache->slots; ++slot) {
        clear_plan(&cache->plans[slot]);
    }
    free(cache->plans);
    free(cache);
}

static bool same_layer(const struct batch_plan *plan, const seabraid_ggml_layer *layer) {
    return plan->layer.gate == layer->gate && plan->layer.up == layer->up && plan->layer.down == layer->down;
}

static bool same_plan(const struct batch_plan *plan, const seabraid_ggml_layer *layer,
                       const seabraid_config_v1 *config, uint32_t tokens) {
    /* The bound tensor shapes already determine hidden, ff and expert count. */
    return plan->buffer && same_layer(plan, layer) && plan->tokens == tokens && plan->top_k == config->top_k;
}

/* True when `plan` should be replaced before `victim`: empty beats full, then older beats newer. */
static bool better_victim(const struct batch_plan *plan, const struct batch_plan *victim) {
    if (!plan->buffer || !victim->buffer) {
        return !plan->buffer && victim->buffer;
    }
    return plan->used < victim->used;
}

/* Slot choice: the layer's own plan (hit, or stale shape to rebuild in place),
 * else an empty slot, else the least recently used. WHY keyed on layer identity
 * rather than operation id: direct execute callers have no id, and a layer
 * reached through two ids shares one plan. */
static struct batch_plan *choose_plan(seabraid_ggml_cache *cache, const seabraid_ggml_layer *layer) {
    struct batch_plan *victim = &cache->plans[0];
    for (uint32_t slot = 0; slot < cache->slots; ++slot) {
        struct batch_plan *plan = &cache->plans[slot];
        if (plan->buffer && same_layer(plan, layer)) {
            return plan;
        }
        if (better_victim(plan, victim)) {
            victim = plan;
        }
    }
    return victim;
}

static int32_t build_plan(seabraid_ggml_cache *cache, struct batch_plan *plan, const seabraid_ggml_layer *layer,
                          const seabraid_config_v1 *config, uint32_t tokens) {
    const size_t arena_size = GRAPH_SLOTS * ggml_tensor_overhead() +
                              ggml_graph_overhead_custom(GRAPH_SLOTS, false);
    plan->metadata = ggml_init((struct ggml_init_params) {arena_size, NULL, true});
    if (!plan->metadata) {
        return report(SEABRAID_LOG_ERROR, SEABRAID_INTERNAL, "prepare failed: ggml metadata arena allocation (%zu bytes)",
                      arena_size);
    }
    plan->job = seabraid_ggml_graph_build(plan->metadata, layer, config, tokens);
    plan->buffer = ggml_backend_alloc_ctx_tensors(plan->metadata, cache->backend);
    if (!plan->buffer) {
        return report(SEABRAID_LOG_ERROR, SEABRAID_PROVIDER, "prepare failed: backend workspace allocation for %u tokens",
                      tokens);
    }
    if (!seabraid_ggml_graph_supported(cache->backend, plan->job.graph)) {
        return report(SEABRAID_LOG_ERROR, SEABRAID_PROVIDER, "prepare failed: backend does not support a routed op");
    }
    plan->layer = *layer;
    plan->tokens = tokens;
    plan->top_k = config->top_k;
    ++cache->preparations;
    return SEABRAID_OK;
}

static int32_t prepare_plan(seabraid_ggml_cache *cache, const seabraid_ggml_layer *layer,
                            const seabraid_config_v1 *config, uint32_t tokens, struct batch_plan **out) {
    struct batch_plan *plan = choose_plan(cache, layer);
    if (!same_plan(plan, layer, config, tokens)) {
        /* Release the victim before allocating to bound peak memory. */
        clear_plan(plan);
        const int32_t status = build_plan(cache, plan, layer, config, tokens);
        if (status != SEABRAID_OK) {
            clear_plan(plan);
            return status;
        }
    }
    plan->used = ++cache->clock;
    *out = plan;
    return SEABRAID_OK;
}

/* NULL when the batch is acceptable, else the specific reason it is not. */
static const char *batch_fault(const seabraid_config_v1 *config, const seabraid_batch_v1 *batch,
                               size_t count, int64_t ff) {
    if (!batch->activations || !batch->expert_ids || !batch->weights) {
        return "a batch buffer is null";
    }
    if (!seabraid_ggml_batch_sizes_valid(config, batch, ff)) {
        return "tokens, hidden size or top-k are outside the supported bounds";
    }
    if (!seabraid_ggml_batch_counts_match(config, batch, count)) {
        return "element counts do not match tokens, hidden size and top-k";
    }
    return seabraid_ggml_routes_fault(config, batch);
}

/* Batch already validated: pick the plan, map skipped picks, run. */
static int32_t run_prepared(seabraid_ggml_cache *cache, const seabraid_ggml_layer *layer,
                            const seabraid_config_v1 *config, const seabraid_batch_v1 *batch, float *output) {
    struct batch_plan *plan = NULL;
    const int32_t status = prepare_plan(cache, layer, config, batch->tokens, &plan);
    if (status != SEABRAID_OK) {
        return status;
    }
    int32_t *owned;
    const int32_t *ids = seabraid_ggml_graph_ids(config, batch, &owned);
    if (!ids) {
        return report(SEABRAID_LOG_ERROR, SEABRAID_INTERNAL, "execute failed: out of memory mapping skipped picks");
    }
    const int32_t result = run_graph(cache->backend, &plan->job, batch, ids, output);
    free(owned);
    return result;
}

int32_t seabraid_ggml_execute(seabraid_ggml_cache *cache, const seabraid_ggml_layer *layer,
                          const seabraid_config_v1 *config, const seabraid_batch_v1 *batch,
                          float *output, size_t output_count) {
    if (!cache || !layer || !config || !batch || !output) {
        return report(SEABRAID_LOG_ERROR, SEABRAID_INVALID, "execute refused: a required argument is null");
    }
    if (config->abi_version != SEABRAID_ABI_VERSION) {
        return report(SEABRAID_LOG_ERROR, SEABRAID_INVALID, "execute refused: config ABI version %u, expected %u",
                      (unsigned) config->abi_version, (unsigned) SEABRAID_ABI_VERSION);
    }
    if (batch->abi_version != SEABRAID_ABI_VERSION) {
        return report(SEABRAID_LOG_ERROR, SEABRAID_INVALID, "execute refused: batch ABI version %u, expected %u",
                      (unsigned) batch->abi_version, (unsigned) SEABRAID_ABI_VERSION);
    }
    if (!seabraid_ggml_layer_matches(cache->backend, layer, config)) {
        return report(SEABRAID_LOG_ERROR, SEABRAID_INVALID,
                      "execute refused: layer tensors are not packed on this backend or do not match the config");
    }
    const char *fault = batch_fault(config, batch, output_count, layer->up->ne[1]);
    if (fault) {
        return report(SEABRAID_LOG_ERROR, SEABRAID_INVALID, "execute refused: %s", fault);
    }
    return run_prepared(cache, layer, config, batch, output);
}

static int32_t execute(void *context, const seabraid_config_v1 *config, const seabraid_batch_v1 *batch,
                       float *output, size_t output_count) {
    const seabraid_ggml_binding *binding = context;
    if (!binding) {
        return report(SEABRAID_LOG_ERROR, SEABRAID_INVALID, "provider call refused: null binding");
    }
    return seabraid_ggml_execute(binding->cache, &binding->layer, config, batch, output, output_count);
}

seabraid_provider_v1 seabraid_ggml_provider(seabraid_ggml_binding *binding) {
    return (seabraid_provider_v1) {SEABRAID_ABI_VERSION, binding, execute};
}

static int32_t execute_operation(void *context, uint32_t operation_id,
                                 const seabraid_config_v1 *config, const seabraid_batch_v1 *batch,
                                 float *output, size_t output_count) {
    const seabraid_ggml_layer_table *table = context;
    if (!table || !table->layers) {
        return report(SEABRAID_LOG_ERROR, SEABRAID_INVALID, "provider call refused: null or empty layer table");
    }
    if (operation_id >= table->count) {
        return report(SEABRAID_LOG_ERROR, SEABRAID_INVALID, "provider call refused: operation %u outside %zu layers",
                      (unsigned) operation_id, (size_t) table->count);
    }
    return seabraid_ggml_execute(table->cache, &table->layers[operation_id], config, batch,
                             output, output_count);
}

seabraid_provider_v2 seabraid_ggml_multi_provider(seabraid_ggml_layer_table *table) {
    return (seabraid_provider_v2) {
        SEABRAID_MULTI_PROVIDER_VERSION, table ? table->count : 0, SEABRAID_PROVIDER_PARTIAL_ROUTES, table, execute_operation
    };
}
