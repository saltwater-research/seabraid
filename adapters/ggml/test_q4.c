// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
/* Each Q4_0 row has one exactly representable coefficient. Cyclic shifts
 * make a wrong expert, tensor order, or output stride disagree with the oracle.
 * This file: the logging stand-ins, the direct-path and plan-cache checks and main; the oracle is in
 * test_q4_oracle.h and the scheduler continuation in test_q4_scheduler.c (a seam of concept, one file each). */
#include "_seabraid_ggml_cache.h"
#include "test_q4_oracle.h"

#include <stdio.h>
#include <string.h>

#ifdef SEABRAID_GGML_TEST_MAIN
/* The standalone fixture links no runtime, so it supplies the log sink and
 * asserts that each refusal wrote its specific record. */
static char last_log[200];

int32_t seabraid_log(uint32_t level, int32_t status, const char *message) {
    (void) level;
    (void) status;
    snprintf(last_log, sizeof(last_log), "%s", message);
    return SEABRAID_OK;
}

/* The fixture stands in for the runtime's epoch too: the adapter's start-up check then passes here. */
uint32_t seabraid_abi_epoch(void) {
    return SEABRAID_ABI_EPOCH;
}

static bool logged(const char *needle) {
    return strstr(last_log, needle) != NULL;
}
#else
/* Embedded in another binary the real runtime sink is linked; nothing to assert. */
static bool logged(const char *needle) {
    (void) needle;
    return true;
}
#endif

/* Repeated picks must leave the output and workspace untouched and say why. */
static bool refuses_duplicates(seabraid_ggml_binding *binding, const seabraid_provider_v1 *provider,
                               const seabraid_config_v1 *config, seabraid_batch_v1 *batch,
                               float *output, size_t count) {
    static const int32_t duplicates[] = {0, 0, 1, 0};
    batch->expert_ids = duplicates;
    output[0] = 99.0f;
    return provider->execute(provider->context, config, batch, output, count) == SEABRAID_INVALID &&
           output[0] == 99.0f && binding->cache->preparations == 1 && logged("repeated expert id");
}

/* An operation past the table is refused with a record naming it. */
static bool refuses_operation(const seabraid_provider_v2 *provider, uint32_t operation,
                              const seabraid_config_v1 *config, const seabraid_batch_v1 *batch) {
    float output[WIDTH] = {0};
    return provider->execute(provider->context, operation, config, batch, output, WIDTH) == SEABRAID_INVALID &&
           logged("operation 2 outside 2 layers");
}

static int run_routes(seabraid_ggml_binding *binding) {
    const seabraid_provider_v1 provider = seabraid_ggml_provider(binding);
    const seabraid_config_v1 config = {SEABRAID_ABI_VERSION, WIDTH, EXPERTS, TOKENS, EXPERTS};
    float output[TOKENS * WIDTH] = {0};
    const size_t count = TOKENS * WIDTH;
    const size_t picks = TOKENS * EXPERTS;
    seabraid_batch_v1 batch = {SEABRAID_ABI_VERSION, TOKENS, activation, count, experts, picks, weights, picks};
    if (provider.execute(provider.context, &config, &batch, output, count) != SEABRAID_OK ||
        !matches_expected(output, &batch, EXPERTS, &gate, &up) || binding->cache->preparations != 1) {
        return 2;
    }

    if (!refuses_duplicates(binding, &provider, &config, &batch, output, count)) {
        return 3;
    }

    /* All three uploads change even though the workspace must stay resident. */
    const float replay_activation[TOKENS * WIDTH] = {[7] = -0.5f, [WIDTH + 2] = 1.25f};
    const int32_t replay_experts[TOKENS * EXPERTS] = {1, 0, 0, 1};
    const float replay_weights[TOKENS * EXPERTS] = {0.8f, 0.2f, 0.3f, 0.7f};
    batch.activations = replay_activation;
    batch.expert_ids = replay_experts;
    batch.weights = replay_weights;
    if (provider.execute(provider.context, &config, &batch, output, count) != SEABRAID_OK ||
        !matches_expected(output, &batch, EXPERTS, &gate, &up)) {
        return 4;
    }
    return binding->cache->preparations == 1 ? 0 : 5;
}

static int run_layer_changes(seabraid_ggml_binding *binding, const seabraid_config_v1 *config,
                             const seabraid_batch_v1 *batch) {
    seabraid_ggml_cache *cache = binding->cache;
    const seabraid_ggml_layer *layer = &binding->layer;
    float output[WIDTH] = {0};
    /* A different immutable layer can share one adapter and backend. */
    const seabraid_ggml_layer layers[] = {*layer, {layer->up, layer->gate, layer->down}};
    seabraid_ggml_layer_table table = {cache, layers, 2};
    const seabraid_provider_v2 provider = seabraid_ggml_multi_provider(&table);
    if (provider.operation_count != 2 || provider.flags != SEABRAID_PROVIDER_PARTIAL_ROUTES ||
        !refuses_operation(&provider, 2, config, batch) ||
        provider.execute(provider.context, 1, config, batch, output, WIDTH) != SEABRAID_OK ||
        !matches_expected(output, batch, 1, &up, &gate) || cache->preparations != 4) {
        return 8;
    }
    if (provider.execute(provider.context, 0, config, batch, output, WIDTH) != SEABRAID_OK ||
        !matches_expected(output, batch, 1, &gate, &up)) {
        return 9;
    }
    return cache->preparations == 5 ? 0 : 10;
}

static int run_shape_changes(seabraid_ggml_binding *binding) {
    seabraid_ggml_cache *cache = binding->cache;
    const seabraid_ggml_layer *layer = &binding->layer;
    const size_t full_buffer = ggml_backend_buffer_get_size(cache->plans[0].buffer);
    seabraid_config_v1 config = {SEABRAID_ABI_VERSION, WIDTH, EXPERTS, TOKENS, EXPERTS};
    seabraid_batch_v1 batch = {SEABRAID_ABI_VERSION, 1, activation, WIDTH, experts, EXPERTS, weights, EXPERTS};
    float output[WIDTH] = {0};
    if (seabraid_ggml_execute(cache, layer, &config, &batch, output, WIDTH) != SEABRAID_OK ||
        !matches_expected(output, &batch, EXPERTS, &gate, &up) || cache->preparations != 2 ||
        ggml_backend_buffer_get_size(cache->plans[0].buffer) > full_buffer) {
        return 6;
    }

    config.top_k = 1;
    batch.expert_id_count = 1;
    batch.weight_count = 1;
    if (seabraid_ggml_execute(cache, layer, &config, &batch, output, WIDTH) != SEABRAID_OK ||
        !matches_expected(output, &batch, 1, &gate, &up) || cache->preparations != 3) {
        return 7;
    }

    return run_layer_changes(binding, &config, &batch);
}

static int run_single_routes(seabraid_ggml_binding *binding) {
    const seabraid_provider_v1 provider = seabraid_ggml_provider(binding);
    const seabraid_config_v1 config = {SEABRAID_ABI_VERSION, WIDTH, EXPERTS, TOKENS, 1};
    const size_t count = TOKENS * WIDTH;
    const int32_t single[] = {0, 1};
    const float single_weights[] = {0.25f, 0.6f};
    const seabraid_batch_v1 batch = {SEABRAID_ABI_VERSION, TOKENS, activation, count, single, TOKENS, single_weights, TOKENS};
    float output[TOKENS * WIDTH] = {0};
    if (provider.execute(provider.context, &config, &batch, output, count) != SEABRAID_OK ||
        !matches_expected(output, &batch, 1, &gate, &up) || binding->cache->preparations != 6) {
        return 11;
    }
    return 0;
}

/* Skipped picks (-1, weight 0) equal the oracle restricted to the real picks,
 * including a token whose picks are all skipped (output stays zero). */
static bool partial_batch_ok(const seabraid_provider_v2 *provider, const seabraid_config_v1 *config,
                             const int32_t *ids, const float *pick_weights) {
    float output[TOKENS * WIDTH] = {0};
    const seabraid_batch_v1 batch = {SEABRAID_ABI_VERSION, TOKENS, activation, TOKENS * WIDTH, ids, TOKENS * EXPERTS,
                                     pick_weights, TOKENS * EXPERTS};
    return provider->execute(provider->context, 0, config, &batch, output, TOKENS * WIDTH) == SEABRAID_OK &&
           matches_expected(output, &batch, EXPERTS, &gate, &up);
}

/* A skip with a nonzero weight would drop real work, so it is refused once. */
static bool refuses_weighted_skip(const seabraid_provider_v2 *provider, const seabraid_config_v1 *config) {
    static const int32_t ids[] = {0, -1, 1, 0};
    static const float bad[] = {0.25f, 0.5f, 0.6f, 0.4f};
    float output[TOKENS * WIDTH] = {99.0f};
    const seabraid_batch_v1 batch = {SEABRAID_ABI_VERSION, TOKENS, activation, TOKENS * WIDTH, ids, TOKENS * EXPERTS,
                                     bad, TOKENS * EXPERTS};
    return provider->execute(provider->context, 0, config, &batch, output, TOKENS * WIDTH) == SEABRAID_INVALID &&
           output[0] == 99.0f && logged("skipped pick has a nonzero weight");
}

static int run_partial_routes(ggml_backend_t backend, seabraid_ggml_layer layer) {
    seabraid_ggml_cache *cache = seabraid_ggml_cache_create(backend);
    seabraid_ggml_layer_table table = {cache, &layer, 1};
    const seabraid_provider_v2 provider = seabraid_ggml_multi_provider(&table);
    const seabraid_config_v1 config = {SEABRAID_ABI_VERSION, WIDTH, EXPERTS, TOKENS, EXPERTS};
    static const int32_t half[] = {0, -1, -1, -1};
    static const float half_weights[] = {0.25f, 0.0f, 0.0f, 0.0f};
    static const int32_t mixed[] = {-1, 1, 1, -1};
    static const float mixed_weights[] = {0.0f, 0.6f, 0.4f, 0.0f};
    int result = 0;
    if (!(provider.flags & SEABRAID_PROVIDER_PARTIAL_ROUTES)) {
        result = 12;
    } else if (!partial_batch_ok(&provider, &config, half, half_weights)) {
        result = 13;
    } else if (!partial_batch_ok(&provider, &config, mixed, mixed_weights)) {
        result = 14;
    } else if (!refuses_weighted_skip(&provider, &config)) {
        result = 15;
    }
    seabraid_ggml_cache_destroy(cache);
    return result;
}

static bool same_bits(seabraid_ggml_cache *roomy, seabraid_ggml_cache *tight, const seabraid_ggml_layer *layer,
                      const seabraid_config_v1 *config, const seabraid_batch_v1 *batch) {
    float fast[TOKENS * WIDTH] = {0};
    float slow[TOKENS * WIDTH] = {0};
    return seabraid_ggml_execute(roomy, layer, config, batch, fast, TOKENS * WIDTH) == SEABRAID_OK &&
           seabraid_ggml_execute(tight, layer, config, batch, slow, TOKENS * WIDTH) == SEABRAID_OK &&
           memcmp(fast, slow, sizeof(fast)) == 0;
}

/* Two layers alternating on a two-slot cache build twice, then only hit, and
 * every result is bit-identical to the one-slot cache that rebuilds each time. */
static int run_plan_reuse(ggml_backend_t backend, seabraid_ggml_layer layer) {
    const seabraid_ggml_layer layers[] = {layer, {layer.up, layer.gate, layer.down}};
    seabraid_ggml_cache *roomy = seabraid_ggml_cache_create_slots(backend, 2);
    seabraid_ggml_cache *tight = seabraid_ggml_cache_create(backend);
    const seabraid_config_v1 config = {SEABRAID_ABI_VERSION, WIDTH, EXPERTS, TOKENS, EXPERTS};
    const seabraid_batch_v1 batch = {SEABRAID_ABI_VERSION, TOKENS, activation, TOKENS * WIDTH, experts, TOKENS * EXPERTS,
                                     weights, TOKENS * EXPERTS};
    int result = roomy && tight ? 0 : 16;
    for (int round = 0; result == 0 && round < 6; ++round) {
        result = same_bits(roomy, tight, &layers[round % 2], &config, &batch) ? 0 : 17;
    }
    if (result == 0 && (roomy->preparations != 2 || tight->preparations != 6)) {
        result = 18;
    }
    seabraid_ggml_cache_destroy(roomy);
    seabraid_ggml_cache_destroy(tight);
    return result;
}

static int run_checks(ggml_backend_t backend, seabraid_ggml_binding *binding) {
    int result = run_routes(binding);
    if (result == 0) {
        result = run_shape_changes(binding);
    }
    if (result == 0) {
        result = run_single_routes(binding);
    }
    if (result == 0) {
        result = run_partial_routes(backend, binding->layer);
    }
    if (result == 0) {
        result = run_plan_reuse(backend, binding->layer);
    }
    if (result == 0 && ggml_backend_dev_type(ggml_backend_get_device(backend)) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        result = seabraid_ggml_q4_run_scheduler(backend, binding->layer);
    }
    return result;
}

int seabraid_ggml_fixture(ggml_backend_t backend) {
    if (!backend) {
        return 1;
    }
    struct ggml_context *ctx = ggml_init((struct ggml_init_params) {
        3 * ggml_tensor_overhead(), NULL, true
    });
    if (!ctx) {
        return 1;
    }
    seabraid_ggml_binding binding = {
        seabraid_ggml_cache_create(backend),
        {
            ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_0, WIDTH, WIDTH, EXPERTS),
            ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_0, WIDTH, WIDTH, EXPERTS),
            ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_0, WIDTH, WIDTH, EXPERTS)
        }
    };
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    int result = 1;
    if (binding.cache && buffer && upload(binding.layer.gate, &gate) &&
        upload(binding.layer.up, &up) && upload(binding.layer.down, &down)) {
        result = run_checks(backend, &binding);
    }
    seabraid_ggml_cache_destroy(binding.cache);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return result;
}

#ifdef SEABRAID_GGML_TEST_MAIN
int main(void) {
    ggml_backend_load_all();
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    const int result = seabraid_ggml_fixture(backend);
    if (backend) {
        ggml_backend_free(backend);
    }
    return result;
}
#endif
