// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
/* The host-scheduler continuation fixture: two layers appended to one caller-owned graph with
 * seabraid_ggml_append_swiglu, computed through ggml_backend_sched twice. WHY split from test_q4.c: it exercises
 * the append path (a graph the host owns); the rest of the fixture covers the direct provider path and the plan cache. */
#include "test_q4_oracle.h"

struct scheduled_batch {
    struct ggml_tensor *input;
    struct ggml_tensor *ids;
    struct ggml_tensor *weights;
    struct ggml_tensor *output;
};

static bool append_two_layers(struct ggml_context *ctx, struct ggml_cgraph *graph,
                              seabraid_ggml_layer layer, struct scheduled_batch *job) {
    job->input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, WIDTH, 1, TOKENS);
    job->ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, EXPERTS, TOKENS);
    job->weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, EXPERTS, TOKENS);
    const int empty_count = ggml_graph_n_nodes(graph);
    if (seabraid_ggml_append_swiglu(ctx, graph, layer, job->input, job->ids, job->weights, 3).output ||
        ggml_graph_n_nodes(graph) != empty_count) {
        return false;
    }
    const seabraid_ggml_layer swapped = {layer.up, layer.gate, layer.down};
    const seabraid_ggml_routed_nodes first = seabraid_ggml_append_swiglu(ctx, graph, layer, job->input,
                                                                job->ids, job->weights, EXPERTS);
    const seabraid_ggml_routed_nodes second = seabraid_ggml_append_swiglu(ctx, graph, swapped, job->input,
                                                                 job->ids, job->weights, 1);
    if (!first.output || !second.output) {
        return false;
    }
    /* Keep the short warmup reduction and residual in the caller's graph. */
    struct ggml_tensor *residual = ggml_reshape_2d(ctx, job->input, WIDTH, TOKENS);
    job->output = ggml_add(ctx, ggml_add(ctx, first.output, residual), second.output);
    ggml_build_forward_expand(graph, job->output);
    return true;
}

static bool scheduler_run(ggml_backend_sched_t sched, struct ggml_cgraph *graph,
                          const struct scheduled_batch *job, const seabraid_batch_v1 *batch) {
    ggml_backend_tensor_set(job->input, batch->activations, 0, ggml_nbytes(job->input));
    ggml_backend_tensor_set(job->ids, batch->expert_ids, 0, ggml_nbytes(job->ids));
    ggml_backend_tensor_set(job->weights, batch->weights, 0, ggml_nbytes(job->weights));
    if (ggml_backend_sched_graph_compute(sched, graph) != GGML_STATUS_SUCCESS) {
        return false;
    }
    float expected[TOKENS * WIDTH];
    float output[TOKENS * WIDTH];
    for (size_t i = 0; i < TOKENS * WIDTH; ++i) {
        expected[i] = batch->activations[i];
    }
    oracle_add(expected, batch, EXPERTS, EXPERTS, &gate, &up);
    oracle_add(expected, batch, EXPERTS, 1, &up, &gate);
    ggml_backend_tensor_get(job->output, output, 0, sizeof(output));
    return rows_match(output, expected, TOKENS * WIDTH);
}

static int scheduler_batches(ggml_backend_sched_t sched, struct ggml_cgraph *graph,
                             const struct scheduled_batch *job) {
    const size_t picks = TOKENS * EXPERTS;
    seabraid_batch_v1 batch = {SEABRAID_ABI_VERSION, TOKENS, activation, TOKENS * WIDTH, experts, picks, weights, picks};
    if (!ggml_backend_sched_alloc_graph(sched, graph) || !scheduler_run(sched, graph, job, &batch)) {
        return 21;
    }
    const float replay[TOKENS * WIDTH] = {[7] = -0.75f, [WIDTH + 2] = 1.25f};
    const int32_t next_ids[TOKENS * EXPERTS] = {1, 0, 0, 1};
    const float next_weights[TOKENS * EXPERTS] = {0.55f, 0.45f, 0.3f, 0.7f};
    batch.activations = replay;
    batch.expert_ids = next_ids;
    batch.weights = next_weights;
    return scheduler_run(sched, graph, job, &batch) ? 0 : 22;
}

int seabraid_ggml_q4_run_scheduler(ggml_backend_t backend, seabraid_ggml_layer layer) {
    enum { SLOTS = 64 };
    const size_t size = SLOTS * ggml_tensor_overhead() + ggml_graph_overhead_custom(SLOTS, false);
    struct ggml_context *ctx = ggml_init((struct ggml_init_params) {size, NULL, true});
    if (!ctx) {
        return 20;
    }
    struct ggml_cgraph *graph = ggml_new_graph_custom(ctx, SLOTS, false);
    struct scheduled_batch job = {0};
    ggml_backend_sched_t sched = ggml_backend_sched_new(&backend, NULL, 1, SLOTS, false, true);
    int result = 20;
    if (sched && append_two_layers(ctx, graph, layer, &job)) {
        result = scheduler_batches(sched, graph, &job);
    }
    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    return result;
}
