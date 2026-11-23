// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
/* The graph half of the adapter: layer and batch validation, and the routed SwiGLU graph the direct path
 * builds and the host's scheduler path appends to. WHY a file of its own: it holds no cache or execution state
 * (execution lives in seabraid_ggml.c), so it is reviewed and size-gated apart from the plan cache.
 * FAILURE: fail-fast. Every refusal is reported through `report` (the runtime sink) and returned as a status. */
#include "_seabraid_ggml_graph.h"
#include "_seabraid_ggml_routes.h"

#include "ggml-alloc.h"

static bool packed_on(ggml_backend_t backend, const struct ggml_tensor *tensor) {
    if (!tensor || !tensor->buffer || !tensor->data) {
        return false;
    }
    if (tensor->view_src || tensor->op != GGML_OP_NONE || tensor->ne[3] != 1) {
        return false;
    }
    return ggml_is_contiguous(tensor) &&
           ggml_backend_supports_buft(backend, ggml_backend_buffer_get_type(tensor->buffer));
}

static bool packed_shapes_match(const seabraid_ggml_layer *layer, const seabraid_config_v1 *config) {
    const int64_t ff = layer->up->ne[1];
    return ff > 0 && ff <= INT32_MAX &&
           layer->up->ne[0] == config->hidden && layer->up->ne[2] == config->experts &&
           ggml_are_same_shape(layer->gate, layer->up) &&
           layer->down->ne[0] == ff && layer->down->ne[1] == config->hidden &&
           layer->down->ne[2] == config->experts;
}

bool seabraid_ggml_layer_matches(ggml_backend_t backend, const seabraid_ggml_layer *layer,
                          const seabraid_config_v1 *config) {
    return packed_on(backend, layer->gate) &&
           packed_on(backend, layer->up) &&
           packed_on(backend, layer->down) &&
           packed_shapes_match(layer, config);
}

bool seabraid_ggml_batch_sizes_valid(const seabraid_config_v1 *config, const seabraid_batch_v1 *batch, int64_t ff) {
    if (!config->hidden || !config->top_k || config->top_k > MAX_PICKS) {
        return false;
    }
    if (!batch->tokens || batch->tokens > config->max_tokens) {
        return false;
    }
    /* The largest intermediate is [max(hidden, ff), top_k, tokens] F32. */
    const size_t width = config->hidden > ff ? config->hidden : (size_t) ff;
    return width <= INT32_MAX &&
           width <= (size_t) PTRDIFF_MAX / sizeof(float) / batch->tokens / config->top_k;
}

bool seabraid_ggml_batch_counts_match(const seabraid_config_v1 *config, const seabraid_batch_v1 *batch,
                         size_t output_count) {
    const size_t rows = (size_t) batch->tokens * config->hidden;
    const size_t picks = (size_t) batch->tokens * config->top_k;
    return output_count == rows && batch->activation_count == rows &&
           batch->expert_id_count == picks && batch->weight_count == picks;
}

static void expand_if_scheduled(struct ggml_cgraph *graph, struct ggml_tensor *node) {
    if (graph) {
        ggml_build_forward_expand(graph, node);
    }
}

static struct ggml_tensor *merge_picks(struct ggml_context *ctx, struct ggml_cgraph *graph,
                                       struct ggml_tensor *weighted, uint32_t count) {
    /* llama-graph.cpp expands all views first for backend fusion and then sums
     * left to right. The direct provider has no scheduler to hint. */
    struct ggml_tensor *views[MAX_PICKS];
    expand_if_scheduled(graph, weighted);
    for (size_t pick = 0; pick < count; ++pick) {
        views[pick] = ggml_view_2d(ctx, weighted, weighted->ne[0], weighted->ne[2],
                                   weighted->nb[2], pick * weighted->nb[1]);
        expand_if_scheduled(graph, views[pick]);
    }
    struct ggml_tensor *sum = views[0];
    for (size_t pick = 1; pick < count; ++pick) {
        sum = ggml_add(ctx, sum, views[pick]);
        expand_if_scheduled(graph, sum);
    }
    return count == 1 ? ggml_cont(ctx, sum) : sum;
}

static seabraid_ggml_routed_nodes swiglu_nodes(struct ggml_context *ctx, seabraid_ggml_layer layer,
                                           struct ggml_tensor *activation, struct ggml_tensor *ids,
                                           struct ggml_tensor *weights) {
    /* Separate-gate LLM_FFN_SILU + post-FFN weight path in llama-graph.cpp. */
    seabraid_ggml_routed_nodes nodes = {0};
    nodes.up = ggml_mul_mat_id(ctx, layer.up, activation, ids);
    nodes.gate = ggml_mul_mat_id(ctx, layer.gate, activation, ids);
    nodes.swiglu = ggml_swiglu_split(ctx, nodes.gate, nodes.up);
    nodes.down = ggml_mul_mat_id(ctx, layer.down, nodes.swiglu, ids);
    nodes.weighted = ggml_mul(ctx, nodes.down, weights);
    return nodes;
}

static bool route_shapes_match(const struct ggml_tensor *ids, const struct ggml_tensor *weights,
                               int64_t tokens) {
    return ids->ne[1] == tokens && ggml_n_dims(ids) <= 2 && weights->ne[0] == 1 &&
           weights->ne[1] == ids->ne[0] && weights->ne[2] == tokens && weights->ne[3] == 1;
}

static bool activation_matches(const struct ggml_tensor *activation) {
    return activation->type == GGML_TYPE_F32 && activation->ne[1] == 1 &&
           activation->ne[2] > 0 && activation->ne[3] == 1;
}

static bool graph_inputs_match(const struct ggml_tensor *activation, const struct ggml_tensor *ids,
                                const struct ggml_tensor *weights, uint32_t reduce_picks) {
    if (!activation || !ids || !weights) {
        return false;
    }
    if (!reduce_picks || reduce_picks > ids->ne[0] || ids->ne[0] > MAX_PICKS) {
        return false;
    }
    return ids->type == GGML_TYPE_I32 && weights->type == GGML_TYPE_F32 &&
           activation_matches(activation) && route_shapes_match(ids, weights, activation->ne[2]);
}

static bool graph_weight_valid(const struct ggml_tensor *weight) {
    return weight && !ggml_is_transposed(weight) && weight->ne[3] == 1;
}

static bool graph_layer_matches(seabraid_ggml_layer layer, int64_t hidden, int64_t picks) {
    if (!graph_weight_valid(layer.gate) || !graph_weight_valid(layer.up) || !graph_weight_valid(layer.down)) {
        return false;
    }
    return ggml_are_same_shape(layer.gate, layer.up) && layer.up->ne[0] == hidden &&
           layer.down->ne[0] == layer.up->ne[1] && layer.down->ne[1] == hidden &&
           layer.down->ne[2] == layer.up->ne[2] && picks <= layer.up->ne[2];
}

static const char *append_fault(struct ggml_context *ctx, struct ggml_cgraph *graph,
                                seabraid_ggml_layer layer, struct ggml_tensor *activation,
                                struct ggml_tensor *ids, struct ggml_tensor *weights,
                                uint32_t reduce_picks) {
    if (!ctx || !graph) {
        return "null context or graph";
    }
    if (!graph_inputs_match(activation, ids, weights, reduce_picks)) {
        return "activation, ids or weights do not match in type, shape or pick count";
    }
    if (!graph_layer_matches(layer, activation->ne[0], ids->ne[0])) {
        return "layer tensors do not match the activation width or pick count";
    }
    return NULL;
}

seabraid_ggml_routed_nodes seabraid_ggml_append_swiglu(struct ggml_context *ctx, struct ggml_cgraph *graph,
                                               seabraid_ggml_layer layer, struct ggml_tensor *activation,
                                               struct ggml_tensor *ids, struct ggml_tensor *weights,
                                               uint32_t reduce_picks) {
    /* WHY once per distinct reason per thread: the host builds this graph for
     * every layer of every batch and falls back to its stock graph on a
     * refusal, so a record per call would bury the log; a changed reason is
     * still reported. */
    static _Thread_local const char *last_reason;
    const char *fault = append_fault(ctx, graph, layer, activation, ids, weights, reduce_picks);
    if (fault) {
        if (fault != last_reason) {
            last_reason = fault;
            report(SEABRAID_LOG_WARN, SEABRAID_INVALID, "append_swiglu refused, host keeps its stock graph: %s", fault);
        }
        return (seabraid_ggml_routed_nodes) {0};
    }
    last_reason = NULL;
    /* Append metadata only; the engine keeps a single resident scheduler graph
     * across layers rather than preparing a competing per-layer workspace. */
    seabraid_ggml_routed_nodes nodes = swiglu_nodes(ctx, layer, activation, ids, weights);
    nodes.output = merge_picks(ctx, graph, nodes.weighted, reduce_picks);
    return nodes;
}

struct batch_graph seabraid_ggml_graph_build(struct ggml_context *ctx, const seabraid_ggml_layer *layer,
                                      const seabraid_config_v1 *config, int64_t tokens) {
    struct batch_graph job;
    /* The singleton pick dimension lets MUL_MAT_ID broadcast each token row. */
    job.activation = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, config->hidden, 1, tokens);
    job.ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, config->top_k, tokens);
    job.weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, config->top_k, tokens);

    const seabraid_ggml_routed_nodes nodes = swiglu_nodes(ctx, *layer, job.activation, job.ids, job.weights);
    job.output = merge_picks(ctx, NULL, nodes.weighted, config->top_k);
    job.graph = ggml_new_graph_custom(ctx, GRAPH_SLOTS, false);
    ggml_build_forward_expand(job.graph, job.output);
    return job;
}

bool seabraid_ggml_graph_supported(ggml_backend_t backend, struct ggml_cgraph *graph) {
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        const struct ggml_tensor *node = ggml_graph_node(graph, i);
        if (!ggml_backend_supports_op(backend, node)) {
            return false;
        }
    }
    return true;
}
