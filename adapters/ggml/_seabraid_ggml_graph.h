// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
/* Private to the adapter: what the graph half (_seabraid_ggml_graph.c) and the execution half (seabraid_ggml.c)
 * share. Pure declarations plus the one logging helper, so neither half owns the other's state. */
#ifndef SEABRAID_GGML_GRAPH_INTERNAL_H
#define SEABRAID_GGML_GRAPH_INTERNAL_H

#include "_seabraid_ggml_cache.h"
#include "_seabraid_ggml_routes.h"

#include <stdarg.h>
#include <stdio.h>

/* Bound metadata and the uniqueness check for the CUDA routed sort path. */
enum { MAX_PICKS = SEABRAID_GGML_MAX_PICKS, GRAPH_SLOTS = 2 * MAX_PICKS + 16 };


/* The one place this adapter logs: every refusal and failure reports through
 * the runtime sink (seabraid_log) and returns its status. Records carry codes,
 * counts and ids only - never tensor data or paths. If the sink itself fails,
 * the record goes to stderr so a failure is never silent. */
static inline int32_t report(uint32_t level, int32_t status, const char *format, ...) {
    char message[200];
    va_list args;
    va_start(args, format);
    vsnprintf(message, sizeof(message), format, args);
    va_end(args);
    if (seabraid_log(level, status, message) != SEABRAID_OK) {
        fprintf(stderr, "seabraid-ggml: level %u: %s\n", (unsigned) level, message);
    }
    return status;
}

bool seabraid_ggml_layer_matches(ggml_backend_t backend, const seabraid_ggml_layer *layer,
                                 const seabraid_config_v1 *config);
bool seabraid_ggml_batch_sizes_valid(const seabraid_config_v1 *config, const seabraid_batch_v1 *batch, int64_t ff);
bool seabraid_ggml_batch_counts_match(const seabraid_config_v1 *config, const seabraid_batch_v1 *batch,
                                      size_t output_count);
struct batch_graph seabraid_ggml_graph_build(struct ggml_context *ctx, const seabraid_ggml_layer *layer,
                                             const seabraid_config_v1 *config, int64_t tokens);
bool seabraid_ggml_graph_supported(ggml_backend_t backend, struct ggml_cgraph *graph);

#endif
