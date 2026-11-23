// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
#ifndef SEABRAID_GGML_CACHE_INTERNAL_H
#define SEABRAID_GGML_CACHE_INTERNAL_H

#include "seabraid_ggml.h"

struct batch_graph {
    struct ggml_tensor *activation;
    struct ggml_tensor *ids;
    struct ggml_tensor *weights;
    struct ggml_tensor *output;
    struct ggml_cgraph *graph;
};

struct batch_plan {
    seabraid_ggml_layer layer;
    uint32_t tokens;
    uint32_t top_k;
    uint64_t used; /* cache clock at last use, for LRU eviction */
    struct ggml_context *metadata;
    ggml_backend_buffer_t buffer;
    struct batch_graph job;
};

struct seabraid_ggml_cache {
    ggml_backend_t backend;
    struct batch_plan *plans; /* `slots` entries, each one workspace */
    uint32_t slots;
    uint64_t clock;
    /* Counts successful preparations so the fixture can distinguish a hit
     * from a free/allocate pair that reuses the same address. */
    uint64_t preparations;
};

#endif
