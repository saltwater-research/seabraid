// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
#ifndef SEABRAID_GGML_H
#define SEABRAID_GGML_H

#include "ggml-backend.h"
#include "seabraid/seabraid.h"

#ifdef __cplusplus
extern "C" {
#endif

/* One layer of separate gate, up, and down expert tensors. Use contiguous
 * backend-resident leaf tensors, in ggml dimension order [hidden, ff, experts]
 * for gate/up and [ff, hidden, experts] for down. Submit only unclamped SwiGLU
 * with post-FFN routing weights. */
typedef struct seabraid_ggml_layer {
    struct ggml_tensor *gate;
    struct ggml_tensor *up;
    struct ggml_tensor *down;
} seabraid_ggml_layer;

/* Intermediate handles let an engine preserve its existing graph labels. */
typedef struct seabraid_ggml_routed_nodes {
    struct ggml_tensor *up;
    struct ggml_tensor *gate;
    struct ggml_tensor *swiglu;
    struct ggml_tensor *down;
    struct ggml_tensor *weighted;
    struct ggml_tensor *output;
} seabraid_ggml_routed_nodes;

/* Append a separate-weight, unclamped SwiGLU to the engine's graph. Activation
 * is F32 [hidden, 1, tokens], IDs are I32 [top_k, tokens], weights are F32
 * [1, top_k, tokens]. Valid IDs and model semantics are the engine's contract.
 * Warmup can project more experts than it reduces: pass 1 <= reduce_picks <=
 * top_k <= 64. An empty result means unsupported geometry and no graph change.
 * The engine owns the arena, tensors, backend selection, and graph execution. */
seabraid_ggml_routed_nodes seabraid_ggml_append_swiglu(struct ggml_context *ctx, struct ggml_cgraph *graph,
                                               seabraid_ggml_layer layer, struct ggml_tensor *activation,
                                               struct ggml_tensor *ids, struct ggml_tensor *weights,
                                               uint32_t reduce_picks);

typedef struct seabraid_ggml_cache seabraid_ggml_cache;

/* One cached batch graph and buffer (a single slot), borrowing the backend. Use a dedicated
 * backend or serialize access to it. Returns NULL for a missing backend or
 * allocation failure. Destroy the cache before freeing its backend or weights. */
seabraid_ggml_cache *seabraid_ggml_cache_create(ggml_backend_t backend);
/* Like seabraid_ggml_cache_create but holds up to `slots` plans (1..1024), one
 * per layer in use, so alternating layers reuse their own graph and buffer.
 * Workspace memory is bounded by slots x one plan; the least recently used
 * plan is replaced when all slots hold other layers. Size it to the layer
 * count of the multi-provider table to avoid per-layer rebuilds. */
seabraid_ggml_cache *seabraid_ggml_cache_create_slots(ggml_backend_t backend, uint32_t slots);
void seabraid_ggml_cache_destroy(seabraid_ggml_cache *cache);

/* Synchronous engine boundary for materialized host rows and engine-owned
 * weights. Serialize calls on the cache. Tensor handles, metadata, and weight
 * contents stay valid and unchanged until the cache is destroyed. */
int32_t seabraid_ggml_execute(seabraid_ggml_cache *cache, const seabraid_ggml_layer *layer,
                          const seabraid_config_v1 *config, const seabraid_batch_v1 *batch,
                          float *output, size_t output_count);

typedef struct seabraid_ggml_binding {
    seabraid_ggml_cache *cache;
    seabraid_ggml_layer layer;
} seabraid_ggml_binding;

/* Pass this descriptor to seabraid_open, then prepare matching dimensions. Keep
 * the binding and borrowed handles valid until seabraid_close drains the worker. */
seabraid_provider_v1 seabraid_ggml_provider(seabraid_ggml_binding *binding);

typedef struct seabraid_ggml_layer_table {
    seabraid_ggml_cache *cache;
    const seabraid_ggml_layer *layers;
    uint32_t count;
} seabraid_ggml_layer_table;

/* The engine provides the table and holds its entries stable until close.
 * Each operation selects a layer using the same cache and batch geometry. */
seabraid_provider_v2 seabraid_ggml_multi_provider(seabraid_ggml_layer_table *table);

/* Process-wide count of what the direct execute path moves between host and device: each batch uploads its
 * activation, ids and weights and downloads one result. The expert weights stay resident and are never
 * staged (weight_bytes_staged stays 0), and the append_swiglu path moves nothing the adapter can see.
 * Library-filled like the runtime's structs: set abi_version = SEABRAID_ABI_VERSION and struct_size = sizeof
 * before the call. Counters are atomic and relaxed; a read taken during a batch may split it across fields. */
typedef struct seabraid_ggml_transfer_stats {
    uint32_t abi_version;
    uint32_t struct_size;
    uint64_t sent_bytes;           /* host to device */
    uint64_t sent_transfers;       /* tensor uploads (three per batch) */
    uint64_t returned_bytes;       /* device to host */
    uint64_t returned_transfers;   /* result downloads (one per successful batch) */
    uint64_t batches;              /* batches that finished upload, compute and download */
    uint64_t weight_bytes_staged;  /* always 0 today */
} seabraid_ggml_transfer_stats;

/* SEABRAID_OK, or SEABRAID_INVALID for a null struct, unknown abi_version or struct_size below sizeof (the v1 size).
 * Writes min(struct_size, sizeof) bytes and sets struct_size to that count; a newer caller reads it back. */
int32_t seabraid_ggml_get_transfer_stats(seabraid_ggml_transfer_stats *out);
/* Zero every counter. */
void seabraid_ggml_reset_transfer_stats(void);

#ifdef __cplusplus
}
#endif

#endif
