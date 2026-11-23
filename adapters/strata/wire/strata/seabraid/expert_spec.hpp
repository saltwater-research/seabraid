// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
#pragma once
// seabraid channel v6: the engine-neutral description of a layer's routed experts, written by the MAIN engine into the
// channel's `spec` region so a helper can load and run experts WITHOUT parsing the model pack itself.
//
// WHY main describes and the helper only executes: the pack format and its parser belong to the engine (for dsv41
// they are private to ds4_model.cu).  A helper that re-parsed the pack would be a second owner of the same format
// and would drift.  So main owns "where is expert e of layer l and what type is it" (describe), the helper owns
// "turn those bytes plus an activation into an output row" (execute).  Qwen's helper predates this and still reads
// its own pack (engine kEngineQwen leaves the spec region unused).
//
// Host memory only, like channel.hpp: both toolchains (nvcc+gcc for the helper, clang/hipcc for main) include it,
// so every wire struct is explicitly padded and size-asserted.
#include <cstddef>
#include <cstdint>

namespace seabraid {

inline constexpr int kSpecMaxFiles = 64;     // distinct model files a spec may reference (dsv41 GGUF shards: < 20)
inline constexpr int kSpecPathMax = 512;     // bytes per path, NUL-terminated
inline constexpr int kRoles = 3;             // gate, up, down - in that order everywhere

/// Which executor the helper must run.  Checked at connect-time by the helper against what it was built to serve.
enum Engine : int32_t { kEngineQwen = 0, kEngineDsv41 = 1 };

/// Result rows the helper writes for one layer request (Header::out_mode).
///   kOutPerPick : n_tok*k rows, one UNWEIGHTED row per owned pick, unowned rows untouched (Qwen, v3/v4 behaviour).
///   kOutPerToken: n_tok rows, row t = sum over t's owned picks of weights[t*k+i] * expert(x_t); a token with no
///                 owned pick gets a zero row.  Cuts the return traffic by k and lets the helper fuse the weighting.
enum OutMode : int32_t { kOutPerPick = 0, kOutPerToken = 1 };

/// Spec flags (LayerSpec::flags).
inline constexpr uint32_t kSpecActQ8K = 1u;  // round activations to Q8_K (256-blocks) before low-bit dots, as the
                                             // main engine's CPU lane does; without it the GPU rounds per 32 and the
                                             // helper's rows would differ from the CPU rows they replace

/// One weight matrix role of a layer: expert e's bytes are [off + e*stride, off + e*stride + bytes) of file `file`.
struct RoleSpec {
    int32_t file;      // index into the spec's path table
    int32_t type;      // the engine's quant type id for this matrix (dsv41: ggml type ids)
    uint64_t off;      // byte offset of expert 0 in the file
    uint64_t stride;   // bytes between consecutive experts
    uint64_t bytes;    // bytes of one expert's matrix
};
static_assert(sizeof(RoleSpec) == 32, "RoleSpec must be 32 bytes in both builds");

/// One MoE layer.  Layers that are not MoE (or that the helper must not serve) have hidden == 0.
struct LayerSpec {
    RoleSpec role[kRoles];   // gate, up, down
    int32_t hidden;          // H: activation width (rows of down, cols of gate/up)
    int32_t ffn;             // F: expert FFN width
    float swiglu_limit;      // > 0: g = min(g, lim), u = clamp(u, -lim, lim) before silu(g)*u (DeepSeek); 0 = none
    uint32_t flags;          // kSpec* bits
};
static_assert(sizeof(LayerSpec) == 112, "LayerSpec must be 112 bytes in both builds");
static_assert(alignof(LayerSpec) == 8, "LayerSpec must be 8-aligned in both builds");

/// Bytes of the spec region for `n_layers` layers: the layer table, then kSpecMaxFiles paths.
inline constexpr size_t spec_region_bytes(int64_t n_layers) {
    return (size_t) n_layers * sizeof(LayerSpec) + (size_t) kSpecMaxFiles * (size_t) kSpecPathMax;
}

} // namespace seabraid
