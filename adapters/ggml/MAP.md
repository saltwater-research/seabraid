# ggml provider

| File | Owns | Does not own |
|---|---|---|
| `seabraid_ggml.h` | Cache lifecycle, graph continuation, and v1/v2 provider bindings | Model loading, routing, or device placement |
| `_seabraid_ggml_cache.h` | Private workspace state shared with its fixture | Engine or runtime-session state |
| `_seabraid_ggml_routes.{h,c}` | Route-row validation and mapping of skipped picks to filler experts | Graph building or caching |
| `_seabraid_ggml_graph.{h,c}` | The routed FFN graph fragment (`seabraid_ggml_append_swiglu`, the direct path's graph), layer and batch-size validation; no cache or execution state. Split from `seabraid_ggml.c` so each stays under the size cap | Plan cache, backend execution, logging records of its own beyond refusals |
| `seabraid_ggml.c` | The plan cache, batch refusal order, direct backend execution and the v1/v2 provider entry points; owns all mutable state | Weight packing, scheduler choices, or model architecture discovery |
| `_seabraid_ggml_stats.{h,c}` | The process-wide transfer counter: bytes and transfers of the direct execute path, read with `seabraid_ggml_get_transfer_stats` and zeroed with `seabraid_ggml_reset_transfer_stats` | Any transfer the host's scheduler makes (the `append_swiglu` path) |
| `test_transfer_stats.c` | Exact byte and transfer counts from the shapes of a known CPU batch run, refusals counting nothing, reset to zero, and the teardown summary line | Accelerator transfer behavior |
| `test_q4.c` | Direct-path routes, shape and layer changes, partial routes, plan reuse and the `main` of the fixture (also embeddable as `seabraid_ggml_fixture`; an embedder compiles `test_q4_scheduler.c` with it) | Full inference or performance qualification |
| `test_q4_oracle.h` | Q4_0 scalar oracle: the exactly representable projections, `upload`, `oracle_add`, the comparison | Any case |
| `test_q4_scheduler.c` | Native scheduler continuation: two layers appended to one caller-owned graph, run through `ggml_backend_sched` twice | The direct provider path |
| `test_edges_fixture.h` | The identity-expert setup, log sink and `CHECK` shared by the two edge executables | Any case |
| `test_edges.c` | Route faults (range, repeat, non-finite weight, skips), token and count bounds (0, 1, max, past max, `UINT32_MAX`), `top_k` equal to and past the expert count | Cache lifecycle, counters |
| `test_edges_cache.c` | Cache slot bounds and destroy order, teardown summary with and without batches, provider misuse, counter races under threads, `append_swiglu` refusal reporting | Route and bounds cases |
| `test_routes_golden.h` | The golden-table half of `test_routes.c`, an include-once fragment (not a library) that shares its helpers and `CHECK`; split out for the file-size limit | `test_routes.c` |
| `test_routes.c` | The adapter against `src/picks/golden.txt` (the pick rule's shared table: every partial-routes row gets its verdict kind; `NARROW_ROW`/`MAX_PICKS` equal the recorded limits), lowest-index-first refusal order (range, finite, repeat per pick), skip-weight bit test (`-0.0` accepted, NaN refused), 64,000-pick row cost; built twice, the second time with `_seabraid_ggml_routes.c` under `-ffast-math` (`seabraid_ggml_routes_fastmath`) | Backend execution |
| `llama-cpp-qwen3moe.patch` | Opt-in Qwen3 MoE graph hook for the pinned llama.cpp source | Other architectures or physical expert placement |
| `CMakeLists.txt` | Build against the caller's ggml target; optional CPU smoke | Fetching another copy of ggml |

Build note: when building llama.cpp with the patch, keep `LLAMA_CONNECT=OFF`
(the default). When ON, llama.cpp's build downloads a prebuilt binary from
GitHub releases without a hash check. We build only the `llama` target with
`LLAMA_CURL=OFF`.

Session status: this adapter ships provider v1 and v2 bindings only; it has no `seabraid_provider_v3`
yet. The v1 session calls below (`seabraid_open`, `seabraid_open_multi`, `seabraid_prepare`) are deprecated
and removed at 1.0; the supported entry points are `seabraid_session_open` and `seabraid_session_submit`,
which take a v3 provider. Until this adapter gains one, a host on the new calls wraps
`seabraid_ggml_execute` in its own v3 `execute` (viewing the batch v2 as a batch v1, as the Strata
adapter's `execute_session` does), or calls `seabraid_ggml_execute` directly.

Create a cache with `seabraid_ggml_cache_create(backend)`. To use the (deprecated) v1 session,
store the cache and the three tensor handles in `seabraid_ggml_binding`, pass
`seabraid_ggml_provider(&binding)` to `seabraid_open`, and `seabraid_prepare` with its hidden
size, expert count, and routed top-k (1–64). Keep the binding alive until
`seabraid_close` returns, then destroy the cache before unloading the tensors or
backend. For engines with their own layer dispatch, select a `seabraid_ggml_layer`
and call `seabraid_ggml_execute` with the materialized batch. Calls on a cache and on
its backend must be serialized. The engine keeps the weights resident and
synchronizes their use. Direct backend execution requires resident weights on a
supported buffer type and checks graph operations before uploading inputs.

A multi-layer host can pass a `seabraid_ggml_layer_table` to
`seabraid_ggml_multi_provider` and open one Seabraid session with
`seabraid_open_multi` (deprecated, see above). The operation ID selects a table entry. All entries in
the session share its prepared routing geometry and backend. This adapter
accepts real expert IDs and `SEABRAID_SKIP_EXPERT_ID` (-1) with weight exactly 0.0, and
advertises `SEABRAID_PROVIDER_PARTIAL_ROUTES`, so a layer can be split across devices.
A skipped pick contributes nothing: the adapter uploads a distinct unused expert id for it
(its zero weight cancels the product), and a skip with a nonzero weight is refused with one
record. A batch with skips needs top-k no larger than the expert count.

The adapter computes the separate-gate, unclamped SwiGLU branch of
`llm_graph_context::build_moe_ffn` (llama.cpp `src/llama-graph.cpp`), including
post-FFN weighting and a left-to-right sum. ggml supplies the original quantized
`MUL_MAT_ID` kernels. The cache keeps up to N plans (a graph and scratch buffer each), one per layer in use,
each keyed by the three immutable tensor handles, token count, and top-k. N is 1 for
`seabraid_ggml_cache_create` and chosen (1..1024) by `seabraid_ggml_cache_create_slots`;
size it to the layer table so alternating layers keep their own plans. Peak workspace is
N x one plan. When every slot holds another layer the least recently used one is replaced. The tensors fix the
rest of the geometry. A hit uploads the new batch without rebuilding. A miss
releases the replaced plan before allocating the next one to bound peak
workspace memory. Keep separate cache handles for execution streams that should
retain separate hot shapes. Invalid geometry or repeated picks leave the
workspace untouched. Allocation failure or unsupported ops leave an empty slot.
The graph call fences device execution before returning host output. Use other
model paths for pre-FFN weighting,
LoRA, biases, scales, clamped SwiGLU, merged gate/up, or shared-expert work.
Model code retains ownership of any shared-expert or residual merge.

The llama.cpp hook uses `seabraid_ggml_append_swiglu` with the engine's existing
activation, selected expert IDs, and normalized weights. It appends operation
metadata to the engine's graph and returns the routed output for the existing
residual. The router, tensor lifetime, named graph nodes, scheduler, and device
allocation stay with llama.cpp. This path has no Seabraid workspace and does not
round trip through host rows, so changing layers cannot evict a Seabraid graph
cache entry. It does not route individual experts to separate physical devices.

The included patch targets the local llama.cpp source at
`b1b770d90d94cad9e9ea5b993bdbc383a522fb28` (release v1.0.0, 2026-11-13). Apply it from that engine's checkout
with `git apply /path/to/seabraid/adapters/ggml/llama-cpp-qwen3moe.patch`. Add
`-DLLAMA_SEABRAID_GGML_DIR=/path/to/seabraid/adapters/ggml` to the engine's CMake
configuration to select the experimental Qwen3 MoE path. An empty path keeps
the stock build. The adapter calls `seabraid_log` and `seabraid_abi_epoch`, which only the runtime defines
(a link without it fails with undefined references), so a build that selects the adapter also needs the
runtime: install the SDK and add `-DSeabraid_DIR=/prefix/lib/cmake/Seabraid` (or `-DCMAKE_PREFIX_PATH=/prefix`);
the adapter's CMake then links `Seabraid::runtime` and stops configuring with a one-line error when it is absent.
`SEABRAID_GGML_TESTS=ON` builds skip that link because the fixtures define both symbols themselves. In an enabled build, other architectures and unsupported
geometry or layer variants still use the stock graph. The hook maintains the
upstream order that lets the scheduler fuse routed ops and the narrower
reduction used during warmup.

Where to get ggml: it lives in the `ggml/` directory of llama.cpp at release v1.0.0, commit
`b1b770d90d94cad9e9ea5b993bdbc383a522fb28` (the same commit the CI `sdk` job fetches; it is the
version the research log's ggml 1.0.0 runs used). Nothing here downloads it. To build and install it
for `-Dggml_DIR` (these three commands were not run end to end offline on the build box; they are
the standard CMake steps for that directory):

```sh
git clone https://github.com/ggml-org/llama.cpp && git -C llama.cpp checkout b1b770d90d94cad9e9ea5b993bdbc383a522fb28
cmake -S llama.cpp/ggml -B build/ggml -DCMAKE_BUILD_TYPE=Release
cmake --build build/ggml --parallel && cmake --install build/ggml --prefix "$HOME/ggml-pkg"
```

The adapter also needs the Seabraid SDK (see the README: `cmake --install` to a prefix; pass it as
`-DSeabraid_DIR=<prefix>/lib/cmake/Seabraid`). To skip the install step, `sdk_check/run.sh <llama.cpp>/ggml`
builds and links the adapter against the ggml source tree directly.

Add this directory after the engine's ggml target, or configure it using a local
ggml CMake installation (`-Dggml_DIR` is the installed directory that holds ggml's CMake package config, under
`$HOME/ggml-pkg/lib/cmake/` for the prefix above; `find` for it if the layout differs):

```sh
cmake -S adapters/ggml -B build/seabraid-ggml -DSEABRAID_GGML_TESTS=ON -Dggml_DIR=/path/to/ggml/package
cmake --build build/seabraid-ggml --config Release
ctest --test-dir build/seabraid-ggml -C Release --output-on-failure
```

The tests-off path (what the llama.cpp hook builds: `find_package(Seabraid)` then `Seabraid::runtime`) is checked by
`sdk_check/run.sh <ggml source dir>`: it installs the root project to a prefix, configures the adapter against it
with the engine's ggml, links `sdk_check/sdk_link_main.c`, runs it, and confirms the adapter refuses to configure
when no SDK is on the path. Offline; needs cmake, cargo and a C compiler.
The CI `sdk` job runs it on every push against the `ggml/` directory of llama.cpp at the commit above, fetched by
`actions/checkout` pinned to that SHA; no CI run of it has happened yet (the CI billing block), so the first one is
the first proof that the pin and the sparse checkout work.

| Contract | Check |
|---|---|
| Routed execution and same-shape refresh | Scalar oracle with changed activations, routes, and weights |
| Cache reuse and replacement | Preparation counts on hits and token, top-k, and layer changes |
| Bounded workspace | Slot count caps plans; release before allocation |
| Plan reuse | Two layers alternating on a two-slot cache prepare twice in total and match the one-slot cache bit for bit |
| Partial routes | Skipped picks equal the oracle over the remaining picks; a weighted skip is refused once |
| Invalid routes | Duplicate picks leave output and preparation count unchanged |
| Scheduler continuation | A native CPU scheduler runs two routed fragments plus residual and replays changed inputs against the scalar oracle |
| Warmup and fallback | Fixture reduces fewer picks than it projects and rejects incompatible geometry without changing the graph |

The native Q4_0 fixture passed on CPU, CUDA, and Vulkan. Disabling the cache
hit failed the preparation-count guard. The earlier uncached fixture also
caught reversed SwiGLU gate and up weights. The scheduled fixture also fails
when the fragment incorrectly reduces every projected pick. Loaded GGUF
parity, scheduler placement on accelerators, and model latency still require
an engine run.

Transfer counter: each successful direct batch adds three uploads (activation, ids, weights) and one
download (the output) to process-wide atomic counters, in bytes and in transfers, and adds nothing when
a batch is refused before upload. A compute failure after the upload still counts the uploads (they moved)
but no download or batch.
The expert weights are resident and never staged, so `weight_bytes_staged` is 0. The `append_swiglu` path
copies nothing through the adapter, so it is not counted; its traffic belongs to the host's scheduler.
`seabraid_ggml_cache_destroy` logs one INFO line with the process totals when any batch ran. The counter
is fail-open and never alters a result. The fixture asserts exact counts for 3-, 3- and 1-token batches.

Every refusal and failure in `seabraid_ggml.c` reports once through `seabraid_log`
(codes, counts and ids only) and returns its status: invalid arguments, config
ABI, layer shape, batch faults (null buffer, bounds, counts, out-of-range expert, repeated expert,
non-finite weight; each its own reason, because the runtime already refuses repeats and non-finite
weights on the session path and these checks only fire for direct `seabraid_ggml_execute` callers),
allocation, unsupported ops, graph compute and operation IDs outside the layer
table. `seabraid_ggml_append_swiglu` refusals warn once per distinct reason per
thread because the host rebuilds that graph per layer and falls back to its
stock graph. The adapter does not link the runtime itself (the host already
does); the fixture defines `seabraid_log` so it can assert on the records.
