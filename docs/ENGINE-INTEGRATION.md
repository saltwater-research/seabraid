# Engine integration: what an inference engine should do

Start here if you maintain an inference engine (or any host that routes MoE work) and want
experts of one layer to run on a second device, possibly from another vendor. This guide says
who does what, in what order to integrate, and how to know you are done. The C mechanics of a
provider, partial routes, the placement loop and diagnostics are in the
[adapter guide](ADAPTER-GUIDE.md); this guide links there instead of repeating them.

> **Alpha: use at your own risk.** The ABI can change before 1.0. Two integrations exist
> (Strata and llama.cpp through ggml). Everything about other engines below is **proposed, not
> implemented**. The Strata integration had live AMD + NVIDIA runs on 2026-11-16 and 2026-11-17
> ([research log](RESEARCH-LOG.md)); the current tree has not been re-measured on GPUs yet.

## 1. Division of labour

| The engine keeps | Seabraid owns | Your provider owns |
|---|---|---|
| Model loading and the expert weights it already has | The session, its worker thread, tickets, buffer reuse | The native kernels and the weights on its device |
| The router (top-k ids and weights) | Validation of every batch, timeouts, status codes | Finishing all device work before the callback returns |
| The scheduler, batching, KV cache, sampling | Placement bookkeeping: owner table, epochs, plans, outcomes, quarantine | Acknowledging a slot copy or write only when bytes and residency index agree |
| The final merge, including its own local experts | Refusing stale epochs, repeated experts, non-finite weights | Reporting its own failures through the log sink |
| The fallback: computing a lost share itself | The log sink and `seabraid_last_error` | Nothing about routing policy |
| When to plan and when to fence | Deciding *what* to move, from the heat you give it | The physical move at the fence |

Seabraid does not load models, run attention, sample, or schedule requests. It never decides
what the engine does with a failure: it returns a status, logs once, and keeps the session usable.

## 2. The integration, in order

Do the steps in this order. Each one can be tested alone (section 8).

1. **Choose the process model** ([section 3](#3-process-model)). Same vendor or CPU: the provider can
   live in the engine's process. Two vendors: one helper process per vendor.
2. **Fix the geometry.** `hidden`, `experts`, `max_tokens` (the largest batch you will submit) and
   `top_k` come from the model; one prepared config per session. Give each layer one operation id
   (`0 .. layers - 1`) and declare `operation_count`. All operations share one geometry and one buffer pool.
3. **Write the provider** for the second device: one callback, `execute`, that computes the experts the
   device holds for the rows it is given. See [ADAPTER-GUIDE section 1](ADAPTER-GUIDE.md#1-a-minimal-provider).
   Declare `SEABRAID_PROVIDER_PARTIAL_ROUTES` when the layer is split, and `COMPACT_ROWS` when
   the device would otherwise be sent many all-skipped rows ([section 2 of the guide](ADAPTER-GUIDE.md#2-split-a-layer-partial-routes)).
4. **Open at model load.** Compare `seabraid_abi_epoch()` with `SEABRAID_ABI_EPOCH` first, install a log
   sink, then `seabraid_session_open`. Open one session per engine thread that runs layers.
5. **Hook the MoE layer forward** (below).
6. **Add placement** if experts will move while serving ([section 4](#4-placement-and-the-engines-weights)).
   A static split needs only `seabraid_placement_fill` once, at load.
7. **Wire shutdown and failure** ([section 6](#6-failure-contract)).
8. **Run the conformance checklist** ([section 8](#8-conformance-checklist)).

### 2.1 Where the hook goes

In the engine's MoE layer forward, after the router has produced its top-k ids and weights for
the batch and before the engine merges the layer output:

```text
router -> ids, weights
  build the helper's masked copy of ids/weights (picks it does not own -> SKIP id, weight 0)
  seabraid_session_submit(helper)         # returns at once; the helper starts working
  engine computes its own picks           # at the same time
  seabraid_collect(helper)                # partial sums come back
  merge: local partial + helper partial (+ shared expert, residual: the engine's)
```

Submit first, compute second, collect last, so both cards work at the same time. The engine's
own fallback stays at the merge: if collect fails, the engine computes the helper's picks with
its own copy of those experts (host RAM, a slower tier) and carries on. The masking loop is in
[ADAPTER-GUIDE section 2](ADAPTER-GUIDE.md#2-split-a-layer-partial-routes). This is the whole hot-path
shape, compiled by `tests/test_guides.py` against the public header:

<!-- compilable -->
```c
#include <seabraid/seabraid.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* Engine-owned, declared here only so the fragment compiles. */
extern void engine_run_picks(int layer, const int32_t *ids, const float *wts, float *partial);
extern void engine_add(float *dst, const float *src, size_t n);

typedef struct {
    seabraid_session *session; /* opened with PARTIAL_ROUTES; one per engine thread that runs layers */
    uint64_t epoch;            /* snap.epoch of the table the engine routed with; 0 only before versioning */
    int retired;               /* set on failure: route everything locally until the session is replaced */
} helper_tier;

/* Start the helper's share of one layer. Returns 1 when a ticket is out, 0 when the engine must do it all. */
int helper_begin(helper_tier *h, uint32_t layer, uint32_t tokens, size_t hidden, size_t top_k,
                 const float *acts, const int32_t *helper_ids, const float *helper_wts, uint64_t *ticket)
{
    seabraid_batch_v2 b;
    if (h->retired) {
        return 0;
    }
    memset(&b, 0, sizeof b); /* unset bytes must be zero */
    b.abi_version = SEABRAID_BATCH_V2_VERSION;
    b.struct_size = sizeof b;
    b.operation_id = layer; /* the operation id selects the layer */
    b.source_modalities = SEABRAID_SOURCE_TEXT;
    b.placement_epoch = h->epoch;
    b.tokens = tokens;
    b.activation_format = SEABRAID_FORMAT_ROUTED_F32;
    b.activations = acts;
    b.activation_count = (size_t)tokens * hidden; /* counts are elements, not bytes */
    b.expert_ids = helper_ids;
    b.expert_id_count = (size_t)tokens * top_k;
    b.weights = helper_wts;
    b.weight_count = (size_t)tokens * top_k;
    /* A refusal (stale epoch, busy) changed nothing: the engine computes the helper's picks itself this layer. */
    return seabraid_session_submit(h->session, &b, ticket) == SEABRAID_OK;
}

/* Merge the helper's share into `partial`. `scratch` holds tokens * hidden floats. Always leaves `partial` complete. */
void helper_finish(helper_tier *h, int have_ticket, uint64_t ticket, int layer, size_t count,
                   const int32_t *helper_ids, const float *helper_wts, float *partial, float *scratch)
{
    if (have_ticket) {
        int32_t status = seabraid_collect(h->session, ticket, 50, scratch, count);
        if (status == SEABRAID_OK) {
            engine_add(partial, scratch, count);
            return;
        }
        /* PROVIDER consumed the ticket; TIMEOUT left it pending and the device may still be writing.
           Either way completion is uncertain: stop using this helper and close its session at a safe point. */
        h->retired = 1;
    }
    engine_run_picks(layer, helper_ids, helper_wts, partial); /* the engine's own fallback, at the merge */
}
```

What to take from it:

- The helper receives the **same rows** as the engine, with the picks it does not own replaced by
  `SEABRAID_SKIP_EXPERT_ID` and weight 0. The engine sums partials from every device.
- Put the placement epoch you routed with in **every** batch once you version (section 4).
- A timeout from `seabraid_collect` leaves the ticket pending. If you cannot prove the device is done
  with the buffers, retire the helper rather than reuse the session.
- A provider whose device holds only some experts should also check that it owns every pick that is not
  a skip, and fail the batch if not. The epoch guard cannot see your table.

## 3. Process model

Seabraid runs each vendor's kernels in a separate vendor-native process, by design: to keep the toolkits
apart, to replace a helper without restarting the engine, and to contain its failures. The design that
follows:

- The **engine process** holds its own vendor's toolkit and the Seabraid library. Its `execute`
  callback runs on the session's worker thread. It does not run the other vendor's kernels.
- Each other vendor gets a **helper process**, built with that vendor's toolkit, holding the experts
  it computes. The callback copies the routed rows into **shared memory** on the same machine and
  waits for the helper's partial sums. No network, no peer-to-peer.
- The Strata adapter shows the details, and they are the rules to copy:
  - The controller creates the shared mapping as a memfd, hands the helper exactly one descriptor
    (`--shm-fd`) and every other descriptor is closed in the child. The mapping is sealed to its size.
  - The helper path must be absolute, a regular file owned by the engine's user or root, not group- or
    world-writable, not setuid; it is executed through the vetted descriptor so a swapped file does not run.
  - The helper gets an allow-listed environment only (toolkit variables such as `CUDA_VISIBLE_DEVICES`,
    `HIP_VISIBLE_DEVICES`, `SEABRAID_*`), never credentials.
  - The helper watches the engine through a pidfd and exits if the engine dies without cleanup. The
    engine watches the helper the same way and retires the lease within milliseconds of a kill.
  - A reply is accepted only if it is current and complete: right row count, finite values, matching
    sequence. A late reply can only write the old mapping.
  - After any wire fault the engine closes the session, reaps the helper, and starts a **new** helper
    with a **fresh** mapping. It never resets a slot in place or reattaches.
- The helper deadline is independent of the `seabraid_collect` timeout. Choose both.
- Match the helper's model pack and geometry to the engine's yourself: the protocol carries shape and
  engine tags and no model fingerprint.

Details and the Linux requirements (pidfd, `/proc`, glibc) are in the
[Strata adapter README](../adapters/strata/README.md). Windows handle handoff has not been run.

For two devices of the **same** backend in one process, or a CPU provider, none of this is needed:
the provider is an ordinary callback in the engine's process (the ggml adapter works that way).

## 4. Placement and the engine's weights

Placement is about who owns which expert (layer, expert) cell. The engine owns the bytes; Seabraid
owns the table and the plan. The loop is in
[ADAPTER-GUIDE section 4](ADAPTER-GUIDE.md#4-the-placement-loop); what the engine must guarantee:

1. **Load and verify before you open.** `seabraid_placement_fill` proposes owners from slot capacities
   and a heat table. Load those experts onto their cards and verify them. Open the placement handle
   only on a layout that is true.
2. **Route by the exported table.** `seabraid_placement_snapshot` gives the executor table and its
   epoch. Route every row by that table and stamp every batch with that epoch.
3. **Heat is yours.** Count routed picks per (layer, expert) over a window and pass them to
   `seabraid_placement_plan`. The planner blends windows, honours cooldown, a promotion margin and a move budget.
4. **Apply only at a fence.** Stop admission, drain every publisher of the old epoch (decode, prefill,
   adaptive copies, blob readers), then call `seabraid_placement_apply`. Publish the new table and
   its epoch before reopening admission, **even when an outcome was a rejection, rollback or quarantine**.
5. **Your provider's `stage` and `replace` are acknowledgements.** `stage` copies a whole verified
   slot out. `replace` writes a new expert into a slot and returns `ACK` only when the bytes and the
   engine's own residency index are both updated. Return the exact status (`ACK`, `REFUSED`, `FAILED`,
   `TIMEOUT`). On uncertain completion return `TIMEOUT`: the rung is quarantined and its cells route
   to the fallback (the engine's own copy). A torn slot belongs in quarantine, not in service.
6. **Keep your own copy of the owner table** and apply each committed outcome to it; the library cannot
   give the table back.
7. **A stale epoch means: re-snapshot, re-route, resubmit.** Resubmitting the old batch fails the same way.
8. **Reopening a handle** restarts epochs at 1; sessions keep their highest accepted epoch. Stamp batches
   with an epoch you own (for example `(generation << 32) | snap.epoch`) or reopen the sessions too.
   See [PLACEMENT-ABI.md](PLACEMENT-ABI.md#reopening-and-the-epoch-mark).
9. **Do not let an old and a new path both write.** In Strata the legacy re-tier scheduler and the
   Seabraid placement driver are selected by one value so that they cannot both run.
10. **Optional: the fallback boundary and multi-hop climbs.** Open with `seabraid_placement_open_v2`
    (`fallback_boundary` 1, `max_hops_per_round` above 1; details in
    [PLACEMENT-ABI.md](PLACEMENT-ABI.md#policy-v2-the-fallback-boundary-and-climbs)). Then `stage` and
    `replace` may be called with the fallback rung, which is your verified host copy (RAM or the model file):
    a `replace` there is all-or-nothing, and on any status but `ACK` the fallback must still serve every cell.
    Keep the boundary between the lead card and the fallback with your own cache, so each boundary has one publisher.
11. **Optional: by-reference moves.** With `seabraid_placement_apply_v2` and `replace_from_fallback` a rung can
    load the expert from the fallback's own store instead of receiving bytes through the library; return
    `REFUSED` when you cannot, and the library falls back to the byte path inside the same swap
    ([PLACEMENT-ABI.md](PLACEMENT-ABI.md#by-reference-moves-provider-v2)).

A live placement run on a real model has **not passed** its first matched-arm test: placement does not
settle at the default margin and interval, and host page-migration stalls hit every arm. Ship a static
split first; add live placement behind an opt-in flag, as Strata does.
Items 10 and 11 and the multi-card ladder have **not yet been shown on GPUs**.
The multi-card ladder ran once ([live run L1, 2026-11-19](RESEARCH-LOG.md#2026-11-19--live-run-l1-the-pyramid-on-real-cards)); the by-reference moves of items 10 and 11 were not part of that run and
remain **not yet shown on GPUs**.

## 5. Threading

- Every session starts one worker thread; every `execute` callback runs on it. Make a thread-bound
  device context (CUDA, HIP) current **inside** the callback; it then stays valid across calls.
- Calls on one session must be serialized by you. Any call on a session while another call is inside
  it (a blocking `seabraid_collect`, say) returns `SEABRAID_BUSY_OR_STATE` and does nothing. This
  includes `seabraid_session_status` from a metrics thread.
- **The worker-thread refusal:** a call into the session made from inside its own provider callback is
  refused with `SEABRAID_BUSY_OR_STATE` too. A provider must not submit, collect, close or poll status
  on its own session. In Python a hosted `provider=fn` must not call back into its own `Runtime`.
- If the engine runs layers on several threads, open one session per thread. Only one batch is in
  flight per session.
- The log sink runs on whichever thread logs: the caller's, or the worker's for a `seabraid_log` made
  inside a callback. Make it thread-safe.
- Close the session before the provider's context goes away. After `seabraid_close_timeout` returns
  `SEABRAID_TIMEOUT` the worker is detached and may still call the provider: keep its context valid for
  the rest of the process.
- A C++ provider must catch every exception inside its callback.
- After `fork()` the child must not use an inherited session.

## 6. Failure contract

- **Open-time mismatch fails fast.** Wrong geometry, ABI version or descriptor: `SEABRAID_INVALID`,
  nothing starts. Check the ABI epoch at start-up.
- **After open, fail open and loud.** A provider that returns nonzero gives `SEABRAID_PROVIDER` at
  collect. The session counts and logs it; the engine decides to compute the picks locally. The ticket
  is consumed, `output` is untouched.
- **Uncertain completion retires the device.** A timeout, a malformed reply, a wrong sequence, a dead
  helper: retire the tier at the request boundary, route future work locally, start a replacement
  only if you choose to.
- **No silent success.** Every outcome is a status or an outcome record. Never swallow one.
- **Keep the engine alive.** On the new runtime a killed helper fails the in-flight request and the
  next request is served from local tiers (recorded on one machine; see the
  [Strata adapter README](../adapters/strata/README.md#opt-in-strata-source-integration)).
- Diagnostics: install the sink once (`seabraid_set_log_sink`), report your own failures with
  `seabraid_log`, read `seabraid_last_error` on the thread that made the failing call. Do not put paths,
  environment values or user data in messages. The level table is in
  [ADAPTER-GUIDE section 5](ADAPTER-GUIDE.md#5-diagnostics); raising `min_level` to ERROR hides every refusal.

## 7. Per engine

### 7.1 llama.cpp and ggml (adapter and patch exist; no expert split yet)

What we have, all in [`adapters/ggml`](../adapters/ggml/MAP.md):

- `seabraid_ggml_execute`: routed SwiGLU on resident Q4_0 expert weights with a cached graph, usable
  as a provider body. The adapter ships provider v1 and v2 bindings; a host on the session calls wraps
  it in its own v3 `execute`.
- `seabraid_ggml_append_swiglu`: appends the same operation to the engine's own graph, so the
  scheduler keeps placement, allocation and continuation.
- [`llama-cpp-qwen3moe.patch`](../adapters/ggml/llama-cpp-qwen3moe.patch): an opt-in hook for the Qwen3 MoE graph,
  pinned to llama.cpp v1.0.0 (`b1b770d90d94cad9e9ea5b993bdbc383a522fb28`). It is selected with
  `-DLLAMA_SEABRAID_GGML_DIR=...`; an empty value keeps the stock build.

What it proves: on a loaded model the hook is bit-identical to stock on CPU, CUDA, HIP and a Vulkan
layer split. What it does **not** do: it does not call the Seabraid runtime and does not route
individual experts to separate devices. No mixed-device expert dispatch has been run in llama.cpp.

What upstream would need (proposed): a way to split one layer's expert picks between two backends or
processes inside the MoE FFN build; a per-layer submit and collect around it (the shape in section 2.1);
a place to start and own a helper process. Today's scheduler places a whole layer's tensors on one backend.

### 7.2 Strata (adapter and patch exist; live runs on one machine with two cards)

Two paths, both documented in the [user guide](USER-GUIDE.md#1-run-a-moe-model-on-an-nvidia-and-an-amd-card):

- The **Strata integration** (`compat/strata`): a patch series on Strata v0.2.15 plus a helper
  process on a named shared-memory channel. Measured; helper loss is not survivable on this path.
- The **Strata adapter** (`adapters/strata`) plus
  [`strata-v7-seabraid.patch`](../adapters/strata/strata-v7-seabraid.patch): `QwenRuntime::start` opens
  a Rust session over a private helper, the patch delegates at Strata's weighted merge and discards a
  serial window whose helper failed, and a placement driver acts at Strata's decode-window boundary
  through its `RungIO` interface (`adapters/strata/placement.hpp`).
  The patch's second commit adds up to three helper rungs, the fallback rung and blob swaps (installer
  `--ladder`; switches in the [options catalog](OPTIONS.md#re-tiering-and-residency)); commit 3 adds the engine side of
  file-reference moves (`--retier-file-moves`), implemented, not yet shown on GPUs. It is a worked example of items 10
  and 11. A `provider_v2` of your own must follow the `from_fallback_rungs` rule in
  [PLACEMENT-ABI](PLACEMENT-ABI.md#by-reference-moves-provider-v2).

Strata is the reference for the process model in section 3 and for the fence in section 4. Copy its
tests, not its code layout: the engine-specific parts are a patch against a pinned commit.

### 7.3 Engines with no adapter: vLLM, SGLang, KTransformers, ik_llama.cpp

**Proposed, not implemented, not tested.** We have not built against these engines. The repository
only plans adapters for them ([roadmap](ROADMAP.md#adapters)). The hook points below are stated at the
level every MoE engine shares; check each against the engine's current source before you start.

| Engine | Hook points to look for | Route |
|---|---|---|
| **ik_llama.cpp** | The MoE FFN graph build and the expert weight tensors, as in llama.cpp | Reuse the ggml adapter and the hook pattern of the llama.cpp patch. We have not applied our patch to it |
| **KTransformers** | The MoE layer forward where experts are split between CPU and GPU, and the process that launches it | A second-device provider next to its CPU expert path |
| **SGLang**, **vLLM** | The MoE layer forward (router output, expert weights), and the worker-process launch that already exists for tensor and expert parallelism | The Python binding with `Runtime(provider=fn, ...)` for the callable, or the C ABI from a native extension. A cross-vendor card still needs its own helper process behind the callable. The Python binding has no split-phase submit today (see the constraints below) |

As an **outline** only, an integration would need these four items: (1) at layer forward, split
the router's picks into "mine" and "helper's"; (2) submit before and collect after the engine's own
expert compute; (3) keep the engine's own path as the failure fallback; (4) own the helper process's
launch and teardown. This is a list of the pieces, not a recipe: a working engine-specific example
also has to bridge buffers, execution ordering and the partial reduction, and none exists yet.

What the repository does not provide today, so item (2) is not available from Python as written:

- **No public split-phase submit from Python.** The `PreparedMoE` weave in
  `bindings/python/seabraid/runtime.py` submits and then collects before it returns, so the call blocks
  for the helper's round trip and cannot overlap the engine's local expert compute. Overlap needs a
  native or threaded prototype that calls `seabraid_session_submit` early and `seabraid_collect` late.
- **No device-tensor boundary.** The C entry points borrow host-readable slices, copy the inputs, and
  collect `F32` output ([ABI.md](ABI.md)). Activations that live on the engine's GPU must be staged to
  host memory first and the result copied back. The staging, the stream or event ordering around it,
  the dtype (the collected output is `F32`; engines often hold BF16 or FP16 activations) and what CUDA or HIP
  graph capture does to a host-side submit and wait are all open constraints, not yet measured here. Time the full staged round trip on the target engine before judging the design or
  changing the ABI.
- **A callback is not enough.** Registering a provider callable gives the helper's work a place to
  run; it does not supply staging, synchronization or overlap with local GPU work.

The engines' own docs show where a boundary could sit, which we have read but not built against: vLLM's
modular fused-MoE kernel separates dispatch (`prepare`) and combine (`finalize`) from the expert compute,
and has an optional split-phase hook (`prepare_no_receive`, which returns a receiver callback) that
backends opt into
([vLLM fused MoE modular kernel](https://docs.vllm.ai/en/stable/design/fused_moe_modular_kernel/));
SGLang stages the MoE forward as dispatch, pre-permute, core runner, post-permute and combine, with
`BaseDispatcher` and `MoeRunner` classes
([SGLang expert parallelism](https://docs.sglang.io/advanced_features/expert_parallelism.html)). Whether
either boundary can carry a cross-vendor helper is untested. Python hosts also read the [Python user section](USER-GUIDE.md#2-python-user) and the
[hosting a provider](ADAPTER-GUIDE.md#hosting-a-provider-from-python) contract: views are valid only
during the call, and an exception becomes status 4.

## 8. Conformance checklist

Run these before you claim an integration works. Each names a test or fixture in this repository you can
copy or run against your build.

| Check | What to reuse |
|---|---|
| Header and library agree; the SDK links from outside the tree | `find_package(Seabraid)`, [`examples/consumer`](../examples/consumer/CMakeLists.txt), `native-tests/installed/check_sdk.sh` |
| A minimal provider round trip | [`examples/minimal_provider.c`](../examples/minimal_provider.c): output must equal the reference transform |
| Every refusal and failure path of a provider (stale epoch, format refusal, `COMPACT_ROWS`, device failure, status counters, log sink) | [`examples/adapter_walkthrough.c`](../examples/adapter_walkthrough.c), which prints `ok` last. `native-tests/provider_smoke.cpp` covers the older v1 multi-operation calls |
| Placement: fill, open, plan, apply, ack, restore, quarantine, snapshot (read the fixture for which cases it covers) | `native-tests/placement_smoke.c` (C11) and `native-tests/placement_smoke_cxx.cpp` (C++17) |
| Your sink and `seabraid_last_error` work from your threads | `native-tests/diag_smoke.c` |
| A scripted helper: wrong row count, NaN reply, late reply, helper killed mid-request, a helper that exits when its controller dies | the fixtures in `adapters/strata` (`static_qwen_test.cpp`, `private_qwen_test.cpp`, `helper_process_test.cpp`); they need a Strata source tree, no GPU |
| Provider math against an oracle | `adapters/ggml/test_q4.c` and `test_q4_oracle.h`: a scalar oracle, changed inputs, partial routes, plan reuse |
| **Parity fuzz (copy the idea):** seeded random batches, same input through your local path and through the Seabraid path, compare every output element and every error status; a failure prints the seed | `bindings/python/tests/test_parity_fuzz.py` and `_parity_cases.py` do this for the reference session against the native library |
| Output quality on a real model | A KL gate against your engine on one card: `compat/strata/tools/kl_gate.sh` and `compat/strata/tools/kl.py` (two configs and a fixed text, noise floor measured by running stock twice) |
| Live placement under load | The matched-arm harness in `compat/strata/scripts/placement-qual`; the A/A arm is the checker for fence invariants |
| Load shape without a model | `tools/sim` (`SEABRAID_BUILD_TOOLS=ON`) |

The minimum for a first integration: the round trip, the stale-epoch and device-failure paths, a killed
helper that leaves the engine serving, and a KL comparison against the same engine on one card. State in
your own write-up what you ran and what you did not.

Build and run the repository's native tests first so that you know your toolchain is sound:

```sh
cmake -S . -B build/seabraid -DSEABRAID_BUILD_NATIVE_TESTS=ON
cmake --build build/seabraid
ctest --test-dir build/seabraid --output-on-failure
```

## 9. Reporting problems

Report as in the [user guide](USER-GUIDE.md#5-reporting-problems), with the version (`seabraid_version()`), the
platform, and what you did and expected. Include `seabraid_last_error` text and the log records; they carry codes,
counts and ids only. If you know prior art we missed, tell us that too.
