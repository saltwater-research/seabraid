# Roadmap

Seabraid is alpha (0.5.0-alpha.8). This is the open work for the runtime, in rough
priority order. The roadmap for the Strata add-on is in
[compat/strata/docs/ROADMAP.md](../compat/strata/docs/ROADMAP.md). An item
here is a plan, not a promise, and nothing listed is claimed to work yet.

## Verification still owed

- Output quality of the new runtime's cross-vendor path, repeated with more
  texts and at least two engine starts per arm. The 2026-11-17 KL gate passed
  pooled in both orders, but with one start per arm and three short texts its
  per-text results are noisy.
- The ggml adapter's transfer counter is done (`seabraid_ggml_get_transfer_stats`;
  the direct execute path only, the scheduler's own copies in a hook run are not
  visible to it). A HIP timing rerun on a quiet host (n>=6) remains. Logits parity is done:
  bit-identical to stock on CPU, CUDA, HIP and a Vulkan two-vendor layer split
  (2026-11-17/18, Qwen3-30B-A3B).
- Windows handle handoff and multiple workers (the sealed descriptor path is
  Linux-only today).
- Live placement publisher (experimental and opt-in; static placement is the
  supported default, see [DECISIONS](DECISIONS.md)), after the first matched-arm
  run (2026-11-18, n=2-4; [figures](../README.md#whats-verified)): long pauses
  (10 s or more) appeared mostly in the live arms and their cause is unproven
  (pinned expert RAM leaves about 285 MiB free), so the next test varies host RAM headroom independently of the moves, with a
  stall-rate A/B at n>=10 with and without moves (a proposed follow-up, not a
  result), and reports stall-inclusive decode, per-text quality, convergence and
  helper-loss recovery separately; a margin and interval sweep, since
  placement never settles at the defaults; why one KL text sits at 4x its
  noise floor after moves; D4 at n>=4; and a helper share and table hash in
  the legacy ladder's round line so it can be compared directly.

## Runtime and ABI

- **Device descriptor.** A provider reports what it is (vendor, memory,
  slot geometry, supported formats) so placement can be planned from the
  descriptor instead of adapter-supplied tables.
- **Pull-based signals.** Memory pressure and step-time signals the engine can
  read, so promotion can weigh cost and not only routing heat.
- **Prompt-start expert hints.** Evaluate hints with held-out traces and
  measure causal lookahead against the engine's own next-layer warm-up, using
  ordered route timestamps (the v1 histogram cannot show when an expert first
  appeared).
- **Directly leased staging slots.** Measure the two activation copies and the
  wait in the private handoff before adding a fast path; any fast path must
  keep geometry validation, storage lifetime and late-reply isolation.
- **begin/finish callbacks appended to `provider_v3`.** Split callbacks that
  run on the engine's thread, removing the two thread hand-offs per layer that
  the worker queue adds today. `seabraid_provider_v3` itself has shipped (it
  is the struct `seabraid_session_open` takes); this is a later append to it,
  not a new version. Measure first.
- **Wait strategy.** A cross-process futex wake after a short spin instead of
  spin-then-nap on both sides, so idle waits stop burning a core.
- **Pinned staging.** Register the shared input/output regions with the
  vendor runtime for asynchronous DMA, and reuse staging buffers for placement
  swaps outside the fence.
- **Cost-aware placement.** Feed measured per-card throughput into the planner
  so a layer's helper share balances helper compute plus transfer against
  the lead card's time, and overlap predicted expert transfers with compute.
- **Device buffers.** A buffer descriptor (host, CUDA, HIP; dtype) so engines
  that keep activations on the GPU avoid a host round trip.
- **Observability.** A pollable completion (eventfd or callback) for async
  runtimes and per-ticket timings from the helper.
- **ABI freeze.** Settle the structs and the placement API, then tag 1.0; the
  list of what is left is [below](#before-an-abi-freeze).

## Before an ABI freeze

The ABI is not frozen. Status of each item, checked against the tree:

1. Open: sized v2 forms of `config` and of the four caller-filled placement structs.
2. Done, one residual: the submit order is not part of the contract, the duplicate-id rule is stated and the runtime refuses a repeated expert id, and `COMPACT_ROWS` is refused on provider v2. The adapters still carry their own copies of the duplicate-id check.
3. Half done: the v1 calls are marked deprecated (`SEABRAID_DEPRECATED`); deleting them is a 1.0 task.
4. Open: a measurement gate (compaction on versus off, KL) before `COMPACT_ROWS` is called stable; the device descriptor and the epoch authority stay open design questions.
5. Open: the Miri job (FFI tests under Stacked Borrows and Tree Borrows; [ABI.md](ABI.md)) has not yet run in CI. Its first run (2026-11-19, on a test machine) found aliasing undefined behavior in the old layout and a use-after-free in one test; after the fix both models report none.

## Adapters

- **vLLM and SGLang.** Python-binding or C ABI provider adapters. Not
  integrated yet.
- **KTransformers.** A second-device provider next to its CPU expert path.
  Not integrated yet.
- **Project Maya and ik_llama.cpp.** The Strata adapter path and the ggml
  adapter respectively. Untested.
- **Multimodal.** Validate encoded audio and video activations with a real
  multimodal MoE adapter. Evaluate a stateless dense operation for audio or
  video backends as a separate operation with its own measured contract.
- **Dense FFN shards (deferred).** Not in the expert ABI until a real consumer
  exists; first prototype one stateless operation against an unsplit oracle
  (see [DECISIONS](DECISIONS.md)).
- **Second-GPU payoff comparison (planned).** The three-arm comparison and its
  status are in [DECISIONS](DECISIONS.md).
- **One native split-phase adapter first.** Before any device-buffer ABI, a
  native or threaded prototype on one documented dispatch/combine boundary
  (vLLM or SGLang), timing the full staged round trip; see the constraints in
  [engine integration](ENGINE-INTEGRATION.md#73-engines-with-no-adapter-vllm-sglang-ktransformers-ik_llamacpp).

## Packaging

- **Platform wheels** for the Python package, so `pip install` works without a
  source build. Done: `scripts/build_wheel.py` builds a Linux x86_64 wheel that
  bundles the library (`seabraid.bundled_library()`) with a glibc-derived
  manylinux tag. Remaining: the Windows and macOS wheels exist only as a CI job
  that has not run, so they are unbuilt until it does. Only the Linux x86_64 wheel is on PyPI.
- A published SDK archive per platform.
- Windows build and handoff support for the Strata adapter.
