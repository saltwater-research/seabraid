# Migration inventory

> Historical snapshot, written on 2026-11-14 during the migration. It is
> superseded by the [README](../../../README.md) and the
> [CHANGELOG](../../../CHANGELOG.md). Counts below are corrected where they were
> plainly wrong; the rest describes the state when it was written.
> Current facts: the patch series is 0001-0006 and 0010-0020 on Strata v0.2.15 (0007 became `adapters/strata/wire` on 2026-11-19; 0008 and 0009 were dropped on 2026-11-17 with Strata's DeepSeek-V4.1 engine); the add-on suite passed
> 434 (11 skipped) on Windows and 445 on Linux; the Strata fixtures now pass in
> CTest on Linux.

The Seabraid tree descends from the first Strata integration.
The compatibility installer pins Strata v0.2.15 (`patches/strata/STRATA_BASE`; it was `628f74955` when this inventory was written). A separately
rebased Strata integration also exists. Keep the base and both
integrations identifiable when recording build and timing evidence. See the
[design log](DESIGN-LOG.md) and the runtime [roadmap](../../../docs/ROADMAP.md).

| Source | Current location and migration destination |
|---|---|
| 0001 external tier | Preserve the Strata adapter and v7 connection during the Rust transport migration. Engine routing, cache integration, and final merge remain adapter responsibilities. |
| 0002 slot fill clamp | Preserve as a Strata correctness fix. |
| 0003 DFlash2 window | Preserve as Strata speculative decode tuning. |
| 0004–0005 re-tier and residency | Rust now implements the same-layer card swap planner and acknowledged updates. Active Strata cache writes and engine-thread fences remain in its adapter. |
| 0006 hot-share | Keep experimental pending an advantage over the default at matched settings. |
| 0007 capability and weight spec | Move the generic contract into the runtime. Keep format descriptors and arithmetic rules provider-owned. |
| 0008–0009 DeepSeek helper and link comparison | **Dropped 2026-11-17.** Strata 0.2.14 removed its DeepSeek-V4.1 engine, so there is nothing to patch; the add-on's dsv41 helper, executor and KL gate were deleted with them. The row stays as history. |
| 0010 weighted output | Retain output semantics and validate them through the native API. Provider kernels implement reduction. |
| 0011 route histograms | Keep engine capture in adapters and move analysis to tooling. |
| 0012 ownership map and promotion | Rust now tracks ownership, applies epoch-bound moves, restores known slots, and quarantines uncertain providers. The engine must drain and fence before applying an update. |
| 0013 D4 prefill | Preserve bounded prefill and the helper path. Strata retains its scheduler and lending fallback. |
| 0014 channel hardening | Carried in the compatibility installer with helper bounds fixes and CPU fixtures. Switch to the sealed descriptor path only after native handoff and lifecycle tests. |
| D4 drain | Kept in the private dev history as `compat/strata/docs/legacy/strata-d4-drain.patch` for independent qualification; not exported (export-ignore). |
| Installer, discovery, card probe, lead ranking | Preserve in compatibility tooling. Extract reusable parts after the second adapter establishes its needs. |
| Prompt expert replay, capture, and route report | Carried as offline tooling. The existing parser and replay tests remain part of the evidence. |
| KL gate, benchmark arms, and exclusion records | Keep with qualification. Retain historical revisions and excluded experiments. |
| Strata fixture and recovery fixes, HIP link fix, Windows draft | Integrate independently once verified on the relevant source and native toolchain. |

## Verification

- The 16 active compatibility patches (0001-0016) applied in sequence to the published
  Strata pin in a disposable Git index. Patch 0014 and the separate D4 drain
  also applied to both local post-0013 candidates.
- The targeted compatibility installer and replay suite passed 39 tests at the time (the add-on suite is now 434 passed, 11 skipped on Windows; 445 passed on Linux).
- Rust formatted and type-checked across all targets in this package.
- C11/C++17 syntax and Windows x64 C layout checks passed with the
  installed Clang. Four Python mock tests passed and four linked tests
  were skipped pending a shared library.
- The Windows Rust linker attempt requires SDK import libraries absent from
  that environment. Native linking, Linux process tests, and GPU/engine parity
  remain open. The local CPU provider's arithmetic serves as a fixture.


## Hybrid adapter and modality verification

- The packaged Q4_0 ggml fixture passed against the installed native CPU,
  Vulkan, and CUDA providers. The swapped gate/up negative control failed
  as expected. Engine-level graph integration and model parity are pending.
- The fixed-layer Strata adapter and its v7 fixture compiled. Running the
  fixture on Linux and qualifying the loaded Qwen helper are pending.
- The default C ABI and Python wrapper expose capability discovery, tagged
  routed input, and session status. Six Python mock tests passed. Running
  the five linked Python integration tests requires a built Seabraid library.
- Rust type-checking covers 14 placement regression tests. The Windows
  linker lacks the SDK startup objects required to execute them locally.

## Transition checkpoints

| Area | Landed | Remaining proof |
|---|---|---|
| Runtime and bindings | One worker, shared buffers, multiple prepared operation IDs, explicit partial-route capability, Python activation metadata | Run linked Rust and Python fixtures on a supported native linker |
| Transport and recovery | Private v7 Linux descriptor handoff, dedicated helper supervision and opt-in weighted Strata routing with serial-window retirement | Run linked Linux fixtures, controller kill, and two-request recovery with a loaded helper |
| Adaptive placement | Rust planner and ownership transactions plus a Strata rung transfer and snapshot bridge | Adopt the engine's live fence and shared route authority, then measure promotion and fallback |
| Engine adapters | Opt-in source patches for Strata weighted serial decode and llama.cpp Qwen3 MoE scheduling, plus the direct ggml cache and layer table | Build both full engines and qualify loaded-model parity and latency |
| Verification | ggml Q4_0 provider passes on CPU, CUDA, and Vulkan. Disabling graph reuse fails the cache regression. C/C++ source checks and Rust Clippy pass. | End-to-end mixed-device model runs, Windows handoff, and broader media providers |

Separate Strata changes exist for a fixture fix,
serial recovery and a Windows stack. The patched v7
baseline lacks their generic expert-process fixtures and interface, so these
changes remain separate from the Seabraid patch series. Seabraid ports the
serial-window discard and next-request recovery behavior to the v7 weighted
merge. The top-8 fixtures compile on both selected geometries. The
pinned engine's weighted reducer still expects ten picks, and the engine
adapter checks that requirement before launching the helper. The reported
CUDA/HIP builds and CPU recovery checks apply to that separate stack.
Seabraid's live second-request and Windows runtime qualifications remain open.
