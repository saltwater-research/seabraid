# Seabraid research and architecture log

## Recording a result

Record the date, Seabraid revision, engine revision, hardware and backend, workload,
comparison, result, and evidence location. Mark unrun hypotheses and failed
controls explicitly. Preserve the original baseline when updating an entry.
The [migration inventory](../compat/strata/docs/MIGRATION-INVENTORY.md) maps every
retained Strata patch to its owner.

## 2026-11-14 · Architecture changes relative to Strata

The compatibility baseline was Strata `628f74955` (v0.2.13.1) when this entry was written; since 2026-11-17 it is
Strata v0.2.15 (`c77572806`). Treat Strata rebases and Strata's own later changes as separate comparison targets, and label every measurement with the Strata version it ran on.

| Change | Ownership and rationale | Evidence and open work |
|---|---|---|
| Engine-independent hybrid runtime | Rust owns the request lease, validation, buffers, and placement. C/C++ providers keep their native tensors and kernels. Python exposes the same runtime for adoption and testing. Engines retain routing and reduction. | A shared session selects a prepared layer by operation ID and keeps one in-flight batch. Source checks and Python mock tests pass. Linked Rust and Python checks are open. |
| Private Strata helper | The adapter starts a dedicated v7 helper with a sealed Linux file descriptor, verifies geometry and the peer PID, and owns shutdown and reaping. A replacement starts a fresh channel generation. A pidfd watcher ends the helper when the controller process dies. Existing Strata flags and helper behavior live in the compatibility path. | Adds a scripted-peer fixture for failed startup, malformed geometry and termination. The controller-kill regression uses a scripted peer and checks that an orphaned helper exits. Adapter and fixture compile. Linux execution, the loaded helper, and the engine fallback path still need verification. |
| Acknowledged expert placement | Rust proposes same-layer exchanges and commits only acknowledged moves. The adapter must supply the engine fence, implement physical transfer, and publish the resulting ownership. Uncertain providers require reconciliation. | Exposes plans, snapshots and apply through the C ABI. Native test sources and Rust checks pass. Real engine fence, failure recovery and promotion outcomes remain open. |
| ggml graph and layer reuse | C owns one prepared graph per provider and reuses it for changed route IDs and input with compatible geometry. A multi-layer provider shares one cache entry. Changing layers replaces it. Backend allocation stays with ggml. | The cache and the v2 layer table passed a synthetic Q4_0 fixture against ggml 0.28.0 on CPU, CUDA and Vulkan. Re-run 2026-11-17 against ggml 1.0.0 (llama.cpp v1.0.0, b1b770d) on CPU, CUDA (RTX 3090), HIP (R9700) and Vulkan (RTX 3090 and R9700): all pass. The deliberate cache miss failed the regression. A single model running mixed CUDA and Vulkan, scheduler integration and model latency are open. |
| Activation provenance for text, images, audio and video | The engine or encoder produces the activation rows. Seabraid exposes provider capabilities and accepts an origin tag for those rows. This keeps media decoding and model input contracts with their owners. | Adds capability and input metadata to the native and Python APIs. Real multimodal expert routing remains untested. Dense audio/video operators would need their own measured contract. |
| Legacy performance and correctness work | The compatibility series carries route histograms, N-card placement, weighted output and D4 bounded prefill. (It carried a DeepSeek helper as patches 0008 and 0009 until Strata 0.2.14 removed its DeepSeek-V4.1 engine; they were dropped on 2026-11-17.) The separate D4 drain patch keeps drain behavior independently testable. | All active patches (0001-0007, 0010-0016) apply to the pin. D4 and promotion need matched quality, failure and performance checks before a default change. |

Fixture elapsed times include setup and are unsuitable as throughput baselines.

## Research hypotheses and prior art

- **Prompt-triggered expert warming.** Use existing route captures and replay
  to compare a prompt-start prediction against static hot-expert placement and
  actual routes. Stratify by layer, prompt class, and modality. Score useful
  bytes arrived before first use, wasted bytes, and TTFT. A prediction may
  warm weights, while the model router still selects every expert.
- **Lookahead during prefill.** Test async warming from routes already
  observed in earlier chunks or layers. The current layer's gate provides
  exact IDs only after its activations are available. Compare useful overlap
  against the bandwidth it takes from active layers and the p95 TTFT. The
  replay and route capture already exist. Online prediction and prefetch
  decisions remain experiments.
- **Compare against upstream scheduling.** Match quantization and memory
  budget against llama.cpp's existing MoE cache and ggml's backend scheduler.
  SGLang expert parallelism and KTransformers expert scheduling are related
  work. Seabraid's candidate differentiator relative to the Strata pin is a
  reusable host-engine boundary for heterogeneous, same-layer experts.
  [llama.cpp cache](https://github.com/ggml-org/llama.cpp/blob/master/docs/moe-cache.md)
  · [ggml backend API](https://github.com/ggml-org/ggml/blob/master/include/ggml-backend.h)
  · [SGLang expert parallelism](https://docs.sglang.io/docs/advanced_features/expert_parallelism)
  · [KTransformers scheduling](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/kt-kernel/experts-sched-Tutorial.md)
- **Multimodal integration.** Qwen3-Omni is a candidate for comparing expert
  demand across modalities after encoding. Select a real adapter and verify
  layer geometry, residency pressure and numerical parity. Consider DLPack
  for tensor interop when a consumer can demonstrate an ownership and stream
  synchronization contract. [Qwen3-Omni](https://github.com/QwenLM/Qwen3-Omni)
  · [DLPack](https://dmlc.github.io/dlpack/latest/python_spec.html)

## 2026-11-14 · Engine transition follow-up

| Change relative to the Strata baseline | Evidence | Open measurement |
|---|---|---|
| v7 serial serving through Seabraid | Binds the private runtime to Strata's existing weighted reducer. The serial drain-and-discard behavior comes from a Strata review fix. Strict source checks and the complete 15-patch replay passed. also fixes a stale operation count exposed by the channel hardening patch. | Qualify the two-process engine on CUDA and HIP. Time local overlap and the first request after helper death. |
| Routed graph continuation | Adds an opt-in llama.cpp Qwen3 MoE patch on llama.cpp v1.0.0 (b1b770d). Native CPU fixtures pass on ggml 0.12, 0.22 and 1.0.0, including two layers, a residual continuation and changed routes on replay. Deliberately using the wrong warmup reduction returns 21. Direct-only checks pass on CUDA and Vulkan. The patched libllama builds on CPU with the adapter enabled, and the hook leaves layers served by llama.cpp's new MoE cache on the stock path. | Measure real-model logits, residency, transfer volume and p95 layer latency with default flags and the opt-in graph path. This path eliminates an extra Seabraid graph and host round trips by construction. |
| Fenced placement bridge | Uses the Strata rung interface for physical acknowledgements and stages a consistent Rust executor snapshot for publication. The fixture compiles. | Qualify actual writes, rollback, and quarantine at a shared live fence. Legacy blob read timeouts can leave a request outstanding, and legacy promotion advances the channel epoch behind the decode lease. |
| Prompt expert replay | Uses actual prefill token counts and adds a static control based on how often each expert appears within training families, with separate held-out evaluation. A synthetic holdout needs an 8-byte on-demand load under the frequency baseline and zero under the new static control. All 29 targeted replay and report tests pass. | Add ordered trace events before measuring causal hints. The v1 histogram cannot tell when an expert first appeared or when its weights became usable. |

The pinned Strata decode dispatch already calls `d.lookahead->submit` to warm
the next layer. Use it as a baseline when testing prompt-start hints and
earlier prefill observations. Compare prefetch bytes that arrive before use,
wasted traffic, p95 TTFT and CPU/GPU overlap at the same memory budget.

The private Qwen handoff currently copies activations through the Rust request
lease and into the v7 mapping. Measure both copies and the wait before
considering a directly leased staging slot. Any later fast path needs to
preserve geometry validation, storage lifetime and late-reply isolation.
A worker-liveness check during prolonged waits could also shorten hard-crash
recovery while keeping the fast-response path free of process syscalls.

## 2026-11-14 · Plugin integration fixes

- The private Strata helper now uses the engine's resolved pack and native
  shard. Legacy overrides must resolve to those same filesystem objects.
  The ten-case checkpoint fixture and both launcher configurations passed
  strict C++17 source checks. The fixture still needs execution on Linux.
- A standalone CMake install supplies `Seabraid::runtime` and the ABI headers to
  downstream `find_package` consumers. Relocation and missing-library checks
  passed with stand-in Windows artifacts. A linked installed consumer still
  needs a host with a complete linker and SDK.
- Python now rejects invalid batch counts before building ctypes arrays.
  All six new malformed-input cases reproduced the problem before the fix.
  Seven Python tests passed after the fix and five linked cases await the
  native library.

Status note (2026-11-20): the "needs execution on Linux", "linked Rust and Python checks are open" and "needs a host with
a complete linker" items in the three 2026-11-14 entries above are later covered by the [2026-11-16 Linux gates](#2026-11-16--linux-gates)
(CTest including the Strata fixtures, linked Python, and an installed-SDK consumer built from outside the tree) and the
[2026-11-17 verification](#2026-11-17--verification-on-strata-v0215). Items those entries do not name stay open.

## Entries 2026-11-16 to 2026-11-18: provenance of the README's verification table

The entries below back every row of the README's "What's verified" table. They
were written on 2026-11-19 from what the repository itself records: the
adapter README, the add-on design log, the roadmap, and the commit messages of
the development history (named by short hash; a squashed public history does
not carry them, so these entries are then the record). Figures not written
down in any of those places are not reproduced: the raw run output stayed
outside the repository, and a figure with no written source was removed from
the README rather than re-derived. Host class for every live run: one Linux
x86_64 machine with an AMD R9700 (HIP, gfx1201) and an NVIDIA RTX 3090 (CUDA,
sm_86). Engine: Strata v0.2.15 (`c77572806`) unless stated.

## 2026-11-16 · Linux gates

- Revision: development tree on 2026-11-16 (commit 07a4e94 records the gates).
- Commands: `cargo test --offline`; the CMake configure, build and `ctest`
  lines of the README's Quickstart with `-DSEABRAID_BUILD_NATIVE_TESTS=ON`
  `-DSEABRAID_BUILD_STRATA_ADAPTER=ON -DSTRATA_SOURCE=<strata checkout>`; an
  ASan+UBSan build of the native tests; a `find_package(Seabraid)` consumer
  built against the installed SDK from outside the tree.
- Result: Rust 53 of 53; CTest 13 of 13 (8 without the Strata adapter);
  ASan/UBSan 8 of 8; installed-SDK consumer built and ran
  ([TECHNICAL-NOTE section 7](TECHNICAL-NOTE.md#7-verification-status) says
  the same).
- Live cross-vendor run of the new runtime, same day (R9700 HIP engine, RTX 3090
  CUDA helper, previous pin Strata v0.2.13.1 `628f74955`): 89.5 to 95.0 tok/s
  greedy decode over six runs; helper kills as in
  [the adapter README](../adapters/strata/README.md#opt-in-strata-source-integration)
  (client error after 0.33 to 0.44 s, next request served, no zombie). Output
  quality was not measured that day.

## 2026-11-17 · Verification on Strata v0.2.15

- Revision: commit 889d83f (the move to v0.2.15); patch files applied with
  plain `git am`.
- Commands: the build and `ctest` steps of the adapter README on the patched
  Strata tree (HIP and CUDA builds, plus a CPU run); `cargo test --offline`;
  the Seabraid CTest run in Release and ASan+UBSan.
- Result: Strata's own ctest on the patched tree, CPU 138 pass / 0 fail, HIP
  291 / 2, CUDA 357 / 2. The failures are the missing model for `ple_parity`
  and two that fail on a pristine tree too
  ([STRATA-INTEGRATION 4.6](../compat/strata/docs/STRATA-INTEGRATION.md)).
  Seabraid CTest 16 of 16 in both Release and ASan+UBSan; Rust 62 tests.
- Live, IQ3_S, greedy 2,000 tokens, n=6 per arm: R9700 main + RTX 3090 helper
  94.8 tok/s mean (sd 3.0, range 91.1 to 98.4; 94.75 on the previous pin);
  RTX 3090 main + R9700 helper 106.7 tok/s mean (sd 2.0, range 104.6 to 110.2,
  16,000 helper slots). SIGKILLed helper (three per arm, fresh engine each
  time): request failed after 0.33 to 0.46 s (HIP main) and 0.51 to 0.73 s
  (CUDA main); the next request was served 1.7 to 1.8 s and 2.5 to 2.7 s after
  the kill; no zombie
  ([adapter README](../adapters/strata/README.md#opt-in-strata-source-integration)).
- D4 remote prefill (patch 0013, opt-in): engaged, but an 8.3k-token prompt
  read slower than lending (n=2 per mode). Off by default.
- KL quality gate against the same main card alone, both card orders, three
  short texts, one engine start per arm: the pooled result passed in both
  orders (commit 7b7122e). The gate's thresholds are `kl_max` 0.03 and
  `top1_min` 0.95 in `compat/strata/scripts/placement-qual/arms.json`. The KL and top-1
  values were not written into the repository and are not reproduced.

## 2026-11-17 · Long prompts on the private helper tier

- Revision: commit e23ce87 (helper experts stay in the engine's RAM copy; RAM
  fills decode's experts first). Host: R9700 main + RTX 3090 helper, Qwen3.8-Flash-Next
  IQ3_S, private tier. Command: long-prompt requests with a needle; launcher
  command not recorded in the repository.
- Result, before to after the fix: 32k-token prompt first token 32.0-35.7 s to
  20.3 and 18.0 s (named channel 19.6 and 17.1 s); 128k 135.5 s to 80.7 s; 261k
  260-269 s to about 128 s; needles found; short-prompt decode 91.8-98.5 to
  91.8-98.4 tok/s with no file reads in decode; KL rows byte-identical to the
  run before the fix, in both card orders. With the RTX 3090 as main there was
  no change (its helper's 31.7 GiB does not fit the RAM budget).

## 2026-11-17/18 · llama.cpp hook logits parity

- Revision: commits 1fd4de4 (2026-11-17, CPU) and 5c535a3 (2026-11-18, CUDA,
  HIP, Vulkan); roadmap entry 2955420. Engine: llama.cpp v1.0.0 with the opt-in
  `adapters/ggml/llama-cpp-qwen3moe.patch`; model Qwen3-30B-A3B.
- Comparison: logits with the hook on against the same backend's stock run, 2,044
  positions over 4 texts.
- Result: bit-identical to stock on CPU, on CUDA (RTX 3090), on HIP (R9700) and
  on an RTX 3090 + R9700 Vulkan layer split. The hook appends ggml operations
  and does not call the Seabraid runtime, and the Vulkan arm is llama.cpp's
  own layer split, not an expert split. The roadmap still lists a quiet HIP
  timing rerun (n>=6). No timing or transfer figure is recorded in the tree.
- Independent audit of the 17 Nov CPU parity run: engagement of the hook is shown
  by a call counter on a 12-layer cut of the model (96 successful calls); for the
  full model the evidence is the numerical equality of the outputs, since the
  full-model counter run ended before completion. On one backend the hook appends
  the same ops as stock, so equality is expected. The audit also found one
  hook-line difference between the source archive of that run and the expected
  patched graph (`graph.arch` versus `graph.model.arch`), so which graph variant
  each historical binary compiled is not established.

## 2026-11-18 · First matched-arm run of the live placement publisher (S8)

- Revision: commits 4b59351 (driver fixes found by the first live smoke) and
  16e32e4 (the README and roadmap record). Strata v0.2.15 with the repository's
  patch files and `strata-v7-seabraid.patch`; R9700 HIP main + RTX 3090 CUDA
  helper. Runner: `compat/strata/scripts/placement-qual/run.py` (arms and criteria in
  `arms.json`).
- Comparison: the Seabraid placement publisher against static placement and the
  legacy publisher, matched arms, 2 to 4 engine starts per cell.
- Result: a first run, not a pass. Mean decode, which includes the stalls, did
  not pass. Long pauses (10 s or more inside one request, archived W1, counts out
  of 12 per arm as recorded by the 2026-11-19 audit; the audit summary does not say
  whether the 12 are requests or starts): private live 3/12, private static 0/12,
  zero-move control 0/12, named-channel new publisher 2/12, named legacy 1/12,
  named static 1/12. Run-wide, 15 of 194 promotion requests and 5 of 140 static or
  control requests had a long pause. Shorter warnings also occurred, so "stalls hit
  every arm" held for warnings, not for long pauses (private static and the
  zero-move control had none). Recomputing the archived request means gives private
  static 93.66 tok/s against private live 75.66 tok/s (audit, 2026-11-19; the audit
  also found failed A/A token-stream equality and variation inside the static
  control itself). Pinned expert RAM leaves about 285 MiB free, and pauses coincide
  with that memory pressure, but whether the moves or the memory pressure cause the
  arm difference is **unproven**: it needs a controlled comparison that varies RAM
  headroom and promotion separately. Verdict kept: stall-inclusive decode did not
  pass, so live promotion stays opt-in and experimental, and static placement is the
  supported baseline. Placement never
  settles at the default margin and interval. One KL text sits at about 4x its
  noise floor after moves. The 100 ms fence budget is provisional. Follow-ups
  are in [the roadmap](ROADMAP.md#verification-still-owed). The per-arm figures
  were not written into the repository and are not reproduced.
- Same day, Linux gates at commit 4b59351: CTest 19 of 19 in Release and
  ASan+UBSan; Rust 63 of 63.

## 2026-11-19 · Live run L1: the pyramid on real cards

- Revision: repository commit e9cc304 and the Strata v7 patch series it carries (the engine side
  of file-reference moves was in the series but not exercised). One Linux test machine. Main card
  R9700 (HIP); helper1 RTX 3090 (CUDA); helper2 a second HIP helper **process** on the same R9700
  (not a third card). Model Qwen3.8-Flash-Next IQ3_S with capped capacity: main 2582 slots, helper1
  1500, helper2 1000, RAM budget 6 GiB, so 66% of the 24576 experts lived on the NVMe. The run sat in a
  cgroup memory limit with the page cache evicted, so SSD reads were real. Evidence: the run's
  per-scenario summaries, ledgers, journeys, latency tables, the KL analysis and its chain report
  (archived outside the repository; the per-request figures are not reproduced here).
- Comparison: placement across all three boundaries (main to helper1, helper1 to helper2, helper2 to
  rung F) against its own verification fields, and decode-path KL against the main card alone.
- Passed, natural traffic (S2): 1539 moves in 513 rounds, all committed with `verified=1`; 0
  stale-epoch refusals; every move's byte count equal to the pack's per-layer blob size. Hops: main to
  helper1 520, helper1 to helper2 481, helper2 to rung F 538. Fallback promotions: 413 from RAM, 125
  from the SSD pack file.
- Passed, traced journey (S6, heat steered with `SEABRAID_PLACEMENT_QUAL` and `--retier-ceiling 1.0`
  to force a full descent): expert 407 of layer 25, 2329600 bytes, `verified=1` at each hop. Climb in
  one round (7): SSD pack file to helper2 to helper1 to main. Descent: main to helper1 (round 290),
  helper1 to helper2 (round 333), helper2 to rung F, landing SSD-only (round 388).
- Passed, decode-path KL against the main card alone: C1a 0.00132, top-1 0.9817; C1b 0.00130,
  top-1 0.9858; the noise floor (a second main-card-alone run) 0.00135. The gate is KL at most 0.03
  and top-1 at least 0.95.
- Fence cost: the placement round fence averaged 18.7 ms (p99 52 ms) in S2, about 1.2% of the
  time between rounds. This is one run, not a measured budget.
- Speed: 4.2 to 7.8 tok/s decode under that SSD starvation. Load-contaminated; **not** a performance
  claim and not a comparison with static placement.
- **Failed:** two successive helper losses. A killed named helper stalled decode for 30 s and the engine
  then exited (HTTP 400, then 503). Helper loss is **not yet survived** for named channel helpers; a
  fix was in progress at the time; see the 2026-11-20 entry on helper loss (HL, HL2).
- Not exercised: file-reference (by-reference) moves, compaction, a third physical card, and a live
  `SEABRAID_TRACE` file. The run used lab-only extra log lines to see the displaced expert and RAM
  against SSD; the shipped trace now records both. These stay "not yet shown on GPUs".

## 2026-11-20 · Live run L2: one expert through every level

- Revision: built from commit f423da7 with Strata patches 0001 to 0018 and the v7 patch. One Linux
  test machine, the same layout as L1: main card R9700 (HIP), helper 1 RTX 3090 (CUDA), helper 2 a
  second HIP helper process on the R9700, rung F = engine RAM + NVMe, capped capacity. The per-step
  RAM and SSD lines come from a lab build with extra logging; the shipped build logs the card moves
  only. Evidence: the raw logs, the diff of the lab build and the runner, kept in the run archive on
  the test machine, and a private chain report (none of it is reproduced here).
- Comparison: one expert followed through every level (S1), file-reference moves between helpers
  (S2), and decode-path KL with file moves live, against the main card alone.
- Passed, S1 one-expert walk: cell layer 39 / expert 16 made all eight transitions in one run (428
  rounds, 7 requests):
  - SSD to RAM, round 28. The cell's page cache was dropped to 20,480 bytes just before the read, and
    the 1,971,200-byte read took 2,761 us.
  - RAM to helper 2, round 79 (staged from RAM).
  - Helper 2 to helper 1, then helper 1 to main, both in round 79, `verified=1`.
  - Main to helper 1, round 118.
  - Helper 1 to helper 2, round 176.
  - Helper 2 to RAM, round 274 (installed in the RAM copy).
  - RAM to SSD only, round 278 (evicted from the RAM copy).

  Seven of 14 steered cells made all eight transitions; a second cell (30/57) is also in the archive.
  S1 counts: 1,281 committed moves and 3 non-committed (1 refused with no write begun, 2 skipped), with
  0 stale epochs. Decode ran at 1.9 to 4.8 tok/s at a box load of 8 to 23: load-contaminated, not a
  performance figure.
- Passed, S2 file-reference moves on GPUs: 4 requests, 248 rounds, 744 moves, 0 non-committed. 124
  hops went by file reference (48 from F to helper 2, 76 from helper 2 to helper 1) and 190 by bytes.
  Helper file io: 256,307,200 direct bytes in 372 reads, `bad_sum=0`. By-reference writes were 2.6 to
  3.0 times slower than byte writes, because the helper reads the NVMe itself; the main card's stage
  time was 12 to 15% lower. Decode 1.9 to 2.3 tok/s, load-contaminated.
- Passed, decode-path KL with file moves live, 6 texts x 400 = 2,400 positions, top-32: ladder arm C2a
  KL 0.00116 (top-1 0.9904), C2b KL 0.00121 (top-1 0.9871), against a noise floor (main card alone,
  two runs) of 0.00116. The gate is KL at most 0.03 and top-1 at least 0.95.
- Round cost: S1 mean 28.8 ms (p99 54.8), S2 mean 35.2 ms (p99 56.6).
- Limits: `verified=1` covers the card-to-card hops. The two hops that touch the fallback log
  `verified=0` by design, because the fallback copy has no read-back. That the bytes came off the
  drive is inferred from the page-cache drop and the 2.76 ms read, not from a block-level trace. The
  RAM install and eviction were driven by a real switch of request type (code, then prose), because
  the RAM copy follows real routing, not the heat bias.
- Not run: compaction with the pyramid. The compaction flag acts only on the private helper tier
  (`adapters/strata/engine_flags.cpp:23`), which has a single helper.
- Unchanged from L1: helper loss had not yet been survived on GPUs; the re-run is recorded in the 2026-11-20 entry on helper loss (HL, HL2).

## 2026-11-20 · Helper loss on GPUs (HL, HL2)

- Revision: run HL used commit f04173e (Strata patches 0001 to 0019 and the v7 patch). Run HL2 used the
  s9fix commit 82c7c50 (patch 0020; merged as 7cc2620). Run L2 above used commit f423da7. Both ran on the same machine and layout as L1 and L2: main card R9700 (HIP),
  helper 1 RTX 3090 (CUDA), helper 2 a second HIP helper process on the R9700, named-channel helpers.
  Evidence: the raw logs and runner, kept in the run archive on the test machine (not reproduced here).
- Comparison: the helper-loss scenarios that failed in L1, re-run against the fixes in patches 0019 and 0020.
- Passed, S7 (a helper killed mid-request): HL, two runs, 7 of 7 requests OK each, one engine start;
  HL2, 7 of 7 OK. The dead helper's card closed 22 to 93 ms after the kill: the log reported the card
  CLOSED (helper exited) with its experts now coming from RAM or the drive, then reported placement degraded
  (helper is gone). The engine did not exit, and the request in flight finished. The engine's log prefix was
  renamed after these runs (L1, L2, HL, HL2 and the smoke run); the raw logs are kept in private history.
- Passed, S8 (a helper killed while idle): 4 of 4 later requests OK. The card is closed at the next layer
  request, not at the kill: 34 to 203 ms into that request, which in that run came 2.5 to 2.7 s after
  the kill. Nothing is logged during the idle gap.
- Passed, S9b (restart after the engine was SIGKILLed with both helpers dead), in HL and in HL2: ready in
  34 to 35 s, request OK, one WARN per card (`the helper on /l1a is gone ...; this card starts closed`).
- Failed in HL, passed in HL2, S9a (restart after a graceful engine stop with both helpers dead): in HL the
  engine exited with "the helper is not ready for a plan (state 5)". A graceful stop leaves the channel in
  its shutdown state, and the code checked that state before it checked the helper's identity. Patch 0020
  binds the identity before it reads the state. A dead helper now opens its card closed with one WARN,
  whatever the state; a live helper still in its shutdown state is waited for, at most 30 s. In HL2 S9a
  was ready in 32 s, 6 of 6 requests OK, two WARN lines, no exit.
- CPU: the `seabraid_helper_loss_test` cases L1 to L12 in Strata pass with patch 0020. L11 and L12 failed
  without it (rc=1, "not ready for a plan (state 5)"). Three mutants of the fix are each caught by the tests.
- Not shown on GPUs: a restart while a live helper is still in its shutdown state (CPU test L12 only).
- Limits: the machine's load was not clean (load1 4 to 14), so no timing here is a performance figure.
  No speed claim comes from these runs.
