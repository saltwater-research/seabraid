# Strata integration — mixed-vendor MoE engine (AMD + NVIDIA in one model): build plan

Version: **v0.0.1** (scaffold: plan only, no engine code yet). Base: Strata v0.2.11-3 (9f7d977), MIT.

Goal (2026-11-06): one inference engine that runs the same large MoEs (Strata's Qwen3.8-Flash-Next,
Maya's GLM-5.3-Flash) across **all** AMD and NVIDIA cards in a host at once. Status: **prepared, not started.**

## What we start from (checked in code, 2026-11-06)

- **Strata** (MIT, a checkout at v0.2.11-3, detached HEAD) builds ONE backend per binary: CUDA (sm_80+)
  or HIP (wave32 RDNA3/4: gfx1100/1101/1200/1201; gfx906 via a separate compat build). `docs/AMD_HIP.md:25`:
  mixed AMD/NVIDIA "not supported". `CMakeLists.txt`: "Do not enable both GPU backends in one build."
- **The seam already exists.** `src/core/remote_experts.cpp` (501 lines) + `remote_expert_opt.cu` (202): helper
  cards (CUDA1-3) each own a disjoint set of experts; per layer the main card ships routed rows to each helper,
  helpers compute, results return **through private pinned host buffers** (no P2P). `docs/SECOND_GPU.md`.
  A helper in *another process* built for *another vendor* is the same data flow with the in-process
  call replaced by shared memory + a doorbell.
- **Maya** is vendored in Strata: `third_party/project-maya` (own LICENSE/NOTICE — read before touching).
- **Prior art to beat, available on the test system:** llama.cpp (built CUDA-only at the time; can be rebuilt with
  Vulkan, which drives AMD+NVIDIA in one process, or RPC). It lacks Strata's expert cache + DFlash2.
- **Measured baseline:** Flash-Next IQ2_XS: 3090+3060 113.4 tok/s e2e; R9700 alone 98.6,
  expert cache 99.8% hit → for THIS model a second card adds nothing. The value is only in models > one card.

## Design: cross-process expert helpers

```
 main engine (vendor A, e.g. HIP on R9700)          helper process (vendor B, e.g. CUDA on 3090/3060)
  dense layers, KV, drafter, its expert cache          owns a disjoint expert slice (its own cache)
  per layer: route -> split rows by owner  ──shm──▶    doorbell: rows + expert ids (pinned, both runtimes
                                                        register the SAME /dev/shm mapping)
  computes own experts + CPU pool          ◀──shm──    writes partial sums, rings back
  sums partials, continues layer
```

- One helper process per vendor runtime (CUDA or HIP), each holding one or more of its vendor's cards
  (in-process multi-card stays Strata's existing path).
- Transport: one `/dev/shm` region per helper, `cudaHostRegister`/`hipHostRegister` on both sides, futex or
  spin doorbell. No network, no P2P. Correctness: helper results are bit-exact to the same helper in-process
  (same kernels, same vendor); the cross-vendor part only moves bytes.
- Ownership/placement: reuse `--expert-cache-remote-placement stripe|layer` + the expert profile.

## Expert placement (design decision, 2026-11-06): hottest experts on the best card, the rest down the tiers

Already Strata's model (expert profile ranks experts; CUDA0 cache takes the top, the peer/helper tier takes the next
ranked pairs; adaptive swaps follow the workload) -- Strata integration reuses it and extends the tiers across vendors:
main card -> other cards by speed -> RAM -> NVMe.
- "Best card" = the MAIN card (dense layers, KV, drafter): a hot expert there costs zero cross-process hops.
- Proposed layout (2026-11-06): **3090 = main** (decode is memory-bandwidth bound: 936 vs 640 GB/s; measured 113.4 vs
  98.6 tok/s) holding the hottest experts it fits after dense+KV+drafter (~18 GB est., unmeasured); **R9700 = cold
  tier** (the long tail, 32 GB); 3060 after it; then RAM, NVMe. Risk: if most layers route >=1 expert to the R9700,
  the hop latency sets decode speed and the main-card choice matters less -- the zero-hop count checks this.
- **PLACEMENT RULE (2026-11-06):** measure every tier's real speed at startup (each GPU, RAM, M.2 NVMe),
  sort fastest first, fill each tier to capacity with the hottest remaining experts for the workload, down to RAM
  then NVMe; adaptive swaps keep it current. Vendor-agnostic, so it scales to any AMD/NVIDIA mix.
  Caveat: a layer waits for its slowest expert, so fastest-first can scatter one layer's experts across cards;
  if the zero-hop count shows that, v0.0.2 adds co-placement of experts that fire together.
- Tiers: 3090 -> R9700 -> 3060 -> RAM (pulled to a GPU on demand; PCIe ~14 GB/s, 20-60x slower than VRAM).
- Swapped in from the start by WORKLOAD: pick the expert profile per job type at load (Strata ships
  expert-profile.bin / -coder.bin), then adaptive swaps follow live use. New work: swaps ACROSS vendors (today
  Strata swaps one card <-> RAM). Check STRATA_ADAPT_OVERLAP_RESIDENT=1 (off by default in resident-RAM mode).
- Later option (not v0.0.1): prefill is compute-bound and may favor the R9700 -> split serving (R9700 reads the
  prompt, 3090 decodes), Strata docs/SPLIT_SERVE.md. Unmeasured.
- Metric that matters: fraction of (token, layer) pairs whose top-k experts ALL sit on the main card (zero-hop
  layers), not the expert hit rate. Computable offline from the expert profile before any code.

## Honest risks / what could kill it

1. **Sync cost.** 48 layers x every decode step x a cross-process round trip. Estimated ~1 ms/token (~13% at
   100 tok/s) — must be MEASURED in phase 1 before anything else is built.
2. **"All AMD and NVIDIA" is bounded by Strata's kernels**, not by our transport: NVIDIA sm_80+ only, AMD
   RDNA3/4 (+gfx906 compat). Older/other cards would need Vulkan — that is a different engine.
3. **Upstream churn.** Strata moves fast (v0.2.10 -> v0.2.11 in days). Keep the patch small and at the seam.
4. **RAM.** The test system has 30 GB of RAM; Maya-XS wants 64 GB RAM. Larger MoEs may be RAM-blocked regardless of VRAM.
5. **Only pays when the model overflows the biggest card.** Phase 0 measures how much.

## Phases (each has a gate; stop on a failed gate and report it)

| # | Work | Gate |
|---|---|---|
| 0a | Cache-cap proxy: R9700 alone with `--expert-cache` limited to ~60% of experts | tok/s drop >= 25% → extra VRAM is worth chasing |
| 0b | llama.cpp Vulkan build, same model, R9700+3060 (later +3090) | the number our engine must beat |
| 1 | Round-trip microbench: HIP proc <-> CUDA proc over shm doorbell, 48 hops | <= 15 us/hop, else redesign (batch layers). **PASSED 2026-11-06: p50 4.76 us, p99 6.30 us (tests/seabraid/channel_pingpong.cpp in the Strata fork, now compat/strata/tests/channel_pingpong.cpp; under build load) = 0.23 ms/token over 48 layers** |
| 2 | Helper process: CUDA build of remote_experts as a standalone server on shm | bit-exact vs in-process helper on CUDA |
| 3 | Main side: HIP engine treats the shm helper as a remote device | KL vs single-card run at noise floor; needle + a 59-task classification bench |
| 4 | R9700 + 3090 (+3060) on a model > 32 GB | beats R9700 alone AND llama.cpp Vulkan |

## Where it lives (reuse-extraction doctrine)

A patch series on a fork of Strata, kept at the remote-expert seam. No new
"framework" repo.

## Open decisions (as of 2026-11-06)

- Put the 3090 back in alongside the R9700 (phase 4 needs it; phases 0-3 can run on R9700 + 3060).
- Where the series is hosted. (Strata is MIT: keep its license and notice.)

## Status 2026-11-06 (a) — v0.0.2 built and measured (R9700 HIP main + RTX 3060 CUDA helper)

- Works end to end: seabraid-helper (CUDA) serves Strata's RemoteExperts tier over /dev/shm; main = HIP engine.
  Channel round trip p50 4.7 us. Crash recovery: a killed engine's restart re-plans the helper (verified live).
- Fits-on-card (cap 11,000): tie with stock (95.9 vs 96.0 tok/s e2e), accuracy within noise.
- Does-not-fit (cap 4,000): stock 0/10 answered (hung); Strata integration v0.0.2 8/10, decode ~70 tok/s (stock greedy 52-60).
- Required for the win: SEABRAID_STRATA_EXCLUDE_HELPER_RAM=1 STRATA_PREFILL_HELPER_SOURCE=1 STRATA_DISJOINT_ADAPT=1
  (helper experts leave RAM, prompt path reads them from the helper's VRAM).
- OPEN: (1) two 60 s no-progress hangs under memory pressure (prompt read / conversation parking) - same class as
  stock; test --conversation-cache-mib 0; (2) correctness gate: teacher-forced KL vs single-backend (greedy text
  diverges at near-ties; garbage was ruled out only by coherence + accuracy); (3) 3090 back in for the real
  3090-main pyramid; (4) GPU helper waits ~80 us/layer-launch (3060 launch+compute), not the channel.
- F4 (conversation cache off): still 2 hangs, 8/10 answered. Hang counts at cap 4,000: stock 9 vs Strata integration 2 -> a stock Strata prompt-path issue on a starved card, made rarer by Strata integration.

## Status 2026-11-06 (b) — v0.0.3: correctness gate, helper timing, hang diagnosis

- KL gate PASSED against Strata's own tolerance (`compat/strata/tools/kl_gate.sh`): the same 441-token text read
  teacher-forced through decode windows (`--short-read 1000`, the path where the helper computes), `STRATA_LOGPOS` +
  `STRATA_LOGPOS_TOPK=32`, tail-skip off in every arm, cap 4,000:
  | | stock vs stock (noise) | Strata integration vs stock |
  |---|---|---|
  | KL(top-32) mean / p99 / max | 0.0025 / 0.039 / 0.18 | 0.0098 / 0.148 / 0.40 |
  | top-1 agreement | 98.4% | 96.1% |
  | mean abs d logp(target) | 0.027 | 0.086 |
  Reading: a real, small numeric difference (the 3060's CUDA kernels reduce in another order) - about 4x the
  run-to-run floor, under the 0.03-0.035 KL Strata itself accepts for its default MMQ / tail-skip paths. One text, n=1.
- v3 channel: the helper reports its own service time per layer request. Prompt via windows (33 rows/launch):
  820 us in the helper, 48 us of it staging, 772 us waiting for the 3060's kernels -> the GPU compute, not the
  channel, is the helper's cost; the main engine waited only 126 us/launch (the rest overlapped).
- Hang diagnosis (STRATA_VERIFY_TRACE run G4, 3 hangs): every hang is after the batched prompt read finished
  ("N of N+5 tokens"), before the next decode window launched; the last decode window had SYNCED cleanly (the
  verify spin waits are NOT the cause). Matches Strata #884 (AMD: long prompt + adaptive swaps + SDMA). Testing
  STRATA_HIP_ADAPT_KERNEL_COPY=1 (run H4); `SEABRAID_STRATA_PTRACER=1` (a ptrace debug switch, removed by patch 0015) + compat/strata/tools/stallwatch.sh dump thread stacks at 40 s.
- Hang hunt result (2026-11-06): NOT fixed; time-boxed after 4 refuted fixes. Evidence (gdb stacks of the live
  hung engine via SEABRAID_STRATA_PTRACER=1 + compat/strata/tools/repro.py, which dumps at a stalled progress counter):
  the engine's main thread busy-waits in hsa_signal_wait_scacquire under a HIP copy - in ExpertCache::fill_slot_queued
  (refilling the slots the prompt path borrowed), in conversation_checkpoint_save (tail checkpoint, 5 tokens before
  the prompt's end), or in a verify window ("GPU waits were released but the GPU did not finish within 5 s").
  Different callers, one symptom: the R9700's queue stops completing. Kernel log clean (no VM fault, no reset).
  Refuted: verify spin waits (trace: last window SYNCED), STRATA_HIP_ADAPT_KERNEL_COPY=1 (#884), HSA_ENABLE_SDMA=0,
  --prompt-cache 0 (moved the stall into decode). Not deterministic per prompt; needs prior long requests.
  Stock Strata hits it MORE (C4: 9 vs Strata integration 2-3); none at cap 11,000 -> Strata + ROCm 7.13 + gfx1201 + a starved
  30 GB host. Mitigation that exists: the watchdog restarts the engine (~105 s lost per hang). Candidate upstream
  issue for Niko1221/Strata with the stacks (not yet reported).

## ROOT CAUSE FOUND 2026-11-06 (c) — the cap-4,000 hang is KFD userptr queue eviction (stock Strata + ROCm on Linux)

- Measured directly (no root): `/sys/class/kfd/kfd/proc/<pid>/stats_<gpuid>/evicted_ms` of the stalled engine jumped
  2,634 -> 41,465 ms (one ~38.8 s eviction credited at restore); the GPU sat at 3% / 17-22 W, then ran 47-100% the
  moment the queues were restored. Independently confirmed by read-only samples.
- Mechanism (Strata PR #3913, ROCm/rocm-systems #12670 -> #14285, TheRock #7832): on Linux, HIP "pinned" host memory
  (hipHostRegister, and hipHostMalloc under the default HSA_USERPTR_FOR_PAGED_MEM) is a KFD userptr tracked by MMU
  notifiers, NOT long-term pinned. RDNA has no XNACK, so any invalidation (reclaim, compaction, migration) evicts ALL
  the process's GPU queues until the range validates again; under memory pressure that lasts seconds to minutes ->
  host spins on a signal, GPU idle, dmesg clean, watchdog fires. Why cap 8,000 never hung: far less host-tier traffic.
- Host facts: ~30 GB of system RAM, the arena is 16.8-22.6 GiB, ~2.9 GiB left for the page cache, GTT ceiling (ttm.pages_limit) 15.13 GiB.
- Fixes (all need root or a reboot): (A) PR #3913 GTT route - HSA_USERPTR_FOR_PAGED_MEM=0
  HSA_SVM_FOR_PAGED_MEM=0 STRATA_HIP_ARENA_NONCOHERENT=1 GPU_PINNED_MIN_XFER_SIZE=1048576, needs ttm.pages_limit >=
  arena + ~6 GiB (boot parameter + reboot); (B) STRATA_HIP_MLOCK=1, needs the memlock limit raised (+ ideally
  compact_unevictable_allowed=0). Both make ~17+ GiB of RAM unmovable on a 30 GB host -> the RAM pressure moves to other processes. Real fix: less host tier (3090 back: R9700+3090+3060 = 68 GB VRAM >
  the 33 GB model) or more RAM.
- PR #3913's 5 commits cherry-pick cleanly: a local series (opt-in env switches; defaults unchanged).
- Refuted along the way (all still hung): #884 kernel copy, HSA_ENABLE_SDMA=0, DEBUG_HIP_DYNAMIC_QUEUES=0,
  GPU_MAX_HW_QUEUES=1, STRATA_HIP_QUEUE_PRIME=0, --prompt-cache 0, STRATA_ARENA_MMAP=1, --pcie-frac 0,
  HSA_USERPTR_FOR_PAGED_MEM=0 alone (does not change existing hipHostRegister registrations).
- D4 hazard (review, 2026-11-06): if the helper ever computes during prefill, size the /dev/shm channel by
  max TOKENS per request, never by pairs (cap = kWindowMaxT*10 is pairs). A 16K chunk at top-10 would reserve ~3.4 GB
  of shm = host RAM, the exact pressure that triggers the KFD userptr eviction hang. Also: `out` is per owned pair
  (remote_experts.cpp:539) so a helper-side per-token reduce is needed; remote_opt_'s reduce is refused for helpers
  by design (remote_experts.cpp:433). Compute-vs-lend crossover ~11.8K prompt tokens (f32) / ~24K (bf16).

## Roadmap 2026-11-06 (d): N cards of any vendor + live cross-card tiering
1. IQ3_S tiered run (GSQ-RCO IQ3_S, 50.3 GB experts > 44 GB VRAM): baseline for the full pyramid, GTT route on
   (PR #3913 env: HSA_USERPTR_FOR_PAGED_MEM=0 HSA_SVM_FOR_PAGED_MEM=0 STRATA_HIP_ARENA_NONCOHERENT=1
   GPU_PINNED_MIN_XFER_SIZE=1048576), RAM tier capped ~6 GiB so it fits today's 15.13 GiB GTT ceiling.
2. Live re-tiering for the existing helper: periodic re-rank of every expert by Strata's usage counters across ALL
   tiers; an expert crossing a tier boundary moves one level. New channel commands: LOAD(expert, slot) / EVICT(expert),
   bytes from the RAM tier via shm or from the GGUF; throttled (a few moves per window). Today the helper's plan is
   static and swaps only run R9700<->RAM (STRATA_DISJOINT_ADAPT).
3. N helpers: SEABRAID_STRATA_SHM=/seabraid0,/seabraid1,... one process per card, built for its vendor (CUDA / HIP / SYCL);
   startup benchmark per card (expert rows/s, free VRAM) orders the pyramid; lift Strata's 3-remote-tier limit.
4. Test N>2 with two helper processes sharing the 3060 until a third card exists.
Guardrails: shm sized by tokens not pairs; GTT on for any RAM tier; placement decided by measurement (clustered
whole-layer placement onto the slower card measured slower, 2026-11-06).

## Optimization ledger 2026-11-06 (d) — every speed/performance lever, with status and evidence
DONE (measured, now default unless noted)
- Whole model in VRAM across both cards (IQ2_XS: R9700 18,733 + 3060 5,843 experts, RAM 0): no RAM tier, no hangs.
- --draft-conf 0.35 via config user_args (Strata's server picked 0.8 by sizing from the 3060; upstream #5109): +4-7%.
- STRATA_ROUTE_TAIL_SKIP=0: same speed, exact routing.
- Helper zero-copy off (STRATA_REMOTE_ZEROCOPY=0 in helper): 77.7 -> 65.3 us/call, +1% decode.
- GTT route for any RAM tier (PR #3913 env): KFD evictions 0 ms with IQ3_S's 6 GiB RAM tier.
- Native packs without experts.bin (GGUF read in place): saves one model-size of disk; IQ3_S pack built in 3 s.
- IQ3_S at --max-context 32768: ~2 GB VRAM back for experts.
- D8 per-card startup benchmark (SEABRAID_STRATA_CARD_BENCH=1): RTX 3060 148,383 rows/s decode (6.7 us/row), 167,364 rows/s at 16 rows/expert.
MEASURED AND REJECTED
- CUDA-graph helper (nsys: kernel time, not launch); GPU_MAX_HW_QUEUES 1/2 (no effect); deeper drafting (0.25/0.15/spec 8 all slower);
  STRATA_LFUSE (needs all-resident graph: n/a); STRATA_VERIFY_QDEDUP / QFUSE / HC_PERSIST (neutral);
  kernel fusion in general (STRATA_DUMMY_NODES: ~2.2 us/node -> ~2% ceiling); STRATA_VERIFY_DEVICE_PLAN=1 and
  clustered whole-layer placement (slower: the 3060 loses its overlap partner); KV streaming to RAM (freeze risk + 45x slower reads).
IN PROGRESS at the time
- D7 live cross-card re-tiering: policy (per-layer helper-share ceiling 0.08 default) ready; transport (channel v4) pending. Needs --adapt-every / --adapt-swaps > 0 for heat counters.
- D2 3060 down-kernel batching (low value: helper overlaps). D4 helper compute during prompt reading (size shm by tokens). D5 tree drafting on the 3060: NO-GO - DFlash2 drafts from the verify window's hidden-state taps and the target's sampled anchor (dflash2.hpp:8,14; generate.cpp:20537), both only known after verify k, so a parallel branch must guess both: ceiling ~1-4%, costs ~1 GB 3060 VRAM + contention. D5b (start the drafter's context update before the accept length is known, ceiling ~1-2%) gated on STRATA_DF2_TIMING / STRATA_DF2_SKIP=ctx / STRATA_DF2_PROFILE runs; if the drafter's 2.17 ms is launch-bound, a cheaper drafter beats any overlap.
OPEN, NOT YET TRIED (ranked)
1. Concurrency: several requests at once (--batch/--slots / server parallel) -> aggregate tok/s; the R9700 idles ~26% of a window.
2. IQ3_S tier tuning now that RAM/NVMe tiers carry traffic: --adapt-min-gain, --adapt-async, --adapt-every, STRATA_COLD_OVERLAP, read-ahead; RAM budget 6 vs 10 GiB.
3. Raise ttm.pages_limit (boot param + reboot) for a bigger GTT RAM tier -> fewer NVMe reads on IQ3_S (costs locked RAM on a 30 GB host).
4. Workload-specific expert profiles (code vs chat vs agent) loaded per job type.
5. --kv k8v4 (~25% smaller KV -> more experts in VRAM; KL-check).
6. Dual-format mmvq <F1,F2> launch (native_mmvq_multi = 25% of R9700 GPU time; shared gate IQ3_S + up IQ4_XS can't pair today).
7. Unsloth UD-IQ4_XS (94 GB): qualifies for Strata's fused MMQ prompt path (+10-17% prefill), different engram -> full download.
8. Helper rebuilt with the CUDA 13 toolkit (driver supports 13.2; built with 12.4).
10. Upstream to Strata: #5109 draft-conf sizing on mixed-vendor hosts; E-6 "exact" doc wording (same math, different reduction order).
- D7 test requirements (2026-11-06): run with --adapt-every and --adapt-swaps > 0 or DriverState::usage is
  empty and re-tiering silently never runs; helper-served experts DO count in usage (expert_source.cpp:4911-4913).
  Acceptance metrics, pre-registered with a static-plan control on the same prompt sequence: misses per 1,000 lookups
  and NVMe bytes per request ("file tier I/O this request"), then tok/s. IQ3_S baseline: ~0.8% misses, ~2.4 MB NVMe per
  miss, 14.8 GB NVMe per 1,606-token reply.

## D7 call site (2026-11-06): `--retier`, patch 0004
- Policy -> transport had a mismatch the policy tests could not see: the wire has no EVICT and a SWAP is same-layer, but
  `plan_tier_round` emits bare EVICTs (full helper: free_slots = 0 for good) and cross-layer LOAD/EVICT pairs. Applied as is,
  a full helper would never move. Fix: `TierPolicyConfig::same_layer_swaps` (default false; the loop forces it) makes the
  policy return only free-slot LOADs and same-layer LOAD+EVICT pairs; `to_wire()` turns pairs into SWAPs and drops the rest.
  Open: a cross-layer move needs a relocate() in ExpertCache and an EVICT on the wire; until then re-tiering is within-layer.
- Scheduling is in strata-ext `retier_loop.*`; Strata gets hooks only (one `tick()` before `adapt_tick()` in the serial
  `--serve` loop). Interval is in windows (heat ages per window). Planner thread + two single-slot mailboxes; ops are applied
  only inside `tick()` on the engine thread. Not hosted: pipelined loop, --batch, generate mode (refused at startup).
- Ceiling semantics, stated truthfully (2026-11-06): `--retier-ceiling` (max_helper_layer_share) is a GATE on admitting LOADs and
  SWAPs, not a restored invariant. restore_ceiling (unpaired EVICTs) is skipped in same-layer mode, so a layer over the ceiling
  admits no swap (a hot-for-cold swap always raises held heat, `after <= cap` never holds) and is never actively lowered (that
  needs a wire EVICT, the cross-layer follow-up above). CORRECTED by the merge/d7 live smoke (ceiling 0.08, 1 swap/round): the
  over-ceiling count swung 4 -> 45 -> 12 -> 45 -> 21 layers and max share 0.117-0.246 across 44 rounds, far more than one swap
  per round can move. The share is driven by the SMOOTHED HEAT, which follows the text being generated, so layers drift over and
  back under on their own; the earlier "0.30 = initial placement, frozen" reading was wrong (it was the same heat swing).
  `TierPolicyState::last_layers_over_ceiling` (computed in the post-op coverage loop only) is printed on every round line.
  OPEN (policy quality, not correctness): with heat this volatile, admission at 0.08 flickers per layer per round; candidates
  are a longer heat window for the gate or hysteresis (admit below c_lo, block above c_hi). Needs an A/B before changing.
- `claimed` is passed as nullptr: generate.cpp's `claimed` is local to the helper-open block and nothing reads it afterwards
  (the disjoint-adapt path asks `holds()` live).

## 2026-11-07 — D7 tests 3-5 + KL gate; residency bug found and fixed

- **Bug:** the helper wrote `held[]` only at plan load. After an applied SWAP, main copied the stale bitmap, routed the
  displaced expert to the helper, and `RemoteExperts::begin` skipped it (slot < 0, no error), so its rows were dropped.
  Symptom: the next round re-proposed displacing it, "SWAP but the displaced expert is not resident here" (21/47 and 27/47 rounds).
- **Fix (patch 0005):** `publish_tier_residency` (the helper writes cache truth after `tier_sync`, before `tier_ack`) plus
  `tier_residency_consistent` (main refuses a round whose `held[]` contradicts its applied ops: "helper residency is STALE").
  `compat/strata/tests/tier_residency_test.cpp` fails 4 checks with the publish removed.
- **Arms (IQ3_S tiered):** off 94.9/101.3 tok/s; on (1 swap/32 windows) 99.3/100.3, 42 applied, 0 failed, fence 4.3 ms avg
  (11 max); on20 91.9/98.4, 43/0. NVMe per-reply bytes are page-cache noise, not comparable across arms.
- **KL gate:** cold off x3 bit-identical (KL 0). With a warm-up, off-vs-off noise is 0.021-0.078 (placement diverges).
  Fixed arms 0.026-0.056, inside the noise. Positive control (the bug put back): KL 0.179, top-1 85.7%, STALE fired 109x,
  so the gate sees this class. PASS, n=1 per arm.
- **Resolved (same day):** `max layer share` hit 0.30 with `--retier-ceiling 0.20` because the ceiling only gates
  admissions; layers the initial plan put over it are frozen (see the ceiling-honesty entry above).

## 2026-11-07 — Rebase onto Strata v0.2.12 (146df3c4); HIP configure gains setup.py's prefill flags

- **Series:** 0001-0006 (hot-share 9de9dfb) applied with `git am -3` onto v0.2.12, no conflicts; regenerated with
  `git format-patch`. Content is unchanged apart from hunk offsets. Applied tree 9a60103f, matched independently twice on the
  the test system (`git am -3`, and by cherry-picks). Upstream also changed `expert_source.cpp`,
  `generate.cpp` and `CMakeLists.txt` in the patched files; its new `STRATA_ADAPT_FETCH` path is disabled whenever
  `remote_count > 0` (`ExpertDispatch::fetch_can`), so it cannot act on Strata integration's remote picks.
- **HIP link break (upstream, not the rebase):** v0.2.12 calls `fused::want_dm_f16` / `dm_f16_used` from `prefill.cpp`
  without stubs for `STRATA_PREFILL_MMQ=OFF`, the CMake default, so a HIP `strata` configured without it fails to link
  (reproduced on unmodified 146df3c4). `compat/strata/install/build.py` now passes setup.py's HIP release flags
  `PREFILL_MMQ`, `Q6K_EXPERTS`, `MMQ_KQUANTS`; `test_amd_main_plus_nvidia_helper_commands` fails with `PREFILL_MMQ` removed.
- **Verified on a CUDA development host (CUDA 13.1, sm_86;89, llama.cpp from a local archive of the 3cf03257 pin):** `strata` and
  `seabraid-helper` build; ctest seabraid_* 5/5 pass (incl. `seabraid_hot_share_test`); `tests/core/fill_bytes_test.cpp`
  (not registered in CMake by patch 0002) built by hand: all checks passed. Installer suite 183 passed, 6 skipped.
- **Not verified:** the HIP build of this series, and any GPU run. The KL gate vs single-vendor Strata plus tok/s is the
  finish line and needs the two-GPU host. Watch: v0.2.12 Auto picks MTP over DFlash2 on gfx1201 (11/50
  DFlash2 prompt stalls on 0.2.11) while Strata integration configs pass `--draft dflash2` explicitly.

## 2026-11-07 — dsv41 (DeepSeek-V4.1-Flash) support: patches 0007-0008, seabraid-helper-dsv41, KL gate

- **What:** the R9700 keeps ~2,907 hottest experts; the helper GPU takes the next ~2,617 (fit and heat from the
  pack's own expert-profile.bin: CPU-lane picks 53.9% -> 34.2%). The hook is in dsv41's `serve_lane` (the host thread
  the device already waits on once per layer), so no main-engine kernel changes; helper-held experts keep vram_tab 0.
- **Wire:** channel v6 (SEABRD1): `expert_spec.hpp` LayerSpec[] + path table (main describes, the helper only executes:
  it never parses the pack), per-token weighted output, router weights region, helper pid. v5 is reserved for the
  neutral STRXHLP tag on the v4 layout (proposed upstream).
- **Failure contract:** mismatch at open fails fast (engine, geometry, ABI); any error after open fails open loudly
  (the CPU lane computes the helper's picks: same math, the experts are still in the GGUF mapping); a helper that dies
  while loading is caught by its pid in seconds.
- **Verified:** CUDA build (RTX 4080, sm_86;89) and HIP build of strata-dsv41 (the two-GPU host, gfx1201, 0 undefined).
  On the 4080: executor parity vs the ds4 CPU reference for IQ1_M / IQ2_XXS / IQ2_XS / MXFP4 / mixed, clamp 10.0,
  nrmse ~1.1e-7; e2e smoke (real link <-> real helper over shm, T=1/4/16, fail-open after SIGKILL) PASS, and the
  no-clamp sabotage FAILS 7/7. The first smoke found a helper startup crash (a printf missing an argument read
  H=5120 as a char*), now a build error under -Werror=format.
- **KL gate:** `compat/strata/tools/kl_gate_dsv41.sh` + `compat/strata/tools/kl_dsv41.py`. `--dump-logits` / `--score` default to 16-token
  windows, which take the host-routed sync path (ds4_model.cu plan_window: async only for T <= 8), so the helper would
  never run and the gate would pass trivially; the gate forces `--window 1` and fails unless the Strata integration arm reports
  helper requests > 0, and fails a bit-identical result.
- **Not verified:** any run on the real model (needs the two-GPU host with any server that holds the main card stopped).
## RTX 3090 replaces the 3060 (2026-11-07): IQ3_S is all-VRAM; the tiered RAM budget is now inert
- Test system: R9700 main (HIP) + RTX 3090 helper (CUDA, PCIe 3.0 x8 root port, 280 W cap), build 1cd8443e.
  The installer planner (`seabraid_setup.py --plan-only --pack iq3_s`) on this system: ALL-VRAM, R9700 24.0 + 3090 22.8 GiB of
  experts, RAM 0, NVMe 0.
- A/B (one variable, cold start per arm, 3 x 2,000-token greedy, run 2 quoted): A0 = `compat/strata/configs/example-iq3s-tiered.json`
  (with `--resident-budget-gib 10`) on the current build, A1 = the same minus the budget
  (`compat/strata/configs/example-iq3s-all-vram.json`). A0 105.5/106.0/105.6, A1 105.9/105.9/105.6 tok/s by wall time; engine lines
  106.5-107.1 in both. Window 26.2 ms in both (GPU-reach wait 18.4-18.8 ms), 2.80 tokens/window, expert cache 100.0 % hit,
  13.5 % of routed experts served across PCIe by the helper, helper 74.6-74.9 us/call (16.5-17.1 staging + ~58 waiting;
  3060: 78-84 us).
- WHY no difference: in A0 the RAM budget placed 0 bytes ("RAM budget 10.00 GiB: 0 of the 0.00 GiB of experts the GPU
  cache does not hold"), because the helper's `--expert-cache-device1 auto` already takes everything the main cannot hold.
  The budget is dead weight on a 24 GB helper, not a cost. A predicted +3-9 % for removing it (an inference from the 3060
  numbers) is RETRACTED. The speed-up over the 3060's tiered IQ3_S (96-99, run 2 99-101) came from the card swap itself.
- KL gate (teacher-forced, 441 positions, `--short-read 1000`): A1 vs A1 0.00000, A0 vs A1 0.00000 (top-1 100 %), i.e.
  identical placement and identical logits, as the log line predicts.
- Found while reading for this (not yet acted on): the CUDA helper never applies an arch default table (`helper.cpp`
  sets only STRATA_REMOTE_ZEROCOPY); STRATA_EXPDEC_DOWN is documented as "default 1 on gfx12" (`iq_kernels.hpp:125`) but
  defaults to 0 in code (`iq_kernels.cu:4091`). Next measurable levers on this system: concurrency (`--parallel 4`, +27 %
  aggregate on the RTX 3060 system) and the prompt-time blob lending, which roughly doubled in bytes with the bigger helper.


## 2026-11-09 — v0.0.3 on the two-GPU host; both role orders; prompt reading; rebase onto Strata v0.2.13.1
- **Setup:** Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S (both shards on the NVMe), default engine args (DFlash2 spec 6 conf 0.35, kv int8, 131072 ctx). Runner: `compat/strata/tools/arms.sh` (one cold start per arm; quote runs 2/3).
  Every figure below is copied from the run logs.
- **Decode, one stream:** R9700 main + 3090 helper 105.4/104.9 tok/s, 26.7 ms/window; 3090 main + R9700 helper 114.5/116.1, 24.0 ms/window
  (same 2.83 tokens/window). Four streams (`--parallel 4`): 119-124 vs 114 tok/s aggregate. So the better MAIN depends on load.
- **Card ranking is measuring the helper's job.** `--bench` (expert rows/s) ranks R9700 270k over 3090 226k and makes the R9700 main, yet the
  3090 is the faster main for one stream (its draft step 1.8 vs 2.3 ms, GPU-reach 14.1 vs 18.9 ms). Installer gaps found: it never builds
  `seabraid-helper` for the MAIN vendor, so `--bench` cannot run from an installer tree (the HIP helper links in ~10 s); a CUDA main needs the
  C/C++ compilers pinned to the CUDA host compiler (gcc-15 host + CUDA 12.4 fails to link); `--no-start` prints "units are in ..." but
  writes none; `rank_note` prints "not measured: not measured:".
- **Placement:** R9700 main: 12,784 experts on the R9700 + 11,792 on the 3090, RAM 0. 3090 main: 8,137 + 16,379, and 60 experts
  (0.12 GiB) page-locked in RAM. The 26.8 GiB PLE table streams from NVMe in both.
- **Hot-share / phase 1 (R9700 main):** hot-share alone 26.9 ms/window (no gain); `--remote-expert-opt` alone 26.3 ms but 2.58 tokens/window;
  both 24.4 ms (-9%), 2.63 tokens/window. KL vs baseline (441 positions, deterministic floor 0): hot-share 0.011, dlogp -0.038 (t -2.1,
  better); phase 1 0.025, dlogp +0.021 (t +0.9, n.s.).
- **Prompt reading is copy-bound.** 75-85% of the prompt's GPU timeline is `wait copy`: the main card pulls every helper-held expert
  (13 GiB for a 3.5k prompt, 46 GiB for 14k) across the 3090's PCIe 3.0 x8. Two-card prompt speed 135 / 360 / 740 tok/s at 0.9k / 3.5k /
  14k tokens vs stock Strata on the R9700 alone (experts in page-locked RAM) 314 / 619 / 1,224, whose decode is only 12.6 tok/s.
  `STRATA_PREFILL_HELPER_SOURCE=0` (read from NVMe instead): within +-6% with the R9700 main, 25-30% slower with the 3090 main; keep 1.
  An earlier quiet-run 1,026 tok/s for it did not reproduce in two quiet repeats (RETRACTED). The hipBLASLt tuning table is not
  applicable (ROCm 7.13 ships hipBLASLt 1.3.0; tables exist for 1.2.2 and 1.5.0). The fix that would remove the copy is the helper
  computing its own experts during prompts (D4 above, not built).
- **Rebase:** series rebased onto Strata v0.2.13.1 (628f74955): CMake hunks moved into `cmake/strata_gpu_engine.cmake` (0008, 0010) and
  the 0001 conflict kept both blocks; plain `git am` of the new files reproduces tree b085a6a8. v0.2.13.1 vs v0.0.3 on the same args:
  decode identical, logits bit-identical (KL 0).

## 2026-11-09 (b) — hot share sweep, the option B regression explained, KL gate
Words from here on: card 1 = the card running the engine, card 2 = the other card. All arms: v0.0.3 tree, RTX 3090 = card 1, R9700 = card 2, IQ3_S, one cold start.
- **Hot share sweep (run7).** `SEABRAID_STRATA_HOT_SHARE` unset / 0.25 / 0.40 / 0.544 (card 2's share of expert rows/s) / auto /
  unset: 23.92 / 24.55 / 24.95 / 25.40 / 26.28 / 24.16 ms/window. Every hot expert handed to card 2 cost time; GPU-reach
  wait stayed ~14 ms, card 1's per-layer host CPU rose 6.3 -> 9.2 ms. Strict heat cascade (hottest on card 1) is the
  default; balancing among cards stays a hypothesis (compat/strata/docs/design/N-CARD-LADDER.md, A/B 1b and 9).
- **Option B regression explained (runs 8 and 9, design A/B 8).** 2x2 of `SEABRAID_STRATA_EXCLUDE_HELPER_RAM` x
  `STRATA_DISJOINT_ADAPT`: both on 24.1 / 24.3 ms/window (115 tok/s), exclusion off 24.4, disjoint off 49.0, both off
  66.3. Both off reproduces option B's 66 ms exactly. Disjoint adapt is the switch that matters for decode (CPU jobs
  22.5 -> 0.1 ms): without it card 1's adaptive tier promotes experts card 2 already holds. With it on, RAM exclusion
  leaves decode unchanged but lifts prompt reading 238 -> 375 tok/s at 3.5k. Run8 on option B itself: exclusion alone
  66 -> 47 ms/window, prompt 430 -> 200 tok/s (option B cannot lend, so excluded experts come from NVMe).
- **KL gate.** `compat/strata/tools/kl.py --baseline --arm` exits PASS / FAIL / INVALID (mean KL <= 0.03, top-1 >= 0.95);
  `arms.sh` marks a kl arm INVALID when `KL_BASELINE` is set and the gate fails. Re-checked on run1's files: hot share
  PASS (0.0114, 95.7%), hot share + phase 1 PASS (0.0129, 96.8%), phase 1 alone FAIL on top-1 (0.0247, 94.8%), run8
  exclusion PASS (KL 0). Sign: the gate reports dlogp as arm minus baseline (positive = arm better), the opposite of the
  hot-share line above.
- **Installer:** five gaps fixed (expert program built for every vendor so `--bench`
  works; CUDA C/C++ pinned to the CUDA host compiler; `--no-start` writes units; single "not measured:"; window test).
  Not yet run on hardware.

## 2026-11-14 - integration of the ladder and the D-series engine changes (BUILT: NO, RUN: NO)
Status first: the GPU host was unreachable for the whole integration session, so NOTHING below has been compiled for CUDA or HIP
and no C++ test has been run on the integrated tree. What was done is git-level: the engine changes were cherry-picked onto the series,
`git am` of the whole series (0001-0013) onto the pinned base was verified to reproduce the working tree, and the Python suite and
the repository checkers pass. Every claim about speed or quality below is from an EARLIER build, cited to run logs.

| Candidate | Decision | Evidence (one line) |
|---|---|---|
| Route histograms (patch 0011) | INCLUDE, opt-in | run10: logging off = default speed (23.93 ms/window), unit test runs the same binary env set / unset |
| N-card ladder, engine side (patch 0012) | INCLUDE, opt-in (`--retier-promote`) | never timed (design A/B 5 pending); host tests only; changes the existing `--retier` path only to fail closed on a failed round |
| N-card ladder, Strata integration side | INCLUDE | `tier_ladder_test` and the retier loop; an earlier build of the tree compiled for CUDA and HIP; arms.sh keeps main's version |
| D4 prompt compute, `d4-precision` build (patch 0013) | INCLUDE, opt-in (`STRATA_PREFILL_REMOTE_COMPUTE`, helper `--prefill-tokens N`) | runs 11, 18: prompt reading 1.6-3x the lending path; KL vs lending 0.0277 (R9700 first) / 0.0213 (3090 first), gate 0.03 / top-1 0.95 PASS; vs stock single-card FAIL for R9700 first (0.0533) |
| D4 `d4-prefill-compute` (v1 of the same engine change) | EXCLUDE | superseded by `d4-precision` (it fixes the fused-path KL regression the first build had: 0.057 -> 0.028) |
| D4 branches (prompt_compute, `chunk_compute`, `SEABRAID_STRATA_CHUNK_KERNEL`) | EXCLUDE | older design on the pre-rebase lineage, "UNBUILT for HIP, untimed" |
| D2 `STRATA_DOWN_RPL` (down-kernel rows per lane) | EXCLUDE | refactors the DEFAULT decode kernel (RR=1 path), bit-equality was shown on an RTX 4080 SUPER only, never timed, its own note says low value (the helper overlaps); needs a GPU run to show the default path did not slow. |
| D5 tree draft | EXCLUDE | verdict NO-GO in its own note (ceiling 1-4%, costs ~1 GB VRAM on card 2) |
| D7 tier policy / wiring / channel v4 | ALREADY IN | patches 0001, 0004, 0005, 0007 and `compat/strata/strata-ext/src/tier_plan.cpp`, `retier_loop.cpp` |
| D8 card bench | ALREADY IN | `compat/strata/strata-ext/src/card_bench.cpp`, measured 2026-11-09 |

What had to be written, not copied (so it is NEW, untested on a GPU): the D4 engine change was measured with a helper that offered the
prefill region and served it, but that helper code was never committed. `compat/strata/strata-ext/include/strata/seabraid/pf_service.hpp`, the
`--prefill-tokens N` option and the wait-loop hook in `compat/strata/strata-ext/src/helper.cpp` are written here (`compat/strata/tests/pf_service_test.cpp` covers the
answer contract on a real channel; the GPU `prefill_serve` it calls is the engine's own). The channel is now v7 (`kVersion` 7); the header
size 2688 in `compat/strata/tests/channel_v6_golden.txt` and `compat/strata/install/_plan_shm.py` is PREDICTED from the field layout (v6 2240 + the appended fields,
rounded to 64) and MUST be confirmed by `seabraid_layout_print` on the first build: `compat/strata/tests/channel_v6_test.cpp` fails if it is wrong.

Not done, and why: no CUDA / HIP build, no C++ unit test run, no GPU or model run, no sabotage run of the new C++ tests (they need the
build). `compat/strata/tests/SABOTAGE.md` holds the earlier ladder records only.

## 2026-11-17 - rebase onto Strata v0.2.15 (c77572806); the DeepSeek-V4.1 helper is dropped
- **Pin:** `STRATA_BASE` is now the full SHA `c77572806996349b49995e4f98d41b5294d36a21` (v0.2.15; was v0.2.13.1 `628f74955`). Patches 0001-0007 and 0010-0016 were rebased with
  `git am -3` and regenerated with `git format-patch` (author Strata integration, fixed date, subjects `[PATCH NN/16]` where NN is the file number). Per-commit +/- line
  multisets equal the originals (checked with a per-file multiset diff) except 2 lines of 0013 (hand-resolved, from the first rebase pass) and patch 0016, which lost its `src/dsv41/ds4_helper_link.cpp` hunk (the only difference the check reports for 0016).
  A fresh `git am` of the repository's files onto `c77572806` reproduces tree `c839daf01` (the same tree as the scratch branch they were made on).
- **Dropped:** 0008 and 0009 (the dsv41 helper link and `SEABRAID_STRATA_LINK_CHECK`). Strata 0.2.14 deleted its DeepSeek-V4.1 engine (`src/dsv41`, `ds4_*`), so there is
  nothing to patch and the CUDA configure failed on the add-on's own dsv41 targets. Removed with them: `seabraid-helper-dsv41`, `dsv41_exec.*`, `dsv41_plan_check.hpp`,
  the parity/e2e tests and their synthetic-data header, `tools/kl_dsv41.py`, `kl_gate_dsv41.sh`, `_kl_paired.py` and their test, the installer's `blobs=False` channel
  geometry, and the dsv41 rows of the channel layout golden file. The DeepSeek measurements stay as history (measured on v0.2.13.1).
- **Flags:** the installer now passes `-DSTRATA_MMQ=ON` on CUDA and HIP. Strata 0.2.14 made `STRATA_MMQ` default OFF (its own prompt kernels only); patch 0013 (D4)
  hooks llama.cpp's MMQ prompt path, so ON keeps the build the patches were validated against. Strata's own release HIP build passes `STRATA_PREFILL_MMQ=ON STRATA_MMQ=OFF`.
- **Upstream findings (pristine v0.2.15, not ours):** `conversation_copy_streams_test` on ROCm 7.13 / gfx1201 (151552 KiB vs 16 MiB) and `cuda_spill_guard` with nvcc 12.4
  (44 kernels not allow-listed); see STRATA-INTEGRATION section 4.6.

## 2026-11-20 - the upstream PR candidates: evidence kept from STRATA-INTEGRATION section 6
STRATA-INTEGRATION section 6 now lists the candidates in one line each. The measured evidence its first version carried is kept here, as
recorded then, with the hardware and build named. These are earlier results (Strata v0.2.12 and before), not re-run on v0.2.15.
- **Channel round trip (item 3, v4 header under the Strata integration names):** R9700 + RTX 3060 and R9700 + RTX 3090; microbench between a HIP and a CUDA
  process under build load: p50 4.76 us, p99 6.30 us; 48 hops = 0.23 ms/token. The microbench's sample count was not recorded.
- **Helper mode (item 4), build 9f7d977:** R9700 + RTX 3060, expert cap 4,000 (the model does not fit the card): stock answered 0 of 10
  classification prompts, Strata integration 8 of 10, decode about 70 tok/s against stock greedy 52-60. At cap 11,000 (fits): 95.9 vs 96.0 tok/s e2e, a tie.
  R9700 + RTX 3090, IQ3_S all-VRAM, 3 x 2,000-token greedy, run 2 quoted: 105.6-105.9 tok/s by wall time (A0 and A1 arms 105.5/106.0/105.6 and
  105.9/105.9/105.6); 13.5% of routed experts served by the helper; helper 74.6-74.9 us per call (16.5-17.1 staging, about 58 waiting).
  KL gate on the 3060 pairing, 441 positions, one text (n=1): Strata integration vs stock mean KL 0.0098, p99 0.148, max 0.40, top-1 96.1%, against stock
  vs stock 0.0025 / 0.039 / 0.18 and 98.4%. The 3090 pairing gave identical placement and identical logits across two arms (KL 0.00000), a null
  result between two Strata integration configurations, not against single-vendor.
- **Tier transition (item 6):** R9700 + RTX 3060, IQ3_S tiered, scheduler and policy from seabraid: off 94.9 / 101.3 tok/s; on (1 swap per 32
  windows) 99.3 / 100.3, 42 applied, 0 failed, fence 4.3 ms average (11 ms max). KL gate: cold off x3 bit-identical; with warm-up off-vs-off
  noise 0.021-0.078; fixed arms 0.026-0.056; positive control (stale residency put back) 0.179 with top-1 85.7% and STALE fired 109 times;
  n=1 per arm. Only within-layer swaps are implemented (no cross-layer relocate, no wire EVICT).
- **Spec region (patch 0007, item 8):** golden header size 2240 bytes at v6 (the golden file is now `adapters/strata/wire_v10_golden.txt`, 2752 bytes at v10).
- **DeepSeek adapter (item 9, dropped 2026-11-17):** RTX 4080 (sm_86;89), synthetic experts: parity vs the CPU reference for IQ1_M, IQ2_XXS, IQ2_XS,
  MXFP4 and a mixed layout, clamp 10.0, nrmse about 1.1e-7; smoke PASS; the no-clamp sabotage fails 7 of 7 checks; HIP build of the engine
  (gfx1201) had 0 undefined symbols. The expected benefit (CPU-lane picks 53.9% -> 34.2%) was derived from the pack's own expert profile and not measured.
- **Placement seam (item 10):** with IQ3_S all-VRAM the RTX 3090 held 45% of the experts (11,110 of 24,576) but did 13.6% of the expert work.
  Hot share did not pay on hardware (2026-11-09, runs 1 and 7): no gain with the R9700 first, and with the 3090 first every share tried (0.25 to auto)
  was slower than leaving the hottest experts on the first card.
- **`STRATA_DF2_WINDOW` (item 11, patch 0003):** R9700, window 512 = 1.957 vs 2.110 ms per round at 59.4% vs 59.9% acceptance (n not stated).
- **Fill clamp (item 2):** with the old clamp restored, `fill_bytes_test` fails 9 checks (review record).
