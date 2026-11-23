# Status: what works today

This one-page status report is updated after each verification. The label **live** means a run on real GPUs, with
numbers in the [research log](RESEARCH-LOG.md). The label **tested** means automated suites cover it on every
commit, CPU only. The label **not shown** means the feature is built and no run has verified it yet. Nothing on
this page is a performance promise. Last updated 2026-11-22.

All live results come from one Linux test machine equipped with an AMD Radeon AI PRO R9700 (ROCm/HIP) and an
NVIDIA RTX 3090 (CUDA). The machine ran Qwen3.8-Flash-Next IQ3_S on Strata v0.2.15 with this repository's
patch series.

## Private alpha.7 rehearsal

On the working snapshot, the installer completed the IQ3_S file-move setup through `--no-start`.
It built and verified the expert checksum manifest from the real pack. The run supplied a fixture
hardware plan and redirected systemd unit files to a test directory. Starting those services and
serving a request through that installation remain separate checks.

A separate mixed-GPU serving run used an existing manifest and a manual launch. On the long turn
the engine processed 32,313 prompt tokens, reported 20,895 cached tokens, and recorded 90.37 GB
of native physical reads. These measurements identify the tested snapshot. Final release
verification uses the commit and annotated tag object from the fresh private export.

## Works, shown live on GPUs

| What | Evidence |
|---|---|
| **Expert braiding** splits one MoE layer's experts between an AMD card and an NVIDIA card. Each card uses its vendor's kernels. The cards coordinate through shared memory once per layer. | In 2,000-token greedy decode with n=6 per arm, R9700 main + RTX 3090 helper reached 94.8 tok/s, and RTX 3090 main + R9700 helper reached 106.7 tok/s. Both orders passed the KL quality gate against the main card alone ([2026-11-17](RESEARCH-LOG.md#2026-11-17--verification-on-strata-v0215)). |
| **Engine recovery after an engine-started helper is killed** (private helper tier) | The in-flight request failed in under 0.75 s. The engine served the next request 1.7 to 2.7 s after the kill. No zombie process remained after any of the three kills per card order (same entry). |
| **Serving requests after losing a named helper** (named-channel helpers, patches 0019 and 0020) | Live runs HL and HL2, 2026-11-20 ([entry](RESEARCH-LOG.md#2026-11-20--helper-loss-on-gpus-hl-hl2)). A helper killed mid-request: 7 of 7 requests OK, the engine stayed up, and the dead helper's card closed 22 to 93 ms after the kill. A helper killed while idle: the card closes at the next layer request, not at the kill, 34 to 203 ms into that request; nothing is logged in the gap. Restart with both helpers dead passed after a SIGKILL of the engine and, with patch 0020, after a graceful stop. Live run L1 had failed this step (a 30 s stall, then the engine exited), and HL failed the graceful-stop restart before patch 0020. |
| **Long prompts** with helper experts kept in the engine's RAM copy | For a 32k-token prompt, time to the first token fell from 35.7 s to 18.0 s. The run found the needles, and decode was unchanged ([2026-11-17](RESEARCH-LOG.md#2026-11-17--long-prompts-on-the-private-helper-tier)). |
| **Live expert pyramid movement** between RAM/SSD, helper 2, helper 1 and the main card, in both directions. Every card write is read back and verified. | Live run L1 verified all 1,539 moves in 513 rounds, with no stale or rejected moves. The path for one expert (layer 25, expert 407) was RAM/SSD → helper 2 → helper 1 → main → helper 1 → helper 2 → RAM/SSD. The decode-path KL values were 0.00132 and 0.00130 against a noise floor of 0.00135 ([L1](RESEARCH-LOG.md#2026-11-19--live-run-l1-the-pyramid-on-real-cards)). |
| **One expert through every level**: SSD → RAM → helper 2 → helper 1 → main and back down to the SSD only, in one run | Live run L2: cell layer 39 / expert 16 made all eight transitions in 428 rounds and 7 requests; the SSD read took 2,761 us after a page-cache drop. Seven of 14 steered cells made all eight. `verified=1` covers the card-to-card hops; the two hops that touch the fallback log `verified=0` by design; the RAM and SSD step lines came from a test build with extra logging ([L2 entry](RESEARCH-LOG.md#2026-11-20--live-run-l2-one-expert-through-every-level)). |
| **File-reference moves** between helpers | L2: 744 moves in 248 rounds, none non-committed; 124 by file reference, 190 by bytes; helper file io 256,307,200 direct bytes, `bad_sum=0`. Decode-path KL with file moves live 0.00116 and 0.00121 against a floor of 0.00116 (same entry). |
| **Real SSD reads** when experts exceed card and RAM capacity | L1 capped card and RAM capacity so 66% of the experts resided on the NVMe drive. The memory limit included the page cache. Disk counters matched the engine's reads within 3% (same entry). |
| **Placement round cost** | The engine paused for a mean of 18.7 ms per round (p99 52 ms), about 1.2% of the time between rounds (same entry). |
| **llama.cpp/ggml adapter** on every backend listed here | Fixtures pass on CPU, CUDA (RTX 3090), HIP (R9700, ROCm 7.13 and 10.1) and Vulkan. With the llama.cpp hook, logits are bit-identical to stock ([2026-11-17/18 entry](RESEARCH-LOG.md)). |

## Works, proven by tests (every commit, CPU only)

Commit e9cc304 passed every verification stage. All 248 Rust tests passed in debug and release. Miri found
0 instances of undefined behaviour with either borrow model. C ABI CTest passed 46 of 46 in three build types.
The checks also covered ASan and UBSan, the Python binding against the native library, a wheel installed with no
index, the oldest supported CMake, and the crash reproducers. Random-input fuzzing ran about 200 million cases
with no crash across the placement ABI, the helper's move handling and the helper-list parser.

The tests also cover the following behavior beyond the core runtime.

- Tests check each of the 13 stated rules in the [demand-driven placement policy](PLACEMENT-ABI.md#demand-driven-placement-policy).
  The rules cover promotion of a hot expert, demotion of the coldest expert in the same layer by exactly one rung,
  cooldown, margin, the share cap, and promotion across several rungs in one round.
- In a file-reference move, the helper reads an expert directly from the model file and checks it against its own
  checksum list. The helper rejects a reference to another expert's bytes before reading anything.
- The tests cover the opt-in move and latency trace (`SEABRAID_TRACE`) and its report tool.
- The installer tests check `--ladder` and `--file-moves` against the engine's startup rules using one shared case list.
- For helper loss (patches 0019 and 0020), only the dead helper's card closes, and the in-flight request finishes on the CPU.
  If a helper is dead at restart, the engine starts with that card closed. The twelve cases of the helper-loss test
  (L1 to L12) pass with patch 0020; the live GPU results are in the table above.

## Built, not yet shown on GPUs

| What | Where it stands |
|---|---|
| Row compaction on the pyramid | The private helper tier run shipped 91.6% of rows, showed no measurable speed change, and passed the quality check. Row compaction has not run with the pyramid: the compaction flag acts only on the private helper tier (`adapters/strata/engine_flags.cpp:23`), which has a single helper. |
| A third physical card | In L1, helper 2 ran as a second process on the R9700. The code supports up to three helpers. |

## Known limits

- **Live placement is opt-in.** Static placement is the default. Live placement decoded more slowly in one earlier
  matched comparison (75.66 vs 93.66 tok/s). There is no speed claim for live placement.
- **Restart while a live helper is still in its shutdown state** passes the CPU test (L12), and the engine waits for that helper for at most 30 s. It has not been run on GPUs.
- **Speeds under heavy SSD use are not performance figures**. L1 decoded at 4.2 to 7.8 tok/s with most experts on
  the drive by design. L2 decoded at 1.9 to 4.8 tok/s (S1) and 1.9 to 2.3 tok/s (S2) at a box load of 8 to 23.
- Engine-started helpers use a fallback launch on C libraries older than glibc 2.39. If another thread of the host
  reaps arbitrary children, the fallback identity check could in principle accept a reused process ID.
  glibc 2.39 or newer has no such window.
- Linux x86-64 is the supported release platform. Windows core source builds are experimental. The source includes
  Windows DLL and import-library build rules. Native Windows inference remains unverified. The Strata installer
  and helper-process transport target Linux. macOS is unverified.
- The ABI can change before 1.0. The product name is final: Seabraid (decided 2026-11-21).

## After the release (updates)

Post-release work includes engine-supplied heat smoothing and engine-supplied move lists, separate RAM and SSD
rungs, a native vLLM or SGLang adapter, and a race-free launch on older C libraries. It also includes support
for reopening a helper card after its process restarts, Windows builds, and CI.

## Other work going on

An independent operator re-ran the pyramid on an earlier commit (e9cc304) and reached the same conclusion.
The re-run observed all eight move directions. Its single-expert walk was incomplete; L2 has since shown the full walk. A group of reviewers
reads each change. Review findings so far have been fixed and merged. They include a checksum trust gap in file
moves, launch and parity edge cases, and documentation mismatches.
