# Decisions

- Start with MoE. The expert path gives us a measured baseline against which
  to verify the extraction.
- Use Rust for host ownership, existing native kernels for execution, and a
  C ABI shared by engines and the Python wrapper.
- Keep one in-flight ticket per worker. Reuse its storage only after
  terminal completion. A wait timeout leaves the lease pending.
- Require the initial in-process provider to return. Process isolation and
  restart are separate acceptance criteria for the native worker transport.
- Preserve the Strata integration's committed baseline and record the original Strata pin,
  the newer rebase, and local patches separately.
- Batch v2 and provider v3 sit beside the v1 calls instead of replacing them.
  The v1 calls are marked deprecated during the alpha and are deleted at 1.0.
- Licensing is split by part: the core is Apache-2.0, the public C headers are
  Apache-2.0 OR MIT, and the patches against MIT upstream code stay MIT. NOTICE
  states each part's license.
- The runtime opens no sockets and talks to no service; `checks/check_no_egress.py`
  is the checker.
- The project is named Seabraid; the original Strata add-on is now the Strata integration under `compat/strata`.

## 2026-11-19 decisions (after the independent audit)

- **Dense FFN shards are not in the expert ABI.** Deferred until a real consumer
  exists. Routed experts produce selected, weighted outputs for the rows that
  picked them. A dense split needs a defined shard axis, an intermediate layout,
  and a reduction over every shard for every token, which is a different
  contract. Next step, before any public interface: prototype one stateless
  operation and check it against an unsplit oracle. Until then "shard braiding"
  of dense layers is a deferred research idea, not a feature.
- **Live placement stays opt-in; static placement is the supported default.**
  In the archived four-start comparison the private static arm averaged 93.66
  tok/s and the private live arm 75.66 tok/s (request means recomputed by the
  audit; see the [research log](RESEARCH-LOG.md#2026-11-18--first-matched-arm-run-of-the-live-placement-publisher-s8)).
  Live promotion becomes a candidate default only with separate results for
  stall-inclusive decode, per-text quality, convergence and helper-loss
  recovery, with host RAM headroom varied independently of the moves. The
  roadmap's n of 10 or more starts per arm is a proposed follow-up, not a result.
- **The payoff of a second GPU is established by a three-arm comparison.** Arm
  one computes the experts from RAM on the CPU. Arm two copies host weights to the
  GPU per use (Strata `--pcie-frac`). Arm three holds the experts on a second GPU.
  The arms match on model, quantization, usable memory, requests, context and
  quality gate; prefill and decode are measured separately; staging, merge-wait
  and local-compute times are recorded so a crossover can be explained. A
  harness for this is planned under `compat/strata/scripts/payoff/` by another
  track; it is not in this tree and no comparison has run.

## 2026-11-19 · The demand-driven placement policy is stated once and pinned by tests

- **Decision.** The planner's behaviour is written as one statement, the
  [Demand-driven placement policy](PLACEMENT-ABI.md#demand-driven-placement-policy), and every sentence of it has a
  behaviour test in `src/placement/tests/policy_demand.rs` (the cross-reference list in the doc names them, and one test
  fails if a test is missing from the list). The statement is read from the code, not from a description of it.
- **Evidence it happens on real cards.** [Live run L1](RESEARCH-LOG.md#2026-11-19--live-run-l1-the-pyramid-on-real-cards),
  natural traffic (S2): 1539 moves, every displacement one rung; in the traced journey (S6) one expert went down one rung
  in each of three later rounds (290, 333, 388). No new measurement was made for this decision.

