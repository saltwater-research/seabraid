# Strata integration roadmap

Open work, in rough priority order. Evidence and measurements live in
[`DESIGN-LOG.md`](DESIGN-LOG.md) and [`research/2026-11-07-server.md`](research/2026-11-07-server.md).

## Integrated 2026-11-14, opt-in and off by default

Opt-in, so they cannot change a run that does not ask for them. Where each stands:
- **D4 prompt reading** (`compat/strata/tools/arms_d4.txt`): earlier builds read prompts 1.6-3x faster than lending with KL inside the gate. On the
  integrated build on v0.2.15 it engaged, but an 8.3k-token prompt read slower than lending (n=2 per mode), so it stays off
  ([RESEARCH-LOG](../../../docs/RESEARCH-LOG.md), 2026-11-17, verification on Strata v0.2.15).
- **Live promotion** (`compat/strata/tools/arms_ab5_promotion.txt`, design A/B 5): the finish line was at least +3% tok/s including fence time and
  zero STALE events. The matched-arm run of 2026-11-18 (S8) compared the Seabraid publisher, the legacy publisher and static
  placement, and stall-inclusive decode did not pass, so live promotion stays experimental and static placement is the supported
  baseline ([RESEARCH-LOG](../../../docs/RESEARCH-LOG.md#2026-11-18--first-matched-arm-run-of-the-live-placement-publisher-s8);
  open follow-ups in [the root roadmap](../../../docs/ROADMAP.md#verification-still-owed)).
- **Route histograms** (`compat/strata/tools/arms_routehist.txt`): measured earlier; logging off costs nothing.

## Correctness

- **DeepSeek-V4.1-Flash perplexity gap (retired).** Strata integration measured +3.0% / +3.6% perplexity over stock on two segments. The DeepSeek helper
  and `SEABRAID_STRATA_LINK_CHECK` were dropped on 2026-11-17 when Strata 0.2.14 removed that engine, so the gap stays unexplained and will not be
  chased here. The work and its status are in [DESIGN-LOG.md](DESIGN-LOG.md) (2026-11-07 and 2026-11-17 entries) and in the research note.
- **KL thresholds.** `compat/strata/tools/kl.py --baseline TSV --arm TSV` is now a gate: exit 0 PASS, 1 FAIL, 2 INVALID, one summary line.
  Defaults are mean KL <= 0.03 (the bound Strata's own MMQ and tail-skip defaults accept) and top-1 agreement >= 0.95;
  `compat/strata/tools/arms.sh` runs it for every `kl` arm when `KL_BASELINE` names the baseline TSV and marks a failing arm INVALID.
  Still open: the bounds come from one 441-token stock-vs-stock sample. Replace them with a paired verdict (mean delta
  log-prob with its standard error over interleaved arms; the gate already prints the paired dlogp and t but does not
  judge on them), and run every arm with the expert profile read-only so arms do not inherit each other's placement.
- **Qwen "worse or just different".** Run a stock arm with only the expert placement shifted, as the reference spread
  for the hot-share KL result.

## Performance

- **Concurrent helper wave (phase 1).** Today the helper's response sits in series with the main card's own work, which
  caps the second GPU at roughly a quarter busy. Overlapping the two is the next lever. Success criterion: helper time
  per request on greedy decode no higher than the unshared baseline, and the time the main card spends blocked on the
  helper below it.
- **Concurrency.** Several requests at once (`--parallel`) switches Strata's drafter from DFlash2 to MTP; aggregate
  throughput rose 27% on one setup. Measure it on the current two-card setup.
- **Helper kernels.** The helper's time is dominated by integer dp4a kernels; a different kernel shape or a CUDA 13
  rebuild may help. Measure before changing.

## Features

- **Measured card ranking.** `--bench` ranks main versus the other card by measured decode rows/s. The live multi-card
  run happened on 2026-11-09 (R9700 + RTX 3090): the installer printed "measured" and picked the R9700. The run also
  showed that expert rows/s measures the wrong job for choosing the first card: the 3090 was the faster first card for
  one stream (114.5 / 116.1 tok/s against 105.4 / 104.9 with the R9700 first). The ranking is therefore not yet a
  trustworthy way to pick the first card; a separate design doc covers what to measure instead. A first-ever install
  still ranks by heuristic until the binaries it needs are built.
- **Cross-layer re-tiering.** Live re-tiering moves experts within a layer only. Moving across layers needs a
  relocate in the expert cache and an EVICT on the wire. The per-layer share ceiling is a gate, not an invariant; a
  longer heat window or hysteresis would stop it flickering.
- **N helpers.** `SEABRAID_STRATA_SHM` takes up to three channels, one process each (patch 0017). Live runs L1 and L2 (2026-11-19 and 2026-11-20) used two helper
  processes, the second on the main card's own GPU, so a third physical card is untested
  ([RESEARCH-LOG](../../../docs/RESEARCH-LOG.md#2026-11-19--live-run-l1-the-pyramid-on-real-cards)).
- **Bandwidth table.** Memory-bandwidth rows in `bindings/python/seabraid/ranking.py` came from vendor spec sheets; re-check them when
  a card is added.
- **Upstream.** Track Strata releases (the patch series is pinned in `compat/strata/patches/strata/STRATA_BASE`), and propose the
  neutral channel tag and the draft-confidence sizing fix upstream.
