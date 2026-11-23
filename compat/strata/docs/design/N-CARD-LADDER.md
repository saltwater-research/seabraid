# The N-card expert ladder: ranking, placement, live moves, prefill

Status: design, written 2026-11-09 against Strata (`628f74955` + `compat/strata/patches/strata/0001-0010`) and the
measurements in `compat/strata/docs/DESIGN-LOG.md`. Sections 1 to 8 keep that date: file:line citations in them point at the
patches as they were then. Since then the ownership map (patch 0012), per-request routing histograms (0011), D4 (0013) and
the channel list (0017) were built; section 9 says what is built and what is not. Every statement about behaviour cites a
file:line in this repository or a dated DESIGN-LOG entry; a statement with no citation is a proposal. Reviewed
independently; section 10 records the review notes.

Words. **Card 1** is the most capable card: it runs the engine (attention, dense layers, KV, router, drafter) and
holds its own share of experts. **Cards 2..N** hold disjoint shares of experts; a card of another make runs its
own process and is served per layer over a shared-memory channel, a card of card 1's make may stay in process
(section 6, one provisional rule). Below the cards: **RAM**, then **NVMe**. Code and env names that
say "helper" (`seabraid-helper`, `SEABRAID_STRATA_EXCLUDE_HELPER_RAM`, `STRATA_PREFILL_HELPER_SOURCE`, `remote_dev >=
1000`) are live contracts and are quoted as code; renaming them needs a maintainer decision and keeps the old names as
aliases.

## 1. What exists (the ladder today, two cards)

| Piece | Where | What it does today |
|---|---|---|
| Card order | `bindings/python/seabraid/ranking.py` `rank` | bench rows/s when every card has one, else VRAM > bandwidth table > arch class > PCIe; a card that cannot hold card 1's resident set is never card 1 (`_card1_key`) |
| Startup bench | `compat/strata/strata-ext/src/card_bench.cpp:40-73` (`order_tiers`), `card_bench.hpp:20-22` | expert rows/s per card, decode and prefill regimes kept apart; Main pinned first, measured cards by rows/s, then RAM, NVMe, unmeasured last |
| Startup fill | `compat/strata/patches/strata/0001:736-748` (`seabraid_open`) | card 1 fills the top of the profile ranking; card 2 walks the same ranking and takes what card 1 does not hold, to its free VRAM less 512 MiB |
| Hot-share | `compat/strata/patches/strata/0006:119-175`, `:261-265` | opt-in f-paced interleave of the ranking between card 1 and card 2; `auto` = card 2's share of BYTE capacity; refused with `--retier`, >1 extra tier, layer split (`:230-245`) |
| Per-layer serve | `0001:787-843` | card 1 publishes rows for the experts card 2 holds, computes its own, waits up to 30 s for the reply |
| Live moves | `compat/strata/strata-ext/src/tier_plan.cpp`, `retier_loop.cpp`, `0001:846-921`, `0004:229-233`, `0005` | `--retier`: heat-ranked LOAD/SWAP into card 2, same layer only, 4 moves per round every 128 windows, margin 1.25, cooldown 8, per-layer share ceiling 0.08 as an admission gate; fenced; residency published back and checked |
| Card 1 <-> RAM | `0006:452-456` | Strata's adaptive swaps, `STRATA_DISJOINT_ADAPT=1` keeps card 2's experts out of card 1 |
| RAM duplicate | `0001:1008-1016` | card 2's experts stay in card 1's RAM copy unless `SEABRAID_STRATA_EXCLUDE_HELPER_RAM=1`; the recommended configuration runs `=1` with `STRATA_PREFILL_HELPER_SOURCE=1` (prompt reads borrow blobs from card 2, `0001:926-955`) |
| Tier count | `0001:999-1000`, `0004:205`, `0006:232` | three remote tiers in Strata (`for (int r = 0; r < 3; ++r)`); only tier 0 maps to a process (`0001:991-994`); `TierWireOp::helper` is always 0 (`0001:111`) |

Measured on the two-GPU test system (32 GB AMD GPU HIP, 24 GB NVIDIA GPU CUDA on PCIe 3.0 x8, about 30 GB of memory, IQ3_S, ~47 GiB of
experts; DESIGN-LOG 2026-11-09 unless noted): rows/s 32 GB AMD GPU ~270k, 24 GB NVIDIA GPU ~226k, 12 GB NVIDIA GPU ~148k (2026-11-06).
32 GB AMD GPU as card 1: 105.4/104.9 tok/s, 26.7 ms/window; 24 GB NVIDIA GPU as card 1: 114.5/116.1, 24.0 ms; four streams:
32 GB AMD GPU-first 119-124 vs 24 GB NVIDIA GPU-first 114 aggregate. Strict cascade left the 24 GB NVIDIA GPU with 45% of the experts and 13.6%
of the work (`0006:7-9`). Hot-share results are in section 3; the fence costs 4.3 ms average, 11 ms max per
one-swap round (2026-11-07); prompt reading is copy-bound (section 5).

Run7 (2026-11-09, 24 GB NVIDIA GPU = card 1, 32 GB AMD GPU = card 2, `--remote-expert-opt` on, one cold start per arm): ms/window rises monotonically with every hot expert handed to card 2.

| card 2 hot share f | tok/s (runs 2/3) | ms/window | tokens/window | GPU-reach wait | card 1 per-layer host CPU |
|---|---|---|---|---|---|
| unset (card 1 takes the top) | 117.4 / 115.6 | 23.92 | 2.80 | 14.13 ms | 6.27 ms |
| 0.25 | 115.8 / 116.2 | 24.55 | 2.88 | 13.97 | 7.07 |
| 0.40 | 106.4 / 108.6 | 24.95 | 2.74 | 13.81 | 7.69 |
| 0.544 (speed ratio) | 108.4 / 108.8 | 25.40 | 2.79 | 13.74 | 8.23 |
| auto (byte ratio ~0.67) | 101.5 / 102.7 | 26.28 | 2.73 | 13.71 | 9.22 |
| unset (control repeat) | 116.8 / 114.4 | 24.16 | 2.80 | 14.19 | 6.39 |

The GPU-reach wait fell only 14.13 -> 13.71 ms (3%) while the window grew 10%, so card 2's compute was not what
the window waited on; the cost landed in card 1's per-layer host CPU time, 6.3 -> 9.2 ms per window: the host
side of sending more rows to the other card (`seabraid_helper_begin` walks every id and copies the whole `n_tok x H`
block whenever card 2 owns any row of the layer, `0001:802-815`, kept by `0010:188-189`; with
`--remote-expert-opt` the reply is `n_tok x H` per-token sums that card 1 adds on the host, `0010:218-228`).
Not tested: hot share with the 32 GB AMD GPU first beyond `auto`; whether a cheaper host path changes the answer.

## 2. Ranking N cards of any make

Two jobs, two measurements. `ranking.py` ranks by expert rows/s, which is the job of cards 2..N; it picked the
32 GB AMD GPU as card 1, and the 24 GB NVIDIA GPU was the faster card 1 for one stream (DESIGN-LOG 2026-11-09, "Card ranking is
measuring the helper's job"). The lead job is latency-bound at one stream (draft step 1.8 vs 2.3 ms, GPU-reach
14.1 vs 18.9 ms) and compute-bound at four.

**Rule R1 (card 1).** Among cards that pass the eligibility gate (dense + KV + drafter fit, `ranking.py` `_card1_key`),
card 1 is the card with the lowest measured LEAD-JOB time per decode window at the configured concurrency: the
non-expert path (attention, dense, router, drafter step, verify) with every expert resident. One regime per
concurrency class (1 stream; >= 4 streams), never blended, the same discipline `card_bench.hpp:20-22` already
applies to decode vs prefill. Until that bench exists the proxy is memory bandwidth for 1 stream (the 2026-11-06
prediction from 936 vs 640 GB/s held) and the measured 4-stream order on the test system; the plan line must say
"proxy", as `rank_note` already distinguishes measured from heuristic (`ranking.py` `rank_note`).

**Rule R2 (cards 2..N).** Order by decode expert rows/s, ties by free VRAM (`card_bench.cpp:58-63`, unchanged).
Rows/s is their job. The per-layer round trip is a floor, not an order key: it decides admission (R3), because a
layer cannot join before every card it used has answered.

**Rule R3 (admission: when a slow card costs more than it gives).** Card k is admitted only if both hold:
(a) it would hold experts that otherwise fall to RAM or NVMe: expert bytes exceed the VRAM of cards 1..k-1 after
    their reserves. When the model is already all-VRAM (IQ3_S on 32 GB AMD GPU + 24 GB NVIDIA GPU: RAM 0, NVMe 0, 2026-11-07) a
    further card adds a per-layer round trip and nothing else; the IQ2_XS baseline said the same for a second card
    ("adds nothing", 2026-11-06).
(b) the layer time with the card is not longer than without it. Model: a layer costs host_k + max(T_1, hop_k +
    T_k), where host_k is card 1's serial work for that card (scan, stage, publish, merge: the term run7 exposed)
    and the max is the overlap of card 1's own experts with the round trip (STRATA-INTEGRATION 2.4 step 4);
    without the card the same experts cost RAM or NVMe pulls instead. Floors measured: 12 GB NVIDIA GPU ~80 us/layer
    (2026-11-06), 24 GB NVIDIA GPU ~75 us/call of which ~58 waiting (2026-11-07), 32 GB AMD GPU as card 2 ~130 us of which 118 on its
    GPU (2026-11-09); host_k ~60 us per shared layer (run7). The model is the weakest inference in this document;
    A/B 4 records every term per layer so it can be checked, not assumed.
A card that fails (b) but passes (a) is still worth more than NVMe as a blob store: it serves prompt lending and
live moves without computing a decode layer. Whether to run such a card in "store only" mode is a product call;
the planner must print which rule excluded a card and what it would have held.

Unknown: whether a cheap lead-job bench predicts the measured order. See A/B 3.

## 3. Placement: the strict heat cascade, at every rung

The design rule, "a hot expert promotes to card 1 and the coldest there drops a rung", is a strict heat cascade.
The code is already that cascade at startup (`0001:736-748`: ranked order, nothing card 1 holds). Run7 measured it
against every weighted alternative with the 24 GB NVIDIA GPU as card 1 and it won at every step (section 1). So:

**Rule P1 (default, measured).** One heat ranking, one cascade: card 1 takes the hottest experts it can hold,
card 2 the next hottest, ... card N, then RAM, then NVMe. Live, an expert that out-heats the coldest ELIGIBLE
resident of the rung above by the promotion margin moves up one rung and that resident drops one rung (section
4); "eligible" means the same layer and out of cooldown while the SWAP is the only primitive, so a globally
hotter expert can sit below a colder one in another layer until cross-layer moves exist.
This holds BETWEEN classes (VRAM | RAM | NVMe) for the obvious reason, costs are strictly ordered and a miss is a
serial pull into card 1, and it held AMONG the cards on the test system because the price of handing a layer's rows to
another card is paid on card 1's host thread per layer (6.3 -> 9.2 ms per window in run7), not on the other
card's GPU; the GPU-reach wait never moved. The 45%/13.6% skew that motivated hot-share is therefore not a loss
to recover: card 2 doing little work is what a cheap card 2 looks like when card 1 is the faster card.

**Hypothesis H1 (not a default): finish-time balance pays when card 1's own expert compute is the critical
path.** The one measured win for sharing was the 32 GB AMD GPU as card 1 with hot-share + `--remote-expert-opt`: 24.4 vs
26.7 ms/window (DESIGN-LOG 2026-11-09), a slower card 1 whose GPU reach (18.9 ms) was the window. Even there the
accepted tokens per window fell (2.63 vs 2.83), so the output gain was about +2%, not 9%, and the arm changed two
things against the baseline; the opt-only arm (26.3 ms, 2.58 tokens/window) is the control sharing must beat.
Window time alone is not the metric: every finish line in section 8 is in tokens per second with tokens/window
reported. H1's trigger, "card 1's own expert compute is the critical path", is not isolated by the GPU-reach
figure (it contains attention and drafter too), so A/B 1b must log card 1's per-layer expert time separately.
H1 is a test (A/B 1b) and a knob that stays off, not a controller.

**The lever run7 exposed is the host path, not the share.** `seabraid_helper_begin` walks all `n_tok x k` ids and copies
the full `n_tok x H` activation block into the channel whenever card 2 owns one row of the layer (`0001:802-815`);
`seabraid_helper_finish` spins, then merges per-pick rows (`0001:838-841`) or per-token sums (`0010:218-228`). A compact
submit (owned tokens only; with `kOutPerToken` card 2 needs exactly those) and a non-blocking collect would lower
the per-layer floor for every card and could move H1's break-even. That is A/B 9, before any balance policy.

The design exception, "unless we can architecturally determine which experts will be needed", has two faces:
prefill, where the routing is known per layer once the router runs and every card computes its own share
(section 5), and the prompt's own routing histogram, known before decode starts, which can promote the experts
this request will need (section 4, prompt-conditioned promotion).

## 4. The live controller

Signals that exist: `DriverState::usage`, decayed routing counts per (layer, expert) across all tiers
(`tier_plan.hpp:25-30`; allocated only with `--adapt-every > 0 && --adapt-swaps > 0`, `retier_loop.cpp:61-69`);
card k's cumulative service time (`svc_n/svc_ns/svc_begin_ns`, `0001:156-159`, printed per request at
`0001:1037-1047`); card 1's `ms_begin`/`ms_wait`. Missing: per-LAYER finish time per card; channel v7 appends one
`uint64_t svc_layer_ns[n_layers]` region per card (append-only, size-asserted, as every v6 field was added:
STRATA-INTEGRATION 2.8). It is collected for A/B 1b, 4 and 9; no controller acts on it until one of them shows a
regime where sharing wins. One controller, the heat cascade, generalised from "card 2 vs the rest" to every rung.

**C1, the cascade (exists for one rung: `tier_plan.cpp`).** Challengers hottest-first, incumbents coldest-first; a
challenger displaces the coldest incumbent it out-heats by `promote_margin` 1.25 (`tier_plan.cpp:147-164`);
cooldown 8 rounds (`:77-80`); 4 moves per round every 128 windows (`retier_loop.cpp:53-58`); heat smoothed with
decay 0.75 (`tier_plan.hpp:62-67`). Today it knows two sets, "on card 2" and "not on card 2", and its challengers
are everything not on card 2 (`tier_plan.cpp:90-96`): it cannot see card 1's residency, so it can LOAD onto card 2
an expert card 1 already holds. Numerically harmless, since `seabraid_helper_begin` skips any expert card 1 has resident
(`0001:807-808`) and card 1's adapt asks `holds()` live (DESIGN-LOG 2026-11-06, "claimed is passed as nullptr"),
but the slot is wasted and the duplicate persists until card 1 happens to swap it out. Startup disjointness does
not prove live disjointness.

Generalised: the planner takes one residency bitmap per rung (card 1, card 2, ..., card N, RAM; NVMe is the
complement) and the same rule runs at each boundary, hottest-first, one rung per move:
- card k -> card k-1 (promotion) when the challenger out-heats the coldest resident of k-1 by the margin; the
  displaced resident drops to card k in the same fenced round (a SWAP on card k, an admit-or-replace on card
  k-1). Card 1 is rung 0 and the most important target: the design rule "hot promotes to card 1".
- card N <-> RAM and card 1 <-> RAM as today (Strata's own swaps for card 1, `STRATA_DISJOINT_ADAPT`;
  `tier_apply` LOAD/SWAP for card N, `0001:622-688`).
- same-layer SWAP stays the only cross-card primitive until `relocate()` and a wire EVICT exist (`0001:101-108`:
  slot sizes differ per layer; `0001:655-658`: cross-layer refused; STRATA-INTEGRATION 8.7). A promotion is a
  same-layer pair by construction (the challenger and the displaced resident are in the same layer only if the
  planner chooses them so; it should, because the freed slot on card k is then the right size for the drop).

**One owner for "who executes (layer, expert)".** Today that fact is split: card 1's `ExpertCache`, each card's
`held[]` bitmap adopted after a round (`0001:889-909`), and the startup-only `claimed[]` (`0004:231-232` passes
`nullptr`; nothing reads it after open). With N rungs the planner and the fence must read one authoritative map
(one entry per cell: which rung executes it, and the placement epoch), and every move is a transition on that map:
reserve the destination, fill, sync, publish, commit the map while admission is closed, then release the source.
For two full cards swapping a pair, both displaced blobs are preserved (a blob slot each) before either slot is
overwritten, so every failure boundary has a defined recovery. "Exactly one execution owner per cell at every
window boundary" is the invariant; a temporary second physical copy during a round is allowed, a second owner is
not. This is M3 in section 9.

**Fail closed on unknown residency.** Two places today reopen serving with a view that may be wrong:
- a fence timeout (60 s, `0001:872-876`) returns through the RAII reopen (`0001:866-868`) with `bl_held_`
  unchanged while the card may still be applying the round; the next request routes from a stale view;
- a failed `tier_sync` keeps the displaced expert marked resident on the ground that "their data is still there"
  (`0001:695-699`), but a SWAP writes the new blob into that same slot before the sync (`0001:682`), so after a
  failed copy the slot may hold neither expert whole.
The N-card contract: after a timeout or a failed sync the affected card is CLOSED (no layer requests, its experts
treated as misses or recomputed from the GGUF) until it republishes `held[]` under a new epoch; a slot whose copy
failed is published as holding nothing, which needs the eviction primitive `ExpertCache` lacks, so until then the
card reports `kFailed` and takes a fresh plan. Logging and continuing is not enough here: a wrong row is
well-formed and nothing downstream detects it (STRATA-INTEGRATION 2.5).

**Hysteresis.** The ceiling admission flickers (over-ceiling layers 4 -> 45 -> 12 -> 45 -> 21 across 44 rounds,
DESIGN-LOG 2026-11-06 D7 call site). The per-layer ceiling was a proxy for "do not pile a layer's work on card 2";
run7 says the cascade already avoids that, so the ceiling becomes a safety gate with two thresholds (admit below
c_lo, block above c_hi, A/B 6) rather than a target. The promotion margin and cooldown are the cascade's own
hysteresis and stay.

**Where the bytes come from, with no RAM duplicate.** The recommended configuration runs `SEABRAID_STRATA_EXCLUDE_HELPER_RAM=1`; the
channel comment already rules that "if a future LOAD ever needs main-supplied bytes it must stage through the
existing `blobs` slots, NOT a new per-op payload region" (`0001:200-204`). So every move is card -> shm blob slot
-> card, through host memory, no P2P (cross-vendor runtimes cannot share device memory):
- up (card k -> card 1): the prompt-path read already exists, 16 blobs per round trip (`copy_cached_blobs`,
  `0001:926-955`); card 1 fills its slot from the blob, then the SWAP/EVICT on card k runs inside the same fence;
- down (card 1 -> card k): today card k reads the blob through its own `ExpertSource` (`tier_apply`,
  `0001:676-686`), file-backed, so it is an NVMe read unless the page cache holds it; a `kTierLoadFromBlob` op
  kind lets card 1 D2H the blob into a slot and card k H2D it, PCIe twice, no drive and no RAM copy;
- card j -> card k (both >= 2): via card 1's host memory across two channels, same primitive twice.

**Units and throttle.** A MOVE is one expert's bytes changing rung; a SWAP is two policy ops and one wire op
(`retier_loop.cpp:55`); a round carries at most 64 wire ops (`0001:84`). The fence was measured at ONE swap per
round, 4.3 ms average, 11 ms max (2026-11-07), against a 24-27 ms window; larger rounds are unmeasured. Rule:
fence time <= 1% of decode time, so `every_windows >= fence_max / (0.01 x window_ms)` ~ 46 at one swap; the
default 128 has margin, and a round of more moves must re-measure its fence before it is allowed. One move
budget per round across all rungs, promotions first (they shorten card 1's misses, the most expensive kind), one
fence per round, never two in one window.

**Promotion card k -> card 1 live is not built** and is the first thing to build: without it
the cascade can only push experts down, and the measured result says the value is in what card 1 holds. It needs
the "up" path above plus one hook where card 1 admits-or-replaces into its own ExpertCache at the window boundary,
inside the same fence that evicts on card k, so the two caches never both hold the expert for a window in which a
request could route to it.

**Prompt-conditioned promotion (a design proposal: for large prompts the attention can promote
experts on demand).** After prefill the engine knows exactly which experts the prompt routed to in every layer
and how often. The long-run heat counts decode-window routing (`tier_plan.hpp:25-30`); whether prefill routing
is counted there is to be checked in `expert_source.cpp`, and either way it is decayed and shared across
requests. The design adds a per-request histogram h_req(layer, expert), zeroed at each prompt, and runs ONE
cascade round between the end of prefill and the first decode window with heat = (1 - w) x long-run + w x h_req,
both normalised to sum to one per layer first (their counting horizons differ), w = 0.5 to start, promoting the
prompt's hot experts toward card 1 and dropping card 1's coldest eligible one rung, same-layer limit as always,
through the ownership map (M3). Why it can pay: a long prompt's decode routes to the experts its context made
hot; every promoted expert saves its per-row host work on card 1, and saves the whole card-k round trip of a
layer only when card k then owns no row of that layer in the window (`0001:814`), so the payoff is counted in
REMOVED CARD-LAYER REQUESTS, not in experts moved (A/B 11a measures both). Cost: one expert over host memory
between makes, ~2 MB for IQ3_S (blobs 1.18-1.51 MB, `0001:104`), D2H + H2D at PCIe 3.0 x8 plus its share of a
fence. Budget: moves x measured cost_per_move <= 2% of the expected decode time of the reply (a 2,000-token reply
at ~115 tok/s is ~17 s, so ~350 ms; at the measured 4.3 ms per one-swap fence that is ~80 moves, fewer if larger
rounds cost more); the round stops at the budget or when no challenger beats the margin. The knob is off until
A/B 11 passes. A later, weaker variant (the attention scores of the current step pick which
past tokens matter, and THOSE tokens' experts are promoted) is noted as an idea and not designed here: it needs
per-step attention statistics the engine does not export, and A/B 11a tells first whether the prompt's own
top-k already covers the decode.

## 5. Prefill (D4) and decode pre-gating

Prefill is copy-bound: 75-85% of the prompt's GPU timeline is `wait copy`; card 1 pulls every expert card 2 holds
across PCIe per chunk, 13 GiB for 3.5k tokens and 46 GiB for 14k, about 3.3 MB per prompt token; two-card prompt
speed 135/360/740 tok/s at 0.9k/3.5k/14k against 314/619/1,224 for stock Strata on the 32 GB AMD GPU alone with experts in
page-locked RAM (DESIGN-LOG 2026-11-09). Routing per layer is exact only after that layer's router runs, and a
long prompt touches nearly every expert, so no prediction cuts prefill copies. Lending more cheaply is not the
fix; not lending is.

**D4, compute in place.** Per MoE layer of a prompt chunk, card 1 runs the router, publishes the chunk's rows,
ids and router weights, each card computes its own experts and returns per-token weighted sums
(`kOutPerToken`, patch 0007/0010, STRATA-INTEGRATION 2.2), card 1 adds its own and continues. Bytes per token:
two rows, each of H = 2,560 floats, per layer per participating card beyond card 1
(`compat/strata/tests/install/test_plan.py:239`): ~1.0 MB per token per card over 48 layers in f32, half in bf16 (a bf16 wire
format is a channel contract to write, not a flag), against ~3.3 MB per token lent today; and card k's compute,
idle during prompts now, runs in parallel with card 1's. Hazards already on record (DESIGN-LOG 2026-11-06 "D4 hazard"): the channel must be
sized by TOKENS, bounded per request, never by pairs (a 16K chunk at top-10 would reserve ~3.4 GB of shm, the host
pressure that evicted the GPU queues); so the wire carries sub-chunks of at most `prefill_max_t` tokens (1,024
to start) and loops. Two things in the code say D4 is not a flag flip: `seabraid_helper_begin` refuses `n_tok >
kWindowMaxT` (`0001:790`), and the `x`/`out` regions are sized by `cap` = window x 10 PAIRS (`0001:196-198`;
STRATA-INTEGRATION 7.10), so D4 needs its own token-sized region and its own entry point, not a bigger window
constant. The per-token reduce already crosses the channel for decode (`kOutPerToken`, accumulated by
`RemoteExpertOpt` since patch 0010, which removed the earlier refusal at `0010:183`). "A long prompt touches
nearly every expert" is an inference from the
lending bytes, not a measured coverage: the D4 arm records unique experts per layer per chunk so the claim gets
a number.

**Decode pre-gating** (router of layer L+1 on layer L's output, to prefetch RAM/NVMe experts) only matters when a
RAM or NVMe tier carries traffic: all-VRAM runs hit 100% (2026-11-07). Lead time is one layer (~0.5 ms at 24 ms
per 48 layers) against a ~0.1 ms RAM pull and a ~0.8 ms NVMe read at 2.4 MB per miss: plausible for RAM, marginal
for NVMe. The router of layer L+1 reads the NORMALISED input after attention L+1, not layer L's output, so the
prediction is speculative by construction; a hint may prefetch, it never replaces the router's decision, and the
scored output must stay bit-identical with hints on. Accuracy is unmeasured and measurable offline from logged
hidden states at no GPU cost (A/B 7). Upstream already has a prefetch path, `STRATA_ADAPT_FETCH`, disabled
whenever `remote_count > 0` (DESIGN-LOG 2026-11-07, rebase entry); read it before writing a second one.

## 6. N > 2 and Strata's three-tier cap

Strata holds three remote tiers in fixed arrays (`0001:999-1000`, `0004:205`); Strata integration maps only tier 0 to a
process (`0001:991-994`) and hot-share refuses more than one (`0006:232`); `TierWireOp::helper` is reserved for
the card index (`0001:109-117`). Lift, in this order:
1. Strata, generic: the remote tier count becomes a runtime vector sized from the flags, no card names, no
   process knowledge; the `--expert-cache-deviceK` family already names tiers by index.
2. seabraid: `SEABRAID_STRATA_SHM=/seabraid0,/seabraid1,...`, one channel and one process per card of another make. Rank
   and identity are separate: a channel is addressed by a stable card id, and `helper` on the wire carries that
   id, so re-ranking cards (a re-bench, a card added) never redirects a request to another card's channel.
   Per layer: submit to every card that holds a routed expert, compute own, collect all. Layer time = max over
   cards, which is why R3 admission tightens as N grows; the aggregate staging across N channels is bounded
   (N x the per-channel region is host RAM, section 5's hazard), and a requested N that exceeds the budget fails
   at startup.
3. Same-make cards: the default is a card of card 1's make IN PROCESS on Strata's own remote tier (no channel
   hop, the path Strata already measures) and every card of another make in its own process (crash isolation,
   per-card measurement, uniform addressing). That default is decided by A/B 10 (in-process vs process on the
   same card), not by the vendor name.
4. Test before a third card exists: two processes on one card, plus fake cards on the host test, qualify the
   addressing, the ownership map and the controllers, not the latency (DESIGN-LOG 2026-11-06 roadmap item 4).
   Every routed contribution must execute exactly once, every active card must serve, and an injected card death
   must fail or recover the request without partial output.

## 7. What belongs upstream in Strata, what stays in Strata integration

Strata owns execution contracts (generic, vendor- and card-neutral; nothing names a machine, a card or a path):
the neutral channel tag and header (STRATA-INTEGRATION 6.3); the helper mode of the remote tier and its wiring
(6.4); N remote tiers addressed by stable id (section 6); the ownership map and its transition
(reserve/fill/sync/publish/commit, M3), of which today's fence is the two-rung case (6.6); the placement seam (the
caller supplies card 1's fill order and the initial map, 6.10); the per-layer fan-out and join used by decode and
by D4 alike, bounded by tokens (M5); the channel v7 appends (per-layer service times, prefill sub-chunk fields,
`kTierLoadFromBlob`); later `relocate()` and a wire EVICT (8.7). Each is a PR of its own, in that dependency
order, with the acceptance tests the integration doc lists. Strata validates every contribution (identity,
geometry, output mode, epoch) whatever transport delivered it.

Strata integration owns choices (policy): R1-R3 and the lead-job bench regime; the heat ranking and the cascade's constants
(margin, cooldown, decay, moves per round, c_lo/c_hi); hot-share and H1 if it ever earns a default; the admission
and plan report; the installer; configs; the KL gate; the arms runner; anything that names a card model or a
workload. The planner proposes moves against the map; it can never mutate residency itself (today the call site
links the Strata integration scheduler straight into `strata`, `0004:77-80`, and the plan walk sits inside the transport,
`0001:729-748`; both move behind the seam).

## 8. One-variable A/B tests, ranked (each decides one open point)

Finish lines are in output tokens per second (tokens/window and ms/window both reported, run7's lesson), with the
KL gate inside the warm off-vs-off noise (0.021-0.078) unless a tighter bound is named. The KL gate is not yet a
gate: `compat/strata/tools/kl.py` prints statistics and nothing fails the arm (no exit code, no threshold); before any arm here
counts, it takes a named statistic (mean KL over the 441 positions and top-1 agreement), the baseline arm, the
bound, and it must mark the arm INVALID when the bound is crossed. That is the first checker to build.

| # | Variable | Fixed | Finish line | Decides |
|---|---|---|---|---|
| 1 | DONE (run7): `SEABRAID_STRATA_HOT_SHARE` unset / 0.25 / 0.40 / 0.544 / auto / unset, 24 GB NVIDIA GPU first | prompts, `--remote-expert-opt` on | result: no f beats unset; ms/window +2.6% .. +9.9%, host CPU 6.3 -> 9.2 ms | P1 wins among cards on the test system |
| 1b | the same sweep, 32 GB AMD GPU first | as 1 | some f gives >= +3% tok/s over unset with tokens/window within 0.05, or H1 is dropped for this host path | H1 |
| 2 | D4 compute in place vs blob lending | prompt set 0.9k / 3.5k / 14k, both cards, same placement | ship bar: >= +20% prompt tok/s at 3.5k and 14k, <= 5% loss at 0.9k, zero steady-state lending, staging <= 64 MiB; goal: >= 314 / 619 / 1,224 (stock single card); PCIe bytes per prompt down >= 2x | section 5 |
| 3 | lead-job bench vs measured card-1 order | same system, 1 and 4 streams | the bench picks the 24 GB NVIDIA GPU at 1 stream and the 32 GB AMD GPU at 4; else R1 falls back to "user states the concurrency" | R1 |
| 4 | third card (12 GB NVIDIA GPU) admitted vs excluded by R3, IQ3_S all-VRAM | two-card placement, card 1 fixed | excluded run tok/s >= two-card run; admitted run not faster (R3a holds); then a pack that overflows VRAM: admitted run >= +3% tok/s and fewer NVMe bytes per reply, p95 latency <= +2% | R3 |
| 5 | live promotion (C1 with the up path) on vs frozen placement | same prompt sequence, same initial placement, moves > 0 | >= +3% tok/s including fence time, zero STALE/duplicate-owner events, fence time <= 1% of decode | C1 generalised |
| 6 | C1 ceiling hysteresis (c_lo 0.06 / c_hi 0.10) vs single 0.08 | `--retier` on, 1 swap / 32 windows | over-ceiling layer count variance across 44 rounds halves; tok/s not lower | hysteresis |
| 7 | pre-gating, offline then live | logged hidden states of one 2,000-token reply; then hints on/off with a RAM tier | offline: top-10 overlap of router L+1 on h_L >= 80% per layer before any code; live: bit-identical scored output, exposed miss wait -20%, tok/s +3%, source bytes <= +10% | section 5 |
| 8 | option B regression: the four combinations of `SEABRAID_STRATA_EXCLUDE_HELPER_RAM` x `STRATA_DISJOINT_ADAPT` | 24 GB NVIDIA GPU first, everything else the recommended configuration | which switch, or which interaction, accounts for 45 -> 114 tok/s and CPU jobs 17 -> ~0 ms; the result fixes the settings every later arm holds constant | RAM-duplicate rule |
| 9 | host path: full/blocking (today) vs compact/blocking vs compact/non-blocking, three arms, one change each | 24 GB NVIDIA GPU first, f = 0.25, opt on, placement fixed | card 1 host CPU per window back to <= 6.5 ms at f = 0.25; if tok/s then beats unset, rerun 1 | the host-path lever, H1's break-even |
| 10 | same card as card 2 in process vs as a process | same experts, kernels, workload | process path kept as the uniform default if tok/s loses <= 2% and p99 layer latency <= +5% | section 6 step 3 |
| 11a | prompt-conditioned coverage, logging only, no moves | a fixed set of long prompts (>= 3.5k tokens), both cards, k = 32 per layer | two numbers per request: the fraction of the decode's routed rows that fall on the prompt's top-k experts per layer, and the card-layer requests that would disappear if those top-k not on card 1 were promoted; proceed to 11b only if the second is >= 10% of the decode's card-layer requests | whether prompt heat predicts decode heat |
| 11b | pre-decode promotion on vs off | same prompts, same budget (2% of expected decode time), same initial placement | >= +3% tok/s on the reply including the round's cost; ms/window and tokens/window reported; KL inside noise; zero STALE/duplicate-owner events | prompt-conditioned promotion |

### Request-start expert hints (offline)

Run the replay before changing the gateway or engine. A task or effort label picks a historical profile for the exact model build.
It does not select which experts execute. The router still makes that decision.

Collect one fresh prompt histogram per request with `STRATA_ROUTE_HIST`. Keep model, quantization, and lead residency fixed.
Choose train and test prompt families before collecting, and keep every repeat, paraphrase, and length variant of a family
on the same side. Cover several independent train and test families per task across the same lengths. Vary effort within
matched families if testing effort. The four repeated Run 10 prompts do not establish task generalization. Assign task and
effort before generation, using the actual decision output or a separately identified labeled baseline.

For the first hardware capture, use an isolated loopback server with one request in flight, `--prompt-cache 0`,
`--adapt-swaps 0`, no retiering or routing overrides, and `STRATA_ADMIT_MERGE_FIRST=0`. Point `STRATA_ROUTE_HIST`
at a new directory. Derive the actual starting lead residency from the launch plan and startup evidence. The capture
CLI checks the reported end map against it on every request. An equal end map is a check, not proof against interim
moves. Do not collect latency from this run because histogram collection synchronizes per layer.

Start with the replay manifest below, add `served_model` at the top and put the full `prompt` in each request
instead of `histogram` and `prompt_tokens`. Then run
`python3 compat/strata/tools/capture_expert_preflight.py PLAN HIST_DIR MANIFEST --port PORT` against the dedicated server.
The capture checks observed identity, no prompt-cache reuse, and one dump per response before publishing the manifest.
It holds generation settings constant, so the effort label tests correlation with prompt routes only.

The replay manifest is version 1 JSON with:

- `model`, a fingerprint identifying the model, quant, and routing configuration; `experts`, the count per layer;
  `picks_per_token`, the routed picks per token; and `placement: "frozen"`.
- `layers`, an ordered array of `{"expert_bytes": 8388608, "resident": [0, 1]}` entries. Use the actual complete expert
  blob size and lead GPU resident IDs **before** the request. The histogram holder map is captured at the end.
- `requests`, entries such as
  `{"histogram":"routes/req-000001.json","family":"review-parser-a","split":"train","task":"coding","effort":"deep","prompt_tokens":4096}`.
  Paths resolve relative to the manifest. Family names identify independent source prompts, and token counts must match
  the actual histogram and be available before routing. On normal serial Strata serve, the last prompt token routes in
  the first decode window. The prompt histogram must therefore contain `(prompt_tokens - 1) * picks_per_token`
  entries per layer. Length-conditioned predictions use the same prefill count.

Run `python3 compat/strata/tools/expert_preflight_replay.py MANIFEST --budget-bytes 67108864`. The reference is no staging with the same
frozen resident set. `static` ranks global train-only pick frequency. `static_presence` ranks the fraction of training
requests that used an expert at least once, giving each family equal weight across repeats. It matches the unit of a
replay miss, since each distinct expert is loaded at most once per request. Both static policies use the same ranking
by score per byte and the same staging cap for every held-out request. The other hints use global frequency plus
length, task plus length, and task plus effort plus length. They use DeepSeek's approximate probability of at least one
use, with a default 0.9 cutoff (`--probability`). A task or effort profile needs two independent training families or falls back to the broader
profile. Use the same byte cap for every arm. Rank by expected benefit per byte with deterministic ties. Choose budgets and
cutoffs on training data before reviewing held-out outcomes.

Read avoided on-demand loads, request-layers with no misses, and wasted bytes per arm and by held-out family. Each load is one
distinct nonresident (layer, expert) pair per request. This is a best-case staging model with no eviction, no extra runtime
traffic, and all staged weights ready by first use. It cannot recover per-chunk dispatches, distinguish host from remote cost,
or measure saved milliseconds. The histogram cannot determine when an earlier layer or chunk would have revealed
routes, or whether a staged transfer would have completed in time. Advance to ordered traces and measured wait/TTFT
only if the task arm improves over the static and global length baselines across held-out families without trading the
gain for more traffic. Require the same output and a positive latency result after charging the classifier and copies
before enabling it in the pipeline.

A hand-counted replay fixture distinguishes the baselines. Under an 8-byte cap, the pick-frequency arm stages an
unused 8-byte expert for a held-out coding request. `static_presence` stages the requested expert and avoids its
8-byte on-demand load. These are synthetic replay counts. Model, traffic, and latency measurements remain open.

Every arm uses `compat/strata/tools/arms.sh` (one cold start; `speed` = warm-up + three 2,000-token greedy runs, run 2 quoted;
`prefill` = three prompt lengths + 400 decode tokens; `conc` = two 1,000-token rounds; `kl` = the teacher-forced
text; `compat/strata/tools/arms.sh:77-93`) and is read only with its VALID marker present and the runner's exit code checked.
Arms that change placement run with the expert profile read-only so one arm cannot inherit another's placement
(ROADMAP "KL thresholds"). p95/p99 latencies and per-layer timings named above need counters the tools do not
record today; each such finish line names its counter before the arm runs.

## 9. Module status

This section was a design gate for six modules (owner, contract, size budget, failure, checker, finish line). Most of them
now exist, so it lists what each owns and where it lives. The size budgets are enforced by `checks/check_size.py` (500 lines,
ratcheted past 400) and `checks/check_size_budgets.py` (the budget a file states in its own header); they are not repeated
here. The proposed finish lines were targets and are not evidence; the measured results are in
[DESIGN-LOG.md](../DESIGN-LOG.md) and [the research log](../../../../docs/RESEARCH-LOG.md). The earlier text of this
section, with its per-module budgets and finish lines, is in git history at commit 1d1b427.

| Module | Owns | State |
|---|---|---|
| M1 ranking (R1-R3) | which card is card 1, and which cards are admitted. `rank(cards, bench_lead, bench_rows, fits_main, concurrency, force_main)`, stdlib only, never raises, and one missing measurement drops every card to the proxy order. | Built: `bindings/python/seabraid/ranking.py` (`lead_class`, `LEAD_NOT_RUN`). |
| M2 card bench, lead-job regime | what a card measured, with decode, prefill and lead regimes never blended. The lead regime is lower-is-better window milliseconds by concurrency, so it is its own struct. An unmeasured card is NaN, ranked last and printed. | Built: `compat/strata/strata-ext/src/card_bench.cpp` (`lead1_`, `lead4_`). |
| M3 ownership map and transition | one map, one epoch: the rung that executes each (layer, expert). `prepare`, `commit`, `abort`; the planner proposes and cannot write the map. A timeout or failed sync fails closed for the affected card. | Built: patch 0012 (`ownership_map`, `ownership_rungs`) and the Seabraid placement adapter. |
| M3b tier plan over N rungs | moves of one rung each, hottest promotion first and coldest drop first, a promotion and its displaced drop in the same layer. | Built: `compat/strata/strata-ext/src/tier_ladder.cpp`; the two-set path in `tier_plan.cpp` is unchanged. |
| M3c per-request routing histogram and the blend | the engine counts routing per request; the policy blends it with the long-run heat under one weight and a byte budget. | Histogram built: patch 0011 (`STRATA_ROUTE_HIST`). The blend into the planner is not built. |
| M4 channel, append-only | the wire. The preamble is never touched; new fields are appended and golden sizes updated. | Built, now v11 (v10's layout under the SEABRD1 tag), header 2752 bytes: `adapters/strata/wire/strata/seabraid/channel_layout.hpp`, `adapters/strata/wire_v10_golden.txt`. |
| M5 per-layer fan-out and join (decode and D4) | `submit`/`collect` for a layer, and the same code for a prompt chunk through a token-sized entry. A rejected sub-chunk falls back to lending; an uncertain one closes the card and the chunk is recomputed from a verified source or the request fails. | Built, opt-in: patch 0013 (`STRATA_PREFILL_REMOTE_COMPUTE`). Measured slower than lending on an 8.3k-token prompt (RESEARCH-LOG 2026-11-17), off by default. |
| M6 N channels in the engine | card id to channel, fixed for the session; `helper` on every op carries the id, never the rank. | Built to three: `SEABRAID_STRATA_SHM` is a list and `kMaxChannels = 3` (`adapters/strata/wire/strata/seabraid/channel_names.hpp:18`), patch 0017. Four fake cards and a host-memory budget at startup are not built. |

Two cross-cutting checks from the original gate are still open. First, a build of `strata` without `SEABRAID_STRATA_SOURCE_DIR`
plus an include check that no Strata file names a `strata/seabraid/` policy header (`retier_loop.hpp` still includes
`tier_plan.hpp`, so the upstream split in section 7 stays a wish until that boundary is a callback the engine owns).
Second, p95 and p99 latencies and per-layer timings named in section 8 need counters the tools do not record; a finish line
that needs one names its counter before the arm runs.

## 10. Review notes

Two independent reviews fed this document: one written before run7's result existed, and one over the merged text with
run7. Every claim in the text was checked against the cited lines before it was taken. What the reviews established and
the text still depends on:

- On the two-set path the planner cannot see card 1's residency and can load onto card 2 an expert card 1 holds
  (`tier_plan.cpp:90-96`, `0004:231-232`). The ownership map (M3) is what closes this for the ladder.
- A fence timeout reopened admission with a possibly stale view, and after a failed `tier_sync` the displaced slot may hold
  neither expert whole. This is why a timeout or failed sync fails closed (section 4); patch 0012 implements it.
- The `x` and `out` regions are sized in pairs and `seabraid_helper_begin` is window-bound, so D4 is an entry point and a region.
- The -9% hot-share figure was window time, and output gained about 2%, so finish lines are in tokens per second.
- The 4.3 ms fence was measured at one swap, not at four moves. The promotion payoff counts removed card-layer requests, not
  experts moved. "Every long prompt touches every expert" is unestablished; the D4 arm measures coverage.
- `compat/strata/tools/kl.py` had no pass/fail when this was written; it is a gate now (ROADMAP, "KL thresholds").

Disagreements and how they were settled:

- Placement among cards. One review proposed the cascade as control and measured finish-cost placement as challenger; the
  author's first draft used finish-time balance within the VRAM class. Run7 settled it against both: the cascade won at every
  share and the cost was card 1's host time, which neither model contained. Balance survives only as H1.
- Finish lines. One review wanted five interleaved paired runs and p95 latency on every arm. Kept: the repository's runner and
  its modes, so arms stay comparable to every DESIGN-LOG number.
- D4's bar. The author: beat the stock single card. One review: +20% over lending with a staging budget. Both kept, the
  review's as the ship bar and the author's as the goal.
- Same-make cards in process. The author: by make. Both reviews: by measurement. The author's default is under A/B 10.
- First test. The author's draft: run7 then D4. The second review: A/B 8, the four-way switch matrix, because it needs no code
  and fixes what every later arm holds constant. Adopted: A/B 8 first, then 11a (logging only), then D4.
