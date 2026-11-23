# Native placement

[`include/seabraid/placement.h`](../include/seabraid/placement.h) binds one engine model layout
to a Rust ownership map and heat planner. It works alongside the batch ABI.
The adapter binds each rank to a stable provider and verifies the initial
residency. For a model with two cards, rank 0 is the lead, rank 1 the helper,
and rank 2 the verified fallback. All arrays are layer-major.

0. **Fill (optional).** Compute the initial placement with
   `seabraid_placement_fill`. It is a pure function of the card slot capacities and a
   heat table: hottest cells first, the lead card (rung 0) fills first, then each
   helper in rung order, and the remainder goes to the fallback rung `cards`. It
   writes `layers * experts` owners and touches no provider. Load the experts it
   names, verify them, and pass the result as the layout of the next step. The
   [ABI reference](ABI.md#spillover-fill) has the struct.
1. Open a handle for the verified layout and export the first executor table
   with `seabraid_placement_snapshot`. Route by the exported table and retain its
   epoch with admitted work. An index into the table needs no ABI call.
2. At the engine's planning cadence, pass all-rung routing observations for
   the current epoch as per-cell heat for this observation window. Planning
   is free of provider operations. It replaces the handle's pending ticket,
   including when the new ticket has zero promotions. The planner keeps a
   running heat per cell across windows, so one window is not judged alone
   (see [Heat across windows](#heat-across-windows)).
3. When a plan has work, stop admission and drain the old epoch's work on
   all affected queues. Apply the ticket with the provider and space for every
   planned outcome. An intervening placement change invalidates its epoch
   before the first callback. Geometry failures are retryable with the same
   pending ticket. A started transaction consumes it even after a refusal.
4. Inspect the ordered outcomes and final epoch (what each means is under
   [Outcomes and epochs](#outcomes-and-epochs) and [Faults](#faults-and-their-outcomes)).
   A completed round can include a refusal, restoration, or quarantine. If the epoch changed, export and
   publish the new executor table while admission is still closed, then resume
   dispatch using that epoch. A quarantined card's cells go to the fallback.
5. Quarantine out-of-band provider loss while admission is closed. Report
   verified external moves with `SEABRAID_PLACEMENT_ADOPT`. Use
   `SEABRAID_PLACEMENT_REBUILD` after draining or replacing failed providers and
   verifying the full layout and bytes. Publish either change at the same
   fence. Providers that still have uncertain device work stay closed until
   rebuild is safe.

The ticket, outcome, applied and snapshot structs are filled by the library, so the
caller sets `abi_version` (`SEABRAID_ABI_VERSION`) and `struct_size` before each call.
In the outcome array the stride is element 0's `struct_size` (a multiple of 8) and
the library writes every element. A caught panic poisons the handle: every later call
except close returns `SEABRAID_INTERNAL`. `seabraid_placement_open` leaves `*out` NULL on
failure. A provider panic still quarantines the rung. See the
[ABI reference](ABI.md#versioning) for the versioning rule.

The provider supplies the exact byte count of a verified slot for each layer.
Each count is at most 2^32 - 1 bytes; a larger one is `SEABRAID_INVALID` at bind,
before any provider call, and the ticket is kept. Staging copies a complete source slot into memory owned by the transaction.
The replace callback publishes its local residency index only with ACK. A
failed write keeps the old index so a restoration can target the same slot.
If residency is uncertain, return timeout. To return while device work may
still finish, arrange for it to use provider-owned storage. All accesses to
Seabraid's callback pointers must end before callback return, including on error.
The engine owns deadlines and the drain, and holds its admission fence through
the final snapshot publication. The ABI stores neither callbacks nor context.

## Policy

`seabraid_placement_open` takes an optional `seabraid_placement_policy_v1`. A NULL policy uses the defaults below.
An explicit struct is used as written: a zero field is zero, not "default". A value outside its range makes open
return `SEABRAID_INVALID`. No call changes the policy of an open handle: close it and open another (see
[Reopening](#reopening-and-the-epoch-mark)).

| Field | Type | Valid range | Default | Unit and meaning |
|---|---|---|---|---|
| `max_moves_per_round` | `uint16_t` | 0 to 128 | 4 | Expert **writes**, not swaps. One swap writes two experts, so it counts 2. A plan holds at most `max_moves_per_round / 2` swaps, rounded down: 4 is two swaps, 3 is one, 0 and 1 plan nothing. |
| `ladder_boundaries` | `uint8_t` | 0 to 255 | 0 | Which card boundaries may swap. 0 allows all of them: rung `h` with rung `h + 1`, both cards, so a layout has `cards - 1` boundaries. A value `n` above 0 allows only the top `n`; a value above `cards - 1` acts as 0. The fallback rung takes part in a live swap only on a handle opened with `fallback_boundary` 1 ([Policy v2](#policy-v2-the-fallback-boundary-and-climbs)). |
| `cooldown_rounds` | `uint64_t` | any | 8 | Plan rounds, that is successful `seabraid_placement_plan` calls on this handle (the count advances at plan, not at apply). A cell that committed a move in round `R` is a candidate again from round `R + cooldown_rounds + 1`. Only committed moves start a cooldown. |
| `promote_margin` | `double` | finite, at least 1.0 | 1.25 | A challenger must have strictly more than `promote_margin` times the heat of the cell it displaces, both as smoothed values. 1.0 means any strictly hotter cell wins. |
| `max_helper_layer_share` | `double` | 0.0 to 1.0, not NaN | 0.08 | A per-swap check on the rung that **receives the hotter cell**, not a global limit on what a helper may hold: at a boundary whose upper rung is a helper (rung 1 or higher, so boundary rung 1 with rung 2 and beyond), a swap is skipped if that upper rung's share of the layer's total heat would exceed this fraction afterwards. The check is **skipped at the top boundary** (rung 0 with rung 1), where the lead card receives the hotter cell, and the initial fill does not apply it either. So a helper can legitimately hold more than this fraction, for example a helper that was filled with hot cells, and a value of 0.0 still allows top-boundary promotions that leave heat on the helper (the planner test with `max_helper_layer_share: 0.0` in `src/placement/tests/planner.rs` expects the promotion layer 0, expert 1 for expert 0). With 2 cards the field never matters; it only matters with 3 or more. 1.0 turns it off. |

Per round the planner visits boundaries from the top (rung 0 against rung 1 first). At each boundary it takes
challengers on the lower rung hottest first, never a cell with zero heat, and pairs each with the coldest eligible
incumbent **in the same layer** on the upper rung that passes the margin and, below the top boundary, the share cap. A cell moves at most once
per round, and planning stops at the move budget.

## Heat across windows

Planning is not a function of one window. The planner keeps one smoothed heat value per cell for the life of the handle:

- The first successful `seabraid_placement_plan` on a handle uses the raw heat it was given, so a fresh handle
  responds to its first window at once.
- Every later successful plan sets `smoothed = 0.75 * smoothed + 0.25 * window`, and `promote_margin` compares those
  smoothed values. This happens whether or not the previous ticket was applied, whatever the epoch, and also when the
  plan has zero promotions.
- Apply does not reset it, and neither does an epoch change; only a new handle starts from raw heat. Apply does start
  the cooldown of the cells that committed.
- A plan that is refused (stale epoch, wrong count, negative or non-finite heat, or a window in which some layer's
  values sum past the largest finite `double`) changes nothing, so a refused window is not counted. The layer sums
  are checked on the window you pass, on every call (not only the first), and `seabraid_placement_fill` refuses the
  same heat.

So after the hot set moves, the new hot cells must climb past the margin against the old values. In an
external review fixture (a scratch script that is not in this repository, so these counts are an observation, not a tested
guarantee; the default policy, windows of the old hot set followed by windows of the new one, no apply) the first
promotion came on the 4th window of new heat after any number from 1 to 20 windows of old heat, and on the 6th with
`promote_margin` 1.5. Those counts belong to that fixture; the rule is the blend above. An all-zero window scales every
value by 0.75, which keeps every ratio between cells, and one very large finite value in a single window decays only by
0.75 per window, so keep outliers out of the heat you pass.

## Demand-driven placement policy

This section is the single written statement of how the planner decides what moves. The fields it mentions are
defined above ([Policy](#policy), [Policy v2](#policy-v2-the-fallback-boundary-and-climbs)); the code is
`src/placement/planner.rs` and `src/placement/planner/climb.rs`; each numbered sentence has a behaviour test, listed in
the [cross-reference](#policy-statement-to-test-cross-reference) below. If the code and this statement ever disagree,
the code is the truth and this section is the bug.

**What "hot" means.**

1. A cell's demand is its **smoothed heat**: the first plan on a handle uses the window's raw counts, and every later
   plan sets `smoothed = 0.75 * smoothed + 0.25 * window` ([Heat across windows](#heat-across-windows)). The planner
   never compares raw windows after the first.

**Who challenges whom, per layer and boundary.**

2. A round visits the card boundaries from the top (rung 0 against rung 1), then the fallback boundary last when
   `fallback_boundary` is 1. At a boundary the **challengers** are the cells on its lower rung, hottest first (equal heat:
   lower flat index first). A cell with zero heat is never a challenger.
3. A challenger is paired only with an **incumbent of its own layer** on the boundary's upper rung. The incumbent is the
   **coldest eligible** cell there that the challenger may displace; if there is none, the challenger stays where it is.
4. The challenger must beat the incumbent **strictly**: `heat(challenger) > promote_margin * heat(incumbent)`, both
   smoothed. Equal to the margin does not move.
5. Below the top boundary the swap must also respect `max_helper_layer_share`. A warmer incumbent lifts the receiving
   helper's share by less than a colder one, so the displaced cell is the coldest incumbent that **passes the cap**, which
   is not always the coldest one. (Rung 0 receives the hot cells and is never capped.)
6. A swap moves the challenger up and the incumbent **down exactly one rung**: the two cells trade places across one
   boundary, never more. A cell is in at most one swap per round, so a displaced cell moves at most one rung per round.
7. A round plans at most `max_moves_per_round / 2` swaps (2 writes each): the top boundary first and, within a boundary,
   the hottest challenger first; the rest wait for a later round.

**Cooldown.**

8. A swap that commits starts the cooldown for **both** cells, the challenger and the displaced incumbent. A cell that
   moved in round `R` is a candidate again from round `R + cooldown_rounds + 1`, as challenger or as incumbent. A swap that
   is rejected, restored or skipped starts nothing.

**Climbs.**

9. With `max_hops_per_round = 1` each cell takes part in at most one boundary per round.
10. With `max_hops_per_round = n > 1` the hottest challengers are taken across the whole table, and each climbs bottom-up
    one boundary at a time, each hop a swap by sentences 3 to 6, until a boundary has no incumbent that passes or `n` hops
    are done. Every displaced cell still drops exactly one rung; only the climber travels farther.

**How a cooled expert goes down.**

11. **Nothing is ever demoted on its own.** A cell that cooled keeps its rung until a hotter cell on the rung **below**
    beats it by sentence 4 and it is the coldest eligible incumbent. It then drops one rung. To drop again it must be the
    displaced choice of a later round, and not before its cooldown (sentence 8) has passed. So the descent is one rung per
    swap and at most one per round; its pace is set by demand from below and by the cooldown, not by a timer.

**What never moves.**

12. Cells on a quarantined (closed) rung are never candidates, on either side of a swap. Cells on the fallback take part
    only when `fallback_boundary` is 1, and the lead-to-fallback boundary is the engine's, never planned here. A cell in
    cooldown, a zero-heat challenger and a cell already used this round do not move.
13. Alternating demand does not thrash: swings that keep the smoothed ratio inside `promote_margin` move nothing, and
    swings that clear it still move any one cell at most once per `cooldown_rounds + 1` rounds.

**Worked example.** One layer, three cards (rungs 0, 1, 2) and the fallback F (rung 3), two experts per rung: lead
`{0, 1}`, helper 1 `{2, 3}`, helper 2 `{4, 5}`, F `{6, 7}`. Fallback boundary on, `cooldown_rounds` 2, `max_hops` 1, the
share cap off, a steady window `[1, 50, 40, 45, 30, 35, 20, 25]` for experts 0 to 7 (steady, so smoothed equals raw).
Expert 0 is cold but sits on the lead.

- Round 1. Top boundary: the challengers on rung 1 are expert 3 (45), then expert 2 (40). Expert 3 beats `1.25 * 1`
  against the coldest lead incumbent, expert 0, so they swap and expert 0 drops to helper 1. Expert 2 (40) does not beat
  `1.25 * 50` against expert 1. The lower boundaries find nothing either (expert 5: 35 against `1.25 * 40`).
- Rounds 2 and 3. Experts 0 and 3 are cooling until round 4 (1 + 2 + 1), and nothing else clears a margin.
- Round 4. Boundary of rung 1 and rung 2: challenger expert 5 (35) against helper 1's incumbents, expert 0 (1) and
  expert 2 (40). The coldest is expert 0 and 35 > 1.25, so expert 0 drops to helper 2.
- Round 7. Fallback boundary: challenger expert 7 (25) against helper 2's coldest, expert 0: 25 > 1.25, so expert 0
  lands on F, where it stays because nothing below it remains.

Smoothing numbers for sentences 1 and 4: with the windows `(100, 100)` then `(100, 204)`, expert 1's smoothed value is
`0.75 * 100 + 0.25 * 204 = 126`, above `1.25 * 100 = 125`, so it promotes; with `(100, 200)` it is exactly 125, which is
not strictly above, so it does not.

**In the live run.** [Live run L1](RESEARCH-LOG.md#2026-11-19--live-run-l1-the-pyramid-on-real-cards) showed this
behaviour on real cards: 1539 moves, every displacement one rung, and a traced expert that went down main to helper 1
(round 290), helper 1 to helper 2 (round 333) and helper 2 to rung F (round 388), one rung per later round with long
gaps between. That is evidence of the behaviour, not of a speed gain.

### Policy statement to test cross-reference

All tests are in `src/placement/tests/policy_demand.rs`. The last one checks that every test in that file is named in
this list, so a rule cannot be pinned without being written down.

| Sentence | Test |
|---|---|
| 1 | `heat_first_window_is_used_raw`, `heat_later_windows_blend_075_old_and_025_new` |
| 2 | `a_zero_heat_cell_is_never_a_challenger` |
| 3 | `a_hot_challenger_displaces_the_coldest_same_layer_incumbent` |
| 4 | `a_challenger_must_beat_the_margin_strictly_or_nothing_moves` |
| 5 | `the_helper_share_cap_stops_a_helper_boundary_swap_but_not_the_top_boundary`, `the_share_cap_can_pass_over_the_coldest_incumbent_for_a_warmer_one` |
| 6 | `every_displaced_cell_drops_exactly_one_rung_and_each_cell_moves_once_per_round` |
| 7 | `the_hottest_challenger_is_served_first_when_the_write_budget_allows_one_swap` |
| 8 | `a_cell_in_cooldown_is_skipped_and_the_next_coldest_incumbent_is_displaced`, `a_moved_cell_is_a_candidate_again_from_round_r_plus_cooldown_plus_one` |
| 9 | `every_displaced_cell_drops_exactly_one_rung_and_each_cell_moves_once_per_round` |
| 10 | `a_climb_makes_one_hop_per_boundary_and_displaced_cells_drop_one_rung` |
| 11 | `a_cooled_expert_walks_down_one_rung_per_round_until_the_fallback`, `the_cooldown_paces_the_descent_of_a_cooled_expert`, `a_cold_expert_with_no_hotter_challenger_below_it_stays_where_it_is` |
| 12 | `a_quarantined_rung_is_never_a_candidate_on_either_side`, `the_fallback_boundary_is_planned_only_when_enabled` |
| 13 | `mild_alternating_demand_inside_the_margin_never_moves_anything`, `hard_alternating_demand_ping_pong_is_bounded_by_the_cooldown` |
| (completeness) | `policy_doc_names_every_test_in_this_file` |

## Outcomes and epochs

Each element of the outcome array describes one swap, in plan order. `layer` is the layer. Expert `promote` was on
rung `upper_rung + 1` and moves up to `upper_rung`; expert `displace` was on `upper_rung` and moves down to
`upper_rung + 1`. Swaps are only between adjacent rungs, so the lower rung is always `upper_rung + 1`, which is the fallback only on a handle opened with `fallback_boundary` 1 (see [Policy v2](#policy-v2-the-fallback-boundary-and-climbs)).
`committed_epoch` is the epoch right after this swap and is zero for every outcome that is not `COMMITTED`.
`initial` is the first provider failure and `restore` the failure of the rollback write; an `error` of 0 means none.
In an issue, `rung` names the card whose callback failed.

`slots` in `seabraid_placement_fill_v1` counts cells across all layers together, not per layer: `slots[0] = 12` puts the
12 hottest (layer, expert) cells of the whole table on the lead card, whatever their layers.

Epoch arithmetic, from a handle that opens at epoch 1:

| Event | Epoch change |
|---|---|
| Each committed swap | plus 1 |
| A swap that is rejected or restored | none |
| A quarantine during apply (one event, even when it closes two rungs) | plus 1 |
| `seabraid_placement_quarantine` of a rung that is open | plus 1 |
| `seabraid_placement_quarantine` of a rung that is already closed | none |
| `seabraid_placement_update` ADOPT with owners that differ | plus 1 (same owners: none) |
| `seabraid_placement_update` REBUILD | plus 1 |

Two committed swaps give epoch 3. Plan and snapshot never change it.

The outcome carries no owner table and the library has no call that returns one. The snapshot gives **executors**: an
open rung's cells show their owner, but every cell of a closed rung shows the fallback, so the true owner of those cells
cannot be read back. An engine therefore keeps its own copy of the owners: start from the layout it opened with, apply
each `COMMITTED` outcome (promote to `upper_rung`, displace to `upper_rung + 1`), and pass that copy to REBUILD.

## Faults and their outcomes

A swap runs four provider steps: `stage` of the promoted expert (lower card), `stage` of the displaced expert (upper
card), `replace` on the upper card (it takes the promoted expert), then `replace` on the lower card (it takes the
displaced expert). If the second replace fails, the upper card is written back from the staged bytes. A status other
than 0 to 3 is treated as `TIMEOUT` and logged.

| Step | Provider returns | Outcome | Rung closed | Rollback write |
|---|---|---|---|---|
| stage (either) | `REFUSED`, `FAILED`, or an ACK whose `bytes_written` is not the layer's slot size | `REJECTED` | none | none (nothing was written) |
| stage (either) | `TIMEOUT` or unknown code | `QUARANTINED` | the card whose stage failed | none |
| replace on upper | `REFUSED` | `REJECTED` | none | none |
| replace on upper | `FAILED` | `RESTORED` if the rollback ACKs, else `QUARANTINED` | none, else the upper card | to the upper card |
| replace on upper | `TIMEOUT` or unknown code | `QUARANTINED` | the upper card | none (it may still be writing) |
| replace on lower | `REFUSED` | `RESTORED` if the rollback ACKs, else `QUARANTINED` | none, else the upper card | to the upper card |
| replace on lower | `FAILED`, `TIMEOUT` or unknown code | `QUARANTINED` | the lower card, plus the upper card if its rollback also fails | to the upper card only |

A quarantine closes every cell of that card to the fallback and ends the round: later pairs of the plan are not
attempted and have no outcome, so `outcome_count` can be below `promotion_count`. The ticket is spent either way. A
closed rung stays closed until `SEABRAID_PLACEMENT_REBUILD`. A panic inside a Rust provider is handled like
`TIMEOUT` and reported as `PANICKED`; a C++ exception escaping a callback is undefined behavior.

The two replace rows look unequal on purpose: a `FAILED` write on the upper card is rolled back and the card stays open,
while a `FAILED` write on the lower card closes that card and never rewrites it. The code's rule is that a lower write
that was not refused leaves that card uncertain, so it is retired instead of retried (the transaction test for this
case is named for it: the uncertain card is never routed to or rewritten). Plan for both.

## Reopening and the epoch mark

A new handle starts at epoch 1, and the engine's sessions are not told. Each session remembers the highest
`placement_epoch` it ever accepted and refuses a lower one with `SEABRAID_BUSY_OR_STATE`. After you close a handle and
open another (the only way to change the policy), batches stamped 1, 2, 3 and so on are stale on every session that had
accepted a higher number. Two workarounds were run end to end:

1. **Reopen the sessions too.** A new session has no mark. This is the simplest choice when the engine can pause.
2. **Own the epoch in the engine.** Keep a generation counter that goes up each time you open a handle, and send
   `placement_epoch = (generation << 32) | snapshot.epoch` with every batch. The calls on the placement handle still take
   the library's own epoch; only the number you put in the batch changes. It works while the library's epoch stays below
   2^32, and any later generation outranks every earlier one.

The same rule is why one accepted huge epoch is dangerous: a session that accepts 2e12, or even `UINT64_MAX`, refuses
every lower epoch until it is closed. The mark is per session and only an accepted submit raises it.

The guard does not tie a batch to the placement handle. The runtime does not know the handle, the layout or the table
that produced an epoch; it can only tell that a number went down. It catches a straggler routed with an old table only
when the engine stamps batches honestly. What keeps routing correct is the engine's fence and drain around apply. A
provider that holds only some experts should also check, for each pick that is not `SEABRAID_SKIP_EXPERT_ID`, that it
owns that expert, and fail the batch if it does not: a wrong table then becomes a logged `SEABRAID_PROVIDER` failure.

## Policy v2: the fallback boundary and climbs

`seabraid_placement_open_v2` takes a `seabraid_placement_policy_v2`, a sized struct (`abi_version` =
`SEABRAID_PLACEMENT_POLICY_V2_VERSION`, `struct_size` = `sizeof`, at least 40; bytes past what the library knows must be
zero). A NULL policy is the v1 defaults with both new fields at their v1 values, and `seabraid_placement_open(layout, p1,
out)` behaves exactly as `open_v2` with `p1`'s fields, `fallback_boundary` 0 and `max_hops_per_round` 1. Every v1 field
means what it means above. A value outside its range, a non-zero `reserved` byte or a wrong version makes open return
`SEABRAID_INVALID` and leaves `*out` NULL.

| Field | Type | Valid range | v1 value | Meaning |
|---|---|---|---|---|
| `fallback_boundary` | `uint8_t` | 0 or 1 | 0 | 1 plans live swaps between the last helper (rung `cards - 1`) and the fallback. It needs `cards >= 2`: the boundary between the lead and the fallback stays the engine's adaptive cache, so each boundary has one publisher. `ladder_boundaries` never limits this boundary. |
| `max_hops_per_round` | `uint8_t` | 1 to 64 | 1 | How many adjacent boundaries one expert may climb in one round (see below). 0 is invalid. |
| `reserved` | `uint8_t[3]` | zero | n/a | Must be zero. |

### The fallback boundary

A swap at (last helper, fallback) is "load the fallback's expert into the helper and evict the helper's expert to the
fallback". It is the same same-layer swap as every other boundary: `promote` leaves the fallback for rung
`cards - 1`, `displace` goes the other way, and the outcome's `upper_rung + 1` is then the fallback rung. The
fallback is the engine's verified host copy (its RAM copy or the pack on NVMe), so a stage there reads verified bytes
whichever of the two holds them. A provider that never sees a handle opened with `fallback_boundary` 1 never sees
`rung == fallback` in `stage` or `replace`.

**The fallback provider rule.** A replace on the fallback is all-or-nothing, and on any status but ACK the fallback must
still serve every cell from verified bytes. The library relies on it: the fallback is where every closed card's cells
go, so it is never quarantined (its bit in `quarantined_rungs` is never set).

| Step on the fallback | Provider returns | Outcome | Rung closed | Rollback write |
|---|---|---|---|---|
| stage | `REFUSED` | `REJECTED` | none | none |
| stage | `FAILED`, `TIMEOUT` or unknown code, or a panic | `REJECTED`, logged at ERROR | none (a stage writes nothing) | none |
| replace | `REFUSED` | `RESTORED` if the rollback ACKs, else `QUARANTINED` | none, else the upper card | to the upper card |
| replace | `FAILED`, `TIMEOUT` or unknown code, or a panic | the same as `REFUSED`, plus an ERROR record naming the broken provider rule | none, else the upper card | to the upper card |

The rows for the helper in the same swap are those of the [fault table](#faults-and-their-outcomes) unchanged.

### Climbs

With `max_hops_per_round` above 1, the planner takes challengers hottest first across the whole table. A challenger
climbs bottom-up, one adjacent boundary at a time, each hop a plain same-layer swap with the coldest eligible incumbent
of the upper rung that passes `promote_margin` and the share cap, and stops at the first boundary where none does or
after `max_hops_per_round` hops. A displaced cell drops exactly one rung and moves once per round; a climber is never
displaced in its own round. Each hop is its own swap: it commits, is rejected, restored or quarantined by the table
above, and a climber that stops rests on its last committed rung, a valid placement. The climber's cooldown starts
after the round. The write budget counts every hop as 2 writes, so a climb of `n` hops needs
`max_moves_per_round >= 2n`; a smaller budget ends the climb early.

A plan may therefore name the same climber in several consecutive entries: entry `i + 1` has its `upper_rung` one
lower than entry `i`'s and promotes the same expert. Nothing else may repeat. `apply` checks the whole plan against a
scratch owner table before any I/O, and checks each hop again against the live map just before it runs. A hop whose
climber is not where the plan expected (an earlier hop of the same climb did not commit) gets the outcome
`SEABRAID_PLACEMENT_SKIPPED` (5): no provider call, `committed_epoch` 0, both issues zero. Hops after a quarantine
get no outcome at all, as for every other swap. The provider is asked for the climber's bytes once per climb: later
hops reuse the bytes the previous hop's acknowledged write put on the climber's new rung.

Epoch arithmetic for the new kinds: a committed hop adds 1 and a `SKIPPED` hop adds 0.

## By-reference moves: provider v2

`seabraid_placement_apply_v2` takes a `seabraid_placement_provider_v2`, a sized struct (`abi_version` =
`SEABRAID_PLACEMENT_PROVIDER_V2_VERSION`, `struct_size` = `sizeof` = 64; bytes past what the library knows must be zero).
`seabraid_placement_apply(p, t, v1, ...)` is exactly the v2 call with v1's fields, `from_fallback_rungs` 0 and
`replace_from_fallback` NULL, and every outcome, issue, epoch rule, ticket rule and the snapshot are unchanged. The two
new fields let an engine whose fallback holds the verified bytes in a file (Strata: the model file) have an upper rung
load the promoted expert itself instead of receiving a copy from the library.

| Field | Meaning |
|---|---|
| `from_fallback_rungs` | Bit r set: rung r may be asked to take the promoted expert by reference. Bits for the fallback rung and above must be zero; a non-zero mask needs `replace_from_fallback`. Either mistake is `SEABRAID_INVALID` before any I/O, and the ticket is kept. |
| `replace_from_fallback(context, rung, layer, old, new)` | Like `replace`, with no bytes: put `new` into the slot of `old`. `ACK`, `FAILED`, `TIMEOUT` and unknown codes mean what they mean for `replace`. |

**`REFUSED` here means "use bytes", not "failed".** No write began, nothing is logged, and the library stages the promoted
expert from the lower rung and calls `replace` on the same upper rung, inside the same swap. An engine may therefore
refuse for any reason it likes (the expert has a RAM copy, its checksum manifest is stale, a helper answered with a
wrong sum before) without costing the move.

When the library asks. For a swap whose upper rung is in the mask **and** for which the library holds no staged bytes of
the promoted expert. A climb's first hop leaves no bytes behind after a by-reference write, so the next hop is asked by
reference again if its rung is masked; the first unmasked rung stages the climber from the rung below, which now holds
it, and later hops reuse those bytes as before. The call is always made on the **upper** rung of the swap, never on the
fallback: the fallback is only the byte source, so it is never closed by this call and its own stage and replace rows
above are untouched.

The displaced expert is still staged from the upper rung before any write, because it is the recovery invariant of every
swap; by-reference moves remove the library's read and copy of the promoted expert, not its read of the displaced one.

| Step of a swap with a masked upper rung | Provider returns | Outcome | Rung closed | Rollback write |
|---|---|---|---|---|
| stage of the displaced expert on the upper rung | not `ACK` | the same as the fault table (`REJECTED`, or `QUARANTINED` on `TIMEOUT`) | as there | none |
| `replace_from_fallback` on the upper rung | `ACK` | continue with the replace on the lower rung (its rows are unchanged, the fallback's included) | none | none |
| `replace_from_fallback` on the upper rung | `REFUSED` | the byte path follows in this swap: stage on the lower rung, replace on the upper, replace on the lower, with the rows above | as there | as there |
| `replace_from_fallback` on the upper rung | `FAILED` | `RESTORED` if the rollback ACKs, else `QUARANTINED` | none, else the upper card | to the upper card, from the staged displaced expert |
| `replace_from_fallback` on the upper rung | `TIMEOUT`, unknown code, or a panic | `QUARANTINED` | the upper card | none (it may still be writing) |

A swap whose by-reference write committed is one `COMMITTED` outcome and one epoch step, like any swap.

## Limits

A layout holds 1 to `SEABRAID_PLACEMENT_MAX_CARDS` (63) cards: `cards` is also the
fallback rung, and the quarantine mask keeps one `u64` bit per rung. It also holds
1 to `SEABRAID_PLACEMENT_MAX_CELLS` (2^26) cells, where cells are
`layers * experts`. Anything outside either bound returns `SEABRAID_INVALID`
before any provider work. The Python `placement.fill` raises `ValueError` for
the same bounds.

## Module ownership

| Module | Owns | Enforcement |
|---|---|---|
| `src/placement.rs`, `src/placement/planner.rs`, `src/placement/planner/climb.rs` and `src/placement/transaction.rs` with `transaction/swap.rs` | Layer geometry, authoritative ownership, policy and transaction rules | Existing Rust placement fixtures |
| `src/ffi/placement.rs` and `src/ffi/placement/open.rs` | Handle, ticket, epoch and pointer preflight; both open entry points | Rust ABI tests and native smoke test |
| `src/ffi/placement/types.rs` | C layout and translation of committed/restored/quarantined outcomes | C11 and C++ fixture against the linked Rust library |
| `src/ffi/placement/provider.rs` | Call-scoped byte buffers and acknowledgement translation; provider v1 and v2 structs and the rung mask | Native fixture checks staged bytes, malformed size, rollback and timeout; `ffi/placement/tests/provider_v2.rs` |
| `src/ffi/placement/apply.rs` | Both apply entries over one body, so their refusals run in one order | Rust ABI tests and native smoke test |
| `src/placement/transaction/by_reference.rs` | The by-reference rows of the failure table | `src/placement/tests/by_reference.rs`, one test per row |

The FFI only converts types and delegates inward. Invalid geometry, stale
epochs, and invalid tickets fail before provider I/O. Core transactions resolve
provider failures without permitting dispatch against a half-applied swap.
Unsafe operations stay within the ABI. Each source file stays under the repository's 500-line limit
(`checks/check_size.py`).

## Verification

On Linux, `ctest --test-dir build/seabraid --output-on-failure` (the [README quickstart](../README.md#quickstart)) runs
`native-tests/placement_smoke.c` as C11 and, from the same file, as C++17 (`seabraid_placement_smoke_cxx`), so the "C11 and
C++17" claim has a checker. The provider v2 cases are in `native-tests/placement_smoke_v2.h`, which `placement_smoke.c`
includes. A hand-built link against `target/debug` needs the `libseabraid.so.0.5` symlink the SONAME asks for
(see [ABI.md](ABI.md) and the [user guide](USER-GUIDE.md#4-troubleshooting)).

In an MSVC developer shell, use `cl /TC /std:c11` and `cl /TP /std:c++17`
respectively, `/Iinclude`, and `target\debug\seabraid.dll.lib`. Place the test
executables alongside `target\debug\seabraid.dll` so the loader resolves it.
