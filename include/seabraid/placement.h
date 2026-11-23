/* Copyright 2026 Saltwater Research LLC
 * SPDX-License-Identifier: Apache-2.0 OR MIT */
#ifndef SEABRAID_PLACEMENT_H
#define SEABRAID_PLACEMENT_H

#include "seabraid.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct seabraid_placement seabraid_placement;

/* `cards` is the card count and also the fallback rung, so valid cards are 1..=63. WHY 63: the quarantine mask is
 * one u64 bit per rung. A larger value returns SEABRAID_INVALID. */
#define SEABRAID_PLACEMENT_MAX_CARDS 63u
/* layers * experts must be 1..=this. WHY: Strata's plausibility bound; it also keeps indexing safe on 32-bit hosts. */
#define SEABRAID_PLACEMENT_MAX_CELLS (1u << 26)
/* Each layer_slot_bytes entry passed to seabraid_placement_apply must be 1..=this (2^32 - 1). WHY: the table is caller
 * memory the library cannot check, and each staged slot is an allocation of that size; a larger entry is a corrupt
 * table and apply returns SEABRAID_INVALID before any I/O. */
#define SEABRAID_PLACEMENT_MAX_SLOT_BYTES 4294967295u

/* Owners and heat use index layer * experts + expert. Rung 0 is the lead,
 * 1 through cards - 1 are helpers, and cards is the verified fallback. */
typedef struct seabraid_placement_layout_v1 {
    uint32_t abi_version;
    uint32_t layers;
    uint32_t experts;
    uint8_t cards;
    const uint8_t *owners;
    size_t owner_count;
} seabraid_placement_layout_v1;

typedef struct seabraid_placement_fill_v1 {
    uint32_t abi_version;
    uint32_t layers;
    uint32_t experts;
    uint8_t cards;              /* rungs 0..cards-1 are cards; rung `cards` is the verified fallback */
    const uint64_t *slots;      /* expert slots per card, `cards` entries; rung 0 is the lead. Each entry counts cells
                                 * (layer, expert pairs) across ALL layers together, not per layer: the hottest `slots[0]`
                                 * cells of the whole table go to the lead, whatever layer they sit in. */
    size_t slot_count;
    const double *heat;         /* layers * experts, layer-major, finite and >= 0, each layer's sum finite */
    size_t heat_count;
} seabraid_placement_fill_v1;
/* Hottest-first spillover: the lead card fills first, then each helper in rung order; every remaining expert goes to
 * the fallback rung. Ties break by lower layer, then lower expert. Writes layers * experts owners, ready for
 * seabraid_placement_layout_v1.owners. Pure: no provider work and no state. */
int32_t seabraid_placement_fill(const seabraid_placement_fill_v1 *request, uint8_t *owners, size_t owner_count);

/* Planner policy. A NULL policy to seabraid_placement_open uses the defaults below; an explicit struct is taken as
 * written (a zero field is a zero, never "use the default"). Any value outside its range makes open return
 * SEABRAID_INVALID. There is no way to change the policy of an open handle: close it and open another (see the epoch
 * note at seabraid_placement_open).
 *   field                   valid range            default  unit / meaning
 *   max_moves_per_round     0..=128                4        expert WRITES, not swaps: one swap is 2 writes, so the
 *                                                           plan holds at most max_moves_per_round / 2 swaps (an odd
 *                                                           value rounds down; 0 or 1 never plans a swap)
 *   ladder_boundaries       0..=255                0        0 plans every boundary between adjacent cards (rung h
 *                                                           with h + 1, both cards); n > 0 plans only the top n of
 *                                                           those boundaries. The fallback is never swapped live
 *                                                           unless seabraid_placement_open_v2 enables it.
 *   cooldown_rounds         0..=2^64-1             8        plan rounds (successful seabraid_placement_plan calls on
 *                                                           this handle). A cell that committed a move in round R
 *                                                           is not a candidate again until round R + cooldown + 1
 *   promote_margin          finite, >= 1.0         1.25     a challenger must have MORE than margin times the
 *                                                           smoothed heat of the cell it displaces
 *   max_helper_layer_share  0.0..=1.0, not NaN     0.08     fraction of one layer's smoothed heat that a helper
 *                                                           (rung >= 1) may hold in that layer after a swap; rung 0
 *                                                           is not capped, so it matters only with 3 or more cards
 * Details and the smoothing rule are in docs/PLACEMENT-ABI.md. */
typedef struct seabraid_placement_policy_v1 {
    uint32_t abi_version;
    uint16_t max_moves_per_round;
    uint8_t ladder_boundaries;
    uint64_t cooldown_rounds;
    double promote_margin;
    double max_helper_layer_share;
} seabraid_placement_policy_v1;

/* Calls on each handle must be serialized. A closed or foreign handle, or a seabraid_session passed as a placement, returns
 * SEABRAID_INVALID and a call that
 * overlaps another on the same handle (another thread, or the provider callback calling back into its own
 * placement) returns SEABRAID_BUSY_OR_STATE and does nothing. This is best effort: it reads the memory behind the
 * handle, so a double close or a garbage pointer is still undefined behavior when the block was freed and reused or
 * is not readable. All pointers must refer to valid,
 * suitably aligned, non-overlapping storage for the duration of the call.
 * The layout and optional policy are copied. A null policy uses Rust defaults.
 * Initial ownership is verified by the caller, and starts at epoch 1. On failure *out is NULL.
 * Reopening starts at epoch 1 again, and the placement handle is not linked to any session. A session that already
 * accepted a higher placement_epoch (seabraid_batch_v2) refuses every batch routed with the new handle's lower epochs
 * as stale. Either reopen the sessions too, or give batches an engine-owned epoch such as
 * (generation << 32) | snapshot.epoch; see docs/PLACEMENT-ABI.md.
 * A panic caught inside any call poisons the handle: every later call except close returns SEABRAID_INTERNAL
 * ("handle poisoned") and does nothing. A C++ exception escaping a provider callback is undefined behavior
 * and aborts the process, so providers must catch everything and return an io status. */
int32_t seabraid_placement_open(const seabraid_placement_layout_v1 *layout,
                            const seabraid_placement_policy_v1 *optional_policy,
                            seabraid_placement **out);
int32_t seabraid_placement_close(seabraid_placement *placement);

#define SEABRAID_PLACEMENT_POLICY_V2_VERSION 2u

/* Caller-filled, versioned like seabraid_batch_v2: abi_version = SEABRAID_PLACEMENT_POLICY_V2_VERSION and
 * struct_size = sizeof; bytes past what the library knows must be zero. Fields shared with v1 mean what they mean in v1.
 * fallback_boundary: 0 is off (v1 behaviour); 1 plans live swaps between the last helper (rung cards - 1) and the
 *   fallback, so the fallback may be the lower rung of an outcome. It needs cards >= 2 (the boundary between the lead
 *   and the fallback stays the engine's own adaptive cache). Any other value is INVALID.
 * max_hops_per_round: 1 plans each boundary on its own (v1 behaviour); 2..=64 lets one expert climb that many adjacent
 *   boundaries in one round, as a chain of ordinary swaps, each committed or rolled back on its own. 0 is INVALID.
 *   A climb of n hops is n swaps, so max_moves_per_round must be at least 2 * n to let it finish.
 * ladder_boundaries limits card boundaries only; it never limits the fallback boundary.
 * reserved must be zero. */
typedef struct seabraid_placement_policy_v2 {
    uint32_t abi_version;
    uint32_t struct_size;            /* >= 40 */
    uint16_t max_moves_per_round;
    uint8_t ladder_boundaries;
    uint8_t fallback_boundary;
    uint8_t max_hops_per_round;
    uint8_t reserved[3];
    uint64_t cooldown_rounds;
    double promote_margin;
    double max_helper_layer_share;
} seabraid_placement_policy_v2;       /* 40 bytes */

/* Same as seabraid_placement_open; a NULL policy equals the v1 defaults with fallback_boundary 0 and
 * max_hops_per_round 1. seabraid_placement_open(layout, p1, out) is exactly this call with {p1's fields, 0, 1}.
 * A provider that never sees a handle opened here never sees rung == fallback in stage or replace.
 * Fallback provider rule: a replace on the fallback is all-or-nothing, and on any status but ACK the fallback must
 * still serve every cell from verified bytes (its RAM copy or the pack). The library never closes the fallback: a stage
 * on it that does not ACK is REJECTED, and a replace on it that does not ACK restores the upper card (RESTORED), or
 * quarantines that card only when its restore fails too. A non-REFUSED status from the fallback is logged at ERROR. */
int32_t seabraid_placement_open_v2(const seabraid_placement_layout_v1 *layout,
                                   const seabraid_placement_policy_v2 *optional_policy,
                                   seabraid_placement **out);

/* Library-filled (see the versioning rule in seabraid.h): set abi_version and struct_size before the call. */
typedef struct seabraid_placement_ticket_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint64_t ticket;
    uint64_t epoch;
    size_t promotion_count;
} seabraid_placement_ticket_v1;

/* The newest successful plan supersedes the previous ticket. A ticket only
 * identifies a plan on its originating handle. Heat describes the stated epoch.
 * No provider work occurs during planning.
 * The planner is not stateless per window. The first successful plan on a handle uses the raw heat it is given.
 * Every later successful plan blends it into a running per-cell value, 0.75 * previous + 0.25 * this window, whether
 * or not the earlier ticket was applied, whatever the epoch, and even when a plan has zero promotions. Neither apply
 * nor an epoch change resets it; only a new handle does. A plan refused for a stale epoch or invalid heat changes
 * nothing. promote_margin compares those blended values. */
int32_t seabraid_placement_plan(seabraid_placement *placement, uint64_t expected_epoch,
                            const double *heat, size_t heat_count,
                            seabraid_placement_ticket_v1 *out);

enum seabraid_placement_io_status {
    SEABRAID_PLACEMENT_IO_ACK = 0,
    SEABRAID_PLACEMENT_IO_REFUSED = 1,
    SEABRAID_PLACEMENT_IO_FAILED = 2,
    SEABRAID_PLACEMENT_IO_TIMEOUT = 3
};

/* Callbacks run synchronously on the applying thread with admission fenced and
 * provider queues drained by the engine. Context and layer_slot_bytes must stay
 * valid for apply, and the size table stays immutable. Each layer_slot_bytes entry must be
 * 1..=SEABRAID_PLACEMENT_MAX_SLOT_BYTES: a larger one is a corrupt table and apply returns SEABRAID_INVALID before any
 * I/O, keeping the ticket. The library
 * allocates one zero-filled buffer of that size per staged slot (two per swap) without touching its pages, so only
 * bytes the stage callback writes are committed. Both supplied buffers
 * and bytes_written expire at callback return, including on timeout. Complete
 * any accesses to these buffers before returning. Do not reenter the placement
 * handle, throw across the ABI, or retain pointers.
 *
 * stage ACK may be returned after copying the entire verified slot and setting
 * bytes_written. replace ACK confirms the bytes and the new residency index.
 * REFUSED guarantees no write began. FAILED may have torn bytes but preserves
 * the old index. TIMEOUT and unknown codes mean device completion is uncertain.
 * old_expert can equal new_expert during restoration. */
typedef struct seabraid_placement_provider_v1 {
    uint32_t abi_version;
    void *context;
    const size_t *layer_slot_bytes;
    size_t layer_count;
    int32_t (*stage)(void *context, uint8_t rung, uint32_t layer, uint32_t expert,
                     uint8_t *output, size_t output_bytes, size_t *bytes_written);
    int32_t (*replace)(void *context, uint8_t rung, uint32_t layer, uint32_t old_expert,
                       uint32_t new_expert, const uint8_t *input, size_t input_bytes);
} seabraid_placement_provider_v1;

#define SEABRAID_PLACEMENT_PROVIDER_V2_VERSION 2u

/* Caller-filled: abi_version = SEABRAID_PLACEMENT_PROVIDER_V2_VERSION, struct_size = sizeof; bytes past what the library
 * knows must be zero. Fields shared with v1 mean what they mean in v1.
 * from_fallback_rungs: bit r set = rung r's replace may take the promoted expert's bytes from the fallback's own
 *   verified store (for Strata: the model file) instead of from the library. Bits for the fallback rung and for rungs
 *   >= cards must be zero; a non-zero mask needs replace_from_fallback, else apply_v2 returns SEABRAID_INVALID before I/O.
 * replace_from_fallback: like replace, but the library supplies no bytes. ACK, FAILED and TIMEOUT mean what they mean for
 *   replace. REFUSED means "not this way": no write began, and the library then stages the promoted expert from the
 *   lower rung and calls replace, inside the same swap. A REFUSED here is not a failure and is not logged as one.
 * The library asks a masked rung only when it holds no staged bytes for the climber (a later hop of a climb reuses the
 * bytes the previous hop wrote), and always on the upper rung of the swap, never on the fallback, so the fallback is
 * never closed by it. The displaced expert is still staged from the upper rung before any write. */
typedef struct seabraid_placement_provider_v2 {
    uint32_t abi_version;
    uint32_t struct_size;
    void *context;
    const size_t *layer_slot_bytes;
    size_t layer_count;
    int32_t (*stage)(void *context, uint8_t rung, uint32_t layer, uint32_t expert,
                     uint8_t *output, size_t output_bytes, size_t *bytes_written);
    int32_t (*replace)(void *context, uint8_t rung, uint32_t layer, uint32_t old_expert,
                       uint32_t new_expert, const uint8_t *input, size_t input_bytes);
    uint64_t from_fallback_rungs;
    int32_t (*replace_from_fallback)(void *context, uint8_t rung, uint32_t layer,
                                     uint32_t old_expert, uint32_t new_expert);
} seabraid_placement_provider_v2;     /* 64 bytes */

enum seabraid_placement_outcome_kind {
    SEABRAID_PLACEMENT_COMMITTED = 1,
    SEABRAID_PLACEMENT_REJECTED = 2,
    SEABRAID_PLACEMENT_RESTORED = 3,
    SEABRAID_PLACEMENT_QUARANTINED = 4,
    /* A later hop of a climb that was not attempted because an earlier hop of the same expert did not commit. No
     * provider call, epoch unchanged, both issues zero. Only a handle opened with max_hops_per_round > 1 can produce
     * it. Hops after a quarantine get no outcome at all, as for every other swap. */
    SEABRAID_PLACEMENT_SKIPPED = 5
};

enum seabraid_placement_io_phase {
    SEABRAID_PLACEMENT_STAGE = 1,
    SEABRAID_PLACEMENT_REPLACE = 2,
    SEABRAID_PLACEMENT_RESTORE = 3
};

enum seabraid_placement_issue_error {
    SEABRAID_PLACEMENT_REFUSED = 1,
    SEABRAID_PLACEMENT_FAILED = 2,
    SEABRAID_PLACEMENT_UNACKNOWLEDGED = 3,
    SEABRAID_PLACEMENT_PANICKED = 4,
    SEABRAID_PLACEMENT_MALFORMED_BLOB = 5
};

typedef struct seabraid_placement_issue_v1 {
    uint32_t phase;
    uint32_t error;
    uint8_t rung;
} seabraid_placement_issue_v1;

/* Library-filled. Embedded issues carry no header of their own: the outcome's abi_version covers them. In an
 * array the element stride is struct_size, read from element 0, which the caller must initialize like a single
 * struct (a multiple of 8); the library then writes every element. As for every library-filled struct, each
 * element's struct_size is overwritten with min(stride, 64), the bytes written, and the bytes past that are not
 * touched. When the stride is wider than the library's 64 bytes the call therefore leaves 64 there: set element 0's
 * struct_size back to the stride before passing the same array to another apply, or the next call strides by 64.
 * Padding bytes inside each element are zero.
 * One outcome describes one swap inside `layer`: expert `promote` moves up from rung upper_rung + 1 to rung
 * upper_rung, and expert `displace` moves down from upper_rung to upper_rung + 1 (swaps are only ever between
 * adjacent rungs, so the lower rung is always upper_rung + 1, which is the fallback only on a handle opened with
 * fallback_boundary = 1). `committed_epoch` is the placement epoch after
 * this swap and is zero for every other kind. `initial` is the first failure and `restore` the failure of the
 * rollback write; `rung` in an issue is the rung whose callback failed. The library does not let you read the owner
 * table back (the snapshot gives executors, and a closed rung's cells read as the fallback), so the engine keeps its
 * own owners and applies COMMITTED outcomes to them. */
typedef struct seabraid_placement_outcome_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint64_t committed_epoch;
    uint32_t kind;
    uint32_t layer;
    uint32_t promote;
    uint32_t displace;
    seabraid_placement_issue_v1 initial;
    seabraid_placement_issue_v1 restore;
    uint8_t upper_rung;
} seabraid_placement_outcome_v1;

/* Library-filled: set abi_version and struct_size before the call. */
typedef struct seabraid_placement_applied_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint64_t epoch;
    uint64_t quarantined_rungs;
    size_t outcome_count;
} seabraid_placement_applied_v1;

/* Supply capacity for the entire plan. `outcome_count` covers attempted pairs
 * in plan order, including a quarantine that may stop the round early (later pairs get no outcome). The epoch
 * grows by 1 per COMMITTED pair and by 1 per quarantine event (one event may close two rungs); a rejection or a
 * restoration leaves it unchanged. The status table is in docs/PLACEMENT-ABI.md. A zero
 * error in either issue denotes absence. `committed_epoch` is zero except on
 * commit. Provider outcomes accompany SEABRAID_OK, including rejection, rollback,
 * and quarantine. A stale or unknown ticket yields SEABRAID_BUSY_OR_STATE before
 * I/O. A stale matching ticket is discarded. Invalid buffer/provider geometry
 * leaves the current ticket available to retry. A transaction consumes it.
 * outcomes may be null for an empty plan. */
int32_t seabraid_placement_apply(seabraid_placement *placement, uint64_t ticket,
                             const seabraid_placement_provider_v1 *provider,
                             seabraid_placement_outcome_v1 *outcomes, size_t outcome_capacity,
                             seabraid_placement_applied_v1 *out);

/* Same as seabraid_placement_apply; seabraid_placement_apply(p, t, v1, ...) is exactly this call with v1's fields,
 * from_fallback_rungs 0 and replace_from_fallback NULL. Works on handles from open and open_v2. Outcomes, epochs,
 * tickets and the snapshot are unchanged: a swap whose upper write went by reference and committed is one COMMITTED
 * outcome and one epoch step. A bad struct_size, a non-zero byte past the library's struct, a bad mask or a NULL
 * replace_from_fallback under a non-zero mask is SEABRAID_INVALID before any I/O, and keeps the ticket. */
int32_t seabraid_placement_apply_v2(seabraid_placement *placement, uint64_t ticket,
                                    const seabraid_placement_provider_v2 *provider,
                                    seabraid_placement_outcome_v1 *outcomes, size_t outcome_capacity,
                                    seabraid_placement_applied_v1 *out);

/* Library-filled: set abi_version and struct_size before the call. */
typedef struct seabraid_placement_snapshot_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint64_t epoch;
    uint64_t quarantined_rungs;
    uint32_t layers;
    uint32_t experts;
    uint8_t fallback_rung;
} seabraid_placement_snapshot_v1;

/* Export exactly layers * experts current executors. A closed rung's cells
 * resolve to the fallback. Publish this table at the same engine fence as the
 * placement change. Its epoch identifies the routing decision for each batch. */
int32_t seabraid_placement_snapshot(seabraid_placement *placement, uint8_t *executors,
                                size_t executor_count, seabraid_placement_snapshot_v1 *out);

enum seabraid_placement_update_mode {
    SEABRAID_PLACEMENT_ADOPT = 0,
    SEABRAID_PLACEMENT_REBUILD = 1
};

/* These calls require a fenced engine. Quarantine records an out-of-band loss.
 * Adopt reports a completed and verified external move while preserving closed
 * rungs. Rebuild reopens rungs only after all providers are drained or replaced
 * and the full layout and bytes are verified. Rank bindings stay fixed for the
 * handle. A stale epoch yields SEABRAID_BUSY_OR_STATE.
 * Epoch arithmetic: quarantine of a rung that is not yet closed adds 1 (a repeat is a no-op, epoch unchanged); adopt
 * adds 1 when the owners differ from the current ones and nothing when they match; rebuild always adds 1. */
int32_t seabraid_placement_quarantine(seabraid_placement *placement, uint64_t expected_epoch,
                                  uint8_t rung);
int32_t seabraid_placement_update(seabraid_placement *placement, uint64_t expected_epoch,
                              uint32_t mode, const uint8_t *owners, size_t owner_count);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* SEABRAID_PLACEMENT_H */
