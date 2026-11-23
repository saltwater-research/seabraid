// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! What one pick, and one row of picks, may be: the single owner of the routing rules `Batch::check` enforces.
//!
//! OWNER: the pick rule (range, skip, weight, no repeated expert in a row) and `SKIP_EXPERT_ID`, the one marker a
//! pick may carry instead of an expert; nothing else decides either. `batch.rs` only calls `check`; `rows.rs` reads
//! the marker from here, so the dependency edges run batch -> picks and rows -> picks and never back.
//! CONTRACT: `check` over a batch's `(expert_ids, weights)` once geometry is proven; knows no format or session.
//! SIZE: under 200 lines. FAILURE: fail-fast `Code::Invalid`, one `fail!` per refusal; the one scratch allocation is
//! `try_reserve`d and reported as `Code::Internal`.
//! CHECKER: src/picks/tests/golden.rs (the table `golden.txt`, shared with the ggml adapter and the Python reference
//! provider), src/picks/tests.rs (a seeded differential test: the wide path names exactly the pick and text the
//! narrow scan names, for batches with several defects of any kind), src/batch/tests.rs and src/batch/tests/edge.rs.

use crate::diag::{fail, Code};
use crate::{Error, Result};

/// The expert id of a skipped pick (the C header's `SEABRAID_SKIP_EXPERT_ID`; src/header_parity.rs holds them equal).
pub(crate) const SKIP_EXPERT_ID: i32 = -1;

/// Checks every pick of a batch whose counts are already proven: `ids` and `weights` are whole rows of `top_k`.
pub(crate) fn check((ids, weights): (&[i32], &[f32]), top_k: usize, rule: PickRule) -> Result<()> {
    let wide = top_k > NARROW_ROW;
    let picks = ids.iter().zip(weights).enumerate();
    let bad_pick = picks.into_iter().find_map(|(index, (&id, &weight))| {
        // The picks of this row before this one. A wide row is checked for repeats afterwards, so it passes none.
        let earlier = if wide {
            &[][..]
        } else {
            &ids[index - index % top_k..index]
        };
        pick_error(index, (id, weight), earlier, rule).map(|error| (index, error))
    });
    let repeat = if wide {
        wide_row_repeat(ids, top_k)?
    } else {
        None
    };
    // WHY the lower index wins, and a tie goes to `bad_pick`: the narrow scan reports the first defect of any kind
    // in pick order, and at one pick it tests range and weight before the repeat; both paths must name the same pick.
    match (bad_pick, repeat) {
        (Some((at, _)), Some((repeat_at, error))) if repeat_at < at => Err(error),
        (Some((_, error)), _) | (None, Some((_, error))) => Err(error),
        (None, None) => Ok(()),
    }
}

#[cfg(test)]
mod tests;

/// Rows the sort path has checked on this thread (tests only).
#[cfg(test)]
pub(crate) fn wide_rows_sorted() -> usize {
    WIDE_ROWS.with(|rows| rows.get())
}

#[cfg(test)]
thread_local! {
    /// Rows `wide_row_repeat` has sorted on this thread. WHY: the narrow and the wide path refuse the same text, so
    /// only a counter can show a test which path ran; a thread-local keeps parallel tests apart.
    static WIDE_ROWS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// The provider-wide limits one pick is checked against.
#[derive(Clone, Copy)]
pub(crate) struct PickRule {
    pub(crate) experts: u32,
    pub(crate) partial: bool,
}

/// The one owner of what a single pick may be: in range (or a skip with zero weight on a partial-routes
/// provider), a finite weight, and an expert id that does not repeat within its row. WHY the repeat is refused
/// here and not in each adapter: the ABI never said what two picks of one expert in a row mean (sum the
/// weights? run twice?) and the adapters disagreed, so one refusal makes "distinct ids per row" a contract
/// every provider can rely on. `earlier` holds the ids of the row's preceding picks (top_k is small, so the
/// scan stays cheap and allocation free).
fn pick_error(
    index: usize,
    (id, weight): (i32, f32),
    earlier: &[i32],
    rule: PickRule,
) -> Option<Error> {
    let PickRule { experts, partial } = rule;
    if id == SKIP_EXPERT_ID && partial {
        return (weight != 0.0).then(|| {
            fail!(
                Code::Invalid,
                "pick {}: skipped expert has a nonzero weight",
                index
            )
        });
    }
    if id < 0 || id as u32 >= experts {
        // WHY two hints: the skips hint is true only for the skip id sent to a provider that lacks PARTIAL_ROUTES;
        // told to any other bad id (or to a provider that has the flag) it sends the adapter author after a
        // capability they already declared.
        let hint = if id == SKIP_EXPERT_ID {
            "skips need a partial-routes provider"
        } else {
            "-1 is the skip marker on a partial-routes provider; any other id must be in range"
        };
        return Some(fail!(
            Code::Invalid,
            "pick {}: expert id {} outside 0..{} ({})",
            index,
            id,
            experts,
            hint
        ));
    }
    if !weight.is_finite() {
        return Some(fail!(Code::Invalid, "pick {}: weight is not finite", index));
    }
    earlier.contains(&id).then(|| {
        fail!(
            Code::Invalid,
            "pick {}: expert id {} repeats an earlier pick of the same row",
            index,
            id
        )
    })
}

/// Rows wider than this are checked for repeats by sorting. WHY: scanning the earlier picks of the row costs
/// O(top_k^2) (one 64,000-pick row took 12.7 s unoptimized), and no configuration bounds top_k below `experts`.
const NARROW_ROW: usize = 64;

/// The repeat rule of `pick_error` for rows too wide to scan, in O(top_k log top_k) per row: the earliest pick of the
/// batch that repeats an earlier pick of its row (its index, and the refusal named exactly as the scan names it).
/// Skips are never repeats. Ids need not be in range here: an out-of-range id is also `bad_pick`'s, at the same or
/// a lower index, so `check` never reports a repeat that the scan would have named differently.
fn wide_row_repeat(ids: &[i32], top_k: usize) -> Result<Option<(usize, Error)>> {
    let mut sorted: Vec<(i32, usize)> = Vec::new();
    sorted.try_reserve_exact(top_k).map_err(|source| {
        fail!(
            Code::Internal,
            "cannot reserve {} picks to check ({})",
            top_k,
            source
        )
    })?;
    for (row, picks) in ids.chunks_exact(top_k).enumerate() {
        #[cfg(test)]
        WIDE_ROWS.with(|rows| rows.set(rows.get() + 1));
        sorted.clear();
        let real = picks
            .iter()
            .enumerate()
            .filter(|&(_, &id)| id != SKIP_EXPERT_ID);
        sorted.extend(real.map(|(at, &id)| (id, at)));
        sorted.sort_unstable();
        let earliest = sorted
            .windows(2)
            .filter(|pair| pair[0].0 == pair[1].0)
            .map(|pair| pair[1].1)
            .min();
        if let Some(at) = earliest {
            let index = row * top_k + at;
            let error = fail!(
                Code::Invalid,
                "pick {}: expert id {} repeats an earlier pick of the same row",
                index,
                picks[at]
            );
            return Ok(Some((index, error)));
        }
    }
    Ok(None)
}
