// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! Which rows of a batch a provider sees: derive them from the skip markers, gather them, scatter results back.
//!
//! OWNER: the row selection rule for COMPACT_ROWS (include/seabraid/seabraid.h states it for callers). A row belongs
//! to the provider exactly when one of its `top_k` picks is not a skip; nothing else decides it, and no caller
//! supplies a list.
//! CONTRACT: `RowMap::derive` after the picks were validated, `gather` per per-row array, `scatter` on collect.
//! Knows nothing of formats, tickets or the host; `batch` calls it, `session::worker` calls `scatter`.
//! SIZE: under 200 lines with tests.
//! FAILURE: fail-fast `Code::Invalid` / `Code::Internal` through `fail!`; storage is reserved with `try_reserve`.

use crate::diag::{fail, Code};
use crate::picks::SKIP_EXPERT_ID;
use crate::Result;

/// The kept rows of one submission, in their original order. Storage is reused across submissions.
#[derive(Debug, Default)]
pub(crate) struct RowMap {
    kept: Vec<u32>,
}

impl RowMap {
    /// Keeps every row with at least one real pick. `expert_ids` is `tokens * top_k` long and already validated.
    pub(crate) fn derive(&mut self, expert_ids: &[i32], top_k: usize) -> Result<()> {
        if top_k == 0 || expert_ids.len() % top_k != 0 {
            return Err(fail!(
                Code::Internal,
                "{} expert ids are not whole rows of {} picks",
                expert_ids.len(),
                top_k
            ));
        }
        self.kept.clear();
        self.kept
            .try_reserve_exact(expert_ids.len() / top_k)
            .map_err(|source| fail!(Code::Internal, "cannot reserve the row map ({})", source))?;
        for (row, picks) in expert_ids.chunks_exact(top_k).enumerate() {
            if picks.iter().any(|&id| id != SKIP_EXPERT_ID) {
                self.kept.push(row as u32);
            }
        }
        Ok(())
    }

    /// Kept row count; 0 means the provider is not called at all.
    pub(crate) fn kept(&self) -> u32 {
        self.kept.len() as u32
    }

    /// Original row numbers, ascending.
    pub(crate) fn rows(&self) -> &[u32] {
        &self.kept
    }

    /// Copies the kept rows of `src` (`width` elements per row) into `dst`, replacing its contents.
    pub(crate) fn gather<T: Copy>(&self, src: &[T], width: usize, dst: &mut Vec<T>) -> Result<()> {
        dst.clear();
        dst.try_reserve_exact(self.kept.len() * width)
            .map_err(|source| fail!(Code::Internal, "cannot reserve gathered rows ({})", source))?;
        for &row in &self.kept {
            dst.extend_from_slice(&src[row as usize * width..][..width]);
        }
        Ok(())
    }

    /// Writes the provider's `kept * width` results into their original rows of `full`; other rows are untouched,
    /// so a zeroed `full` leaves zeros exactly in the skipped rows.
    pub(crate) fn scatter(&self, compact: &[f32], width: usize, full: &mut [f32]) -> Result<()> {
        let last_end = self
            .kept
            .last()
            .map_or(0, |&row| (row as usize + 1) * width);
        if compact.len() != self.kept.len() * width || full.len() < last_end {
            return Err(fail!(
                Code::Internal,
                "scatter of {} results into {} outputs does not fit {} kept rows of {}",
                compact.len(),
                full.len(),
                self.kept.len(),
                width
            ));
        }
        for (result, &row) in compact.chunks_exact(width).zip(&self.kept) {
            full[row as usize * width..][..width].copy_from_slice(result);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn derived(ids: &[i32], top_k: usize) -> RowMap {
        let mut map = RowMap::default();
        map.derive(ids, top_k).unwrap();
        map
    }

    #[test]
    fn no_skips_keeps_every_row() {
        let map = derived(&[0, 1, 2, 0, 1, 2], 2);
        assert_eq!(map.rows(), [0, 1, 2]);
        assert_eq!(map.kept(), 3);
    }

    #[test]
    fn all_skipped_keeps_nothing_and_scatter_leaves_zeros() {
        let map = derived(&[-1, -1, -1, -1], 2);
        assert_eq!(map.kept(), 0);
        let mut full = [0.0_f32; 4];
        map.scatter(&[], 2, &mut full).unwrap();
        assert_eq!(full, [0.0; 4]);
    }

    #[test]
    fn alternating_rows_gather_and_scatter_round_trip() {
        // Rows 0 and 2 are skipped entirely; row 1 has one real pick and one skip, which still keeps it.
        let ids = [-1, -1, 3, -1, -1, -1, 0, 1];
        let map = derived(&ids, 2);
        assert_eq!(map.rows(), [1, 3]);
        let mut gathered = Vec::new();
        map.gather(
            &[10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
            2,
            &mut gathered,
        )
        .unwrap();
        assert_eq!(gathered, [12.0, 13.0, 16.0, 17.0]);
        let mut full = [0.0_f32; 8];
        map.scatter(&[1.0, 2.0, 3.0, 4.0], 2, &mut full).unwrap();
        assert_eq!(full, [0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 3.0, 4.0]);
    }

    #[test]
    fn a_mismatched_scatter_is_refused_not_silently_truncated() {
        let map = derived(&[0, 0], 1);
        let mut full = [0.0_f32; 2];
        let error = map.scatter(&[1.0], 1, &mut full).unwrap_err();
        assert_eq!(error.code(), Code::Internal);
    }
}
