// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! Hottest-first spillover fill: the initial layout an engine hands to `OwnershipMap::new`.
//!
//! * OWNER: this file is the only definition of the fill order; the C ABI forwards to it.
//! * CONTRACT: `fill(layers, experts, cards, slots, heat)` is pure. The lead card (rung 0)
//!   takes its `slots[0]` hottest cells, each helper the next hottest in rung order, and
//!   every remaining cell goes to the fallback rung `cards`. Ties break by lower layer, then
//!   lower expert, so equal heat is deterministic. Shape bounds are those of `layout_size`.
//! * SIZE BUDGET: at most 200 lines including tests.
//! * FAILURE CONTRACT: fail-fast, `InvalidLayout`, `InvalidHeat` or `HeatOverflow` (layer sums, as the planner), before the
//!   result is allocated (the only earlier allocation is `layer_totals`' one f64 per layer); no partial output exists.
//! * CHECKER: the unit tests below (exact assignment, ties, slack, zero slots, bad input, and
//!   that the result opens an `OwnershipMap`) plus `native-tests/placement_smoke.c` over the C ABI.
//! * FINISH LINE: `seabraid_placement_fill` output is accepted unchanged by `seabraid_placement_open`.

use super::heat::layer_totals;
use super::{layout_size, PlacementError, Rung};
use std::cmp::Ordering;

pub fn fill(
    layers: u32,
    experts: u32,
    cards: Rung,
    slots: &[u64],
    heat: &[f64],
) -> Result<Vec<Rung>, PlacementError> {
    let cells = layout_size(layers, experts, cards)?;
    if slots.len() != usize::from(cards) {
        return Err(PlacementError::InvalidLayout);
    }
    if heat.len() != cells || heat.iter().any(|value| !value.is_finite() || *value < 0.0) {
        return Err(PlacementError::InvalidHeat);
    }
    layer_totals(heat, experts as usize)?; // the planner's rule; `layout_size` bounded experts >= 1
    let capacity = slots.iter().fold(0usize, |sum, &slot| {
        sum.saturating_add(usize::try_from(slot).unwrap_or(usize::MAX))
    });
    let mut owners = vec![cards; cells];
    let hottest = hottest_cells(heat, capacity.min(cells));
    let mut next = hottest.into_iter();
    for (card, &slot) in slots.iter().enumerate() {
        let take = usize::try_from(slot).unwrap_or(usize::MAX);
        for index in next.by_ref().take(take) {
            owners[index as usize] = card as Rung;
        }
    }
    Ok(owners)
}

/// The `count` hottest cell indices, hottest first. Heat is finite (checked by the caller),
/// so `partial_cmp` cannot fail; the index tie-break makes the order total and deterministic.
fn hottest_cells(heat: &[f64], count: usize) -> Vec<u32> {
    if count == 0 {
        return Vec::new();
    }
    // Cells are bounded by 2^26 in `layout_size`, so indices fit u32 and halve the memory.
    let mut order: Vec<u32> = (0..heat.len() as u32).collect();
    let hotter = |a: &u32, b: &u32| {
        heat[*b as usize]
            .partial_cmp(&heat[*a as usize])
            .unwrap_or(Ordering::Equal)
            .then(a.cmp(b))
    };
    if count < order.len() {
        // Selecting first keeps a small lead card from paying for a full sort of every cell.
        order.select_nth_unstable_by(count - 1, hotter);
        order.truncate(count);
    }
    order.sort_unstable_by(hotter);
    order
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::placement::OwnershipMap;

    #[test]
    fn hottest_cells_fill_the_lead_then_the_helpers_then_the_fallback() {
        let owners = fill(1, 5, 2, &[1, 2], &[1.0, 5.0, 3.0, 2.0, 0.0]).unwrap();
        assert_eq!(owners, [2, 0, 1, 1, 2]);
    }

    #[test]
    fn ties_break_by_lower_layer_then_lower_expert() {
        let owners = fill(2, 2, 1, &[3], &[1.0; 4]).unwrap();
        assert_eq!(owners, [0, 0, 0, 1]);
        let owners = fill(2, 2, 1, &[2], &[0.0, 7.0, 7.0, 0.0]).unwrap();
        assert_eq!(owners, [1, 0, 0, 1]);
    }

    #[test]
    fn slots_beyond_the_cells_leave_nothing_for_the_fallback() {
        assert_eq!(fill(1, 4, 2, &[100, 100], &[1.0; 4]).unwrap(), [0; 4]);
        assert_eq!(
            fill(1, 2, 2, &[u64::MAX, u64::MAX], &[1.0; 2]).unwrap(),
            [0; 2]
        );
    }

    #[test]
    fn zero_slots_skip_a_card() {
        assert_eq!(fill(1, 3, 2, &[0, 0], &[1.0; 3]).unwrap(), [2; 3]);
        assert_eq!(fill(1, 3, 2, &[0, 2], &[1.0, 3.0, 2.0]).unwrap(), [2, 1, 1]);
    }

    #[test]
    fn invalid_input_is_rejected() {
        let bad = |heat: &[f64]| fill(1, 2, 1, &[1], heat);
        assert_eq!(bad(&[1.0, f64::NAN]), Err(PlacementError::InvalidHeat));
        assert_eq!(bad(&[1.0, f64::INFINITY]), Err(PlacementError::InvalidHeat));
        assert_eq!(bad(&[1.0, -0.5]), Err(PlacementError::InvalidHeat));
        assert_eq!(bad(&[1.0]), Err(PlacementError::InvalidHeat));
        assert!(bad(&[-0.0, 0.0]).is_ok());
        assert_eq!(
            fill(1, 2, 1, &[1, 1], &[1.0; 2]),
            Err(PlacementError::InvalidLayout)
        );
        assert_eq!(
            fill(1, 2, 0, &[], &[1.0; 2]),
            Err(PlacementError::InvalidLayout)
        );
        assert_eq!(fill(0, 2, 1, &[1], &[]), Err(PlacementError::InvalidLayout));
        assert_eq!(
            fill(1, 2, 64, &[0; 64], &[1.0; 2]),
            Err(PlacementError::InvalidLayout)
        );
    }

    #[test]
    fn the_result_opens_an_ownership_map() {
        let heat: Vec<f64> = (0..24).map(|index| f64::from((index * 7) % 11)).collect();
        let owners = fill(4, 6, 3, &[3, 4, 5], &heat).unwrap();
        let map = OwnershipMap::new(4, 6, 3, owners.clone()).unwrap();
        let used = |card| owners.iter().filter(|&&owner| owner == card).count();
        assert_eq!([used(0), used(1), used(2), used(3)], [3, 4, 5, 12]);
        assert_eq!(map.snapshot().fallback(), 3);
    }
}
