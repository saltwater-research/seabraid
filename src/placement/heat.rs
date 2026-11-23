// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! The one definition of the per-layer heat-sum rule shared by `fill` and the planner.
//!
//! * OWNER: this file; `include/seabraid/placement.h` states the same rule for C callers.
//! * CONTRACT: `layer_totals(heat, width)` sums each `width`-long layer of finite values and returns
//!   `HeatOverflow` if any sum is not finite. Per-value finiteness stays with each caller (`InvalidHeat`).
//! * FAILURE CONTRACT: fail-fast, no partial result.
//! * CHECKER: `tests/fill_heat.rs` (fill and plan refuse the same heat) and the planner tests.

use super::PlacementError;

/// WHY shared: fill and plan accept the same heat, so a table one refuses the other must refuse too; the
/// planner needs the totals themselves, fill only the verdict.
pub(super) fn layer_totals(heat: &[f64], width: usize) -> Result<Vec<f64>, PlacementError> {
    let total: Vec<f64> = heat
        .chunks_exact(width)
        .map(|layer| layer.iter().sum())
        .collect();
    if total.iter().any(|v| !v.is_finite()) {
        return Err(PlacementError::HeatOverflow); // callers checked every value was finite, so a sum overflowed
    }
    Ok(total)
}
