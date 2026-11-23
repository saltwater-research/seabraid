// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! Fill and plan apply one heat rule: finite values AND finite per-layer sums (journey 5 found fill accepting
//! two cells of 1.7e308 that plan refused). Checker for `placement::heat::layer_totals`.

use super::*;

const HUGE: f64 = 1.7e308;

#[test]
fn fill_refuses_a_layer_whose_heat_sum_overflows_like_plan() {
    let place = OwnershipMap::new(1, 2, 1, vec![0, 1]).unwrap();
    let mut planner = PlacementPlanner::new(PlannerConfig::default()).unwrap();
    let heat = [HUGE, HUGE];
    assert_eq!(
        planner.plan(place.snapshot(), &heat).err(),
        Some(PlacementError::HeatOverflow)
    );
    assert_eq!(
        fill(1, 2, 1, &[1], &heat),
        Err(PlacementError::HeatOverflow)
    );
}

#[test]
fn fill_checks_each_layer_and_still_accepts_large_finite_sums() {
    // The second layer overflows even though the first is fine.
    assert_eq!(
        fill(2, 2, 1, &[1], &[1.0, 2.0, HUGE, HUGE]),
        Err(PlacementError::HeatOverflow)
    );
    // Large values whose sum stays finite are not refused.
    assert!(fill(1, 2, 1, &[1], &[HUGE, 1.0e306]).is_ok());
    // The per-value rule still wins when a value is not finite.
    assert_eq!(
        fill(1, 2, 1, &[1], &[f64::INFINITY, 1.0]),
        Err(PlacementError::InvalidHeat)
    );
}

/// R3 finding 22: the planner used to check only the BLENDED heat, so a window whose raw layer sum overflowed passed
/// whenever the blend (0.75 * old + 0.25 * new) stayed finite, while `fill` refused the same window.
#[test]
fn plan_refuses_an_overflowing_raw_window_after_the_first_window() {
    let place = OwnershipMap::new(1, 2, 1, vec![0, 1]).unwrap();
    let mut planner = PlacementPlanner::new(PlannerConfig::default()).unwrap();
    assert_eq!(
        planner.plan(place.snapshot(), &[1.0, 1.0]).unwrap().round,
        1
    );
    // blend = 0.75 + 0.25 * 1.7e308 per cell: its sum is finite, the raw window's is not.
    assert_eq!(
        planner.plan(place.snapshot(), &[HUGE, HUGE]).err(),
        Some(PlacementError::HeatOverflow)
    );
    assert_eq!(
        fill(1, 2, 1, &[1], &[HUGE, HUGE]),
        Err(PlacementError::HeatOverflow)
    );
    // The refused window changed nothing: the next success is round 2 and still blends from [1, 1].
    assert_eq!(
        planner.plan(place.snapshot(), &[1.0, 1.0]).unwrap().round,
        2
    );
}

#[test]
fn plan_checks_every_layer_of_a_later_window() {
    let place = OwnershipMap::new(2, 2, 1, vec![0, 1, 0, 1]).unwrap();
    let mut planner = PlacementPlanner::new(PlannerConfig::default()).unwrap();
    planner.plan(place.snapshot(), &[1.0; 4]).unwrap();
    assert_eq!(
        planner
            .plan(place.snapshot(), &[1.0, 2.0, HUGE, HUGE])
            .err(),
        Some(PlacementError::HeatOverflow)
    );
}
