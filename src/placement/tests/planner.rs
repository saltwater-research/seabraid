// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! Paired live subset of `tests/tier_ladder_test.cpp`; execute each plan on the
//! same byte-aware fake providers used by the transition checks.

use super::*;

fn config() -> PlannerConfig {
    PlannerConfig {
        max_helper_layer_share: 1.0,
        cooldown_rounds: 0,
        ..PlannerConfig::default()
    }
}

#[test]
fn three_card_cascade_is_top_first_adjacent_and_uses_each_cell_once() {
    let mut place = OwnershipMap::new(1, 8, 3, vec![0, 0, 1, 1, 2, 2, 3, 3]).unwrap();
    let mut io = Providers::new(&place);
    let mut planner = PlacementPlanner::new(config()).unwrap();
    let heat = [10., 60., 40., 2., 100., 1., 0., 0.];
    let first = planner.plan(place.snapshot(), &heat).unwrap();
    assert_eq!(first.promotions(), [p(0, 2, 0, 0), p(0, 4, 3, 1)]);
    planner
        .record(&place.apply_plan(first, &mut io).unwrap())
        .unwrap();
    assert_eq!(place.snapshot().epoch(), 3);
    assert_eq!(
        place.snapshot().executor(ExpertId {
            layer: 0,
            expert: 4
        }),
        Some(1)
    );
    io.reference_matches(&place);

    let next = planner.plan(place.snapshot(), &heat).unwrap();
    assert_eq!(next.promotions(), [p(0, 4, 2, 0)]);
    planner
        .record(&place.apply_plan(next, &mut io).unwrap())
        .unwrap();
    assert_eq!(
        place.snapshot().executor(ExpertId {
            layer: 0,
            expert: 4
        }),
        Some(0)
    );
    io.reference_matches(&place);
}

#[test]
fn budget_never_splits_pairs_and_layer_mismatch_does_not_displace() {
    let place = OwnershipMap::new(3, 2, 2, vec![0, 1, 0, 1, 0, 1]).unwrap();
    let heat = [90., 100., 1., 20., 1., 30.];
    let mut planner = PlacementPlanner::new(PlannerConfig {
        max_moves_per_round: 3,
        ..config()
    })
    .unwrap();
    assert_eq!(
        planner.plan(place.snapshot(), &heat).unwrap().promotions(),
        [p(2, 1, 0, 0)]
    );
    let mut planner = PlacementPlanner::new(PlannerConfig {
        max_moves_per_round: 1,
        ..config()
    })
    .unwrap();
    assert!(planner
        .plan(place.snapshot(), &heat)
        .unwrap()
        .promotions()
        .is_empty());

    let mut planner = PlacementPlanner::new(config()).unwrap();
    let tied = planner
        .plan(place.snapshot(), &[1., 10., 1., 10., 1., 10.])
        .unwrap();
    assert_eq!(tied.promotions(), [p(0, 1, 0, 0), p(1, 1, 0, 0)]);
}

#[test]
fn cooldown_applies_to_committed_pairs_only_and_margin_prevents_thrash() {
    let mut place = OwnershipMap::new(1, 2, 2, vec![0, 1]).unwrap();
    let mut io = Providers::new(&place);
    let mut planner = PlacementPlanner::new(PlannerConfig {
        cooldown_rounds: 3,
        ..config()
    })
    .unwrap();
    let below_margin = planner.plan(place.snapshot(), &[10., 12.]).unwrap();
    assert!(below_margin.promotions().is_empty());

    let plan = planner.plan(place.snapshot(), &[10., 16.]).unwrap();
    assert_eq!(plan.promotions(), [p(0, 1, 0, 0)]);
    io.write_fault = [Some(IoError::Refused)].into();
    let report = place.apply_plan(plan, &mut io).unwrap();
    assert!(matches!(
        report.outcomes()[0].outcome,
        MoveOutcome::Rejected(_)
    ));
    planner.record(&report).unwrap();

    let retry = planner.plan(place.snapshot(), &[10., 16.]).unwrap();
    assert_eq!(retry.promotions(), [p(0, 1, 0, 0)]);
    planner
        .record(&place.apply_plan(retry, &mut io).unwrap())
        .unwrap();
    for _ in 0..3 {
        assert!(planner
            .plan(place.snapshot(), &[100., 1.])
            .unwrap()
            .promotions()
            .is_empty());
    }
    assert_eq!(
        planner
            .plan(place.snapshot(), &[100., 1.])
            .unwrap()
            .promotions(),
        [p(0, 0, 1, 0)]
    );
    io.reference_matches(&place);
}

#[test]
fn helper_share_and_boundary_caps_do_not_cap_card_one() {
    let mut place = OwnershipMap::new(1, 5, 3, vec![0, 1, 1, 2, 3]).unwrap();
    let mut planner = PlacementPlanner::new(PlannerConfig {
        max_helper_layer_share: 0.15,
        ..config()
    })
    .unwrap();
    assert_eq!(
        planner
            .plan(place.snapshot(), &[70., 1., 7., 13., 9.])
            .unwrap()
            .promotions(),
        [p(0, 3, 2, 1)]
    );
    let mut planner = PlacementPlanner::new(PlannerConfig {
        ladder_boundaries: 1,
        ..config()
    })
    .unwrap();
    assert!(planner
        .plan(place.snapshot(), &[70., 1., 7., 13., 9.])
        .unwrap()
        .promotions()
        .is_empty());

    let mut planner = PlacementPlanner::new(PlannerConfig {
        max_helper_layer_share: 0.0,
        ..config()
    })
    .unwrap();
    assert_eq!(
        planner
            .plan(place.snapshot(), &[1., 80., 0., 0., 100.])
            .unwrap()
            .promotions(),
        [p(0, 1, 0, 0)]
    );
    place.quarantine(1).unwrap();
    assert!(planner
        .plan(place.snapshot(), &[1., 80., 0., 100., 100.])
        .unwrap()
        .promotions()
        .is_empty());
}

#[test]
fn stale_plan_never_stages_and_quarantine_stops_current_round() {
    let mut place = OwnershipMap::new(3, 2, 2, vec![0, 1, 0, 1, 0, 1]).unwrap();
    let mut io = Providers::new(&place);
    let mut planner = PlacementPlanner::new(PlannerConfig {
        max_moves_per_round: 6,
        ..config()
    })
    .unwrap();
    let heat = [1., 50., 1., 30., 1., 20.];
    let stale = planner.plan(place.snapshot(), &heat).unwrap();
    let owner = place.snapshot().owner.clone();
    place.rebuild(1, owner).unwrap();
    assert!(matches!(
        place.apply_plan(stale, &mut io),
        Err(PlacementError::StaleEpoch {
            expected: 1,
            actual: 2
        })
    ));
    assert!(io.stages.is_empty() && io.writes.is_empty());

    let plan = planner.plan(place.snapshot(), &heat).unwrap();
    assert_eq!(
        plan.promotions(),
        [p(0, 1, 0, 0), p(1, 1, 0, 0), p(2, 1, 0, 0)]
    );
    io.write_fault = [None, None, None, Some(IoError::Timeout)].into();
    let report = place.apply_plan(plan, &mut io).unwrap();
    assert_eq!(report.outcomes().len(), 2);
    assert_eq!(
        report.outcomes()[0].outcome,
        MoveOutcome::Committed { epoch: 3 }
    );
    assert!(matches!(
        report.outcomes()[1].outcome,
        MoveOutcome::Quarantined { .. }
    ));
    assert_eq!(io.stages, [1, 0, 1, 0]);
    assert_eq!(io.writes, [0, 1, 0, 1, 0]);
    assert_eq!(place.snapshot().epoch(), 4);
    assert_eq!(
        place.snapshot().owner(ExpertId {
            layer: 2,
            expert: 1
        }),
        Some(1)
    );
    io.reference_matches(&place);
}

#[test]
fn bad_heat_and_policy_fail_closed_without_seeding_smoother() {
    assert!(matches!(
        PlacementPlanner::new(PlannerConfig {
            promote_margin: f64::NAN,
            ..config()
        }),
        Err(PlacementError::InvalidPolicy)
    ));
    let place = OwnershipMap::new(1, 2, 2, vec![0, 1]).unwrap();
    let mut planner = PlacementPlanner::new(config()).unwrap();
    for heat in [&[1.][..], &[1., f64::NAN], &[-1., 10.]] {
        assert!(matches!(
            planner.plan(place.snapshot(), heat),
            Err(PlacementError::InvalidHeat)
        ));
    }
    assert!(matches!(
        planner.plan(place.snapshot(), &[f64::MAX; 2]),
        Err(PlacementError::HeatOverflow)
    ));
    let plan = planner.plan(place.snapshot(), &[10., 13.]).unwrap();
    assert_eq!(plan.promotions(), [p(0, 1, 0, 0)]);
}
