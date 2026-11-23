// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! The climb planner and the plan validation behind it, on the four-helper pyramid (`pyramid.rs`).

use super::pyramid::*;
use super::*;

pub(super) fn climbing(hops: u8) -> PlannerConfig {
    PlannerConfig {
        max_helper_layer_share: 1.0,
        cooldown_rounds: 0,
        max_moves_per_round: 128,
        fallback_boundary: true,
        max_hops: hops,
        ..PlannerConfig::default()
    }
}

/// Layer 0's expert 11 (the coldest, on the fallback) becomes by far the hottest.
fn spike() -> Vec<f64> {
    let mut heat = baseline_heat();
    heat[11] = 1e6;
    heat
}

#[test]
fn a_climb_is_bottom_up_same_layer_and_each_displaced_cell_drops_one_rung() {
    let mut map = pyramid_map(true);
    let mut planner = PlacementPlanner::new(climbing(8)).unwrap();
    let plan = planner.plan(map.snapshot(), &spike()).unwrap();
    // The climber, then the coldest cell of each rung it passes, hottest rung last.
    assert_eq!(
        plan.promotions(),
        [
            p(0, 11, 9, 4),
            p(0, 11, 7, 3),
            p(0, 11, 5, 2),
            p(0, 11, 3, 1),
            p(0, 11, 1, 0)
        ]
    );
    let mut io = Pyramid::new(&map);
    let applied = map.apply_plan(plan, &mut io).unwrap();
    assert!(applied
        .outcomes()
        .iter()
        .all(|o| matches!(o.outcome, MoveOutcome::Committed { .. })));
    assert_eq!(owner_of(&map, 0, 11), 0);
    for (expert, rung) in [(9, 5), (7, 4), (5, 3), (3, 2), (1, 1)] {
        assert_eq!(
            owner_of(&map, 0, expert),
            rung,
            "expert {expert} drops exactly one rung"
        );
    }
    // Staged-bytes reuse: the climber is read once, then one displaced cell per hop.
    assert_eq!(io.count(Op::Stage, CARDS), 1);
    assert_eq!(io.calls.iter().filter(|c| c.0 == Op::Stage).count(), 6);
    io.audit(&map);
    assert_eq!(map.snapshot().epoch(), 6);
}

#[test]
fn max_hops_and_the_write_budget_end_a_climb_at_a_consistent_rung() {
    for (hops, moves, rung) in [(2, 128, 3), (8, 6, 2), (8, 5, 3)] {
        let mut map = pyramid_map(true);
        let config = PlannerConfig {
            max_moves_per_round: moves,
            ..climbing(hops)
        };
        let mut planner = PlacementPlanner::new(config).unwrap();
        let mut io = Pyramid::new(&map);
        run_round(&mut map, &mut planner, &mut io, &spike());
        assert_eq!(owner_of(&map, 0, 11), rung, "hops {hops}, moves {moves}");
    }
}

#[test]
fn a_climb_stops_at_the_first_boundary_whose_incumbents_are_not_beaten_by_the_margin() {
    let mut heat = baseline_heat();
    heat[11] = 21.0; // beats the coldest cell of rung 4 (cell 9, heat 4) and of rung 3 (cell 7, heat 16), not rung 2's (cell 5, 64)
    let map = pyramid_map(true);
    let mut planner = PlacementPlanner::new(climbing(8)).unwrap();
    let plan = planner.plan(map.snapshot(), &heat).unwrap();
    assert_eq!(plan.promotions(), [p(0, 11, 9, 4), p(0, 11, 7, 3)]);
}

#[test]
fn without_the_fallback_boundary_a_climb_starts_at_the_last_helper() {
    let mut map = pyramid_map(false);
    let config = PlannerConfig {
        fallback_boundary: false,
        ..climbing(8)
    };
    let mut planner = PlacementPlanner::new(config).unwrap();
    let mut heat = spike();
    heat[9] = 1e5; // the last helper's cell
    let plan = planner.plan(map.snapshot(), &heat).unwrap();
    assert_eq!(plan.promotions().first(), Some(&p(0, 9, 7, 3)));
    assert!(
        plan.promotions().iter().all(|m| m.promote != 11),
        "a fallback cell is never a challenger"
    );
    let mut io = Pyramid::new(&map);
    map.apply_plan(plan, &mut io).unwrap();
    io.audit(&map);
}

#[test]
fn ladder_boundaries_never_limit_the_fallback_boundary_and_limit_the_card_boundaries() {
    let mut map = pyramid_map(true);
    let config = PlannerConfig {
        ladder_boundaries: 1,
        ..climbing(8)
    };
    let mut planner = PlacementPlanner::new(config).unwrap();
    let plan = planner.plan(map.snapshot(), &spike()).unwrap();
    // Only boundary (0,1) is a card boundary left open, so the climber crosses the fallback boundary and no more.
    assert_eq!(plan.promotions(), [p(0, 11, 9, 4)]);
    let mut io = Pyramid::new(&map);
    map.apply_plan(plan, &mut io).unwrap();
    assert_eq!(owner_of(&map, 0, 11), 4);
}

#[test]
fn the_share_cap_applies_at_the_fallback_boundary() {
    let map = pyramid_map(true);
    let config = PlannerConfig {
        max_helper_layer_share: 0.0,
        ..climbing(1)
    };
    let mut planner = PlacementPlanner::new(config).unwrap();
    // Any swap raises the last helper's share of the layer above a cap of zero.
    assert!(planner
        .plan(map.snapshot(), &spike())
        .unwrap()
        .promotions()
        .is_empty());
}

#[test]
fn plan_validation_allows_a_climber_to_reappear_one_rung_higher_and_nothing_else() {
    let place = || pyramid_map(true);
    let plan = |promotions: Vec<Promotion>| PlacementPlan {
        epoch: 1,
        round: 1,
        promotions,
    };
    let refused = |promotions: Vec<Promotion>| {
        let mut map = place();
        let mut io = Pyramid::new(&map);
        let result = map.apply_plan(plan(promotions), &mut io);
        assert!(matches!(result, Err(PlacementError::InvalidPromotion)));
        assert!(io.calls.is_empty(), "refused before any I/O");
        assert_eq!(map.snapshot().epoch(), 1);
    };
    // The climber skips a rung, a displaced cell is displaced twice, a climber is displaced, the climber repeats a hop.
    refused(vec![p(0, 11, 9, 4), p(0, 11, 5, 2)]);
    refused(vec![p(0, 11, 9, 4), p(0, 10, 9, 4)]);
    refused(vec![p(0, 11, 9, 4), p(0, 8, 11, 3)]);
    refused(vec![p(0, 11, 9, 4), p(0, 11, 9, 4)]);
    // A displaced cell is never promoted back in its own round, and a climber is never displaced after its climb.
    refused(vec![p(0, 11, 9, 4), p(0, 9, 8, 4)]);
    refused(vec![p(0, 11, 9, 4), p(0, 10, 11, 4)]);
    let mut map = place();
    let mut io = Pyramid::new(&map);
    let ok = plan(vec![p(0, 11, 9, 4), p(0, 11, 7, 3)]);
    assert_eq!(map.apply_plan(ok, &mut io).unwrap().outcomes().len(), 2);
}

#[test]
fn a_hop_after_an_uncommitted_hop_is_skipped_without_any_provider_call() {
    for fault in [IoError::Refused, IoError::Failed] {
        let mut map = pyramid_map(true);
        let mut io = Pyramid::new(&map);
        io.fail(3, Op::Replace, &[Some(fault)]); // the second hop's upper card refuses, or tears and restores
        let plan = PlacementPlan {
            epoch: 1,
            round: 1,
            promotions: vec![
                p(0, 11, 9, 4),
                p(0, 11, 7, 3),
                p(0, 11, 5, 2),
                p(0, 11, 3, 1),
            ],
        };
        let applied = map.apply_plan(plan, &mut io).unwrap();
        let kinds: Vec<_> = applied.outcomes().iter().map(|o| o.outcome).collect();
        assert!(matches!(kinds[0], MoveOutcome::Committed { .. }));
        assert!(
            matches!(
                kinds[1],
                MoveOutcome::Rejected(_) | MoveOutcome::Restored(_)
            ),
            "{kinds:?}"
        );
        assert_eq!(&kinds[2..], [MoveOutcome::Skipped, MoveOutcome::Skipped]);
        // Hop 1 committed (epoch 2); the failed hop and the skipped ones add nothing; the climber rests on rung 4.
        assert_eq!(map.snapshot().epoch(), 2);
        assert_eq!(owner_of(&map, 0, 11), 4);
        assert_eq!(
            io.count(Op::Replace, 2),
            0,
            "skipped hops reached no provider"
        );
        io.audit(&map);
    }
}

#[test]
fn a_quarantine_ends_the_round_and_later_hops_get_no_outcome() {
    let mut map = pyramid_map(true);
    let mut io = Pyramid::new(&map);
    io.fail(3, Op::Replace, &[Some(IoError::Timeout)]);
    let plan = PlacementPlan {
        epoch: 1,
        round: 1,
        promotions: vec![p(0, 11, 9, 4), p(0, 11, 7, 3), p(0, 11, 5, 2)],
    };
    let applied = map.apply_plan(plan, &mut io).unwrap();
    assert_eq!(applied.outcomes().len(), 2);
    assert!(map.snapshot().is_quarantined(3));
}

#[test]
fn a_climb_converges_to_the_per_layer_ranking_the_margin_implies() {
    // Brute force: rank each layer by heat and deal the cells out by each rung's per-layer count. Heats are 2x apart,
    // so every inversion beats the 1.25 margin and the fixed point of the climb must be exactly this ranking.
    for seed in 1..=40u64 {
        for hops in [1u8, 3, 8] {
            let mut rng = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
            let mut next = move || {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                rng
            };
            let mut heat = vec![0.0; (LAYERS * EXPERTS) as usize];
            for layer in 0..LAYERS as usize {
                // A distinct power of two per cell of the layer, dealt in a random order.
                let mut shuffled: Vec<u32> = (0..EXPERTS).collect();
                for i in (1..shuffled.len()).rev() {
                    shuffled.swap(i, next() as usize % (i + 1));
                }
                for (expert, power) in shuffled.into_iter().enumerate() {
                    heat[layer * EXPERTS as usize + expert] = 2f64.powi(power as i32);
                }
            }
            let mut map = pyramid_map(true);
            let mut planner = PlacementPlanner::new(climbing(hops)).unwrap();
            let mut io = Pyramid::new(&map);
            let mut rounds = 0;
            while !run_round(&mut map, &mut planner, &mut io, &heat)
                .outcomes()
                .is_empty()
            {
                rounds += 1;
                assert!(rounds < 200, "seed {seed} hops {hops} did not settle");
            }
            for layer in 0..LAYERS {
                let mut ranked: Vec<u32> = (0..EXPERTS).collect();
                ranked.sort_by(|&a, &b| {
                    heat[(layer * EXPERTS + b) as usize]
                        .total_cmp(&heat[(layer * EXPERTS + a) as usize])
                });
                for (place, expert) in ranked.into_iter().enumerate() {
                    let target = (place / 2) as Rung;
                    assert_eq!(
                        owner_of(&map, layer, expert),
                        target,
                        "seed {seed} hops {hops} layer {layer}"
                    );
                }
            }
        }
    }
}
