// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! Demand-policy behaviour tests for cooldown pacing, ping-pong bounds, share caps, the fallback boundary and climbs.
//! Fixture and helpers come from `policy_demand.rs`.

use super::policy_demand::*;
use super::*;

#[test]
fn a_cooled_expert_walks_down_one_rung_per_round_until_the_fallback() {
    // Expert 0 starts on the lead with the least heat in its layer; every colder-than-the-rest layer-mate below it is
    // hotter than it. With no cooldown it still drops only one rung per round: lead, helper 1, helper 2, fallback.
    let mut map = ladder_map(1);
    let mut io = Io::new(&map);
    let mut planner = PlacementPlanner::new(policy()).unwrap();
    let heat = [1., 50., 40., 45., 30., 35., 20., 25.];
    let mut path = vec![rung(&map, 0)];
    for _ in 0..3 {
        round(&mut map, &mut planner, &mut io, &heat);
        path.push(rung(&map, 0));
    }
    assert_eq!(path, [0, 1, 2, 3]);
    // At the fallback it has nowhere lower to go, and nothing moves it up while it stays the coldest.
    for _ in 0..4 {
        round(&mut map, &mut planner, &mut io, &heat);
        assert_eq!(rung(&map, 0), 3);
    }
}

#[test]
fn the_cooldown_paces_the_descent_of_a_cooled_expert() {
    // Same window, cooldown 2: a displaced cell is frozen too, so the expert drops in rounds 1, 4 and 7 and holds
    // its rung in between. This is the pace the live run showed, scaled down.
    let mut map = ladder_map(1);
    let mut io = Io::new(&map);
    let mut planner = PlacementPlanner::new(PlannerConfig {
        cooldown_rounds: 2,
        ..policy()
    })
    .unwrap();
    let heat = [1., 50., 40., 45., 30., 35., 20., 25.];
    let mut moved_in = Vec::new();
    for r in 1..=9u64 {
        let before = rung(&map, 0);
        round(&mut map, &mut planner, &mut io, &heat);
        let after = rung(&map, 0);
        assert!(
            after == before || after == before + 1,
            "round {r}: {before} -> {after}"
        );
        if after != before {
            moved_in.push(r);
        }
    }
    assert_eq!((moved_in, rung(&map, 0)), (vec![1, 4, 7], 3));
}

#[test]
fn a_cold_expert_with_no_hotter_challenger_below_it_stays_where_it_is() {
    // Cooling alone demotes nothing: expert 0 cools from 100 to 2 on the lead, but every cell below it is colder
    // still (at most 1), so no challenger clears the margin and it keeps its rung for as long as that holds.
    let mut map = ladder_map(1);
    let mut io = Io::new(&map);
    let mut planner = PlacementPlanner::new(policy()).unwrap();
    assert!(round(
        &mut map,
        &mut planner,
        &mut io,
        &[100., 80., 1., 1., 0.5, 0.5, 0.1, 0.1]
    )
    .is_empty());
    let heat = [2., 80., 1., 1., 0.5, 0.5, 0.1, 0.1];
    for _ in 0..12 {
        assert!(round(&mut map, &mut planner, &mut io, &heat).is_empty());
    }
    assert_eq!(rung(&map, 0), 0);
}

#[test]
fn mild_alternating_demand_inside_the_margin_never_moves_anything() {
    // Expert 1 (rung 1) and expert 0 (rung 0) trade the lead of demand every window, 60 against 40. The blended
    // values settle at about 51.4 and 48.6, a ratio near 1.06, far from the 1.25 margin.
    let mut map = small_map();
    let mut io = Io::new(&map);
    let mut planner = PlacementPlanner::new(policy()).unwrap();
    for n in 0..40 {
        let heat = if n % 2 == 0 {
            [60., 40., 0.]
        } else {
            [40., 60., 0.]
        };
        assert!(
            round(&mut map, &mut planner, &mut io, &heat).is_empty(),
            "window {n}"
        );
    }
}

#[test]
fn hard_alternating_demand_ping_pong_is_bounded_by_the_cooldown() {
    // Windows alternate between (100, 0) and (0, 100). The blends do clear the margin, so swaps happen, but a cell
    // that moved cannot move again for the cooldown, so the swap count is bounded and the gaps are at least 9 rounds.
    let mut map = small_map();
    let mut io = Io::new(&map);
    let mut planner = PlacementPlanner::new(PlannerConfig {
        cooldown_rounds: 8,
        ..policy()
    })
    .unwrap();
    let mut rounds_with_a_swap = Vec::new();
    for n in 1..=60u64 {
        let heat = if n % 2 == 1 {
            [100., 0., 0.]
        } else {
            [0., 100., 0.]
        };
        if !round(&mut map, &mut planner, &mut io, &heat).is_empty() {
            rounds_with_a_swap.push(n);
        }
    }
    assert!(
        !rounds_with_a_swap.is_empty(),
        "the scenario must actually exercise the swap"
    );
    assert!(
        rounds_with_a_swap.windows(2).all(|w| w[1] - w[0] >= 9),
        "{rounds_with_a_swap:?}"
    );
    assert!(rounds_with_a_swap.len() <= 7, "{rounds_with_a_swap:?}");
}

#[test]
fn the_helper_share_cap_stops_a_helper_boundary_swap_but_not_the_top_boundary() {
    // Boundary (rung 1, rung 2): expert 4 (500) would lift helper 1 from 200 to 600 of ~2700, over 8 percent.
    let heat = [1000., 1000., 100., 100., 500., 1., 0., 0.];
    let map = ladder_map(1);
    let capped = PlannerConfig {
        max_helper_layer_share: 0.08,
        ..policy()
    };
    let plan = PlacementPlanner::new(capped)
        .unwrap()
        .plan(map.snapshot(), &heat)
        .unwrap();
    assert!(plan.promotions().is_empty(), "{:?}", plan.promotions());
    let open = PlacementPlanner::new(policy())
        .unwrap()
        .plan(map.snapshot(), &heat)
        .unwrap();
    assert_eq!(
        open.promotions(),
        [p(0, 4, 2, 1)],
        "uncapped, the coldest helper cell (lowest index on a tie)"
    );
    // The top boundary is exempt: a share of 0.0 still allows it.
    let exempt = PlannerConfig {
        max_helper_layer_share: 0.0,
        ..top_only()
    };
    let plan = PlacementPlanner::new(exempt)
        .unwrap()
        .plan(map.snapshot(), &[100., 80., 500., 1., 0., 0., 0., 0.]);
    assert_eq!(plan.unwrap().promotions(), [p(0, 2, 1, 0)]);
}

#[test]
fn the_share_cap_can_pass_over_the_coldest_incumbent_for_a_warmer_one() {
    // Helper 1 holds expert 2 (100) and expert 3 (300); expert 4 (500) challenges. Swapping with the coldest, expert 2,
    // would lift helper 1 to 800 of 2901 (27.6 percent), over the cap of 22 percent; swapping with expert 3 lifts it
    // to 600 (20.7 percent) and passes. So the displaced cell is the coldest one that passes, not always the coldest.
    let heat = [1000., 1000., 100., 300., 500., 1., 0., 0.];
    let map = ladder_map(1);
    let config = PlannerConfig {
        max_helper_layer_share: 0.22,
        ..policy()
    };
    let plan = PlacementPlanner::new(config)
        .unwrap()
        .plan(map.snapshot(), &heat)
        .unwrap();
    assert_eq!(plan.promotions(), [p(0, 4, 3, 1)]);
}

#[test]
fn the_fallback_boundary_is_planned_only_when_enabled() {
    // Expert 6 (on the fallback) is far hotter than expert 4 (helper 2): it only moves with the boundary on.
    let heat = [1000., 900., 800., 700., 10., 20., 5000., 1.];
    let off = PlannerConfig {
        fallback_boundary: false,
        ..policy()
    };
    let on = PlannerConfig {
        fallback_boundary: true,
        ..policy()
    };
    let map = ladder_map(1);
    let plan = PlacementPlanner::new(off)
        .unwrap()
        .plan(map.snapshot(), &heat)
        .unwrap();
    assert!(
        plan.promotions().iter().all(|m| m.upper_rung < 2),
        "{:?}",
        plan.promotions()
    );
    let plan = PlacementPlanner::new(on)
        .unwrap()
        .plan(map.snapshot(), &heat)
        .unwrap();
    assert!(
        plan.promotions().contains(&p(0, 6, 4, 2)),
        "{:?}",
        plan.promotions()
    );
}

#[test]
fn a_quarantined_rung_is_never_a_candidate_on_either_side() {
    let mut map = ladder_map(1);
    map.quarantine(1).unwrap();
    let mut planner = PlacementPlanner::new(policy()).unwrap();
    let plan = planner
        .plan(map.snapshot(), &[1., 1., 900., 900., 500., 500., 0., 0.])
        .unwrap();
    // Experts 2 and 3 sit on the closed rung: they neither climb nor are displaced, and no swap crosses it.
    for m in plan.promotions() {
        assert!(
            ![m.promote, m.displace].iter().any(|e| [2, 3].contains(e)),
            "{m:?}"
        );
    }
}

#[test]
fn a_climb_makes_one_hop_per_boundary_and_displaced_cells_drop_one_rung() {
    // With max_hops above 1 the hot expert rises several rungs in one round; every displaced cell still drops one.
    let mut map = ladder_map(1);
    let mut io = Io::new(&map);
    let config = PlannerConfig {
        max_hops: 8,
        ..policy()
    };
    let mut planner = PlacementPlanner::new(config).unwrap();
    let heat = [100., 90., 80., 70., 60., 50., 40., 9000.];
    let plan = round(&mut map, &mut planner, &mut io, &heat);
    assert_eq!(plan, [p(0, 7, 5, 2), p(0, 7, 3, 1), p(0, 7, 1, 0)]);
    assert_eq!(
        [rung(&map, 7), rung(&map, 5), rung(&map, 3), rung(&map, 1)],
        [0, 3, 2, 1]
    );
}
