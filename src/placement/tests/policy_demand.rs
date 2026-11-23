// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! The demand-driven placement policy, one behaviour test per sentence of its statement.
//! OWNER: `docs/PLACEMENT-ABI.md#demand-driven-placement-policy` owns the wording; this file owns the proof, and the
//! doc's cross-reference list names every test here (`policy_doc_names_every_test_in_this_file` checks it). The
//! rules themselves live in `planner.rs` and `planner/climb.rs` and are never restated here.
//! FIXTURE: one layer, three cards plus the fallback (rungs 0, 1, 2, F=3), eight experts, two per rung, on the
//! byte-keeping `Providers` of `tests.rs`; the multi-layer cases use two layers of it.

use super::*;

/// `Providers` plus the fallback rung the pyramid fixture also has: it serves every cell from the source and acks a
/// replace (the provider rule of PLACEMENT-ABI.md), so swaps at the fallback boundary run for real.
pub(super) struct Io {
    pub(super) inner: Providers,
    pub(super) fallback: Rung,
}

impl Io {
    pub(super) fn new(map: &OwnershipMap) -> Self {
        Self {
            inner: Providers::new(map),
            fallback: map.snapshot().fallback(),
        }
    }

    pub(super) fn reference_matches(&self, map: &OwnershipMap) {
        self.inner.reference_matches(map);
    }
}

impl PlacementIo for Io {
    fn stage(&mut self, rung: Rung, cell: ExpertId) -> Result<Vec<u8>, IoError> {
        if rung == self.fallback {
            return Ok(source(cell));
        }
        self.inner.stage(rung, cell)
    }

    fn replace(
        &mut self,
        rung: Rung,
        old: ExpertId,
        new: ExpertId,
        bytes: &[u8],
    ) -> Result<(), IoError> {
        if rung == self.fallback {
            return Ok(());
        }
        self.inner.replace(rung, old, new, bytes)
    }
}

pub(super) const LADDER: [Rung; 8] = [0, 0, 1, 1, 2, 2, 3, 3];

pub(super) fn ladder_map(layers: u32) -> OwnershipMap {
    let owners = (0..layers).flat_map(|_| LADDER).collect();
    let mut map = OwnershipMap::new(layers, 8, 3, owners).unwrap();
    map.enable_fallback_boundary().unwrap();
    map
}

/// No share cap, no cooldown, room for every swap, the fallback boundary on, adjacent swaps only: each test changes only what it studies.
pub(super) fn policy() -> PlannerConfig {
    PlannerConfig {
        max_helper_layer_share: 1.0,
        cooldown_rounds: 0,
        max_moves_per_round: 128,
        fallback_boundary: true,
        ..PlannerConfig::default()
    }
}

/// Plan, apply and record one window; returns the promotions of that round.
pub(super) fn round(
    map: &mut OwnershipMap,
    planner: &mut PlacementPlanner,
    io: &mut Io,
    heat: &[f64],
) -> Vec<Promotion> {
    let plan = planner.plan(map.snapshot(), heat).unwrap();
    let promotions = plan.promotions().to_vec();
    let applied = map.apply_plan(plan, io).unwrap();
    planner.record(&applied).unwrap();
    io.reference_matches(map);
    promotions
}

pub(super) fn rung(map: &OwnershipMap, expert: u32) -> Rung {
    map.snapshot().owner(ExpertId { layer: 0, expert }).unwrap()
}

pub(super) fn small_map() -> OwnershipMap {
    let mut map = OwnershipMap::new(1, 3, 2, vec![0, 1, 2]).unwrap();
    map.enable_fallback_boundary().unwrap();
    map
}

/// Boundary 0 only (the top one), so a test can read one decision without the others.
pub(super) fn top_only() -> PlannerConfig {
    PlannerConfig {
        ladder_boundaries: 1,
        ..policy()
    }
}

#[test]
fn heat_first_window_is_used_raw() {
    // Expert 3 is only 1.26 times expert 1 (both on a rung pair that can swap): a seeded raw window already promotes.
    let map = small_map();
    let mut planner = PlacementPlanner::new(policy()).unwrap();
    let plan = planner.plan(map.snapshot(), &[100., 126., 0.]).unwrap();
    assert_eq!(plan.promotions(), [p(0, 1, 0, 0)]);
}

#[test]
fn heat_later_windows_blend_075_old_and_025_new() {
    // Window 1 is (100, 100); window 2 is (100, X). The blend gives expert 1 the value 0.75 * 100 + 0.25 * X.
    // X = 200 blends to exactly 125.0, which is not strictly above 1.25 * 100. X = 204 blends to 126.0, which is.
    for (x, moves) in [(200., 0), (204., 1)] {
        let map = small_map();
        let mut planner = PlacementPlanner::new(policy()).unwrap();
        assert!(planner
            .plan(map.snapshot(), &[100., 100., 0.])
            .unwrap()
            .promotions()
            .is_empty());
        let plan = planner.plan(map.snapshot(), &[100., x, 0.]).unwrap();
        assert_eq!(plan.promotions().len(), moves, "window 2 value {x}");
    }
}

#[test]
fn the_hottest_challenger_is_served_first_when_the_write_budget_allows_one_swap() {
    // Experts 2 and 3 (rung 1) both beat the margin over expert 1; only one swap fits in 2 writes.
    let mut map = ladder_map(1);
    let mut io = Io::new(&map);
    let config = PlannerConfig {
        max_moves_per_round: 2,
        ..top_only()
    };
    let mut planner = PlacementPlanner::new(config).unwrap();
    let heat = [100., 50., 300., 500., 1., 1., 0., 0.];
    assert_eq!(
        round(&mut map, &mut planner, &mut io, &heat),
        [p(0, 3, 1, 0)]
    );
}

#[test]
fn a_hot_challenger_displaces_the_coldest_same_layer_incumbent() {
    // Layer 0: incumbents 100 (expert 0) and 50 (expert 1); the challenger, expert 2, has 500.
    // Layer 1 is colder everywhere (all 1) and equal, so nothing there moves and nothing there may be displaced.
    let mut map = ladder_map(2);
    let mut io = Io::new(&map);
    let mut planner = PlacementPlanner::new(top_only()).unwrap();
    let mut heat = vec![100., 50., 500., 1., 1., 1., 0., 0.];
    heat.extend([1., 1., 1., 1., 1., 1., 0., 0.]);
    let plan = round(&mut map, &mut planner, &mut io, &heat);
    assert_eq!(
        plan,
        [p(0, 2, 1, 0)],
        "the colder of {{0, 1}}, in layer 0, and nothing else"
    );
    assert_eq!((rung(&map, 2), rung(&map, 1), rung(&map, 0)), (0, 1, 0));
}

#[test]
fn a_cell_in_cooldown_is_skipped_and_the_next_coldest_incumbent_is_displaced() {
    // Round 1 moves expert 1 (50) down and expert 2 up. In round 2 expert 4 (smoothed 100) beats the margin over
    // expert 1 (50, the coldest on rung 1) and over expert 3 (60), but expert 1 is cooling: expert 3 is displaced.
    // With no cooldown the same window displaces expert 1, so the cooldown is what changed the choice.
    for (cooldown, victim) in [(8, 3), (0, 1)] {
        let mut map = ladder_map(1);
        let mut io = Io::new(&map);
        let mut planner = PlacementPlanner::new(PlannerConfig {
            cooldown_rounds: cooldown,
            ..policy()
        })
        .unwrap();
        let window1 = [100., 50., 500., 60., 1., 1., 0., 0.];
        assert_eq!(
            round(&mut map, &mut planner, &mut io, &window1),
            [p(0, 2, 1, 0)]
        );
        let window2 = [100., 50., 500., 60., 397., 1., 0., 0.]; // expert 4 blends to 0.75 * 1 + 0.25 * 397 = 100
        let plan = round(&mut map, &mut planner, &mut io, &window2);
        assert_eq!(plan, [p(0, 4, victim, 1)], "cooldown {cooldown}");
    }
}

#[test]
fn a_challenger_must_beat_the_margin_strictly_or_nothing_moves() {
    // Expert 2 against the coldest incumbent, expert 1 (80): exactly 1.25 * 80 = 100 is not enough, 100.5 is.
    for (hot, moves) in [(100., 0), (100.5, 1)] {
        let map = ladder_map(1);
        let mut planner = PlacementPlanner::new(top_only()).unwrap();
        let plan = planner
            .plan(map.snapshot(), &[100., 80., hot, 1., 1., 1., 0., 0.])
            .unwrap();
        assert_eq!(plan.promotions().len(), moves, "challenger heat {hot}");
    }
}

#[test]
fn a_zero_heat_cell_is_never_a_challenger() {
    let map = ladder_map(1);
    let mut planner = PlacementPlanner::new(policy()).unwrap();
    // Every incumbent is colder than the margin allows a zero to beat, and zero is excluded outright.
    let plan = planner
        .plan(map.snapshot(), &[0., 0., 0., 0., 0., 0., 0., 0.])
        .unwrap();
    assert!(plan.promotions().is_empty());
}

#[test]
fn a_moved_cell_is_a_candidate_again_from_round_r_plus_cooldown_plus_one() {
    // One hot pair swaps in round 1; with cooldown 2 the same two experts are frozen for rounds 2 and 3 even though
    // the window now asks for the swap back, and may move again in round 4 (1 + 2 + 1).
    let mut map = ladder_map(1);
    let mut io = Io::new(&map);
    let mut planner = PlacementPlanner::new(PlannerConfig {
        cooldown_rounds: 2,
        ..top_only()
    })
    .unwrap();
    let first = [100., 10., 500., 1., 1., 1., 0., 0.];
    assert_eq!(
        round(&mut map, &mut planner, &mut io, &first),
        [p(0, 2, 1, 0)]
    );
    // Demand reverses at once and hard: expert 1 is now hot, 2 and 0 are cold. Smoothing needs a few windows too.
    let flip = [1., 4000., 1., 1., 1., 1., 0., 0.];
    let mut moved_in = None;
    for r in 2..=8 {
        if !round(&mut map, &mut planner, &mut io, &flip).is_empty() {
            moved_in = Some(r);
            break;
        }
    }
    assert_eq!(
        moved_in,
        Some(4),
        "frozen in rounds 2 and 3, first free round is 1 + 2 + 1"
    );
}

#[test]
fn every_displaced_cell_drops_exactly_one_rung_and_each_cell_moves_once_per_round() {
    // Adjacent mode (one hop), all three card boundaries active in one round. Each boundary pairs its hottest
    // challenger with the coldest incumbent; no cell appears twice in the round's plan.
    let mut map = ladder_map(1);
    let mut io = Io::new(&map);
    let mut planner = PlacementPlanner::new(policy()).unwrap();
    let before: Vec<Rung> = (0..8).map(|e| rung(&map, e)).collect();
    let heat = [100., 90., 200., 10., 300., 1., 50., 0.5];
    let plan = round(&mut map, &mut planner, &mut io, &heat);
    assert_eq!(
        plan.len(),
        3,
        "one swap at each of the three boundaries: {plan:?}"
    );
    let mut touched = std::collections::HashSet::new();
    for promotion in &plan {
        assert!(
            touched.insert(promotion.promote) && touched.insert(promotion.displace),
            "{plan:?}"
        );
        let (up, down) = (promotion.promote as usize, promotion.displace as usize);
        assert_eq!(
            rung(&map, promotion.promote),
            before[up] - 1,
            "the climber rises one rung"
        );
        assert_eq!(
            rung(&map, promotion.displace),
            before[down] + 1,
            "the incumbent drops exactly one rung"
        );
    }
}

#[test]
fn policy_doc_names_every_test_in_this_file() {
    // The doc is the statement and this file the proof; a test the doc does not name is a rule nobody wrote down.
    let doc = include_str!("../../../docs/PLACEMENT-ABI.md");
    let source = [
        include_str!("policy_demand.rs"),
        include_str!("policy_demand_cooldown.rs"),
    ]
    .concat();
    let mut seen = 0;
    let mut lines = source.lines().peekable();
    while let Some(line) = lines.next() {
        let Some(next) = lines.peek().and_then(|n| n.strip_prefix("fn ")) else {
            continue;
        };
        if line == "#[test]" {
            let name = next.split('(').next().unwrap();
            assert!(
                doc.contains(name),
                "docs/PLACEMENT-ABI.md does not name the test {name}"
            );
            seen += 1;
        }
    }
    assert!(
        seen >= 20,
        "the scan found only {seen} tests; it is not reading this file"
    );
}
