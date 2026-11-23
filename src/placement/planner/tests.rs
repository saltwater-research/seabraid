// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! The planner's displacement search against the linear scan it replaced, and its cost when no swap can pass.

use super::*;
use crate::placement::OwnershipMap;
use std::time::{Duration, Instant};

/// The search as it was before the binary search (src/placement/planner.rs at 7dbc475), kept verbatim as the oracle:
/// every incumbent in the margin prefix is tried with the used and ceiling tests. WHY not deleted: a differential
/// test is only worth something against an implementation nobody has "optimised" alongside the one under test.
fn scan_displacement(
    round: &Round<'_>,
    hi: Rung,
    up: usize,
    incumbents: &Incumbents,
) -> Option<usize> {
    let layer = up / round.width;
    incumbents
        .get(&layer)?
        .iter()
        .copied()
        .take_while(|&down| round.heat[up] > round.config.promote_margin * round.heat[down])
        .find(|&down| {
            !round.used[down] && {
                hi == 0 || {
                    let before = round.held[&(hi, layer)];
                    before - round.heat[down] + round.heat[up]
                        <= round.config.max_helper_layer_share * round.total[layer]
                }
            }
        })
}

/// xorshift64*: deterministic, so a failing case number reproduces.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn pick<T: Copy>(&mut self, items: &[T]) -> T {
        items[self.below(items.len() as u64) as usize]
    }
}

fn random_heat(rng: &mut Rng, cells: usize) -> Vec<f64> {
    // A small integer alphabet makes exact ties common (the tie-break by index is part of what is compared);
    // the other branch gives fractional values so the ceiling arithmetic is not exact.
    let tied = rng.below(2) == 0;
    (0..cells)
        .map(|_| match (rng.below(10), tied) {
            (0..=2, _) => 0.0,
            (_, true) => rng.below(6) as f64,
            _ => rng.below(100_000) as f64 / 97.0,
        })
        .collect()
}

#[test]
fn the_search_equals_the_linear_scan_on_random_rounds() {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut with_promotions = 0;
    let mut capped = 0;
    for case in 0..20_000 {
        let layers = 1 + rng.below(3) as u32;
        let experts = 2 + rng.below(30) as u32;
        let cards = 2 + rng.below(4) as u8;
        let cells = (layers * experts) as usize;
        let owners: Vec<Rung> = (0..cells)
            .map(|_| rng.below(u64::from(cards) + 1) as Rung)
            .collect();
        let mut map = OwnershipMap::new(layers, experts, cards, owners).unwrap();
        if rng.below(5) == 0 {
            map.quarantine(rng.below(u64::from(cards)) as Rung).unwrap();
        }
        let heat = random_heat(&mut rng, cells);
        let config = PlannerConfig {
            max_moves_per_round: rng.pick(&[0, 1, 2, 4, 8, 128]),
            promote_margin: rng.pick(&[1.0, 1.25, 1.5, 3.0]),
            cooldown_rounds: 0,
            max_helper_layer_share: rng.pick(&[0.0, 0.02, 0.08, 0.3, 0.6, 1.0]),
            ladder_boundaries: rng.pick(&[0, 0, 1, 2]),
            ..PlannerConfig::default()
        };
        let mask: Vec<bool> = (0..cells).map(|_| rng.below(5) != 0).collect();
        let build = || Round::new(map.snapshot(), &heat, config, |i| mask[i]).unwrap();
        let fast = build().run();
        let slow = build().run_with(scan_displacement);
        assert_eq!(fast, slow, "case {case}: {config:?} heat {heat:?}");
        with_promotions += usize::from(!fast.is_empty());
        capped += usize::from(config.max_helper_layer_share < 0.1 && cards >= 3);
    }
    // The generator must reach the interesting branches, or equality proves little.
    assert!(
        with_promotions > 3000,
        "only {with_promotions} rounds promoted"
    );
    assert!(
        capped > 2000,
        "only {capped} rounds had a helper cap in play"
    );
}

/// R6a finding P1: one layer, three cards, every helper cell colder than its challenger and a helper share of zero,
/// so no swap can ever pass the ceiling. The plan used to try every incumbent for every challenger (30,000 cells:
/// 3.7 s measured in release on the box, 26 s at 80,000); it is now a sort plus one binary search per challenger.
/// WHY a wall-clock bound in a unit test: the quadratic cost is the thing under test and there is no work counter
/// to read; the bound is two orders above the fixed cost (about 10 ms release) and below the old cost on any host.
#[test]
fn a_plan_where_no_swap_passes_the_ceiling_is_not_quadratic() {
    let cells = 30_000usize;
    let owners: Vec<Rung> = (0..cells).map(|i| if i % 2 == 1 { 1 } else { 2 }).collect();
    let heat: Vec<f64> = (0..cells)
        .map(|i| {
            if i % 2 == 1 {
                1000.0
            } else {
                3000.0 + (i % 7) as f64
            }
        })
        .collect();
    let map = OwnershipMap::new(1, cells as u32, 3, owners).unwrap();
    let mut planner = PlacementPlanner::new(PlannerConfig {
        max_helper_layer_share: 0.0,
        ..PlannerConfig::default()
    })
    .unwrap();
    let started = Instant::now();
    let plan = planner.plan(map.snapshot(), &heat).unwrap();
    let took = started.elapsed();
    assert!(plan.promotions().is_empty(), "{:?}", plan.promotions());
    assert!(took < Duration::from_secs(1), "plan took {took:?}");
}
