// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! R6: the Rust planner chooses the same moves as the legacy C++ ladder.
//! `ladder_golden.txt` is written by `compat/strata/tests/ladder_golden_gen.cpp`
//! (C++ `plan_tier_round` ladder plus `to_pairs`, 200 seeded cases, 6 rounds each,
//! every move acknowledged). Each case carries its inputs (shape, knobs, owners,
//! per-round heat), so no second PRNG can drift from the first. WHY all-ACK and no
//! fills: Rust never proposes fills, and a refusal path has no C++ twin.
//! Regenerate: see the generator header, then diff the file in review.

use super::*;

const GOLDEN: &str = include_str!("ladder_golden.txt");

struct Case {
    seed: u64,
    shape: (u32, u32, Rung),
    config: PlannerConfig,
    owner: Vec<Rung>,
    rounds: Vec<(Vec<f64>, Vec<Promotion>)>,
    last: Vec<Rung>,
}

fn nums<T: std::str::FromStr>(line: &str, tag: &str) -> Vec<T> {
    let rest = line
        .strip_prefix(tag)
        .unwrap_or_else(|| panic!("{tag}: {line}"));
    rest.split_whitespace()
        .map(|t| t.parse().unwrap_or_else(|_| panic!("{tag}: bad {t}")))
        .collect()
}

fn promotion(token: &str) -> Promotion {
    let f: Vec<u32> = token.split(':').map(|t| t.parse().unwrap()).collect();
    p(f[0], f[1], f[2], f[3] as Rung)
}

fn parse_case(head: &str, lines: &mut std::iter::Peekable<std::str::Lines>) -> Case {
    let h: Vec<&str> = head.split_whitespace().collect();
    let num = |i: usize| h[i].parse::<f64>().unwrap();
    let config = PlannerConfig {
        max_moves_per_round: num(5) as u16,
        promote_margin: num(6),
        cooldown_rounds: num(7) as u64,
        max_helper_layer_share: num(8),
        ladder_boundaries: num(9) as Rung,
        ..PlannerConfig::default()
    };
    let owner = nums(lines.next().unwrap(), "owner");
    let mut rounds = Vec::new();
    while lines.peek().is_some_and(|l| l.starts_with("heat")) {
        let heat = nums(lines.next().unwrap(), "heat");
        let moves = lines.next().unwrap().strip_prefix("moves").unwrap();
        rounds.push((heat, moves.split_whitespace().map(promotion).collect()));
    }
    let last = nums(lines.next().unwrap(), "final");
    Case {
        seed: num(1) as u64,
        shape: (num(2) as u32, num(3) as u32, num(4) as Rung - 1),
        config,
        owner,
        rounds,
        last,
    }
}

fn parse(text: &str) -> Vec<Case> {
    let mut lines = text.lines().peekable();
    let mut cases = Vec::new();
    while let Some(head) = lines.next() {
        cases.push(parse_case(head, &mut lines));
    }
    cases
}

fn replay(case: &Case) -> Result<usize, String> {
    let (layers, experts, fallback) = case.shape;
    let at = |round: usize| format!("seed {} round {round}", case.seed);
    let mut place = OwnershipMap::new(layers, experts, fallback, case.owner.clone()).unwrap();
    let mut io = Providers::new(&place);
    let mut planner = PlacementPlanner::new(case.config).unwrap();
    let mut moved = 0;
    for (round, (heat, want)) in case.rounds.iter().enumerate() {
        let plan = planner
            .plan(place.snapshot(), heat)
            .map_err(|e| format!("{}: {e:?}", at(round)))?;
        if plan.promotions() != want.as_slice() {
            let rust = plan.promotions();
            return Err(format!("{}: rust {rust:?} != c++ {want:?}", at(round)));
        }
        moved += want.len();
        let applied = place
            .apply_plan(plan, &mut io)
            .map_err(|e| format!("{}: {e:?}", at(round)))?;
        planner
            .record(&applied)
            .map_err(|e| format!("{}: {e:?}", at(round)))?;
    }
    let owner: Vec<Rung> = cells(place.snapshot())
        .map(|c| place.snapshot().owner(c).unwrap())
        .collect();
    if owner != case.last {
        return Err(format!("seed {}: final owners differ", case.seed));
    }
    Ok(moved)
}

#[test]
fn rust_planner_chooses_the_moves_of_the_cpp_ladder() {
    let cases = parse(GOLDEN);
    assert_eq!(cases.len(), 200);
    let mut moved = 0;
    for case in &cases {
        match replay(case) {
            Ok(n) => moved += n,
            Err(first) => panic!("first divergence: {first}"),
        }
    }
    // A golden of empty rounds would pass vacuously: the corpus must exercise real moves.
    assert!(moved >= 500, "only {moved} promotions in the golden");
}
