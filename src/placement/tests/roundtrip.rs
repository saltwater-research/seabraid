// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! The owner's release gate on the host: one expert travels NVMe -> RAM -> helper 4 -> 3 -> 2 -> 1 -> lead and all the
//! way back down, with every byte of every cell verified after every round. A scripted heat walk drives it, once as a
//! climb (the whole way in one round) and once adjacent-only (one boundary per cooldown window).

use super::climb::climbing;
use super::pyramid::*;
use super::*;

const X: u32 = 11; // layer 0's coldest expert: on the fallback and NOT in RAM at the start
const FALLBACK: Rung = CARDS;

fn heat_with(x: f64) -> Vec<f64> {
    let mut heat = baseline_heat();
    heat[X as usize] = x;
    heat
}

/// Runs rounds on `heat` until `x` sits on `rung`; returns how many rounds that took.
fn rounds_until(
    (map, planner, io): (&mut OwnershipMap, &mut PlacementPlanner, &mut Pyramid),
    heat: &[f64],
    rung: Rung,
) -> usize {
    for round in 1..=300 {
        run_round(map, planner, io, heat);
        if owner_of(map, 0, X) == rung {
            return round;
        }
    }
    panic!("the expert never reached rung {rung}");
}

fn walk(config: PlannerConfig) -> (usize, usize) {
    let mut map = pyramid_map(true);
    let mut planner = PlacementPlanner::new(config).unwrap();
    let mut io = Pyramid::new(&map);
    let x = cell(0, X);
    assert!(
        !io.ram.contains(&x) && owner_of(&map, 0, X) == FALLBACK,
        "starts NVMe-only"
    );

    // NVMe -> RAM: the engine's 1:1 step gives the hot cell the RAM place of a colder fallback cell.
    io.ram_adapt(x, cell(0, X - 1));
    assert!(io.ram.contains(&x));

    // RAM -> helper 4 -> 3 -> 2 -> 1 -> lead.
    let up = rounds_until((&mut map, &mut planner, &mut io), &heat_with(1e6), 0);
    assert_eq!(
        io.store[&(0, x)],
        source(x),
        "its bytes on the lead are the pack's"
    );
    assert!(
        !io.ram.contains(&x),
        "its RAM place went to the cell that left the helper"
    );
    assert_eq!(io.ram.len(), 2, "RAM stays the same size");

    // And back down: the heat walk reverses, the expert is the coldest again and sinks to the fallback.
    let down = rounds_until(
        (&mut map, &mut planner, &mut io),
        &heat_with(0.25),
        FALLBACK,
    );
    // Arriving on the fallback it keeps the RAM place of the cell it replaced; the engine's own 1:1 step later gives
    // that place to a hotter cell that has none, and the expert is NVMe-only again.
    assert!(io.ram.contains(&x));
    let hot = cells(map.snapshot())
        .find(|c| owner_of(&map, c.layer, c.expert) == FALLBACK && !io.ram.contains(c))
        .unwrap();
    io.ram_adapt(hot, x);
    assert!(!io.ram.contains(&x), "back to NVMe-only");
    io.audit(&map);
    (up, down)
}

#[test]
fn an_expert_travels_the_whole_pyramid_up_and_down_in_climb_mode() {
    let (up, down) = walk(climbing(8));
    assert_eq!(
        up, 1,
        "a climb takes the expert all the way up in one round"
    );
    // Down is paced by the planner's smoothing (0.75 old + 0.25 new), not by the climb: a heat of 1e6 has to decay
    // below its neighbours' before the expert is the coldest in its layer, then it drops a rung per round.
    assert!(down > up);
}

#[test]
fn an_expert_travels_the_whole_pyramid_up_and_down_adjacent_only() {
    let config = PlannerConfig {
        cooldown_rounds: 2,
        ..climbing(1)
    };
    let (up, _) = walk(config);
    // Five boundaries; the climber hops again when its cooldown has passed: rounds 1, 4, 7, 10, 13.
    let (hops, cooldown) = (5, 2);
    assert_eq!(up, 1 + (hops - 1) * (cooldown + 1));
}

#[test]
fn every_boundary_and_provider_step_fails_into_a_documented_outcome_with_every_byte_intact() {
    // Climb the full chain; fail each boundary's replace and stage in turn with each status. Boundaries are named by
    // the upper rung (4 is the helper with the fallback, 0 is the lead with helper 1).
    for upper in (0..CARDS).rev() {
        for op in [Op::Stage, Op::Replace] {
            for error in [IoError::Refused, IoError::Failed, IoError::Timeout] {
                let mut map = pyramid_map(true);
                let mut planner = PlacementPlanner::new(climbing(8)).unwrap();
                let mut io = Pyramid::new(&map);
                // A card's own stage/replace of this boundary fails; the lower rung of the climb is `upper + 1`.
                io.fail(upper, op, &[Some(error)]);
                let applied = run_round(&mut map, &mut planner, &mut io, &heat_with(1e6));
                let kinds: Vec<_> = applied.outcomes().iter().map(|o| o.outcome).collect();
                let at = (CARDS - 1 - upper) as usize; // hop number of this boundary in the climb
                assert!(
                    kinds[..at]
                        .iter()
                        .all(|k| matches!(k, MoveOutcome::Committed { .. })),
                    "{kinds:?}"
                );
                match (&kinds[at], error) {
                    (MoveOutcome::Quarantined { .. }, _) => {
                        assert_eq!(error, IoError::Timeout, "{upper} {op:?}");
                        assert!(map.snapshot().is_quarantined(upper), "{upper} {op:?}");
                        assert_eq!(kinds.len(), at + 1, "a quarantine ends the round");
                    }
                    (MoveOutcome::Rejected(_) | MoveOutcome::Restored(_), _) => {
                        assert!(
                            kinds[at + 1..].iter().all(|k| *k == MoveOutcome::Skipped),
                            "{kinds:?}"
                        );
                        assert_eq!(map.snapshot().quarantined, 0);
                    }
                    other => panic!("{upper} {op:?} {error:?}: unexpected {other:?}"),
                }
                assert!(!map.snapshot().is_quarantined(FALLBACK));
                io.audit(&map); // run_round audited already; the fallback and every open card serve source bytes
            }
        }
    }
}
