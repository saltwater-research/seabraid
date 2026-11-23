// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! By-reference rows for masked climbs and heat walks: mask handling across hops, zero-mask equivalence, byte conservation.
//! Fixture and helpers come from `by_reference.rs`.

use super::by_reference::world;
use super::by_reference::*;
use super::pyramid::*;
use super::*;

fn spike() -> Vec<f64> {
    let mut heat = baseline_heat();
    heat[11] = 1e6;
    heat
}

fn plan_and_apply(
    map: &mut OwnershipMap,
    planner: &mut PlacementPlanner,
    io: &mut ByRef,
    heat: &[f64],
) -> AppliedRound {
    let plan = planner.plan(map.snapshot(), heat).unwrap();
    let applied = map.apply_plan(plan, io).unwrap();
    planner.record(&applied).unwrap();
    io.io.audit(map);
    applied
}

#[test]
fn a_masked_rung_is_not_asked_when_the_climber_bytes_are_already_held() {
    // Rung 4 is unmasked, so hop 1 stages the climber; hop 2's rung 3 is masked but must reuse those bytes.
    let (mut map, mut io) = world(1 << 3);
    let mut planner = PlacementPlanner::new(super::climb::climbing(2)).unwrap();
    plan_and_apply(&mut map, &mut planner, &mut io, &spike());
    assert_eq!(owner_of(&map, 0, 11), 3);
    assert_eq!(
        io.refs,
        [],
        "no reference call while the climber's bytes are staged"
    );
    assert_eq!(io.count(T::Stage(FALLBACK)), 1, "the climber is read once");
}

#[test]
fn a_climb_goes_by_reference_through_the_masked_rungs_and_by_bytes_after_them() {
    let mask = (1 << 4) | (1 << 3) | (1 << 2);
    let (mut map, mut io) = world(mask);
    let mut planner = PlacementPlanner::new(super::climb::climbing(8)).unwrap();
    let applied = plan_and_apply(&mut map, &mut planner, &mut io, &spike());
    assert!(applied
        .outcomes()
        .iter()
        .all(|o| matches!(o.outcome, MoveOutcome::Committed { .. })));
    assert_eq!(owner_of(&map, 0, 11), 0);
    let by_ref: Vec<Rung> = io.refs.iter().map(|r| r.0).collect();
    assert_eq!(
        by_ref,
        [4, 3, 2],
        "the first unmasked rung (1) ends the references"
    );
    assert_eq!(
        io.count(T::Stage(FALLBACK)),
        0,
        "the fallback is never asked for the climber's bytes"
    );
    // Hop 4 (rung 1) stages the climber from rung 2, where hop 3's reference put it; hop 5 reuses those bytes.
    assert_eq!(
        io.count(T::Stage(2)),
        1 + 1,
        "rung 2's displaced cell, then the climber for hop 4"
    );
    assert_eq!(map.snapshot().epoch(), 6);
    io.io.audit(&map);
}

/// The v1 oracle: an `io` that never answers `from_fallback` (here: a zero mask) must do exactly what the pyramid alone
/// does, call for call, on every row of the fault table.
#[test]
fn a_zero_mask_equals_the_provider_without_the_new_calls_on_every_row() {
    let errors = [
        None,
        Some(IoError::Refused),
        Some(IoError::Failed),
        Some(IoError::Timeout),
        Some(IoError::Panicked),
    ];
    for rung in [4, FALLBACK] {
        for op in [Op::Stage, Op::Replace] {
            for error in errors {
                let map = pyramid_map(true);
                let mut plain = Pyramid::new(&map);
                let mut wrapped = ByRef::new(&map, 0);
                plain.fail(rung, op, &[error]);
                wrapped.io.fail(rung, op, &[error]);
                let (mut left, mut right) = (pyramid_map(true), pyramid_map(true));
                let one = left.apply(1, SWAP(), &mut plain).unwrap();
                let two = right.apply(1, SWAP(), &mut wrapped).unwrap();
                let row = format!("{rung} {op:?} {error:?}");
                assert_eq!(one, two, "{row}");
                assert_eq!(plain.calls, wrapped.io.calls, "{row}");
                assert_eq!(plain.store, wrapped.io.store, "{row}");
                assert_eq!(
                    (left.snapshot().epoch(), left.snapshot().quarantined),
                    (right.snapshot().epoch(), right.snapshot().quarantined),
                    "{row}"
                );
                assert!(wrapped.refs.is_empty(), "{row}");
            }
        }
    }
}

/// The PY round trip with a pack behind the fallback: an expert goes up and back down by reference on the heat walk and
/// every byte of every cell is audited after every round.
#[test]
fn a_heat_walk_by_reference_keeps_every_byte_of_every_cell() {
    let (mut map, mut io) = world((1 << CARDS) - 1);
    let mut planner = PlacementPlanner::new(super::climb::climbing(8)).unwrap();
    for _ in 0..4 {
        plan_and_apply(&mut map, &mut planner, &mut io, &spike());
    }
    assert_eq!(
        owner_of(&map, 0, 11),
        0,
        "the spiked expert reached the lead"
    );
    let mut cold = baseline_heat();
    cold[11] = 0.25;
    for _ in 0..40 {
        plan_and_apply(&mut map, &mut planner, &mut io, &cold);
    }
    assert!(owner_of(&map, 0, 11) > 0, "and sank again");
    assert!(
        io.refs.len() >= 9,
        "the walk moved experts by reference: {}",
        io.refs.len()
    );
    never_closed(&map);
    assert_eq!(map.snapshot().quarantined, 0);
}
