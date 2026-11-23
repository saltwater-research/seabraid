// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! The fallback rules table of docs/PLACEMENT-ABI.md, one test per row: a swap at (last helper, fallback) is
//! "load into the helper, evict to the fallback", the fallback is never closed, and any non-ACK from the fallback's
//! replace undoes the helper's write.

use super::pyramid::*;
use super::*;
use crate::diag::testing::Capture;
use crate::diag::Level;

/// Layer 0's expert 11 (fallback) takes the place of expert 9 (rung 4, the last helper).
const SWAP: fn() -> Promotion = || p(0, 11, 9, 4);
const FALLBACK: Rung = CARDS;

fn world() -> (OwnershipMap, Pyramid) {
    let map = pyramid_map(true);
    let io = Pyramid::new(&map);
    (map, io)
}

fn outcome(map: &mut OwnershipMap, io: &mut Pyramid) -> MoveOutcome {
    map.apply(1, SWAP(), io).unwrap()
}

fn never_closed(map: &OwnershipMap) {
    assert!(!map.snapshot().is_quarantined(FALLBACK));
    assert_eq!(
        map.snapshot().quarantined & (1 << FALLBACK),
        0,
        "the fallback's bit must never be set"
    );
}

#[test]
fn an_acknowledged_swap_loads_the_helper_and_evicts_to_the_fallback() {
    let (mut map, mut io) = world();
    assert_eq!(
        outcome(&mut map, &mut io),
        MoveOutcome::Committed { epoch: 2 }
    );
    assert_eq!((owner_of(&map, 0, 11), owner_of(&map, 0, 9)), (4, FALLBACK));
    assert_eq!(
        io.calls,
        [
            (Op::Stage, FALLBACK),
            (Op::Stage, 4),
            (Op::Replace, 4),
            (Op::Replace, FALLBACK)
        ]
    );
    io.audit(&map);
}

#[test]
fn a_swap_naming_the_fallback_is_refused_on_a_map_without_the_fallback_boundary() {
    let mut map = pyramid_map(false);
    let mut io = Pyramid::new(&map);
    assert_eq!(
        map.apply(1, SWAP(), &mut io),
        Err(PlacementError::InvalidPromotion)
    );
    assert!(io.calls.is_empty());
    // And one card has no such boundary: the lead with the fallback stays the engine's.
    let mut one = OwnershipMap::new(1, 2, 1, vec![0, 1]).unwrap();
    assert_eq!(
        one.enable_fallback_boundary(),
        Err(PlacementError::InvalidPolicy)
    );
}

#[test]
fn a_stage_on_the_fallback_that_does_not_ack_is_rejected_and_never_closes_it() {
    for error in [
        IoError::Refused,
        IoError::Failed,
        IoError::Timeout,
        IoError::Panicked,
    ] {
        let (mut map, mut io) = world();
        io.fail(FALLBACK, Op::Stage, &[Some(error)]);
        let capture = Capture::start(Level::Debug);
        let issue = ProviderIssue {
            rung: FALLBACK,
            phase: IoPhase::Stage,
            error,
        };
        assert_eq!(
            outcome(&mut map, &mut io),
            MoveOutcome::Rejected(issue),
            "{error:?}"
        );
        assert_eq!((map.snapshot().epoch(), map.snapshot().quarantined), (1, 0));
        assert_eq!(
            io.count(Op::Replace, 4) + io.count(Op::Replace, FALLBACK),
            0,
            "a stage writes nothing"
        );
        let level = if error == IoError::Refused {
            Level::Warn
        } else {
            Level::Error
        };
        assert!(
            capture.mine().iter().any(|r| r.level == level as u32),
            "{error:?}"
        );
        never_closed(&map);
        io.audit(&map);
    }
}

#[test]
fn stage_and_replace_on_the_helper_keep_the_card_rules() {
    let issue = |phase, error| ProviderIssue {
        rung: 4,
        phase,
        error,
    };
    let cases = [
        (
            Op::Stage,
            IoError::Refused,
            MoveOutcome::Rejected(issue(IoPhase::Stage, IoError::Refused)),
        ),
        (
            Op::Replace,
            IoError::Refused,
            MoveOutcome::Rejected(issue(IoPhase::Replace, IoError::Refused)),
        ),
        (
            Op::Replace,
            IoError::Failed,
            MoveOutcome::Restored(issue(IoPhase::Replace, IoError::Failed)),
        ),
    ];
    for (op, error, expected) in cases {
        let (mut map, mut io) = world();
        io.fail(4, op, &[Some(error)]);
        assert_eq!(outcome(&mut map, &mut io), expected);
        assert_eq!(map.snapshot().quarantined, 0);
        io.audit(&map);
    }
    for op in [Op::Stage, Op::Replace] {
        let (mut map, mut io) = world();
        io.fail(4, op, &[Some(IoError::Timeout)]);
        let phase = if op == Op::Stage {
            IoPhase::Stage
        } else {
            IoPhase::Replace
        };
        let initial = issue(phase, IoError::Timeout);
        assert_eq!(
            outcome(&mut map, &mut io),
            MoveOutcome::Quarantined {
                initial,
                restore: None
            }
        );
        assert_eq!(map.snapshot().quarantined, 1 << 4);
        never_closed(&map);
    }
}

#[test]
fn a_replace_on_the_fallback_that_does_not_ack_restores_the_helper_and_closes_nothing() {
    for error in [
        IoError::Refused,
        IoError::Failed,
        IoError::Timeout,
        IoError::Panicked,
    ] {
        let (mut map, mut io) = world();
        io.fail(FALLBACK, Op::Replace, &[Some(error)]);
        let capture = Capture::start(Level::Debug);
        let issue = ProviderIssue {
            rung: FALLBACK,
            phase: IoPhase::Replace,
            error,
        };
        assert_eq!(
            outcome(&mut map, &mut io),
            MoveOutcome::Restored(issue),
            "{error:?}"
        );
        assert_eq!((map.snapshot().epoch(), map.snapshot().quarantined), (1, 0));
        assert_eq!((owner_of(&map, 0, 11), owner_of(&map, 0, 9)), (FALLBACK, 4));
        // Upper write, the failed fallback write, and the restore of the helper from the staged original.
        assert_eq!(io.count(Op::Replace, 4), 2);
        // mine() drains, so take the records once.
        let records = capture.mine();
        let breach = records
            .iter()
            .any(|r| r.level == Level::Error as u32 && r.message.contains("fallback"));
        assert_eq!(
            breach,
            error != IoError::Refused,
            "{error:?}: a contract breach is logged at ERROR, a refusal is not"
        );
        // One failure, one record: every record about the fallback replace (rung 5) is the single breach line
        // (non-refused) or the single refusal warning, never both.
        let about_it: Vec<_> = records
            .iter()
            .filter(|r| r.message.contains("rung 5 ") || r.message.contains("fallback rung"))
            .collect();
        assert_eq!(about_it.len(), 1, "{error:?}: {about_it:?}");
        never_closed(&map);
        io.audit(&map);
    }
}

#[test]
fn a_failed_restore_closes_the_helper_only() {
    let (mut map, mut io) = world();
    io.fail(FALLBACK, Op::Replace, &[Some(IoError::Failed)]);
    io.fail(4, Op::Replace, &[None, Some(IoError::Failed)]); // the upper write acks, its restore fails
    let initial = ProviderIssue {
        rung: FALLBACK,
        phase: IoPhase::Replace,
        error: IoError::Failed,
    };
    let restore = ProviderIssue {
        rung: 4,
        phase: IoPhase::Restore,
        error: IoError::Failed,
    };
    assert_eq!(
        outcome(&mut map, &mut io),
        MoveOutcome::Quarantined {
            initial,
            restore: Some(restore)
        }
    );
    assert_eq!(map.snapshot().quarantined, 1 << 4);
    assert_eq!(map.snapshot().epoch(), 2);
    never_closed(&map);
    // The closed card's cells now execute on the fallback, which serves them from verified bytes.
    assert_eq!(map.snapshot().executor(cell(0, 9)), Some(FALLBACK));
}

#[test]
fn the_fallback_ram_place_follows_the_swap_one_to_one() {
    let (mut map, mut io) = world();
    let before = io.ram.len();
    // Expert 10 holds RAM on the fallback; promoting it gives its place to the cell that leaves the helper.
    map.apply(1, p(0, 10, 9, 4), &mut io).unwrap();
    assert_eq!(io.ram.len(), before);
    assert!(io.ram.contains(&cell(0, 9)) && !io.ram.contains(&cell(0, 10)));
}
