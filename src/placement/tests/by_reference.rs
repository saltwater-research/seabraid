// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! The by-reference rows of docs/PLACEMENT-ABI.md ("By-reference moves"), one test per row, on the pyramid fixture
//! with a "pack" behind the fallback: a rung asked to load an expert by reference reads `source(new)` itself.
//! OWNER: this file owns the by-reference fixture (`ByRef`); the rules under test live in `transaction/by_reference.rs`.

use super::pyramid::*;
use super::*;
use crate::diag::testing::Capture;
use crate::diag::Level;

/// Layer 0's expert 11 (fallback) takes the place of expert 9 (rung 4, the last helper).
pub(super) const SWAP: fn() -> Promotion = || p(0, 11, 9, 4);
pub(super) const FALLBACK: Rung = CARDS;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum T {
    Stage(Rung),
    Ref(Rung),
    Replace(Rung),
}

/// The pyramid plus the file-reference answer: rung r in `mask` is asked by reference. `ref_faults` is queued like
/// the pyramid's own faults (None is an ACK).
pub(super) struct ByRef {
    pub(super) io: Pyramid,
    pub(super) mask: u64,
    pub(super) trace: Vec<T>,
    pub(super) refs: Vec<(Rung, ExpertId, ExpertId)>,
    pub(super) ref_faults: Faults,
}

impl ByRef {
    pub(super) fn new(map: &OwnershipMap, mask: u64) -> Self {
        Self {
            io: Pyramid::new(map),
            mask,
            trace: Vec::new(),
            refs: Vec::new(),
            ref_faults: Faults::new(),
        }
    }

    pub(super) fn count(&self, call: T) -> usize {
        self.trace.iter().filter(|&&seen| seen == call).count()
    }

    pub(super) fn touched_fallback(&self) -> bool {
        self.trace
            .iter()
            .any(|call| matches!(call, T::Stage(FALLBACK) | T::Replace(FALLBACK)))
    }
}

impl PlacementIo for ByRef {
    fn stage(&mut self, rung: Rung, cell: ExpertId) -> Result<Vec<u8>, IoError> {
        self.trace.push(T::Stage(rung));
        self.io.stage(rung, cell)
    }

    fn replace(
        &mut self,
        rung: Rung,
        old: ExpertId,
        new: ExpertId,
        bytes: &[u8],
    ) -> Result<(), IoError> {
        self.trace.push(T::Replace(rung));
        self.io.replace(rung, old, new, bytes)
    }

    fn takes_from_fallback(&self, rung: Rung) -> bool {
        (self.mask >> rung) & 1 == 1
    }

    fn replace_from_fallback(
        &mut self,
        rung: Rung,
        old: ExpertId,
        new: ExpertId,
    ) -> Result<(), IoError> {
        self.trace.push(T::Ref(rung));
        self.refs.push((rung, old, new));
        let fault = self.ref_faults.pop_front().flatten();
        let slot = self
            .io
            .store
            .get_mut(&(rung, old))
            .ok_or(IoError::Refused)?;
        match fault {
            Some(IoError::Refused) => return Err(IoError::Refused),
            Some(error) => {
                slot[0] = !slot[0]; // the card began loading from the file: the slot is torn
                if error == IoError::Panicked {
                    panic!("injected by-reference panic");
                }
                return Err(error);
            }
            None => {}
        }
        // The rung read the file and put the pack's bytes for `new` into the slot of `old`.
        self.io.store.remove(&(rung, old));
        self.io.store.insert((rung, new), source(new));
        Ok(())
    }
}

pub(super) fn world(mask: u64) -> (OwnershipMap, ByRef) {
    let map = pyramid_map(true);
    let io = ByRef::new(&map, mask);
    (map, io)
}

pub(super) const ON_4: u64 = 1 << 4;

pub(super) fn go(map: &mut OwnershipMap, io: &mut ByRef) -> MoveOutcome {
    map.apply(1, SWAP(), io).unwrap()
}

pub(super) fn never_closed(map: &OwnershipMap) {
    assert_eq!(
        map.snapshot().quarantined & (1 << FALLBACK),
        0,
        "the fallback's bit must never be set"
    );
}

#[test]
fn an_acknowledged_reference_loads_the_upper_card_without_staging_the_climber() {
    let (mut map, mut io) = world(ON_4);
    assert_eq!(go(&mut map, &mut io), MoveOutcome::Committed { epoch: 2 });
    // Displaced staged on the upper card, the reference, then the displaced written to the fallback: no stage of the climber.
    assert_eq!(io.trace, [T::Stage(4), T::Ref(4), T::Replace(FALLBACK)]);
    assert_eq!(io.refs, [(4, cell(0, 9), cell(0, 11))]);
    assert_eq!((owner_of(&map, 0, 11), owner_of(&map, 0, 9)), (4, FALLBACK));
    io.io.audit(&map);
}

#[test]
fn a_refusal_runs_the_byte_path_in_the_same_swap_and_is_not_logged() {
    let (mut map, mut io) = world(ON_4);
    io.ref_faults.push_back(Some(IoError::Refused));
    let capture = Capture::start(Level::Debug);
    assert_eq!(go(&mut map, &mut io), MoveOutcome::Committed { epoch: 2 });
    assert_eq!(
        io.trace,
        [
            T::Stage(4),
            T::Ref(4),
            T::Stage(FALLBACK),
            T::Replace(4),
            T::Replace(FALLBACK)
        ]
    );
    assert!(
        capture.mine().is_empty(),
        "a refusal is the byte path's cue, not a failure: {:?}",
        capture.mine()
    );
    assert_eq!((map.snapshot().epoch(), map.snapshot().quarantined), (2, 0));
    io.io.audit(&map);
}

#[test]
fn a_failed_reference_restores_the_upper_card_and_never_touches_the_fallback() {
    let (mut map, mut io) = world(ON_4);
    io.ref_faults.push_back(Some(IoError::Failed));
    let capture = Capture::start(Level::Debug);
    assert_eq!(
        go(&mut map, &mut io),
        MoveOutcome::Restored(issue(4, IoPhase::Replace, IoError::Failed))
    );
    assert_eq!(
        io.trace,
        [T::Stage(4), T::Ref(4), T::Replace(4)],
        "the restore is the only write"
    );
    assert!(!io.touched_fallback());
    assert!(
        capture
            .mine()
            .iter()
            .any(|r| r.level == Level::Error as u32),
        "a failure is logged at ERROR"
    );
    assert_eq!((map.snapshot().epoch(), map.snapshot().quarantined), (1, 0));
    never_closed(&map);
    io.io.audit(&map); // the torn slot holds the displaced expert's pack bytes again
}

#[test]
fn a_failed_reference_whose_restore_fails_closes_the_upper_card_only() {
    let (mut map, mut io) = world(ON_4);
    io.ref_faults.push_back(Some(IoError::Failed));
    io.io.fail(4, Op::Replace, &[Some(IoError::Failed)]); // the restore
    let initial = issue(4, IoPhase::Replace, IoError::Failed);
    let restore = issue(4, IoPhase::Restore, IoError::Failed);
    assert_eq!(
        go(&mut map, &mut io),
        MoveOutcome::Quarantined {
            initial,
            restore: Some(restore)
        }
    );
    assert_eq!(
        (map.snapshot().quarantined, map.snapshot().epoch()),
        (1 << 4, 2)
    );
    assert!(!io.touched_fallback());
    never_closed(&map);
}

#[test]
fn a_timeout_or_panic_on_the_reference_closes_the_upper_card_and_sends_no_restore() {
    for error in [IoError::Timeout, IoError::Panicked] {
        let (mut map, mut io) = world(ON_4);
        io.ref_faults.push_back(Some(error));
        let initial = issue(4, IoPhase::Replace, error);
        assert_eq!(
            go(&mut map, &mut io),
            MoveOutcome::Quarantined {
                initial,
                restore: None
            },
            "{error:?}"
        );
        assert_eq!(
            io.trace,
            [T::Stage(4), T::Ref(4)],
            "{error:?}: nothing may be sent to a card that may still be writing"
        );
        assert_eq!(
            (map.snapshot().quarantined, map.snapshot().epoch()),
            (1 << 4, 2)
        );
        never_closed(&map);
    }
}

#[test]
fn a_displaced_stage_failure_on_the_upper_card_makes_no_reference_call() {
    for (error, closed) in [
        (IoError::Refused, 0),
        (IoError::Failed, 0),
        (IoError::Timeout, 1 << 4),
    ] {
        let (mut map, mut io) = world(ON_4);
        io.io.fail(4, Op::Stage, &[Some(error)]);
        let got = go(&mut map, &mut io);
        let initial = issue(4, IoPhase::Stage, error);
        let expected = if closed == 0 {
            MoveOutcome::Rejected(initial)
        } else {
            MoveOutcome::Quarantined {
                initial,
                restore: None,
            }
        };
        assert_eq!(got, expected, "{error:?}");
        assert_eq!(io.trace, [T::Stage(4)], "{error:?}");
        assert_eq!(map.snapshot().quarantined, closed);
    }
}

#[test]
fn a_refusal_followed_by_a_failing_fallback_stage_is_rejected_and_never_closes_the_fallback() {
    for error in [
        IoError::Refused,
        IoError::Failed,
        IoError::Timeout,
        IoError::Panicked,
    ] {
        let (mut map, mut io) = world(ON_4);
        io.ref_faults.push_back(Some(IoError::Refused));
        io.io.fail(FALLBACK, Op::Stage, &[Some(error)]);
        let rejected = MoveOutcome::Rejected(issue(FALLBACK, IoPhase::Stage, error));
        assert_eq!(go(&mut map, &mut io), rejected, "{error:?}");
        assert_eq!(
            io.trace,
            [T::Stage(4), T::Ref(4), T::Stage(FALLBACK)],
            "{error:?}"
        );
        assert_eq!((map.snapshot().epoch(), map.snapshot().quarantined), (1, 0));
        never_closed(&map);
        io.io.audit(&map);
    }
}

#[test]
fn a_failing_fallback_write_after_an_acknowledged_reference_restores_the_upper_card_from_the_staged_displaced_bytes(
) {
    for error in [
        IoError::Refused,
        IoError::Failed,
        IoError::Timeout,
        IoError::Panicked,
    ] {
        let (mut map, mut io) = world(ON_4);
        io.io.fail(FALLBACK, Op::Replace, &[Some(error)]);
        let restored = MoveOutcome::Restored(issue(FALLBACK, IoPhase::Replace, error));
        assert_eq!(go(&mut map, &mut io), restored, "{error:?}");
        assert_eq!(
            io.trace,
            [T::Stage(4), T::Ref(4), T::Replace(FALLBACK), T::Replace(4)],
            "{error:?}"
        );
        assert_eq!((map.snapshot().epoch(), map.snapshot().quarantined), (1, 0));
        assert_eq!((owner_of(&map, 0, 11), owner_of(&map, 0, 9)), (FALLBACK, 4));
        never_closed(&map);
        io.io.audit(&map); // the displaced expert is back on rung 4, the climber's copy is gone from it
    }
}
