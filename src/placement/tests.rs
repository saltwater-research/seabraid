// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! Host reference checks ported from Strata `tests/core/ownership_map_test.cpp`.
//! Providers keep actual slot bytes and mutate half a slot on failed copies.

mod by_reference;
mod by_reference_climb;
mod climb;
mod fallback;
mod fill_heat;
mod layout;
mod parity;
mod planner;
mod policy_demand;
mod policy_demand_cooldown;
mod pyramid;
mod roundtrip;

use super::*;
use std::collections::{HashMap, VecDeque};

type Faults = VecDeque<Option<IoError>>;

#[derive(Default)]
struct Providers {
    slots: HashMap<(Rung, ExpertId), Vec<u8>>,
    stages: Vec<Rung>,
    writes: Vec<Rung>,
    stage_fault: Faults,
    write_fault: Faults,
}

impl Providers {
    fn new(place: &OwnershipMap) -> Self {
        let place = place.snapshot();
        let mut io = Self::default();
        for cell in cells(place) {
            let owner = place.owner(cell).unwrap();
            if owner != place.fallback() {
                io.slots.insert((owner, cell), source(cell));
            }
        }
        io
    }

    fn reference_matches(&self, place: &OwnershipMap) {
        let place = place.snapshot();
        for cell in cells(place) {
            let exec = place.executor(cell).unwrap();
            let actual = if exec == place.fallback() {
                source(cell)
            } else {
                assert!(!place.is_quarantined(exec), "unsafe route for {cell:?}");
                self.slots[&(exec, cell)].clone()
            };
            assert_eq!(actual, source(cell), "wrong bytes for {cell:?}");
        }
        for &(rung, cell) in self.slots.keys() {
            if !place.is_quarantined(rung) {
                assert_eq!(
                    place.owner(cell),
                    Some(rung),
                    "open card has duplicate or stale slot"
                );
            }
        }
    }
}

impl PlacementIo for Providers {
    fn stage(&mut self, rung: Rung, cell: ExpertId) -> Result<Vec<u8>, IoError> {
        self.stages.push(rung);
        if let Some(Some(error)) = self.stage_fault.pop_front() {
            return Err(error);
        }
        self.slots
            .get(&(rung, cell))
            .cloned()
            .ok_or(IoError::Refused)
    }

    fn replace(
        &mut self,
        rung: Rung,
        old: ExpertId,
        new: ExpertId,
        bytes: &[u8],
    ) -> Result<(), IoError> {
        self.writes.push(rung);
        let slot = self.slots.get_mut(&(rung, old)).ok_or(IoError::Refused)?;
        match self.write_fault.pop_front().flatten() {
            Some(IoError::Refused) => return Err(IoError::Refused),
            Some(error) => {
                // A torn in-place restore could otherwise write already
                // correct bytes, making it impossible to tell the path ran.
                slot[0] = !bytes[0];
                assert_ne!(slot, bytes);
                if error == IoError::Panicked {
                    panic!("injected adapter panic after a partial write");
                }
                return Err(error);
            }
            None => {}
        }
        assert_eq!(bytes, source(new), "provider received wrong staged bytes");
        self.slots.remove(&(rung, old));
        self.slots.insert((rung, new), bytes.to_vec());
        Ok(())
    }
}

fn cells(place: &PlacementSnapshot) -> impl Iterator<Item = ExpertId> {
    let (layers, experts) = place.shape();
    (0..layers).flat_map(move |layer| (0..experts).map(move |expert| ExpertId { layer, expert }))
}

fn source(cell: ExpertId) -> Vec<u8> {
    (0..48)
        .map(|offset| (cell.layer * 131 + cell.expert * 29 + offset * 7 + 1) as u8)
        .collect()
}

fn p(layer: u32, promote: u32, displace: u32, upper_rung: Rung) -> Promotion {
    Promotion {
        layer,
        promote,
        displace,
        upper_rung,
    }
}

fn issue(rung: Rung, phase: IoPhase, error: IoError) -> ProviderIssue {
    ProviderIssue { rung, phase, error }
}

fn world() -> (OwnershipMap, Providers) {
    let place = OwnershipMap::new(2, 4, 2, vec![0, 0, 1, 2, 0, 1, 1, 2]).unwrap();
    let io = Providers::new(&place);
    (place, io)
}

#[test]
fn commits_two_layered_cells_only_after_both_acknowledgements() {
    let (mut place, mut io) = world();
    let before = place.snapshot().clone();
    let exchange = p(1, 2, 0, 0);
    assert_eq!(
        place.apply(before.epoch(), exchange, &mut io),
        Ok(MoveOutcome::Committed { epoch: 2 })
    );
    assert_eq!(io.stages, [1, 0]);
    assert_eq!(io.writes, [0, 1]);
    let [up, down] = exchange.cells();
    assert_eq!((before.owner(up), before.owner(down)), (Some(1), Some(0)));
    assert_eq!(
        (place.snapshot().owner(up), place.snapshot().owner(down)),
        (Some(0), Some(1))
    );
    assert_eq!(
        place.snapshot().owner(ExpertId {
            layer: 0,
            expert: 2
        }),
        Some(1)
    );
    io.reference_matches(&place);
}

#[test]
fn rejected_stage_is_untouched_but_stage_timeout_quarantines() {
    let (mut place, mut io) = world();
    io.stage_fault = [Some(IoError::Refused), Some(IoError::Timeout)].into();
    let promotion = p(0, 2, 1, 0);
    assert_eq!(
        place.apply(1, promotion, &mut io),
        Ok(MoveOutcome::Rejected(issue(
            1,
            IoPhase::Stage,
            IoError::Refused
        )))
    );
    assert_eq!(place.snapshot().epoch(), 1);
    assert_eq!(
        place.apply(1, promotion, &mut io),
        Ok(MoveOutcome::Quarantined {
            initial: issue(1, IoPhase::Stage, IoError::Timeout),
            restore: None,
        })
    );
    assert_eq!(place.snapshot().epoch(), 2);
    assert!(io.writes.is_empty());
    io.reference_matches(&place);
}

#[test]
fn torn_upper_and_refused_lower_restore_original_bytes_and_epoch() {
    for (faults, failed_rung, error, writes) in [
        (vec![Some(IoError::Failed)], 0, IoError::Failed, vec![0, 0]),
        (
            vec![None, Some(IoError::Refused)],
            1,
            IoError::Refused,
            vec![0, 1, 0],
        ),
    ] {
        let (mut place, mut io) = world();
        io.write_fault = faults.into();
        assert_eq!(
            place.apply(1, p(0, 2, 0, 0), &mut io),
            Ok(MoveOutcome::Restored(issue(
                failed_rung,
                IoPhase::Replace,
                error
            )))
        );
        assert_eq!(io.writes, writes);
        assert_eq!(place.snapshot().epoch(), 1);
        io.reference_matches(&place);
    }
}

#[test]
fn lower_copy_failure_never_routes_to_or_rewrites_the_uncertain_card() {
    for fault in [IoError::Failed, IoError::Timeout, IoError::Panicked] {
        let (mut place, mut io) = world();
        io.write_fault = [None, Some(fault)].into();
        let exchange = p(0, 2, 1, 0);
        assert_eq!(
            place.apply(1, exchange, &mut io),
            Ok(MoveOutcome::Quarantined {
                initial: issue(1, IoPhase::Replace, fault),
                restore: None,
            })
        );
        let [up, down] = exchange.cells();
        let snapshot = place.snapshot();
        assert_eq!(
            (snapshot.owner(up), snapshot.owner(down)),
            (Some(1), Some(0))
        );
        assert_eq!(snapshot.executor(up), Some(snapshot.fallback()));
        assert_eq!(snapshot.epoch(), 2);
        assert_eq!(
            place.apply(2, exchange, &mut io),
            Err(PlacementError::InvalidPromotion)
        );
        assert_eq!(io.writes, [0, 1, 0]);
        io.reference_matches(&place);
    }
}

#[test]
fn timeout_never_rewrites_its_rung_and_failed_restore_quarantines_both() {
    let (mut place, mut io) = world();
    io.write_fault = [Some(IoError::Timeout)].into();
    assert_eq!(
        place.apply(1, p(0, 2, 0, 0), &mut io),
        Ok(MoveOutcome::Quarantined {
            initial: issue(0, IoPhase::Replace, IoError::Timeout),
            restore: None,
        })
    );
    assert_eq!(io.writes, [0]);
    io.reference_matches(&place);

    let (mut place, mut io) = world();
    io.write_fault = [None, Some(IoError::Timeout), Some(IoError::Failed)].into();
    assert_eq!(
        place.apply(1, p(0, 2, 0, 0), &mut io),
        Ok(MoveOutcome::Quarantined {
            initial: issue(1, IoPhase::Replace, IoError::Timeout),
            restore: Some(issue(0, IoPhase::Restore, IoError::Failed)),
        })
    );
    assert!(place.snapshot().is_quarantined(0) && place.snapshot().is_quarantined(1));
    assert_eq!(place.snapshot().epoch(), 2);
    io.reference_matches(&place);
}
