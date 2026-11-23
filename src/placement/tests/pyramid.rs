// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! A byte-keeping fake of the whole pyramid for the climb, fallback and round-trip tests: four helper cards plus the
//! fallback (rungs 0..=4 are cards, rung 5 is the fallback). The fallback serves every cell from `source` (its
//! RAM copy or the pack) and keeps a RAM set whose size only changes through a 1:1 step, as the engine's does.
//! OWNER: this file owns the fixture only; the rules under test are in `placement/` and are never restated here.

use super::*;
use std::collections::HashSet;

pub(super) const CARDS: Rung = 5;
pub(super) const LAYERS: u32 = 2;
pub(super) const EXPERTS: u32 = 12;

pub(super) fn cell(layer: u32, expert: u32) -> ExpertId {
    ExpertId { layer, expert }
}

/// Two experts per rung in every layer: 0,1 on the lead, then each helper, 10,11 on the fallback.
pub(super) fn pyramid_owners() -> Vec<Rung> {
    (0..LAYERS * EXPERTS)
        .map(|i| ((i % EXPERTS) / 2) as Rung)
        .collect()
}

/// Strictly distinct heat, hotter for a lower expert number: ratio 2 per step, so every pair beats the margin.
pub(super) fn baseline_heat() -> Vec<f64> {
    (0..LAYERS * EXPERTS)
        .map(|i| 2f64.powi(11 - (i % EXPERTS) as i32))
        .collect()
}

pub(super) fn pyramid_map(fallback_live: bool) -> OwnershipMap {
    let mut map = OwnershipMap::new(LAYERS, EXPERTS, CARDS, pyramid_owners()).unwrap();
    if fallback_live {
        map.enable_fallback_boundary().unwrap();
    }
    map
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) enum Op {
    Stage,
    Replace,
}

pub(super) struct Pyramid {
    pub(super) store: HashMap<(Rung, ExpertId), Vec<u8>>,
    /// Cells with a RAM copy on the fallback; the others are served from the pack on NVMe.
    pub(super) ram: HashSet<ExpertId>,
    pub(super) calls: Vec<(Op, Rung)>,
    faults: HashMap<(Rung, Op), Faults>,
}

impl Pyramid {
    pub(super) fn new(map: &OwnershipMap) -> Self {
        let snapshot = map.snapshot();
        let mut pyramid = Self {
            store: HashMap::new(),
            ram: HashSet::new(),
            calls: Vec::new(),
            faults: HashMap::new(),
        };
        for cell in cells(snapshot) {
            let owner = snapshot.owner(cell).unwrap();
            if owner != snapshot.fallback() {
                pyramid.store.insert((owner, cell), source(cell));
            } else if cell.expert == EXPERTS - 2 {
                pyramid.ram.insert(cell); // one RAM-resident cell per layer on the fallback
            }
        }
        pyramid
    }

    /// The next `op` on `rung` answers with `fault` (None is an ACK). Queued: later calls see later entries.
    pub(super) fn fail(&mut self, rung: Rung, op: Op, faults: &[Option<IoError>]) {
        self.faults
            .entry((rung, op))
            .or_default()
            .extend(faults.iter().copied());
    }

    pub(super) fn count(&self, op: Op, rung: Rung) -> usize {
        self.calls
            .iter()
            .filter(|&&call| call == (op, rung))
            .count()
    }

    /// The engine's own RAM step at a window boundary: `hot` takes `cold`'s RAM place, so the RAM size is constant.
    pub(super) fn ram_adapt(&mut self, hot: ExpertId, cold: ExpertId) {
        assert!(
            self.ram.remove(&cold) && self.ram.insert(hot),
            "not a 1:1 RAM step"
        );
    }

    /// Every cell an open card owns holds exactly its source bytes, nothing stale is left on an open card, and every
    /// cell has one executor.
    pub(super) fn audit(&self, map: &OwnershipMap) {
        let snapshot = map.snapshot();
        for cell in cells(snapshot) {
            let owner = snapshot.owner(cell).unwrap();
            if owner != snapshot.fallback() && !snapshot.is_quarantined(owner) {
                assert_eq!(
                    self.store.get(&(owner, cell)),
                    Some(&source(cell)),
                    "{cell:?} on rung {owner}"
                );
            }
        }
        for &(rung, cell) in self.store.keys() {
            if !snapshot.is_quarantined(rung) {
                assert_eq!(
                    snapshot.owner(cell),
                    Some(rung),
                    "stale copy of {cell:?} on rung {rung}"
                );
            }
        }
    }
}

impl PlacementIo for Pyramid {
    fn stage(&mut self, rung: Rung, cell: ExpertId) -> Result<Vec<u8>, IoError> {
        self.calls.push((Op::Stage, rung));
        if let Some(Some(error)) = self
            .faults
            .get_mut(&(rung, Op::Stage))
            .and_then(|q| q.pop_front())
        {
            return Err(error);
        }
        if rung == CARDS {
            return Ok(source(cell));
        }
        self.store
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
        self.calls.push((Op::Replace, rung));
        let fault = self
            .faults
            .get_mut(&(rung, Op::Replace))
            .and_then(|q| q.pop_front())
            .flatten();
        if rung == CARDS {
            // The fallback provider rule: all-or-nothing, and on any non-ACK it still serves everything.
            if let Some(error) = fault {
                if error == IoError::Panicked {
                    panic!("injected fallback panic");
                }
                return Err(error);
            }
            assert_eq!(bytes, source(new), "fallback was handed wrong bytes");
            if self.ram.remove(&old) {
                self.ram.insert(new); // the leaving cell's RAM place goes to the cell that replaces it
            }
            return Ok(());
        }
        let slot = self.store.get_mut(&(rung, old)).ok_or(IoError::Refused)?;
        match fault {
            Some(IoError::Refused) => return Err(IoError::Refused),
            Some(error) => {
                slot[0] = !bytes[0]; // a failed write tears the slot, so a restore is visible
                if error == IoError::Panicked {
                    panic!("injected card panic");
                }
                return Err(error);
            }
            None => {}
        }
        assert_eq!(bytes, source(new), "card {rung} was handed wrong bytes");
        self.store.remove(&(rung, old));
        self.store.insert((rung, new), bytes.to_vec());
        Ok(())
    }
}

/// One planned and applied round on the shared fixture; returns the applied round.
pub(super) fn run_round(
    map: &mut OwnershipMap,
    planner: &mut PlacementPlanner,
    io: &mut Pyramid,
    heat: &[f64],
) -> AppliedRound {
    let plan = planner.plan(map.snapshot(), heat).unwrap();
    let applied = map.apply_plan(plan, io).unwrap();
    planner.record(&applied).unwrap();
    io.audit(map);
    applied
}

pub(super) fn owner_of(map: &OwnershipMap, layer: u32, expert: u32) -> Rung {
    map.snapshot().owner(cell(layer, expert)).unwrap()
}
