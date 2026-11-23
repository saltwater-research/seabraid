// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! One execution owner per layered expert. Invalid input leaves placement unchanged;
//! uncertain provider residency routes to the verified fallback until a fresh layout.
//!
//! Ports Strata `include/strata/core/ownership_map.hpp` and `src/core/ownership_map.cpp`.
//! The engine drains layer work and fences admission before a transition or layout
//! update; this module serializes ownership, but cannot drain a vendor queue itself.

mod fill;
mod heat;
mod planner;
#[cfg(test)]
mod tests;
mod transaction;

use crate::diag::{self, Code, Level};
pub use fill::fill;
pub use planner::{PlacementPlanner, PlannerConfig};
pub use transaction::{IoError, IoPhase, MoveOutcome, ProviderIssue};

/// Rank within one session: 0 is the lead card; `snapshot.fallback()` is the
/// verified host/model source. The adapter binds ranks to stable provider IDs.
pub type Rung = u8;

/// Adapter operations on drained provider queues, mapped by rank. The adapter
/// owns the deadline and must return `Timeout` when a completion is uncertain.
pub trait PlacementIo {
    /// Stage verified bytes from the current owner without changing residency.
    fn stage(&mut self, rung: Rung, cell: ExpertId) -> Result<Vec<u8>, IoError>;

    /// Success acknowledges verified bytes and the new residency index. A
    /// failed write keeps the old index so `old == new` can restore its slot.
    fn replace(
        &mut self,
        rung: Rung,
        old: ExpertId,
        new: ExpertId,
        bytes: &[u8],
    ) -> Result<(), IoError>;

    /// Whether `rung`'s replace may take a promoted expert's bytes from the fallback's own verified store (for Strata:
    /// the model file) instead of from the library. WHY a question per rung and not a flag on the call: the library asks
    /// only rungs that can ever say yes, so every attempt is real and every refusal is counted honestly.
    fn takes_from_fallback(&self, _rung: Rung) -> bool {
        false
    }

    /// Like `replace`, but the library supplies no bytes: the rung puts `new` into the slot of `old` itself. ACK,
    /// `Failed` and `Timeout` mean what they mean for `replace`. `Refused` means "not this way": no write began, and
    /// the transaction then stages `new` from the lower rung and calls `replace` inside the same swap (it is not a
    /// failure and is not logged as one). The default refuses, so an adapter that never answers `from_fallback` with
    /// true never sees this call.
    fn replace_from_fallback(
        &mut self,
        _rung: Rung,
        _old: ExpertId,
        _new: ExpertId,
    ) -> Result<(), IoError> {
        Err(IoError::Refused)
    }

    /// What `IoError` alone cannot say about the failure the latest call returned (an unknown status code, a
    /// short write, an allocation that failed), with the status the record should carry. Taken, so each failure
    /// is described once; the transaction logs it as part of its one record for that failure. WHY a method and
    /// not logging inside the adapter: two records for one failure (one from each layer) is what this replaced.
    fn take_detail(&mut self) -> Option<(Code, String)> {
        None
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ExpertId {
    pub layer: u32,
    pub expert: u32,
}

/// A same-layer exchange. Sharing the layer and deriving the lower rung makes
/// the only supported live primitive unambiguous at the provider boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Promotion {
    pub layer: u32,
    pub promote: u32,
    pub displace: u32,
    pub upper_rung: Rung,
}

impl Promotion {
    fn cells(self) -> [ExpertId; 2] {
        [self.promote, self.displace].map(|expert| ExpertId {
            layer: self.layer,
            expert,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlacementError {
    InvalidLayout,
    InvalidPromotion,
    InvalidHeat,
    /// Every value is finite but one layer's total is not. WHY its own variant: the planner compares layer
    /// totals, so a non-finite one is refused, and saying "heat must be finite" to a caller whose every value
    /// is finite would send them looking for a NaN that is not there.
    HeatOverflow,
    InvalidPolicy,
    PlannerShapeChanged,
    StaleEpoch {
        expected: u64,
        actual: u64,
    },
    EpochExhausted,
}

impl std::fmt::Display for PlacementError {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidLayout => out.write_str("layout or owner table is invalid"),
            Self::InvalidPromotion => {
                out.write_str("promotion does not match current ownership or touches a closed rung")
            }
            Self::InvalidHeat => {
                out.write_str("heat must be finite, non-negative and layers * experts long")
            }
            Self::HeatOverflow => {
                out.write_str("heat of one layer sums past f64::MAX; scale the heat down")
            }
            Self::InvalidPolicy => out.write_str("planner policy is out of range"),
            Self::PlannerShapeChanged => {
                out.write_str("planner state does not match this layout shape")
            }
            Self::StaleEpoch { expected, actual } => {
                write!(out, "epoch {expected} is stale, current epoch is {actual}")
            }
            Self::EpochExhausted => out.write_str("epoch counter exhausted"),
        }
    }
}

/// Quarantine is the one placement event an operator must be able to see in the log.
pub(crate) fn announce_quarantine(rungs: u64, epoch: u64) {
    let text = format!(
        "placement rung mask {rungs:#x} quarantined; its cells now route to the fallback (epoch {epoch})"
    );
    diag::emit(Level::Warn, Code::Provider as i32, &text);
}

#[derive(Clone, Debug)]
pub struct PlacementSnapshot {
    pub(crate) layers: u32,
    experts: u32,
    fallback: Rung,
    pub(crate) owner: Vec<Rung>,
    pub(crate) quarantined: u64,
    epoch: u64,
    /// Set only by `OwnershipMap::enable_fallback_boundary`: the fallback may then be the lower rung of a swap.
    fallback_live: bool,
}

impl PlacementSnapshot {
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    pub fn shape(&self) -> (u32, u32) {
        (self.layers, self.experts)
    }

    pub fn fallback(&self) -> Rung {
        self.fallback
    }

    pub fn is_quarantined(&self, rung: Rung) -> bool {
        rung < self.fallback && self.quarantined & (1 << rung) != 0
    }

    /// Committed owner, retained for diagnostics even when the rung is closed. WHY test-only: with the module
    /// crate-private nothing outside the placement tests reads one cell; the C ABI exports whole tables.
    #[cfg(test)]
    pub fn owner(&self, cell: ExpertId) -> Option<Rung> {
        self.index(cell).map(|index| self.owner[index])
    }

    /// The only answer an engine should use when dispatching routed rows. WHY test-only: see `owner`.
    #[cfg(test)]
    pub fn executor(&self, cell: ExpertId) -> Option<Rung> {
        self.owner(cell).map(|rung| self.executor_for_owner(rung))
    }

    pub(crate) fn executor_for_owner(&self, rung: Rung) -> Rung {
        if self.is_quarantined(rung) {
            self.fallback
        } else {
            rung
        }
    }

    fn index(&self, cell: ExpertId) -> Option<usize> {
        (cell.layer < self.layers && cell.expert < self.experts)
            .then(|| cell.layer as usize * self.experts as usize + cell.expert as usize)
    }

    fn validate_layout(&self, owner: &[Rung]) -> Result<(), PlacementError> {
        let cells = layout_size(self.layers, self.experts, self.fallback)?;
        if owner.len() != cells || owner.iter().any(|rung| *rung > self.fallback) {
            return Err(PlacementError::InvalidLayout);
        }
        Ok(())
    }
}

/// Most cards in one layout. The fallback rung equals the card count, so this is also the
/// highest rung; the quarantine mask is one u64 bit per rung. Mirrors `SEABRAID_PLACEMENT_MAX_CARDS`.
pub(crate) const MAX_CARDS: Rung = 63;
/// Most (layer, expert) cells in one layout. Same plausibility bound as Strata; it also
/// bounds indexing on 32-bit hosts. Mirrors `SEABRAID_PLACEMENT_MAX_CELLS`.
pub(crate) const MAX_CELLS: u64 = 1 << 26;

pub(crate) fn layout_size(
    layers: u32,
    experts: u32,
    fallback: Rung,
) -> Result<usize, PlacementError> {
    let cells = u64::from(layers) * u64::from(experts);
    // `fallback` is the card count (see `OwnershipMap::new`), so this bounds cards to 1..=MAX_CARDS.
    if !(1..=MAX_CELLS).contains(&cells) || !(1..=MAX_CARDS).contains(&fallback) {
        return Err(PlacementError::InvalidLayout);
    }
    Ok(cells as usize)
}

/// The engine owns one map. Policy receives only a read-only snapshot.
#[derive(Debug)]
pub struct OwnershipMap(PlacementSnapshot);

impl OwnershipMap {
    pub fn new(
        layers: u32,
        experts: u32,
        cards: Rung,
        owner: Vec<Rung>,
    ) -> Result<Self, PlacementError> {
        let snapshot = PlacementSnapshot {
            layers,
            experts,
            fallback: cards,
            owner,
            quarantined: 0,
            epoch: 1,
            fallback_live: false,
        };
        snapshot.validate_layout(&snapshot.owner)?;
        Ok(Self(snapshot))
    }

    /// Let the fallback be the lower rung of a live swap (rung `cards - 1` with the fallback). WHY it needs two cards:
    /// with one card that boundary would be main with the fallback, which stays the engine's adaptive cache, so one
    /// publisher owns each boundary. Fails before changing anything.
    pub fn enable_fallback_boundary(&mut self) -> Result<(), PlacementError> {
        if self.0.fallback < 2 {
            return Err(PlacementError::InvalidPolicy);
        }
        self.0.fallback_live = true;
        Ok(())
    }

    pub fn snapshot(&self) -> &PlacementSnapshot {
        &self.0
    }

    /// Adopt a completed and verified external move (such as card 1 ↔ host
    /// adaptation) at a drained boundary. Adoption never reopens a card.
    pub fn adopt(&mut self, expected_epoch: u64, owner: Vec<Rung>) -> Result<(), PlacementError> {
        self.check_epoch(expected_epoch)?;
        self.0.validate_layout(&owner)?;
        if self.0.owner != owner {
            self.update_layout(owner, self.0.quarantined)?;
        }
        Ok(())
    }

    /// Reopen only after the adapter drained or replaced the old providers and
    /// verified the full layout and bytes, including all formerly closed rungs.
    pub fn rebuild(&mut self, expected_epoch: u64, owner: Vec<Rung>) -> Result<(), PlacementError> {
        self.check_epoch(expected_epoch)?;
        self.0.validate_layout(&owner)?;
        self.update_layout(owner, 0)
    }

    /// Record an out-of-band provider loss with admission closed. One closed
    /// card redirects every cell it owned, so the routing epoch changes once.
    pub fn quarantine(&mut self, rung: Rung) -> Result<(), PlacementError> {
        if rung >= self.0.fallback {
            return Err(PlacementError::InvalidLayout);
        }
        if !self.0.is_quarantined(rung) {
            self.0.epoch = self.next_epoch()?;
            self.0.quarantined |= 1 << rung;
            announce_quarantine(1 << rung, self.0.epoch);
        }
        Ok(())
    }

    fn update_layout(&mut self, owner: Vec<Rung>, closed: u64) -> Result<(), PlacementError> {
        self.0.epoch = self.next_epoch()?;
        self.0.owner = owner;
        self.0.quarantined = closed;
        Ok(())
    }

    /// Test hook for the epoch-exhaustion case. WHY a method and not a widened field: the epoch stays writable only
    /// by the transaction code in this module, and the ffi test reaches it through this one door.
    #[cfg(test)]
    pub(crate) fn set_epoch_for_test(&mut self, epoch: u64) {
        self.0.epoch = epoch;
    }

    pub(crate) fn check_epoch(&self, expected: u64) -> Result<(), PlacementError> {
        let actual = self.0.epoch;
        if expected != actual {
            return Err(PlacementError::StaleEpoch { expected, actual });
        }
        Ok(())
    }

    fn next_epoch(&self) -> Result<u64, PlacementError> {
        self.0
            .epoch
            .checked_add(1)
            .ok_or(PlacementError::EpochExhausted)
    }
}

#[derive(Debug)]
pub struct PlacementPlan {
    epoch: u64,
    pub(crate) round: u64,
    promotions: Vec<Promotion>,
}

impl PlacementPlan {
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    pub fn promotions(&self) -> &[Promotion] {
        &self.promotions
    }
}

#[derive(Debug)]
pub struct AppliedRound {
    round: u64,
    shape: (u32, u32),
    outcomes: Vec<PromotionOutcome>,
}

impl AppliedRound {
    /// Outcomes in plan order. A quarantined pair ends the fenced round early.
    pub fn outcomes(&self) -> &[PromotionOutcome] {
        &self.outcomes
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PromotionOutcome {
    pub promotion: Promotion,
    pub outcome: MoveOutcome,
}
