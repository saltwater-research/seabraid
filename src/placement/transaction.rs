// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! Physical swaps are fenced and fail closed on uncertain residency. The
//! ownership map is published only after both provider writes acknowledge.
//! The per-swap failure table lives in `transaction/swap.rs`; this file validates plans and publishes the map.

mod by_reference;
mod swap;

use super::{
    AppliedRound, ExpertId, OwnershipMap, PlacementError, PlacementIo, PlacementPlan, Promotion,
    PromotionOutcome, Rung,
};
use std::collections::{HashMap, HashSet};
use swap::{Exchange, Swap};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IoError {
    /// No write started; the old slot is known whole.
    Refused,
    /// The operation completed with an error. A write may have torn its slot,
    /// but the residency index still names the old expert and can be restored.
    Failed,
    /// No acknowledgement; neither old bytes nor the residency index can be trusted.
    Timeout,
    /// An unwinding callback left its provider state unknown.
    Panicked,
    /// The provider returned an empty blob or did not match the layer's slot size.
    MalformedBlob,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IoPhase {
    Stage,
    Replace,
    Restore,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProviderIssue {
    pub rung: Rung,
    pub phase: IoPhase,
    pub error: IoError,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MoveOutcome {
    Committed {
        epoch: u64,
    },
    Rejected(ProviderIssue),
    Restored(ProviderIssue),
    Quarantined {
        initial: ProviderIssue,
        restore: Option<ProviderIssue>,
    },
    /// A later hop of a climb that was not attempted because an earlier hop of the same expert did not commit. No
    /// provider call, the epoch is unchanged. Only a plan with repeated climbers can produce it.
    Skipped,
}

impl OwnershipMap {
    /// Apply a custom promotion without exposing a half-prepared state. The
    /// adapter must keep admission closed until this call returns. WHY test-only: the C ABI applies whole plans
    /// (`apply_plan`), and with the module crate-private no other caller of the single-promotion form remains.
    #[cfg(test)]
    pub fn apply(
        &mut self,
        expected_epoch: u64,
        promotion: Promotion,
        io: &mut dyn PlacementIo,
    ) -> Result<MoveOutcome, PlacementError> {
        self.check_epoch(expected_epoch)?;
        self.next_epoch()?;
        let swap = self.validate_promotion(promotion)?;
        Ok(self.apply_swap(swap, io, None).0)
    }

    /// Apply one plan at a drained boundary. Its epoch and all pairs are
    /// checked before any I/O. Each pair commits independently; a quarantine
    /// stops the round so no remaining operation starts under degraded routing.
    /// A hop whose climber is not where the plan expected it (an earlier hop of the same climb did not commit) is
    /// `Skipped` and does no I/O: the live map is checked again just before each hop.
    pub fn apply_plan(
        &mut self,
        plan: PlacementPlan,
        io: &mut dyn PlacementIo,
    ) -> Result<AppliedRound, PlacementError> {
        self.check_epoch(plan.epoch)?;
        self.0
            .epoch
            .checked_add(plan.promotions.len() as u64)
            .ok_or(PlacementError::EpochExhausted)?;
        self.validate_plan(&plan.promotions)?;
        let mut outcomes = Vec::with_capacity(plan.promotions.len());
        // The last committed climber and its bytes; climbs are contiguous in a plan, so one slot is enough.
        let mut climber: Option<(ExpertId, Vec<u8>)> = None;
        for promotion in plan.promotions {
            let outcome = match self.validate_promotion(promotion) {
                Ok(swap) => {
                    let (outcome, staged) = self.apply_swap(swap, io, climber.take());
                    climber = staged;
                    outcome
                }
                Err(_) => MoveOutcome::Skipped,
            };
            outcomes.push(PromotionOutcome { promotion, outcome });
            if matches!(outcome, MoveOutcome::Quarantined { .. }) {
                break;
            }
        }
        Ok(AppliedRound {
            round: plan.round,
            shape: self.0.shape(),
            outcomes,
        })
    }

    /// Every pair is valid in plan order against a scratch owner table that has the earlier pairs applied. A climber
    /// may reappear one rung higher; nothing else may repeat: a displaced cell appears once, and never after it
    /// climbed.
    fn validate_plan(&self, promotions: &[Promotion]) -> Result<(), PlacementError> {
        let mut scratch = HashMap::new();
        let (mut climbed, mut displaced) = (HashSet::new(), HashSet::new());
        for &promotion in promotions {
            let swap = self.validate_against(promotion, &scratch)?;
            if displaced.contains(&swap.up_index)
                || displaced.contains(&swap.down_index)
                || climbed.contains(&swap.down_index)
            {
                return Err(PlacementError::InvalidPromotion);
            }
            climbed.insert(swap.up_index);
            displaced.insert(swap.down_index);
            scratch.insert(swap.up_index, swap.hi);
            scratch.insert(swap.down_index, swap.lo);
        }
        Ok(())
    }

    fn validate_promotion(&self, promotion: Promotion) -> Result<Swap, PlacementError> {
        self.validate_against(promotion, &HashMap::new())
    }

    /// `pending` holds the owners earlier pairs of the same plan would have written; a cell not in it has its map owner.
    fn validate_against(
        &self,
        promotion: Promotion,
        pending: &HashMap<usize, Rung>,
    ) -> Result<Swap, PlacementError> {
        let [up, down] = promotion.cells();
        let hi = promotion.upper_rung;
        // WHY the fallback boundary is a map property: only a handle opened with it may name the fallback as the
        // lower rung, so a provider that never sees that open never sees `rung == fallback`.
        let lowest_upper = self
            .0
            .fallback
            .saturating_sub(u8::from(!self.0.fallback_live));
        if hi >= lowest_upper || up == down {
            return Err(PlacementError::InvalidPromotion);
        }
        let lo = hi + 1;
        let up_index = self.0.index(up).ok_or(PlacementError::InvalidPromotion)?;
        let down_index = self.0.index(down).ok_or(PlacementError::InvalidPromotion)?;
        let owner = |index: usize| pending.get(&index).copied().unwrap_or(self.0.owner[index]);
        if owner(up_index) != lo || owner(down_index) != hi {
            return Err(PlacementError::InvalidPromotion);
        }
        if self.0.is_quarantined(hi) || self.0.is_quarantined(lo) {
            return Err(PlacementError::InvalidPromotion);
        }
        Ok(Swap {
            hi,
            lo,
            up,
            down,
            up_index,
            down_index,
            lo_fallback: lo == self.0.fallback,
        })
    }

    /// Runs one swap and publishes its result. Returns the outcome and, on a commit, the climber's bytes.
    fn apply_swap(
        &mut self,
        swap: Swap,
        io: &mut dyn PlacementIo,
        climber: Option<(ExpertId, Vec<u8>)>,
    ) -> (MoveOutcome, Option<(ExpertId, Vec<u8>)>) {
        let reuse = climber.and_then(|(cell, bytes)| (cell == swap.up).then_some(bytes));
        match swap.run(io, reuse) {
            Exchange::Acknowledged(bytes) => {
                self.0.owner[swap.up_index] = swap.hi;
                self.0.owner[swap.down_index] = swap.lo;
                self.0.epoch += 1; // capacity checked before any provider write
                let committed = MoveOutcome::Committed {
                    epoch: self.0.epoch,
                };
                // No staged climber bytes after a by-reference write: the next hop asks its rung again.
                (committed, bytes.map(|bytes| (swap.up, bytes)))
            }
            Exchange::Aborted {
                outcome,
                quarantine,
            } => {
                if quarantine != 0 {
                    self.0.quarantined |= quarantine;
                    self.0.epoch += 1;
                    super::announce_quarantine(quarantine, self.0.epoch);
                }
                (outcome, None)
            }
        }
    }
}
