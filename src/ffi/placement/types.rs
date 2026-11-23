// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! Version-one C values. Output tags are explicit because Rust enum layouts
//! are private to the placement core.
//! FAILURE: none of its own: these are data and infallible conversions; a struct that gained a field without its
//! padding stops the build (the size asserts below), so no wrong layout can reach a caller.

use crate::diag::{fail, Code};
use crate::ffi::ABI_VERSION;
use crate::layout::assert_layout;
use crate::placement::{
    IoError, IoPhase, MoveOutcome, PlacementPlan, PlacementSnapshot, PlannerConfig,
    PromotionOutcome, ProviderIssue, Rung,
};
use crate::Result;

pub(crate) const OUTCOME_COMMITTED: u32 = 1;
pub(crate) const OUTCOME_REJECTED: u32 = 2;
pub(crate) const OUTCOME_RESTORED: u32 = 3;
pub(crate) const OUTCOME_QUARANTINED: u32 = 4;
pub(crate) const OUTCOME_SKIPPED: u32 = 5;
/// `seabraid_placement_policy_v2.abi_version`; its `struct_size` is at least `POLICY_V2_BASE_SIZE`.
pub(crate) const POLICY_V2_VERSION: u32 = 2;
pub(crate) const POLICY_V2_BASE_SIZE: usize = 40;

pub(crate) const PHASE_STAGE: u32 = 1;
pub(crate) const PHASE_REPLACE: u32 = 2;
pub(crate) const PHASE_RESTORE: u32 = 3;

pub(crate) const ISSUE_REFUSED: u32 = 1;
pub(crate) const ISSUE_FAILED: u32 = 2;
pub(crate) const ISSUE_UNACKNOWLEDGED: u32 = 3;
pub(crate) const ISSUE_PANICKED: u32 = 4;
pub(crate) const ISSUE_MALFORMED_BLOB: u32 = 5;

#[derive(Clone, Copy)]
#[repr(C)]
pub struct Layout {
    pub(super) abi_version: u32,
    pub(super) layers: u32,
    pub(super) experts: u32,
    pub(super) cards: Rung,
    pub(super) owners: *const Rung,
    pub(super) owner_count: usize,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct Policy {
    pub(super) abi_version: u32,
    pub(super) max_moves_per_round: u16,
    pub(super) ladder_boundaries: Rung,
    pub(super) cooldown_rounds: u64,
    pub(super) promote_margin: f64,
    pub(super) max_helper_layer_share: f64,
}

impl From<Policy> for PlannerConfig {
    fn from(value: Policy) -> Self {
        Self {
            max_moves_per_round: value.max_moves_per_round,
            promote_margin: value.promote_margin,
            cooldown_rounds: value.cooldown_rounds,
            max_helper_layer_share: value.max_helper_layer_share,
            ladder_boundaries: value.ladder_boundaries,
            fallback_boundary: false,
            max_hops: 1,
        }
    }
}

/// The caller-filled, sized policy of `seabraid_placement_open_v2`. Fields shared with v1 mean what they mean there.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct PolicyV2 {
    pub(super) abi_version: u32,
    pub(super) struct_size: u32,
    pub(super) max_moves_per_round: u16,
    pub(super) ladder_boundaries: Rung,
    pub(super) fallback_boundary: u8,
    pub(super) max_hops_per_round: u8,
    pub(super) reserved: [u8; 3],
    pub(super) cooldown_rounds: u64,
    pub(super) promote_margin: f64,
    pub(super) max_helper_layer_share: f64,
}

impl PolicyV2 {
    /// Checks what the planner cannot see in its own types: the byte flag, the reserved bytes and the hop range.
    /// WHY here and not in the planner: a `bool` and a `u8` that must be 0 or 1 exist only at this boundary.
    pub(super) fn config(self) -> Result<PlannerConfig> {
        if self.reserved != [0; 3] {
            return Err(fail!(
                Code::Invalid,
                "policy.reserved must be zero, got {:?}",
                self.reserved
            ));
        }
        if self.fallback_boundary > 1 {
            return Err(fail!(
                Code::Invalid,
                "policy.fallback_boundary {} is not 0 or 1",
                self.fallback_boundary
            ));
        }
        if !(1..=64).contains(&self.max_hops_per_round) {
            return Err(fail!(
                Code::Invalid,
                "policy.max_hops_per_round {} is outside 1..=64",
                self.max_hops_per_round
            ));
        }
        Ok(PlannerConfig {
            max_moves_per_round: self.max_moves_per_round,
            promote_margin: self.promote_margin,
            cooldown_rounds: self.cooldown_rounds,
            max_helper_layer_share: self.max_helper_layer_share,
            ladder_boundaries: self.ladder_boundaries,
            fallback_boundary: self.fallback_boundary == 1,
            max_hops: self.max_hops_per_round,
        })
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct Ticket {
    pub(super) abi_version: u32,
    pub(super) struct_size: u32,
    pub(super) ticket: u64,
    pub(super) epoch: u64,
    pub(super) promotion_count: usize,
}

impl From<&PlacementPlan> for Ticket {
    fn from(plan: &PlacementPlan) -> Self {
        Self {
            abi_version: ABI_VERSION,
            struct_size: header_size::<Self>(),
            ticket: plan.round,
            epoch: plan.epoch(),
            promotion_count: plan.promotions().len(),
        }
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct Applied {
    pub(super) abi_version: u32,
    pub(super) struct_size: u32,
    pub(super) epoch: u64,
    pub(super) quarantined_rungs: u64,
    pub(super) outcome_count: usize,
}

impl Applied {
    pub(super) fn new(snapshot: &PlacementSnapshot, outcome_count: usize) -> Self {
        Self {
            abi_version: ABI_VERSION,
            struct_size: header_size::<Self>(),
            epoch: snapshot.epoch(),
            quarantined_rungs: snapshot.quarantined,
            outcome_count,
        }
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct Snapshot {
    pub(super) abi_version: u32,
    pub(super) struct_size: u32,
    pub(super) epoch: u64,
    pub(super) quarantined_rungs: u64,
    pub(super) layers: u32,
    pub(super) experts: u32,
    pub(super) fallback_rung: Rung,
    /// Explicit zero tail padding (the C struct ends in 7 bytes of it); see `no_implicit_padding` below.
    pub(super) _pad: [u8; 7],
}

impl From<&PlacementSnapshot> for Snapshot {
    fn from(snapshot: &PlacementSnapshot) -> Self {
        let (layers, experts) = snapshot.shape();
        Self {
            abi_version: ABI_VERSION,
            struct_size: header_size::<Self>(),
            epoch: snapshot.epoch(),
            quarantined_rungs: snapshot.quarantined,
            layers,
            experts,
            fallback_rung: snapshot.fallback(),
            _pad: [0; 7],
        }
    }
}

#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct Issue {
    pub(super) phase: u32,
    pub(super) error: u32,
    pub(super) rung: Rung,
    pub(super) _pad: [u8; 3],
}

impl From<ProviderIssue> for Issue {
    fn from(issue: ProviderIssue) -> Self {
        let phase = match issue.phase {
            IoPhase::Stage => PHASE_STAGE,
            IoPhase::Replace => PHASE_REPLACE,
            IoPhase::Restore => PHASE_RESTORE,
        };
        let error = match issue.error {
            IoError::Refused => ISSUE_REFUSED,
            IoError::Failed => ISSUE_FAILED,
            IoError::Timeout => ISSUE_UNACKNOWLEDGED,
            IoError::Panicked => ISSUE_PANICKED,
            IoError::MalformedBlob => ISSUE_MALFORMED_BLOB,
        };
        Self {
            phase,
            error,
            rung: issue.rung,
            _pad: [0; 3],
        }
    }
}

#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct Outcome {
    pub(super) abi_version: u32,
    pub(super) struct_size: u32,
    pub(super) committed_epoch: u64,
    pub(super) kind: u32,
    pub(super) layer: u32,
    pub(super) promote: u32,
    pub(super) displace: u32,
    pub(super) initial: Issue,
    pub(super) restore: Issue,
    pub(super) upper_rung: Rung,
    pub(super) _pad: [u8; 7],
}

impl From<PromotionOutcome> for Outcome {
    fn from(value: PromotionOutcome) -> Self {
        let mut result = Self {
            abi_version: ABI_VERSION,
            struct_size: header_size::<Self>(),
            layer: value.promotion.layer,
            promote: value.promotion.promote,
            displace: value.promotion.displace,
            upper_rung: value.promotion.upper_rung,
            ..Self::default()
        };
        result.kind = match value.outcome {
            MoveOutcome::Committed { epoch } => {
                result.committed_epoch = epoch;
                OUTCOME_COMMITTED
            }
            MoveOutcome::Rejected(initial) => {
                result.initial = initial.into();
                OUTCOME_REJECTED
            }
            MoveOutcome::Restored(initial) => {
                result.initial = initial.into();
                OUTCOME_RESTORED
            }
            MoveOutcome::Quarantined { initial, restore } => {
                result.initial = initial.into();
                result.restore = restore.map(Issue::from).unwrap_or_default();
                OUTCOME_QUARANTINED
            }
            MoveOutcome::Skipped => OUTCOME_SKIPPED,
        };
        result
    }
}

/// WHY explicit pad fields: `write_out` copies a whole struct into caller memory, and the bytes the C layout leaves
/// between and after fields would be whatever the library's stack held (stack garbage in the caller's
/// struct). With every padding byte a named field, a library-filled struct has none of its own to leak. This is the
/// checker: the size of each library-filled struct must equal the sum of its fields' sizes, so a field added
/// without its padding (or a pad that stopped being needed) stops the build. `SessionStatus` and `Capabilities` are
/// checked beside their definitions.
const _: () = {
    use core::mem::size_of;
    assert!(size_of::<Issue>() == 4 + 4 + 1 + 3);
    assert!(size_of::<Ticket>() == 4 + 4 + 8 + 8 + 8);
    assert!(size_of::<Applied>() == 4 + 4 + 8 + 8 + 8);
    assert!(size_of::<Snapshot>() == 4 + 4 + 8 + 8 + 4 + 4 + 1 + 7);
    assert!(size_of::<Outcome>() == 4 + 4 + 8 + 4 * 4 + 2 * size_of::<Issue>() + 1 + 7);
};

/// The v1 size a library-filled struct reports in its `struct_size` field.
fn header_size<T>() -> u32 {
    core::mem::size_of::<T>() as u32
}

assert_layout!(
    Layout,
    "seabraid_placement_layout_v1",
    [abi_version, layers, experts, cards, owners, owner_count]
);
assert_layout!(
    Policy,
    "seabraid_placement_policy_v1",
    [
        abi_version,
        max_moves_per_round,
        ladder_boundaries,
        cooldown_rounds,
        promote_margin,
        max_helper_layer_share
    ]
);
assert_layout!(
    PolicyV2,
    "seabraid_placement_policy_v2",
    [
        abi_version,
        struct_size,
        max_moves_per_round,
        ladder_boundaries,
        fallback_boundary,
        max_hops_per_round,
        reserved,
        cooldown_rounds,
        promote_margin,
        max_helper_layer_share
    ]
);
assert_layout!(
    Ticket,
    "seabraid_placement_ticket_v1",
    [abi_version, struct_size, ticket, epoch, promotion_count]
);
assert_layout!(
    Applied,
    "seabraid_placement_applied_v1",
    [
        abi_version,
        struct_size,
        epoch,
        quarantined_rungs,
        outcome_count
    ]
);
assert_layout!(
    Snapshot,
    "seabraid_placement_snapshot_v1",
    [
        abi_version,
        struct_size,
        epoch,
        quarantined_rungs,
        layers,
        experts,
        fallback_rung
    ]
);
assert_layout!(Issue, "seabraid_placement_issue_v1", [phase, error, rung]);
assert_layout!(
    Outcome,
    "seabraid_placement_outcome_v1",
    [
        abi_version,
        struct_size,
        committed_epoch,
        kind,
        layer,
        promote,
        displace,
        initial,
        restore,
        upper_rung
    ]
);
