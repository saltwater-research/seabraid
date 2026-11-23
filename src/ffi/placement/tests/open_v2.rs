// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! `seabraid_placement_open_v2`: the sized policy's refusals, v1 parity, and a climb through the fallback boundary
//! over the C provider table, including a SKIPPED hop.

use super::*;
use crate::diag::testing::Capture;
use crate::diag::Level;
use crate::ffi::placement::types::{OUTCOME_REJECTED, OUTCOME_SKIPPED};
use std::ffi::c_void;

const INVALID: i32 = Code::Invalid as i32;

fn v2(fallback: u8, hops: u8) -> PolicyV2 {
    PolicyV2 {
        abi_version: 2,
        struct_size: mem::size_of::<PolicyV2>() as u32,
        max_moves_per_round: 12,
        ladder_boundaries: 0,
        fallback_boundary: fallback,
        max_hops_per_round: hops,
        reserved: [0; 3],
        cooldown_rounds: 0,
        promote_margin: 1.25,
        max_helper_layer_share: 1.0,
    }
}

/// Three cards (rungs 0..=2) and the fallback (rung 3), one expert each; expert 3 is on the fallback.
const OWNERS: [Rung; 4] = [0, 1, 2, 3];

fn layout(cards: Rung, owners: &[Rung]) -> Layout {
    Layout {
        abi_version: ABI_VERSION,
        layers: 1,
        experts: owners.len() as u32,
        cards,
        owners: owners.as_ptr(),
        owner_count: owners.len(),
    }
}

fn open_v2(layout: &Layout, policy: *const PolicyV2) -> (i32, *mut PlacementHandle) {
    let mut handle = ptr::null_mut();
    let rc = unsafe { seabraid_placement_open_v2(layout, policy, &mut handle) };
    (rc, handle)
}

fn closed(layout: &Layout, policy: &PolicyV2) -> i32 {
    let (rc, handle) = open_v2(layout, policy);
    if rc == 0 {
        assert_eq!(unsafe { seabraid_placement_close(handle) }, 0);
    } else {
        assert!(handle.is_null(), "*out is NULL on every failure");
    }
    rc
}

#[test]
fn open_v2_refuses_every_out_of_range_field_and_a_dirty_tail() {
    let _capture = Capture::start(Level::Off);
    let layout = layout(3, &OWNERS);
    assert_eq!(closed(&layout, &v2(0, 1)), 0);
    assert_eq!(closed(&layout, &v2(1, 64)), 0);
    let bad = [
        PolicyV2 {
            abi_version: 1,
            ..v2(0, 1)
        },
        PolicyV2 {
            struct_size: 32,
            ..v2(0, 1)
        },
        PolicyV2 {
            struct_size: 44,
            ..v2(0, 1)
        },
        PolicyV2 {
            reserved: [0, 1, 0],
            ..v2(0, 1)
        },
        v2(2, 1),
        v2(0, 0),
        v2(0, 65),
        PolicyV2 {
            max_moves_per_round: 129,
            ..v2(0, 1)
        },
        PolicyV2 {
            promote_margin: 0.5,
            ..v2(0, 1)
        },
    ];
    for (n, policy) in bad.iter().enumerate() {
        assert_eq!(closed(&layout, policy), INVALID, "case {n}");
    }
    // The fallback boundary needs two cards: with one, that boundary is the lead with the fallback.
    let one = [0u8, 1];
    assert_eq!(closed(&self::layout(1, &one), &v2(1, 1)), INVALID);
    assert_eq!(closed(&self::layout(1, &one), &v2(0, 1)), 0);
    assert_eq!(closed(&self::layout(2, &[0, 1, 2]), &v2(1, 1)), 0);
}

#[test]
fn open_v2_reads_a_wider_caller_struct_only_when_its_tail_is_zero() {
    #[repr(C)]
    struct Wide {
        policy: PolicyV2,
        tail: [u8; 8],
    }
    let _capture = Capture::start(Level::Off);
    let layout = layout(3, &OWNERS);
    let wide = |byte| {
        let mut policy = v2(1, 2);
        policy.struct_size = 48;
        Wide {
            policy,
            tail: [byte; 8],
        }
    };
    // WHY a pointer to the whole struct: the library reads `struct_size` (48) bytes, past the 40-byte field.
    let (clean, dirty) = (wide(0), wide(1));
    let (rc, handle) = open_v2(&layout, ptr::from_ref(&clean).cast());
    assert_eq!(rc, 0);
    assert_eq!(unsafe { seabraid_placement_close(handle) }, 0);
    let (rc, handle) = open_v2(&layout, ptr::from_ref(&dirty).cast());
    assert_eq!((rc, handle.is_null()), (INVALID, true));
}

#[test]
fn a_null_policy_and_v1_open_never_plan_a_swap_with_the_fallback() {
    let _capture = Capture::start(Level::Off);
    let owners = [0u8, 1, 2];
    let layout = layout(2, &owners);
    let heat = [1.0, 2.0, 100.0];
    let promotions = |handle: *mut PlacementHandle| {
        let mut ticket = blank_ticket();
        assert_eq!(
            unsafe { seabraid_placement_plan(handle, 1, heat.as_ptr(), 3, &mut ticket) },
            0
        );
        ticket.promotion_count
    };
    let (rc, null) = open_v2(&layout, ptr::null());
    assert_eq!(rc, 0);
    let mut v1 = ptr::null_mut();
    assert_eq!(
        unsafe { seabraid_placement_open(&layout, ptr::null(), &mut v1) },
        0
    );
    // Only the (0, 1) boundary exists: one swap. Expert 2 sits on the fallback and is not a challenger.
    assert_eq!((promotions(null), promotions(v1)), (1, 1));
    for handle in [null, v1] {
        assert_eq!(unsafe { seabraid_placement_close(handle) }, 0);
    }
}

struct Script {
    calls: Vec<(char, Rung)>,
    refuse_replace_once: Option<Rung>,
}

unsafe extern "C" fn stage_script(
    context: *mut c_void,
    rung: u8,
    _: u32,
    _: u32,
    bytes: *mut u8,
    size: usize,
    written: *mut usize,
) -> i32 {
    unsafe {
        (*context.cast::<Script>()).calls.push(('s', rung));
        ptr::write_bytes(bytes, 7, size);
        *written = size;
    }
    0
}

unsafe extern "C" fn replace_script(
    context: *mut c_void,
    rung: u8,
    _: u32,
    _: u32,
    _: u32,
    _: *const u8,
    _: usize,
) -> i32 {
    let script = unsafe { &mut *context.cast::<Script>() };
    script.calls.push(('r', rung));
    if script.refuse_replace_once == Some(rung) {
        script.refuse_replace_once = None;
        return 1; // SEABRAID_PLACEMENT_IO_REFUSED
    }
    0
}

/// Plans and applies one round on the three-card layout; returns the outcome kinds, the upper rungs and the script.
fn round(policy: &PolicyV2, refuse: Option<Rung>) -> (Vec<(u32, Rung)>, Script, Applied) {
    let _capture = Capture::start(Level::Off);
    let layout = layout(3, &OWNERS);
    let (rc, handle) = open_v2(&layout, policy);
    assert_eq!(rc, 0);
    let heat = [1.0, 2.0, 4.0, 100.0];
    let mut ticket = blank_ticket();
    assert_eq!(
        unsafe { seabraid_placement_plan(handle, 1, heat.as_ptr(), 4, &mut ticket) },
        0
    );
    let mut script = Script {
        calls: Vec::new(),
        refuse_replace_once: refuse,
    };
    let slot = [8usize];
    let io = RawIo {
        abi_version: ABI_VERSION,
        context: ptr::from_mut(&mut script).cast(),
        layer_slot_bytes: slot.as_ptr(),
        layer_count: 1,
        stage: Some(stage_script),
        replace: Some(replace_script),
    };
    let mut outcomes = vec![
        Outcome {
            abi_version: ABI_VERSION,
            struct_size: 64,
            ..Outcome::default()
        };
        4
    ];
    let mut applied = Applied {
        abi_version: ABI_VERSION,
        struct_size: mem::size_of::<Applied>() as u32,
        epoch: 0,
        quarantined_rungs: 0,
        outcome_count: 0,
    };
    let rc = unsafe {
        seabraid_placement_apply(
            handle,
            ticket.ticket,
            ptr::from_ref(&io).cast(),
            outcomes.as_mut_ptr(),
            4,
            &mut applied,
        )
    };
    assert_eq!(rc, 0);
    assert_eq!(unsafe { seabraid_placement_close(handle) }, 0);
    let kinds = outcomes[..applied.outcome_count]
        .iter()
        .map(|o| (o.kind, o.upper_rung))
        .collect();
    (kinds, script, applied)
}

#[test]
fn a_three_hop_climb_through_the_fallback_commits_and_stages_the_climber_once() {
    let (kinds, script, applied) = round(&v2(1, 3), None);
    // Hop 1 has the fallback as its lower rung (outcome upper_rung 2 = cards - 1): the lower rung is rung 3.
    assert_eq!(
        kinds,
        [
            (OUTCOME_COMMITTED, 2),
            (OUTCOME_COMMITTED, 1),
            (OUTCOME_COMMITTED, 0)
        ]
    );
    assert_eq!(script.calls.iter().filter(|c| **c == ('s', 3)).count(), 1);
    assert_eq!(script.calls[0], ('s', 3));
    assert_eq!(applied.epoch, 4);
}

#[test]
fn a_hop_after_an_uncommitted_hop_reports_skipped_with_no_provider_call_and_no_epoch() {
    let (kinds, script, applied) = round(&v2(1, 3), Some(1));
    assert_eq!(
        kinds,
        [
            (OUTCOME_COMMITTED, 2),
            (OUTCOME_REJECTED, 1),
            (OUTCOME_SKIPPED, 0)
        ]
    );
    assert_eq!(
        applied.epoch, 2,
        "one commit; a rejection and a skipped hop add nothing"
    );
    assert!(
        !script.calls.contains(&('r', 0)),
        "the skipped hop reached no provider"
    );
}

#[test]
fn without_the_fallback_boundary_the_same_heat_plans_only_card_swaps() {
    let (kinds, script, _) = round(&v2(0, 1), None);
    assert_eq!(kinds, [(OUTCOME_COMMITTED, 0)]);
    assert!(
        script.calls.iter().all(|c| c.1 != 3),
        "the fallback never sees a call"
    );
}
