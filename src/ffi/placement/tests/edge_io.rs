// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! Overnight hardening O1, rows P34-P59 and P74 onward: apply, snapshot, quarantine, struct sizes and slot sizes (see
//! edge.rs; policy and fill are in edge_policy_fill.rs).

use super::edge::{plan, BUSY, INVALID};

use super::*;
use crate::diag::testing::Capture;
use crate::diag::Level;

// case P34-P41: apply argument errors leave the plan pending so a corrected call still works.
#[test]
fn apply_with_too_small_a_capacity_a_bad_out_or_a_bad_provider_keeps_the_plan_pending() {
    let _capture = Capture::start(Level::Off);
    let placement = TestPlacement::new();
    let ticket = plan(&placement, 1, &[1.0, 20.0]).1;
    assert_eq!(ticket.promotion_count, 1);
    let slot = [4usize];
    let io = RawIo {
        abi_version: ABI_VERSION,
        context: ptr::null_mut(),
        layer_slot_bytes: slot.as_ptr(),
        layer_count: 1,
        stage: Some(stage_ok),
        replace: Some(replace_ok),
    };
    let mut outcome = Outcome {
        abi_version: ABI_VERSION,
        struct_size: mem::size_of::<Outcome>() as u32,
        ..Outcome::default()
    };
    let mut applied = Applied {
        abi_version: ABI_VERSION,
        struct_size: mem::size_of::<Applied>() as u32,
        epoch: 0,
        quarantined_rungs: 0,
        outcome_count: 0,
    };
    let pending = || unsafe { handle(placement.0) }.unwrap().pending.is_some();
    let provider = ptr::from_ref(&io).cast();
    let call = |provider, outcome: *mut Outcome, capacity, applied: *mut Applied| unsafe {
        seabraid_placement_apply(
            placement.0,
            ticket.ticket,
            provider,
            outcome,
            capacity,
            applied,
        )
    };
    assert_eq!(
        call(provider, &mut outcome, 0, &mut applied),
        INVALID,
        "capacity 0 of 1"
    );
    assert_eq!(
        call(provider, ptr::null_mut(), 1, &mut applied),
        INVALID,
        "null outcomes"
    );
    assert_eq!(
        call(provider, &mut outcome, 1, ptr::null_mut()),
        INVALID,
        "null out"
    );
    assert_eq!(
        call(ptr::null(), &mut outcome, 1, &mut applied),
        INVALID,
        "null provider"
    );
    for size in [0, 63, 64 - 4, 65] {
        outcome.struct_size = size;
        assert_eq!(
            call(provider, &mut outcome, 1, &mut applied),
            INVALID,
            "outcome struct_size {size}"
        );
    }
    outcome.struct_size = 64;
    applied.struct_size = 31;
    assert_eq!(
        call(provider, &mut outcome, 1, &mut applied),
        INVALID,
        "applied below its base"
    );
    applied.struct_size = mem::size_of::<Applied>() as u32;
    assert!(pending(), "every refusal above left the plan pending");
    assert_eq!(call(provider, &mut outcome, 1, &mut applied), 0);
    assert!(!pending(), "a successful apply consumes the plan");
    assert_eq!(applied.outcome_count, 1);
    assert_eq!(
        call(provider, &mut outcome, 1, &mut applied),
        BUSY,
        "results are single use"
    );
}

// case P42-P47: snapshot and update argument edges.
#[test]
fn snapshot_and_update_refuse_a_wrong_count_a_bad_mode_and_an_unknown_rung_without_changing_state()
{
    let _capture = Capture::start(Level::Off);
    let placement = TestPlacement::new();
    let mut executors = [9u8; 3];
    let mut out = MaybeUninit::new(blank_snapshot());
    for count in [0, 1, 3, usize::MAX] {
        let rc = unsafe {
            seabraid_placement_snapshot(
                placement.0,
                executors.as_mut_ptr(),
                count,
                out.as_mut_ptr(),
            )
        };
        assert_eq!(rc, INVALID, "executor_count {count}");
    }
    let rc =
        unsafe { seabraid_placement_snapshot(placement.0, ptr::null_mut(), 2, out.as_mut_ptr()) };
    assert_eq!(rc, INVALID, "null executors");
    assert_eq!(executors, [9; 3]);
    for mode in [2, 3, u32::MAX] {
        let owners = [0u8, 1];
        let rc = unsafe { seabraid_placement_update(placement.0, 1, mode, owners.as_ptr(), 2) };
        assert_eq!(rc, INVALID, "mode {mode}");
    }
    for owners in [[0u8, 3], [255, 0]] {
        let rc = unsafe {
            seabraid_placement_update(placement.0, 1, UPDATE_REBUILD, owners.as_ptr(), 2)
        };
        assert_eq!(rc, INVALID, "{owners:?}");
    }
    for rung in [2, 3, 63, 255] {
        assert_eq!(
            unsafe { seabraid_placement_quarantine(placement.0, 1, rung) },
            INVALID,
            "rung {rung}"
        );
    }
    assert_eq!(
        plan(&placement, 1, &[1.0, 20.0]).0,
        0,
        "epoch 1 is still current after every refusal"
    );
}

// case P48-P50: quarantining the same rung again is a no-op, and closing every card still leaves a valid map.
#[test]
fn quarantine_is_idempotent_and_closing_every_card_routes_everything_to_the_fallback() {
    let _capture = Capture::start(Level::Off);
    let placement = TestPlacement::new();
    assert_eq!(
        unsafe { seabraid_placement_quarantine(placement.0, 1, 0) },
        0
    );
    assert_eq!(
        unsafe { seabraid_placement_quarantine(placement.0, 2, 0) },
        0,
        "already closed: no epoch change"
    );
    assert_eq!(
        unsafe { seabraid_placement_quarantine(placement.0, 2, 1) },
        0
    );
    let mut executors = [0u8; 2];
    let mut out = MaybeUninit::new(blank_snapshot());
    let rc = unsafe {
        seabraid_placement_snapshot(placement.0, executors.as_mut_ptr(), 2, out.as_mut_ptr())
    };
    assert_eq!(rc, 0);
    assert_eq!(executors, [2, 2]);
    let snapshot = unsafe { out.assume_init() };
    assert_eq!((snapshot.epoch, snapshot.quarantined_rungs), (3, 3));
    let (rc, ticket) = plan(&placement, 3, &[1.0, 20.0]);
    assert_eq!(
        (rc, ticket.promotion_count),
        (0, 0),
        "no open card pair can take a promotion"
    );
}

// case P51-P59: the library-filled structs, every struct_size, through the placement entry points.
#[test]
fn ticket_and_snapshot_struct_size_matrix() {
    let _capture = Capture::start(Level::Off);
    for size in [0u32, 31, 32, 33, 40, 41, 64, 1024, 1025, u32::MAX] {
        let placement = TestPlacement::new();
        let mut words = vec![0xA5A5_A5A5_A5A5_A5A5u64; 160];
        words[0] = u64::from(ABI_VERSION) | (u64::from(size) << 32);
        let heat = [1.0, 20.0];
        let rc = unsafe {
            seabraid_placement_plan(placement.0, 1, heat.as_ptr(), 2, words.as_mut_ptr().cast())
        };
        let own = mem::size_of::<Ticket>() as u32;
        if size < 32 {
            assert_eq!(rc, INVALID, "ticket struct_size {size}");
            assert_eq!(words[1], 0xA5A5_A5A5_A5A5_A5A5, "untouched");
            continue;
        }
        if size % 8 != 0 {
            assert_eq!(
                rc, INVALID,
                "ticket struct_size {size} is not a multiple of 8"
            );
            assert_eq!(words[1], 0xA5A5_A5A5_A5A5_A5A5, "untouched");
            continue;
        }
        assert_eq!(rc, 0, "ticket struct_size {size}");
        assert_eq!(
            (words[0] >> 32) as u32,
            size.min(own),
            "written size reads back"
        );
        let past = (size.min(own) as usize).div_ceil(8);
        assert!(
            words[past.max(1)..]
                .iter()
                .all(|&w| w == 0xA5A5_A5A5_A5A5_A5A5 || size < own),
            "nothing past the caller's size"
        );
    }
    for size in [0u32, 39, 40, 41, 48, 4096] {
        let placement = TestPlacement::new();
        let mut words = vec![0x5A5A_5A5A_5A5A_5A5Au64; 160];
        words[0] = u64::from(ABI_VERSION) | (u64::from(size) << 32);
        let mut executors = [0u8; 2];
        let rc = unsafe {
            seabraid_placement_snapshot(
                placement.0,
                executors.as_mut_ptr(),
                2,
                words.as_mut_ptr().cast(),
            )
        };
        assert_eq!(
            rc == 0,
            size >= 40 && size % 8 == 0,
            "snapshot struct_size {size}"
        );
    }
}

// case P74-P80: the provider's slot sizes and an empty plan. A size past the cap (R6a P4, provider::MAX_SLOT_BYTES) is
// refused at bind like zero is; the cap and what stage does with a huge size are tested in edge_slots.rs.
#[test]
fn slot_sizes_of_zero_or_past_the_cap_are_invalid() {
    let _capture = Capture::start(Level::Off);
    let placement = TestPlacement::new();
    let ticket = plan(&placement, 1, &[1.0, 20.0]).1.ticket;
    let mut outcome = Outcome {
        abi_version: ABI_VERSION,
        struct_size: mem::size_of::<Outcome>() as u32,
        ..Outcome::default()
    };
    let mut applied = Applied {
        abi_version: ABI_VERSION,
        struct_size: mem::size_of::<Applied>() as u32,
        epoch: 0,
        quarantined_rungs: 0,
        outcome_count: 0,
    };
    let mut call = |slots: &[usize], count: usize| {
        let io = RawIo {
            abi_version: ABI_VERSION,
            context: ptr::null_mut(),
            layer_slot_bytes: slots.as_ptr(),
            layer_count: count,
            stage: Some(stage_ok),
            replace: Some(replace_ok),
        };
        unsafe {
            seabraid_placement_apply(
                placement.0,
                ticket,
                ptr::from_ref(&io).cast(),
                &mut outcome,
                1,
                &mut applied,
            )
        }
    };
    for bad in [
        [0usize],
        [usize::MAX],
        [isize::MAX as usize + 1],
        [isize::MAX as usize],
        [provider::MAX_SLOT_BYTES + 1],
    ] {
        assert_eq!(call(&bad, 1), INVALID, "slot size {}", bad[0]);
    }
    assert_eq!(call(&[4, 4], 2), INVALID, "two slot sizes for one layer");
    assert_eq!(call(&[4], 0), INVALID, "layer_count 0 for one layer");
    assert!(
        unsafe { handle(placement.0) }.unwrap().pending.is_some(),
        "the plan survived every refusal"
    );
}

// case P81: an empty plan needs no outcome array and no out-of-range capacity.
#[test]
fn applying_an_empty_plan_needs_no_outcome_array() {
    let _capture = Capture::start(Level::Off);
    let placement = TestPlacement::new();
    let ticket = plan(&placement, 1, &[5.0, 5.0]).1;
    assert_eq!(ticket.promotion_count, 0);
    let slot = [4usize];
    let io = RawIo {
        abi_version: ABI_VERSION,
        context: ptr::null_mut(),
        layer_slot_bytes: slot.as_ptr(),
        layer_count: 1,
        stage: Some(stage_ok),
        replace: Some(replace_ok),
    };
    let mut applied = Applied {
        abi_version: ABI_VERSION,
        struct_size: mem::size_of::<Applied>() as u32,
        epoch: 0,
        quarantined_rungs: 0,
        outcome_count: 9,
    };
    let rc = unsafe {
        seabraid_placement_apply(
            placement.0,
            ticket.ticket,
            ptr::from_ref(&io).cast(),
            ptr::null_mut(),
            0,
            &mut applied,
        )
    };
    assert_eq!((rc, applied.outcome_count, applied.epoch), (0, 0, 1));
}
