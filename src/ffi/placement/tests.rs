// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
mod edge;
mod edge_io;
mod edge_policy_fill;
mod edge_slots;
mod edge_structs;
mod handle;
mod open_v2;
mod provider_v2;
mod provider_v2_rules;
mod table;

use super::apply::seabraid_placement_apply;
use super::types::{Applied, Outcome, OUTCOME_COMMITTED};
use super::*;
use crate::diag::testing::Capture;
use crate::diag::{self, Level};
use std::mem::{self, MaybeUninit};
use std::ptr;

/// Caller-allocated output headers: abi_version and struct_size are set, as the ABI requires.
fn blank_ticket() -> Ticket {
    Ticket {
        abi_version: ABI_VERSION,
        struct_size: mem::size_of::<Ticket>() as u32,
        ticket: 0,
        epoch: 0,
        promotion_count: 0,
    }
}

fn blank_snapshot() -> Snapshot {
    Snapshot {
        abi_version: ABI_VERSION,
        struct_size: mem::size_of::<Snapshot>() as u32,
        epoch: 0,
        quarantined_rungs: 0,
        layers: 0,
        experts: 0,
        fallback_rung: 0,
        _pad: [0; 7],
    }
}

struct TestPlacement(*mut PlacementHandle);

impl TestPlacement {
    fn new() -> Self {
        let owners = [0, 1];
        let layout = Layout {
            abi_version: ABI_VERSION,
            layers: 1,
            experts: 2,
            cards: 2,
            owners: owners.as_ptr(),
            owner_count: owners.len(),
        };
        let mut handle = ptr::null_mut();
        assert_eq!(
            unsafe { seabraid_placement_open(&layout, ptr::null(), &mut handle) },
            0
        );
        Self(handle)
    }

    fn plan(&self, epoch: u64) -> Ticket {
        let heat = [1.0, 20.0];
        let mut out = MaybeUninit::new(blank_ticket());
        assert_eq!(
            unsafe {
                seabraid_placement_plan(self.0, epoch, heat.as_ptr(), heat.len(), out.as_mut_ptr())
            },
            0
        );
        unsafe { out.assume_init() }
    }
}

impl Drop for TestPlacement {
    fn drop(&mut self) {
        assert_eq!(unsafe { seabraid_placement_close(self.0) }, 0);
    }
}

#[test]
fn rejects_shape_before_reading_input_arrays() {
    let mut layout = Layout {
        abi_version: ABI_VERSION,
        layers: u32::MAX,
        experts: u32::MAX,
        cards: 2,
        owners: ptr::null(),
        owner_count: 0,
    };
    let mut out = ptr::null_mut();
    assert_eq!(
        unsafe { seabraid_placement_open(&layout, ptr::null(), &mut out) },
        Code::Invalid as i32
    );
    layout.layers = 1;
    layout.experts = 2;
    layout.owner_count = 2;
    assert_eq!(
        unsafe { seabraid_placement_open(&layout, ptr::null(), &mut out) },
        Code::Invalid as i32
    );
    assert!(out.is_null());

    let placement = TestPlacement::new();
    let heat = [5.0, 10.0];
    let mut ticket = MaybeUninit::new(blank_ticket());
    assert_eq!(
        unsafe { seabraid_placement_plan(placement.0, 1, heat.as_ptr(), 1, ticket.as_mut_ptr()) },
        Code::Invalid as i32
    );
    assert_eq!(placement.plan(1).ticket, 1);
}

#[test]
fn stale_ticket_short_circuits_preflight_and_exports_fallback() {
    let placement = TestPlacement::new();
    let ticket = placement.plan(1);
    assert_eq!(ticket.promotion_count, 1);
    assert_eq!(
        unsafe { seabraid_placement_quarantine(placement.0, 1, 1) },
        0
    );
    // No output or provider table is needed to reject a stale ticket.
    let stale = unsafe {
        seabraid_placement_apply(
            placement.0,
            ticket.ticket,
            ptr::null(),
            ptr::null_mut(),
            0,
            ptr::null_mut(),
        )
    };
    assert_eq!(stale, Code::BusyOrState as i32);
    assert!(unsafe { handle(placement.0) }.unwrap().pending.is_none());
    let mut out = MaybeUninit::new(blank_snapshot());
    let mut executors = [255, 255];
    assert_eq!(
        unsafe {
            seabraid_placement_snapshot(
                placement.0,
                executors.as_mut_ptr(),
                executors.len(),
                out.as_mut_ptr(),
            )
        },
        0
    );
    assert_eq!(executors, [0, 2]);
    let snapshot = unsafe { out.assume_init() };
    assert_eq!(snapshot.epoch, 2);
    assert_eq!(snapshot.quarantined_rungs, 2);
}

fn expect_internal(rc: i32, text: &str) {
    assert_eq!(rc, Code::Internal as i32, "{text}");
    assert_eq!(diag::last(), text);
}

#[test]
fn a_failed_open_always_leaves_out_null() {
    let _capture = Capture::start(Level::Off);
    let mut out = 0x8 as *mut PlacementHandle; // a stale pointer a careless caller might close
    let bad = Layout {
        owner_count: 7,
        ..TestLayout::new().layout()
    };
    assert_eq!(
        unsafe { seabraid_placement_open(&bad, ptr::null(), &mut out) },
        Code::Invalid as i32
    );
    assert!(out.is_null());
    let mut out = 0x8 as *mut PlacementHandle;
    assert_eq!(
        unsafe { seabraid_placement_open(ptr::null(), ptr::null(), &mut out) },
        Code::Invalid as i32
    );
    assert!(out.is_null());
}

struct TestLayout([Rung; 2]);

impl TestLayout {
    fn new() -> Self {
        Self([0, 1])
    }

    fn layout(&self) -> Layout {
        Layout {
            abi_version: ABI_VERSION,
            layers: 1,
            experts: 2,
            cards: 2,
            owners: self.0.as_ptr(),
            owner_count: 2,
        }
    }
}

#[test]
fn a_caught_panic_poisons_the_placement_and_close_still_frees_it() {
    let _capture = Capture::start(Level::Off);
    let placement = TestPlacement::new();
    let rc = unsafe {
        status_on("seabraid_test_panic", placement.0, || -> Result<()> {
            panic!("injected panic")
        })
    };
    assert_eq!(rc, Code::Internal as i32);
    let heat = [1.0, 20.0];
    let mut out = blank_ticket();
    let rc = unsafe { seabraid_placement_plan(placement.0, 1, heat.as_ptr(), 2, &mut out) };
    expect_internal(
        rc,
        "seabraid_placement_plan: placement handle poisoned by earlier panic; close it",
    );
    let rc = unsafe { seabraid_placement_quarantine(placement.0, 1, 1) };
    expect_internal(
        rc,
        "seabraid_placement_quarantine: placement handle poisoned by earlier panic; close it",
    );
    // TestPlacement::drop asserts that seabraid_placement_close still succeeds.
}

#[test]
fn quarantine_reports_epoch_exhaustion_as_itself() {
    let _capture = Capture::start(Level::Off);
    let placement = TestPlacement::new();
    unsafe { handle(placement.0) }
        .unwrap()
        .map
        .set_epoch_for_test(u64::MAX);
    let rc = unsafe { seabraid_placement_quarantine(placement.0, u64::MAX, 1) };
    assert_eq!(rc, Code::BusyOrState as i32);
    assert_eq!(
        diag::last(),
        "seabraid_placement_quarantine: epoch counter exhausted"
    );
}

#[repr(C)]
struct RawIo {
    abi_version: u32,
    context: *mut std::ffi::c_void,
    layer_slot_bytes: *const usize,
    layer_count: usize,
    stage: Option<
        unsafe extern "C" fn(
            *mut std::ffi::c_void,
            u8,
            u32,
            u32,
            *mut u8,
            usize,
            *mut usize,
        ) -> i32,
    >,
    replace: Option<
        unsafe extern "C" fn(*mut std::ffi::c_void, u8, u32, u32, u32, *const u8, usize) -> i32,
    >,
}

unsafe extern "C" fn stage_ok(
    _: *mut std::ffi::c_void,
    _: u8,
    _: u32,
    _: u32,
    bytes: *mut u8,
    size: usize,
    written: *mut usize,
) -> i32 {
    unsafe {
        ptr::write_bytes(bytes, 7, size);
        *written = size;
    }
    0
}

unsafe extern "C" fn replace_ok(
    _: *mut std::ffi::c_void,
    _: u8,
    _: u32,
    _: u32,
    _: u32,
    _: *const u8,
    _: usize,
) -> i32 {
    0
}

/// A caller whose outcome struct is wider than v1 (a newer header): the stride is its struct_size and the
/// unknown tail is never written; a stride that would misalign the next element is refused before any I/O.
#[test]
fn outcomes_use_the_callers_struct_size_as_stride() {
    #[repr(C)]
    struct Wide {
        outcome: Outcome,
        tail: [u8; 8],
    }
    let _capture = Capture::start(Level::Off);
    let placement = TestPlacement::new();
    let ticket = placement.plan(1).ticket;
    let slot = [4usize];
    let io = RawIo {
        abi_version: ABI_VERSION,
        context: ptr::null_mut(),
        layer_slot_bytes: slot.as_ptr(),
        layer_count: 1,
        stage: Some(stage_ok),
        replace: Some(replace_ok),
    };
    let mut wide = Wide {
        outcome: Outcome {
            abi_version: ABI_VERSION,
            struct_size: 66,
            ..Outcome::default()
        },
        tail: [0xAB; 8],
    };
    let mut applied = Applied {
        abi_version: ABI_VERSION,
        struct_size: mem::size_of::<Applied>() as u32,
        epoch: 0,
        quarantined_rungs: 0,
        outcome_count: 0,
    };
    let call = |wide: &mut Wide, applied: &mut Applied| unsafe {
        seabraid_placement_apply(
            placement.0,
            ticket,
            ptr::from_ref(&io).cast(),
            &mut wide.outcome,
            1,
            applied,
        )
    };
    assert_eq!(call(&mut wide, &mut applied), Code::Invalid as i32);
    assert_eq!(
        diag::last(),
        "seabraid_placement_apply: outcomes struct_size 66 is not a multiple of 8"
    );
    wide.outcome.struct_size = 72;
    assert_eq!(call(&mut wide, &mut applied), 0);
    assert_eq!(wide.outcome.kind, OUTCOME_COMMITTED);
    assert_eq!(
        wide.outcome.struct_size, 64,
        "the library reports what it wrote"
    );
    assert_eq!(wide.tail, [0xAB; 8]);
    assert_eq!(applied.outcome_count, 1);
}
