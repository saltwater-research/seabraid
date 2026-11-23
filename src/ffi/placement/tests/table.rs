// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! Failure table for the placement entry points: one invalid input each, exact text.
//! Run `cargo test failure_table -- --nocapture` to print the TABLE rows.

use super::*;
use crate::diag::testing::{table_row, Capture};
use crate::diag::{self, Level};
use fill::{seabraid_placement_fill, FillRequest};

fn expect(entry: &str, rc: i32, code: Code, text: &str) {
    assert_eq!(rc, code as i32, "{entry}: {text}");
    let message = diag::last();
    table_row(entry, rc, &message);
    assert_eq!(message, format!("{entry}: {text}"));
}

fn layout(layers: u32, owners: &[Rung]) -> Layout {
    Layout {
        abi_version: ABI_VERSION,
        layers,
        experts: 2,
        cards: 2,
        owners: owners.as_ptr(),
        owner_count: owners.len(),
    }
}

#[test]
fn failure_table_placement_open_plan_apply() {
    let _capture = Capture::start(Level::Off);
    let mut out = ptr::null_mut();
    let rc = unsafe { seabraid_placement_open(ptr::null(), ptr::null(), &mut out) };
    expect(
        "seabraid_placement_open",
        rc,
        Code::Invalid,
        "layout is null",
    );
    let rc = unsafe { seabraid_placement_open(&layout(0, &[0, 1]), ptr::null(), &mut out) };
    let message = diag::last();
    table_row("seabraid_placement_open", rc, &message);
    assert!(
        message.contains("layout 0 layers x 2 experts with 2 cards"),
        "{message}"
    );
    let rc = unsafe { seabraid_placement_open(&layout(1, &[0, 1, 0]), ptr::null(), &mut out) };
    expect(
        "seabraid_placement_open",
        rc,
        Code::Invalid,
        "owners count 3 != expected 2",
    );
    let rc = unsafe { seabraid_placement_open(&layout(1, &[0, 3]), ptr::null(), &mut out) };
    expect(
        "seabraid_placement_open",
        rc,
        Code::Invalid,
        "layout or owner table is invalid",
    );

    let placement = TestPlacement::new();
    let heat = [1.0, 20.0];
    let mut ticket = MaybeUninit::new(blank_ticket());
    let rc =
        unsafe { seabraid_placement_plan(placement.0, 7, heat.as_ptr(), 2, ticket.as_mut_ptr()) };
    expect(
        "seabraid_placement_plan",
        rc,
        Code::BusyOrState,
        "epoch 7 is stale, current epoch is 1",
    );
    let rc =
        unsafe { seabraid_placement_plan(placement.0, 1, heat.as_ptr(), 1, ticket.as_mut_ptr()) };
    expect(
        "seabraid_placement_plan",
        rc,
        Code::Invalid,
        "heat count 1 != expected 2",
    );
    let rc = unsafe {
        seabraid_placement_apply(
            placement.0,
            42,
            ptr::null(),
            ptr::null_mut(),
            0,
            ptr::null_mut(),
        )
    };
    expect(
        "seabraid_placement_apply",
        rc,
        Code::BusyOrState,
        "ticket 42 is not the pending plan",
    );
    let ticket = placement.plan(1).ticket;
    let rc = unsafe {
        seabraid_placement_apply(
            placement.0,
            ticket,
            ptr::null(),
            ptr::null_mut(),
            0,
            ptr::null_mut(),
        )
    };
    expect("seabraid_placement_apply", rc, Code::Invalid, "out is null");
}

#[test]
fn failure_table_placement_snapshot_update_close() {
    let _capture = Capture::start(Level::Off);
    let placement = TestPlacement::new();
    let mut out = MaybeUninit::new(blank_snapshot());
    let mut executors = [0u8; 1];
    let rc = unsafe {
        seabraid_placement_snapshot(placement.0, executors.as_mut_ptr(), 1, out.as_mut_ptr())
    };
    expect(
        "seabraid_placement_snapshot",
        rc,
        Code::Invalid,
        "executor_count 1 != cells 2",
    );
    let rc = unsafe { seabraid_placement_quarantine(placement.0, 1, 2) };
    expect(
        "seabraid_placement_quarantine",
        rc,
        Code::Invalid,
        "rung 2 is not a card, fallback is rung 2 (layout or owner table is invalid)",
    );
    let owners = [0u8, 1];
    let rc = unsafe { seabraid_placement_update(placement.0, 1, 9, owners.as_ptr(), 2) };
    expect(
        "seabraid_placement_update",
        rc,
        Code::Invalid,
        "update mode 9 is not 0 or 1",
    );
    let rc = unsafe { seabraid_placement_update(placement.0, 1, 0, owners.as_ptr(), 1) };
    expect(
        "seabraid_placement_update",
        rc,
        Code::Invalid,
        "owners count 1 != expected 2",
    );
    let rc = unsafe { seabraid_placement_close(ptr::null_mut()) };
    expect(
        "seabraid_placement_close",
        rc,
        Code::Invalid,
        "placement is null",
    );
}

#[test]
fn failure_table_placement_fill() {
    let _capture = Capture::start(Level::Off);
    let (slots, heat) = ([1u64, 2], [1.0f64, 5.0, 3.0, 2.0]);
    let request = FillRequest {
        abi_version: ABI_VERSION,
        layers: 1,
        experts: 4,
        cards: 2,
        slots: slots.as_ptr(),
        slot_count: 2,
        heat: heat.as_ptr(),
        heat_count: 4,
    };
    let mut owners = [255u8; 4];
    let rc = unsafe { seabraid_placement_fill(ptr::null(), owners.as_mut_ptr(), 4) };
    expect(
        "seabraid_placement_fill",
        rc,
        Code::Invalid,
        "request is null",
    );
    let rc = unsafe { seabraid_placement_fill(&request, owners.as_mut_ptr(), 3) };
    expect(
        "seabraid_placement_fill",
        rc,
        Code::Invalid,
        "owner_count 3 != cells 4",
    );
    let short = FillRequest {
        slot_count: 3,
        ..request
    };
    let rc = unsafe { seabraid_placement_fill(&short, owners.as_mut_ptr(), 4) };
    expect(
        "seabraid_placement_fill",
        rc,
        Code::Invalid,
        "slots count 3 != expected 2",
    );
    let cold = [1.0f64, -1.0, 3.0, 2.0];
    let negative = FillRequest {
        heat: cold.as_ptr(),
        ..request
    };
    let rc = unsafe { seabraid_placement_fill(&negative, owners.as_mut_ptr(), 4) };
    expect(
        "seabraid_placement_fill",
        rc,
        Code::Invalid,
        "heat must be finite, non-negative and layers * experts long",
    );
    assert_eq!(owners, [255; 4], "a rejected fill must not write");
    assert_eq!(
        unsafe { seabraid_placement_fill(&request, owners.as_mut_ptr(), 4) },
        0
    );
    assert_eq!(owners, [2, 0, 1, 1]);
}

#[test]
fn quarantine_through_the_c_abi_is_logged_at_warn() {
    let capture = Capture::start(Level::Debug);
    let placement = TestPlacement::new();
    assert_eq!(
        unsafe { seabraid_placement_quarantine(placement.0, 1, 1) },
        0
    );
    let records = capture.mine();
    assert_eq!(records.len(), 1, "{records:?}");
    assert_eq!(records[0].level, Level::Warn as u32);
    assert!(
        records[0].message.contains("0x2 quarantined"),
        "{}",
        records[0].message
    );
}
