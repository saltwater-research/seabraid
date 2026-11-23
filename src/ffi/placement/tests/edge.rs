// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! Overnight hardening O1 for the placement C entry points: degenerate shapes, degenerate heat, epoch extremes,
//! output-struct sizes and the order in which a refused call leaves state alone. Case ids refer to
//! scratch/2026-11-19-overnight/O1-cases.md (rows P*).

use super::*;
use crate::diag::testing::Capture;
use crate::diag::Level;

pub(super) const INVALID: i32 = Code::Invalid as i32;
pub(super) const BUSY: i32 = Code::BusyOrState as i32;

fn open_layout(
    layers: u32,
    experts: u32,
    cards: Rung,
    owners: &[Rung],
) -> (i32, *mut PlacementHandle) {
    let layout = Layout {
        abi_version: ABI_VERSION,
        layers,
        experts,
        cards,
        owners: owners.as_ptr(),
        owner_count: owners.len(),
    };
    let mut handle = ptr::NonNull::<PlacementHandle>::dangling().as_ptr();
    (
        unsafe { seabraid_placement_open(&layout, ptr::null(), &mut handle) },
        handle,
    )
}

pub(super) fn plan(placement: &TestPlacement, epoch: u64, heat: &[f64]) -> (i32, Ticket) {
    let mut out = MaybeUninit::new(blank_ticket());
    let rc = unsafe {
        seabraid_placement_plan(
            placement.0,
            epoch,
            heat.as_ptr(),
            heat.len(),
            out.as_mut_ptr(),
        )
    };
    (rc, unsafe { out.assume_init() })
}

// case P1-P9: shapes at and beyond the limits all fail with INVALID and leave `*out` null.
#[test]
fn open_with_zero_or_unrepresentable_layers_experts_and_cards_is_invalid_with_a_null_out() {
    let _capture = Capture::start(Level::Off);
    let owners = [0u8; 4];
    for (layers, experts, cards) in [
        (0, 2, 2),
        (2, 0, 2),
        (0, 0, 0),
        (1, 2, 0),
        (1, 2, 64),
        (1, 2, 255),
        (u32::MAX, 2, 2),
        (1 << 13, (1 << 13) + 1, 2), // one cell past 2^26
    ] {
        let (rc, handle) = open_layout(layers, experts, cards, &owners);
        assert_eq!(rc, INVALID, "{layers}x{experts} cards {cards}");
        assert!(handle.is_null());
    }
    // The smallest legal shapes: one cell and one card; and the largest card count.
    let (rc, handle) = open_layout(1, 1, 1, &[0]);
    assert_eq!((rc, handle.is_null()), (0, false));
    assert_eq!(unsafe { seabraid_placement_close(handle) }, 0);
    let owners = [63u8, 0];
    let (rc, handle) = open_layout(1, 2, 63, &owners);
    assert_eq!((rc, handle.is_null()), (0, false));
    assert_eq!(unsafe { seabraid_placement_close(handle) }, 0);
}

// case P10-P12: owner tables: wrong count, a rung above the fallback, a null array.
#[test]
fn open_refuses_a_wrong_owner_count_an_out_of_range_rung_and_a_null_table() {
    let _capture = Capture::start(Level::Off);
    for owners in [&[0u8][..], &[0u8, 1, 0][..], &[0u8, 3][..], &[0u8, 255][..]] {
        let (rc, handle) = open_layout(1, 2, 2, owners);
        assert_eq!(rc, INVALID, "{owners:?}");
        assert!(handle.is_null());
    }
    let (rc, handle) = open_layout(1, 2, 2, &[]);
    assert_eq!(
        (rc, handle.is_null()),
        (INVALID, true),
        "an empty slice has a dangling non-null pointer"
    );
}

// case P13-P24: heat of every float class, and the count.
#[test]
fn plan_refuses_nan_infinite_negative_and_miscounted_heat_and_accepts_the_degenerate_finite_ones() {
    let _capture = Capture::start(Level::Off);
    for bad in [
        [f64::NAN, 1.0],
        [1.0, f64::INFINITY],
        [f64::NEG_INFINITY, 1.0],
        [1.0, -1e-300],
        [-1.0, 1.0],
        [f64::from_bits(0xFFF8_0000_0000_1234), 0.0],
    ] {
        let placement = TestPlacement::new();
        let (rc, ticket) = plan(&placement, 1, &bad);
        assert_eq!(rc, INVALID, "{bad:?}");
        assert_eq!(ticket.ticket, 0, "a refused plan writes nothing");
        assert!(unsafe { handle(placement.0) }.unwrap().pending.is_none());
        assert_eq!(
            plan(&placement, 1, &[1.0, 20.0]).0,
            0,
            "the planner state was not seeded by the refusal"
        );
    }
    for wrong in [&[][..], &[1.0][..], &[1.0, 2.0, 3.0][..]] {
        let placement = TestPlacement::new();
        assert_eq!(
            plan(&placement, 1, wrong).0,
            INVALID,
            "{} heat values",
            wrong.len()
        );
    }
    for good in [
        [0.0, 0.0],
        [-0.0, 0.0],
        [5.0, 5.0],
        [f64::MAX / 2.0, f64::MAX / 2.0],
        [f64::MAX, 0.0],
        [5e-324, 1e-323],
    ] {
        let placement = TestPlacement::new();
        let (rc, _) = plan(&placement, 1, &good);
        assert_eq!(rc, 0, "{good:?}");
    }
}

// case P24: every value finite and in range, but one layer's total is not: refused loudly, by name, with the
// planner untouched (a NaN sum would make every comparison in the round meaningless).
#[test]
fn a_layer_whose_heat_overflows_f64_is_refused_with_a_message_that_says_so() {
    let capture = Capture::start(Level::Warn);
    let placement = TestPlacement::new();
    let (rc, ticket) = plan(&placement, 1, &[f64::MAX, f64::MAX]);
    assert_eq!((rc, ticket.ticket), (INVALID, 0));
    let message = diag::last();
    assert!(message.contains("sums past f64::MAX"), "{message}");
    assert!(
        !message.contains("must be finite, non-negative"),
        "the finite-input message would mislead: {message}"
    );
    assert_eq!(capture.mine().len(), 1, "logged once");
    assert_eq!(
        plan(&placement, 1, &[1.0, 20.0]).0,
        0,
        "the refusal did not seed the planner"
    );
}

// case P25-P26: all-equal and all-zero heat never promote anything.
#[test]
fn equal_and_zero_heat_produce_empty_plans() {
    let _capture = Capture::start(Level::Off);
    for heat in [
        [0.0, 0.0],
        [7.0, 7.0],
        [f64::MAX, 0.0],
        [f64::MAX / 2.0, f64::MAX / 2.0],
    ] {
        let placement = TestPlacement::new();
        let (rc, ticket) = plan(&placement, 1, &heat);
        assert_eq!((rc, ticket.promotion_count), (0, 0), "{heat:?}");
    }
}

// case P27-P33: epochs: 0, u64::MAX, one ahead, and a plan made against an epoch that then moved.
#[test]
fn plan_quarantine_and_update_refuse_every_epoch_but_the_current_one() {
    let _capture = Capture::start(Level::Off);
    let placement = TestPlacement::new();
    for epoch in [0, 2, u64::MAX] {
        assert_eq!(
            plan(&placement, epoch, &[1.0, 20.0]).0,
            BUSY,
            "plan at {epoch}"
        );
        assert_eq!(
            unsafe { seabraid_placement_quarantine(placement.0, epoch, 1) },
            BUSY
        );
        let owners = [0u8, 1];
        let rc = unsafe {
            seabraid_placement_update(placement.0, epoch, UPDATE_ADOPT, owners.as_ptr(), 2)
        };
        assert_eq!(rc, BUSY, "update at {epoch}");
    }
    let (rc, first) = plan(&placement, 1, &[1.0, 20.0]);
    assert_eq!((rc, first.ticket), (0, 1));
    // A second plan replaces the first: the old ticket is no longer the pending one.
    let (rc, second) = plan(&placement, 1, &[1.0, 20.0]);
    assert_eq!((rc, second.ticket), (0, 2));
    let rc = unsafe {
        seabraid_placement_apply(
            placement.0,
            first.ticket,
            ptr::null(),
            ptr::null_mut(),
            0,
            ptr::null_mut(),
        )
    };
    assert_eq!(rc, BUSY, "the replaced ticket");
    assert!(
        unsafe { handle(placement.0) }.unwrap().pending.is_some(),
        "a refused apply keeps the live plan"
    );
}
