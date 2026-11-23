// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! `seabraid_placement_apply_v2` v1 parity and rejection of malformed provider structs (version, size, mask, null, tail).
//! Fixture and helpers come from `provider_v2.rs`.

use super::edge::{plan, INVALID};
use super::provider_v2::*;
use super::*;
use crate::diag::testing::Capture;
use crate::diag::Level;
use std::ffi::c_void;

fn bytes_of<T>(value: &T) -> Vec<u8> {
    unsafe { std::slice::from_raw_parts(ptr::from_ref(value).cast::<u8>(), mem::size_of::<T>()) }
        .to_vec()
}

#[test]
fn a_zero_mask_equals_apply_on_a_v1_provider() {
    let _capture = Capture::start(Level::Off);
    for lower in [ACK, REFUSED, FAILED, TIMEOUT] {
        let (mut one, mut two) = (Fx::new(), Fx::new());
        (one.lower_code, two.lower_code) = (lower, lower);
        // v1: the same callbacks, no struct_size, no mask.
        let first = TestPlacement::new();
        let ticket = plan(&first, 1, &[1.0, 20.0]).1.ticket;
        let io = RawIoV1 {
            abi_version: ABI_VERSION,
            context: ptr::from_mut(&mut one).cast(),
            layer_slot_bytes: SLOTS.as_ptr(),
            layer_count: 1,
            stage: Some(stage),
            replace: Some(replace),
        };
        let mut o1 = Outcome {
            abi_version: ABI_VERSION,
            struct_size: 64,
            ..Outcome::default()
        };
        let mut a1 = blank_applied();
        let rc1 = unsafe {
            seabraid_placement_apply(
                first.0,
                ticket,
                ptr::from_ref(&io).cast(),
                &mut o1,
                1,
                &mut a1,
            )
        };
        let got = run(0, &mut two);
        assert_eq!((rc1, got.rc), (0, 0), "lower answers {lower}");
        assert_eq!(
            bytes_of(&o1),
            bytes_of(&got.outcome),
            "lower answers {lower}"
        );
        assert_eq!(
            bytes_of(&a1),
            bytes_of(&got.applied),
            "lower answers {lower}"
        );
        assert_eq!(
            (&one.calls, one.bytes, one.resident),
            (&two.calls, two.bytes, two.resident),
            "lower answers {lower}"
        );
    }
}

#[repr(C)]
struct RawIoV1 {
    abi_version: u32,
    context: *mut c_void,
    layer_slot_bytes: *const usize,
    layer_count: usize,
    stage: Option<StageFn>,
    replace: Option<ReplaceFn>,
}

fn blank_applied() -> Applied {
    Applied {
        abi_version: ABI_VERSION,
        struct_size: mem::size_of::<Applied>() as u32,
        epoch: 0,
        quarantined_rungs: 0,
        outcome_count: 0,
    }
}

/// Every refusal below happens before the first provider call and keeps the ticket: a corrected call then works.
fn refused_then_corrected(mutate: impl FnOnce(&mut RawV2), why: &str) {
    let _capture = Capture::start(Level::Off);
    let placement = TestPlacement::new();
    let ticket = plan(&placement, 1, &[1.0, 20.0]).1.ticket;
    let mut fx = Fx::new();
    let mut bad = raw(&mut fx, 0);
    mutate(&mut bad);
    assert_eq!(
        apply_v2(&placement, ticket, ptr::from_ref(&bad).cast()).rc,
        INVALID,
        "{why}"
    );
    assert!(
        fx.calls.is_empty(),
        "{why}: no provider call before the refusal"
    );
    let good = raw(&mut fx, 1);
    let got = apply_v2(&placement, ticket, ptr::from_ref(&good).cast());
    assert_eq!(
        (got.rc, got.outcome.kind),
        (0, OUTCOME_COMMITTED),
        "{why}: the ticket was kept"
    );
}

#[test]
fn a_bad_version_or_struct_size_or_mask_is_invalid_before_any_io_and_keeps_the_ticket() {
    refused_then_corrected(|p| p.abi_version = 1, "version 1");
    refused_then_corrected(|p| p.abi_version = 3, "version 3");
    for size in [0, 8, 56, 63, 68, 1025] {
        refused_then_corrected(|p| p.struct_size = size, &format!("struct_size {size}"));
    }
    refused_then_corrected(
        |p| p.from_fallback_rungs = 1 << 2,
        "the fallback rung (2) is masked",
    );
    refused_then_corrected(
        |p| p.from_fallback_rungs = 1 << 63,
        "a rung past the cards is masked",
    );
    refused_then_corrected(|p| p.from_fallback_rungs = u64::MAX, "every bit");
    refused_then_corrected(
        |p| (p.from_fallback_rungs, p.replace_from_fallback) = (1, None),
        "a non-zero mask without the callback",
    );
    refused_then_corrected(|p| p.layer_slot_bytes = ptr::null(), "null slot table");
    refused_then_corrected(|p| p.stage = None, "null stage");
}

#[test]
fn a_null_provider_is_invalid_and_keeps_the_ticket() {
    let _capture = Capture::start(Level::Off);
    let placement = TestPlacement::new();
    let ticket = plan(&placement, 1, &[1.0, 20.0]).1.ticket;
    assert_eq!(apply_v2(&placement, ticket, ptr::null()).rc, INVALID);
    let mut fx = Fx::new();
    let good = raw(&mut fx, 0);
    assert_eq!(
        apply_v2(&placement, ticket, ptr::from_ref(&good).cast()).rc,
        0
    );
}

#[test]
fn a_mask_with_the_callback_null_only_when_zero_and_the_callback_set_with_a_zero_mask_both_work() {
    let _capture = Capture::start(Level::Off);
    let mut fx = Fx::new();
    let mut provider = raw(&mut fx, 0);
    provider.replace_from_fallback = None;
    let placement = TestPlacement::new();
    let ticket = plan(&placement, 1, &[1.0, 20.0]).1.ticket;
    assert_eq!(
        apply_v2(&placement, ticket, ptr::from_ref(&provider).cast()).rc,
        0
    );
    let mut other = Fx::new();
    let provider = raw(&mut other, 0); // callback set, mask 0: never called
    let placement = TestPlacement::new();
    let ticket = plan(&placement, 1, &[1.0, 20.0]).1.ticket;
    assert_eq!(
        apply_v2(&placement, ticket, ptr::from_ref(&provider).cast()).rc,
        0
    );
    assert!(other.calls.iter().all(|call| call.0 != "ref"));
}

#[test]
fn a_wider_struct_with_a_zero_tail_works_and_a_dirty_tail_is_refused_before_io() {
    let _capture = Capture::start(Level::Off);
    let placement = TestPlacement::new();
    let ticket = plan(&placement, 1, &[1.0, 20.0]).1.ticket;
    let mut fx = Fx::new();
    let mut wide = Wide {
        v2: raw(&mut fx, 1),
        tail: [0; 16],
    };
    wide.v2.struct_size = mem::size_of::<Wide>() as u32;
    for at in [0, 7, 15] {
        wide.tail[at] = 1;
        assert_eq!(
            apply_v2(&placement, ticket, ptr::from_ref(&wide).cast()).rc,
            INVALID,
            "tail byte {at}"
        );
        assert!(
            fx.calls.is_empty(),
            "tail byte {at}: refused before any provider call"
        );
        wide.tail[at] = 0;
    }
    let got = apply_v2(&placement, ticket, ptr::from_ref(&wide).cast());
    assert_eq!((got.rc, got.outcome.kind), (0, OUTCOME_COMMITTED));
    assert_eq!(
        fx.calls,
        [("stage", 0), ("ref", 0), ("replace", 1)],
        "the mask in a wider struct is read"
    );
}
