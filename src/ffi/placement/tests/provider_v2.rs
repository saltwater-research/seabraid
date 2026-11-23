// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! `seabraid_placement_apply_v2` through the C entry point, with the provider laid out as a C caller lays it out
//! (`RawV2`, no Rust type from the library): the sized-struct rules, the mask rules, the by-reference calls and v1 parity.
//! The fixture is the two-card layout of `TestPlacement`: heat [1, 20] promotes expert 1 (rung 1) over expert 0 (rung 0).

use super::apply::seabraid_placement_apply_v2;
use super::edge::plan;
use super::provider::{PROVIDER_V2_BASE_SIZE, PROVIDER_V2_VERSION};
use super::types::{OUTCOME_QUARANTINED, OUTCOME_RESTORED, PHASE_REPLACE};
use super::*;
use crate::diag::testing::Capture;
use crate::diag::Level;
use std::ffi::c_void;

pub(super) const SLOT: usize = 4;
pub(super) const ACK: i32 = 0;
pub(super) const REFUSED: i32 = 1;
pub(super) const FAILED: i32 = 2;
pub(super) const TIMEOUT: i32 = 3;

/// Expert e's bytes in the pack: what a by-reference write puts into a slot.
pub(super) fn pack(expert: u32) -> u8 {
    10 + 20 * expert as u8
}

pub(super) struct Fx {
    pub(super) bytes: [[u8; SLOT]; 2],
    pub(super) resident: [u32; 2],
    pub(super) calls: Vec<(&'static str, u8)>,
    pub(super) ref_code: i32,
    pub(super) unknown_code: bool,
    /// What a replace on the lower card (rung 1) answers without writing; ACK writes.
    pub(super) lower_code: i32,
    /// Calls that broke the fixture's own expectations. WHY recorded and not asserted: these callbacks are
    /// `extern "C"`, so a panic in one aborts the whole test process instead of failing one test.
    pub(super) bad: Vec<String>,
}

impl Drop for Fx {
    fn drop(&mut self) {
        assert!(
            std::thread::panicking() || self.bad.is_empty(),
            "unexpected provider calls: {:?}",
            self.bad
        );
    }
}

impl Fx {
    pub(super) fn new() -> Self {
        Self {
            bytes: [[pack(0); SLOT], [pack(1); SLOT]],
            resident: [0, 1],
            calls: Vec::new(),
            ref_code: ACK,
            unknown_code: false,
            lower_code: ACK,
            bad: Vec::new(),
        }
    }

    pub(super) fn names(&self) -> Vec<&'static str> {
        self.calls.iter().map(|call| call.0).collect()
    }
}

pub(super) unsafe extern "C" fn stage(
    context: *mut c_void,
    rung: u8,
    _layer: u32,
    expert: u32,
    output: *mut u8,
    size: usize,
    written: *mut usize,
) -> i32 {
    let fx = unsafe { &mut *context.cast::<Fx>() };
    if (fx.resident[rung as usize], size) != (expert, SLOT) {
        fx.bad
            .push(format!("stage rung {rung} expert {expert} size {size}"));
        return FAILED;
    }
    fx.calls.push(("stage", rung));
    unsafe { ptr::copy_nonoverlapping(fx.bytes[rung as usize].as_ptr(), output, size) };
    unsafe { *written = size };
    ACK
}

pub(super) unsafe extern "C" fn replace(
    context: *mut c_void,
    rung: u8,
    _layer: u32,
    old: u32,
    new: u32,
    input: *const u8,
    size: usize,
) -> i32 {
    let fx = unsafe { &mut *context.cast::<Fx>() };
    if (fx.resident[rung as usize], size) != (old, SLOT) {
        fx.bad
            .push(format!("replace rung {rung} old {old} size {size}"));
        return FAILED;
    }
    fx.calls.push(("replace", rung));
    if rung == 1 && fx.lower_code != ACK {
        return std::mem::replace(&mut fx.lower_code, ACK);
    }
    unsafe { ptr::copy_nonoverlapping(input, fx.bytes[rung as usize].as_mut_ptr(), size) };
    fx.resident[rung as usize] = new;
    ACK
}

unsafe extern "C" fn by_reference(
    context: *mut c_void,
    rung: u8,
    _layer: u32,
    old: u32,
    new: u32,
) -> i32 {
    let fx = unsafe { &mut *context.cast::<Fx>() };
    if fx.resident[rung as usize] != old {
        fx.bad.push(format!("reference rung {rung} old {old}"));
        return FAILED;
    }
    fx.calls.push(("ref", rung));
    if fx.unknown_code {
        return 77;
    }
    match fx.ref_code {
        ACK => {
            fx.bytes[rung as usize] = [pack(new); SLOT];
            fx.resident[rung as usize] = new;
        }
        FAILED | TIMEOUT => fx.bytes[rung as usize][0] ^= 0xFF, // began loading, then died: torn
        _ => {}
    }
    fx.ref_code
}

pub(super) type StageFn =
    unsafe extern "C" fn(*mut c_void, u8, u32, u32, *mut u8, usize, *mut usize) -> i32;
pub(super) type ReplaceFn =
    unsafe extern "C" fn(*mut c_void, u8, u32, u32, u32, *const u8, usize) -> i32;
pub(super) type RefFn = unsafe extern "C" fn(*mut c_void, u8, u32, u32, u32) -> i32;

/// seabraid_placement_provider_v2 as a C caller writes it.
#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct RawV2 {
    pub(super) abi_version: u32,
    pub(super) struct_size: u32,
    pub(super) context: *mut c_void,
    pub(super) layer_slot_bytes: *const usize,
    pub(super) layer_count: usize,
    pub(super) stage: Option<StageFn>,
    pub(super) replace: Option<ReplaceFn>,
    pub(super) from_fallback_rungs: u64,
    pub(super) replace_from_fallback: Option<RefFn>,
}

/// A caller built against a newer library: the same struct and a tail this library does not know.
#[repr(C)]
pub(super) struct Wide {
    pub(super) v2: RawV2,
    pub(super) tail: [u8; 16],
}

pub(super) static SLOTS: [usize; 1] = [SLOT];

pub(super) fn raw(fx: &mut Fx, mask: u64) -> RawV2 {
    RawV2 {
        abi_version: PROVIDER_V2_VERSION,
        struct_size: mem::size_of::<RawV2>() as u32,
        context: ptr::from_mut(fx).cast(),
        layer_slot_bytes: SLOTS.as_ptr(),
        layer_count: 1,
        stage: Some(stage),
        replace: Some(replace),
        from_fallback_rungs: mask,
        replace_from_fallback: Some(by_reference),
    }
}

pub(super) struct Run {
    pub(super) rc: i32,
    pub(super) outcome: Outcome,
    pub(super) applied: Applied,
}

pub(super) fn apply_v2(placement: &TestPlacement, ticket: u64, provider: *const c_void) -> Run {
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
    let rc = unsafe {
        seabraid_placement_apply_v2(
            placement.0,
            ticket,
            provider.cast(),
            &mut outcome,
            1,
            &mut applied,
        )
    };
    Run {
        rc,
        outcome,
        applied,
    }
}

pub(super) fn run(mask: u64, fx: &mut Fx) -> Run {
    let placement = TestPlacement::new();
    let ticket = plan(&placement, 1, &[1.0, 20.0]).1.ticket;
    let provider = raw(fx, mask);
    apply_v2(&placement, ticket, ptr::from_ref(&provider).cast())
}

#[test]
fn the_struct_is_64_bytes_with_the_documented_version_and_base_size() {
    assert_eq!(
        (
            mem::size_of::<RawV2>(),
            PROVIDER_V2_BASE_SIZE,
            PROVIDER_V2_VERSION
        ),
        (64, 64, 2)
    );
    assert_eq!(mem::size_of::<provider::ProviderV2>(), 64);
}

#[test]
fn an_acknowledged_reference_commits_without_staging_the_climber() {
    let _capture = Capture::start(Level::Off);
    let mut fx = Fx::new();
    let got = run(1 << 0, &mut fx);
    assert_eq!(
        (got.rc, got.outcome.kind, got.outcome.committed_epoch),
        (0, OUTCOME_COMMITTED, 2)
    );
    assert_eq!(
        fx.names(),
        ["stage", "ref", "replace"],
        "one stage (the displaced), the reference, one replace"
    );
    assert_eq!(fx.calls, [("stage", 0), ("ref", 0), ("replace", 1)]);
    assert_eq!(
        (fx.resident, fx.bytes),
        ([1, 0], [[pack(1); SLOT], [pack(0); SLOT]])
    );
    assert_eq!((got.applied.epoch, got.applied.outcome_count), (2, 1));
}

#[test]
fn a_refusal_runs_the_byte_path_in_the_same_swap() {
    let capture = Capture::start(Level::Debug);
    let mut fx = Fx::new();
    fx.ref_code = REFUSED;
    let got = run(1 << 0, &mut fx);
    assert_eq!(
        (got.rc, got.outcome.kind, got.applied.epoch),
        (0, OUTCOME_COMMITTED, 2)
    );
    assert_eq!(
        fx.calls,
        [
            ("stage", 0),
            ("ref", 0),
            ("stage", 1),
            ("replace", 0),
            ("replace", 1)
        ]
    );
    assert_eq!(
        (fx.resident, fx.bytes),
        ([1, 0], [[pack(1); SLOT], [pack(0); SLOT]])
    );
    assert!(
        capture.mine().is_empty(),
        "REFUSED from replace_from_fallback is not a failure: {:?}",
        capture.mine()
    );
}

#[test]
fn failed_timeout_and_an_unknown_code_follow_the_replace_rules() {
    let _capture = Capture::start(Level::Off);
    // (code, unknown?, kind, closed rungs, calls after the reference)
    let rows = [
        (FAILED, false, OUTCOME_RESTORED, 0u64, 1usize),
        (TIMEOUT, false, OUTCOME_QUARANTINED, 1, 0),
        (ACK, true, OUTCOME_QUARANTINED, 1, 0),
    ];
    for (code, unknown, kind, closed, after) in rows {
        let mut fx = Fx::new();
        (fx.ref_code, fx.unknown_code) = (code, unknown);
        let got = run(1 << 0, &mut fx);
        let row = format!("code {code} unknown {unknown}");
        assert_eq!(
            (got.rc, got.outcome.kind, got.applied.quarantined_rungs),
            (0, kind, closed),
            "{row}"
        );
        assert_eq!(got.outcome.initial.phase, PHASE_REPLACE, "{row}");
        assert_eq!(
            fx.calls.len(),
            2 + after,
            "{row}: stage, ref, then only a restore after FAILED"
        );
        assert!(
            fx.calls.iter().all(|call| call.1 == 0),
            "{row}: the fallback and the lower card are never called"
        );
        if kind == OUTCOME_RESTORED {
            assert_eq!(
                (fx.resident, fx.bytes[0]),
                ([0, 1], [pack(0); SLOT]),
                "{row}: the torn slot is rewritten"
            );
        }
    }
}
