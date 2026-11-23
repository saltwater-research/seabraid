// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! R6a H1 (a handle of the wrong kind), V1 (the version word of a v1 batch) and V2 (which refusal wins).
//!
//! CHECKER for the rule in crate::gate: the gate is the first thing in every handle, so the one read made before a
//! pointer is known to be a handle is at offset 0 and exists in a session and in a placement alike.

use super::v2::{batch, submit};
use super::*;
use crate::ffi::lease::Handle;
use crate::ffi::placement::{
    seabraid_placement_close, seabraid_placement_snapshot, types::Snapshot, Placement,
    PlacementHandle,
};
use crate::gate::{Gate, PLACEMENT_COOKIE, SESSION_COOKIE};

const NOT_A_SESSION: &str = "session handle is not live (already closed, or not a session)";
const NOT_A_PLACEMENT: &str = "placement handle is not live (already closed, or not a placement)";

/// The gate leads both handles and its cookie leads the gate (both are also const-asserted at their definitions):
/// what makes the offset-0 read possible.
#[test]
fn the_gate_is_the_first_thing_in_both_handles_and_the_kinds_have_different_cookies() {
    assert_eq!(SessionHandle::GATE_OFFSET, 0);
    assert_eq!(PlacementHandle::GATE_OFFSET, 0);
    assert_ne!(SESSION_COOKIE, PLACEMENT_COOKIE);
    assert_ne!(<Session as Handle>::COOKIE, <Placement as Handle>::COOKIE);
    let session = Gate::new(SESSION_COOKIE, None);
    assert!(session.is_live(SESSION_COOKIE) && !session.is_live(PLACEMENT_COOKIE));
    let call = session.enter().expect("an idle gate is entered");
    assert!(
        !call.retire(PLACEMENT_COOKIE),
        "a wrong-kind retire does nothing"
    );
    let call = session
        .enter()
        .expect("a refused retire ends the call it was made in");
    assert!(call.retire(SESSION_COOKIE));
    assert!(!session.is_live(SESSION_COOKIE));
    assert!(
        session.enter().is_none(),
        "a retired gate stays busy until it is freed"
    );
}

#[cfg(target_os = "linux")]
mod wrong_kind {
    use super::super::guard_page::EdgeBlock;
    use super::*;

    /// A block that holds only `cookie` in its first 8 bytes, at the very end of mapped memory. A real handle is
    /// hundreds of bytes; this one proves what is read before the cookie matched: nothing beyond offset 0.
    fn cookie_only(cookie: u64) -> EdgeBlock {
        let block = EdgeBlock::new(8);
        unsafe { block.start.cast::<u64>().write(cookie) };
        block
    }

    #[test]
    #[cfg_attr(
        miri,
        ignore = "EdgeBlock maps a guard page (mmap, mprotect), which Miri does not run"
    )]
    fn a_placement_pointer_given_to_a_session_call_is_refused_without_reading_past_its_cookie() {
        let _capture = Capture::start(Level::Off);
        let block = cookie_only(PLACEMENT_COOKIE);
        let handle = block.start.cast::<SessionHandle>();
        expect(
            "seabraid_session_status",
            status_code(handle),
            Code::Invalid,
            NOT_A_SESSION,
        );
        expect(
            "seabraid_close",
            unsafe { seabraid_close(handle) },
            Code::Invalid,
            NOT_A_SESSION,
        );
        expect(
            "seabraid_close_timeout",
            unsafe { seabraid_close_timeout(handle, 5) },
            Code::Invalid,
            NOT_A_SESSION,
        );
        let (rc, _) = submit(handle, &batch());
        expect("seabraid_session_submit", rc, Code::Invalid, NOT_A_SESSION);
    }

    #[test]
    #[cfg_attr(
        miri,
        ignore = "EdgeBlock maps a guard page (mmap, mprotect), which Miri does not run"
    )]
    fn a_session_pointer_given_to_a_placement_call_is_refused_without_reading_past_its_cookie() {
        let _capture = Capture::start(Level::Off);
        let block = cookie_only(SESSION_COOKIE);
        let handle = block.start.cast::<PlacementHandle>();
        let mut executors = [0u8; 2];
        let mut out = std::mem::MaybeUninit::<Snapshot>::zeroed();
        let rc = unsafe {
            seabraid_placement_snapshot(handle, executors.as_mut_ptr(), 2, out.as_mut_ptr())
        };
        expect(
            "seabraid_placement_snapshot",
            rc,
            Code::Invalid,
            NOT_A_PLACEMENT,
        );
        expect(
            "seabraid_placement_close",
            unsafe { seabraid_placement_close(handle) },
            Code::Invalid,
            NOT_A_PLACEMENT,
        );
    }

    /// V1: a v1 batch pointer whose memory ends after its version word: the version is read as a word first, so a
    /// caller on another version is refused without the whole struct being treated as present. (The old code formed
    /// `&BatchView` over the missing bytes first; no read of them happened, so only Miri sees the difference.)
    #[test]
    #[cfg_attr(
        miri,
        ignore = "EdgeBlock maps a guard page (mmap, mprotect), which Miri does not run"
    )]
    fn a_v1_batch_whose_memory_ends_after_the_version_word_is_refused_by_version() {
        let _capture = Capture::start(Level::Off);
        let session = open_session(ptr::null());
        let block = EdgeBlock::new(8);
        unsafe { block.start.cast::<u64>().write(7) }; // abi_version 7 (little endian), then 4 spare bytes
        let mut ticket = 0u64;
        let rc =
            unsafe { seabraid_submit_operation(session, 0, block.start.cast(), 0, &mut ticket) };
        expect(
            "seabraid_submit_operation",
            rc,
            Code::Invalid,
            "batch abi_version 7 != 1",
        );
        assert_eq!(ticket, 0);
        assert_eq!(unsafe { seabraid_close(session) }, 0);
    }
}

fn status_code(session: *mut SessionHandle) -> i32 {
    let mut out = SessionStatus {
        abi_version: 1,
        struct_size: core::mem::size_of::<SessionStatus>() as u32,
        ..SessionStatus::default()
    };
    unsafe { seabraid_session_status(session, &mut out) }
}

/// V2: a poisoned session answers INTERNAL on a v1 submit even when the call also has bad source modalities; the
/// v1 and v2 submit entries agree (the lease comes first on both).
#[test]
fn a_poisoned_session_reports_internal_before_bad_modalities_on_both_submit_entries() {
    let _capture = Capture::start(Level::Off);
    let session = open_session(ptr::null());
    let rc = unsafe {
        status_on("seabraid_test_panic", session, || -> Result<()> {
            panic!("injected panic")
        })
    };
    assert_eq!(rc, Code::Internal as i32);
    let view = view(1, [2, 2, 2]);
    let mut ticket = 0u64;
    let rc = unsafe { seabraid_submit_operation(session, 0, &view, 1 << 20, &mut ticket) };
    assert_eq!(
        rc,
        Code::Internal as i32,
        "v1: poisoned outranks the modality refusal"
    );
    assert!(diag::last().contains("poisoned"), "{}", diag::last());
    let mut bad_batch = batch();
    bad_batch.source_modalities = 1 << 20;
    let (rc, _) = submit(session, &bad_batch);
    assert_eq!(
        rc,
        Code::Internal as i32,
        "v2: the same call, the same answer"
    );
    assert_eq!(unsafe { seabraid_close(session) }, 0);
}
