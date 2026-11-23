// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! Placement handle misuse that is detectable (the session twin is src/ffi/tests/handle.rs): a dead or foreign
//! pointer (the state a second close sees), a call from inside the provider callback, and two threads on one
//! handle. Each refusal is a status, never a crash. The residual undefined behavior (reading a freed or unrelated
//! allocation) is documented in src/gate.rs and docs/ABI.md.

use super::edge::{plan, BUSY, INVALID};
use super::*;
use crate::diag::testing::Capture;
use crate::diag::Level;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::time::{Duration, Instant};

const DEAD: &str = "placement handle is not live (already closed, or not a placement)";

/// One call's name, status and the message it left (read at once: only the latest message is kept per thread).
type Call = (&'static str, i32, String);

fn refused(call: &Call, code: i32, text: &str) {
    assert_eq!(call.1, code, "{}", call.0);
    assert_eq!(call.2, format!("{}: {text}", call.0));
}

/// Every exported call that takes a handle, called on `target` with otherwise valid arguments.
fn every_call(target: *mut PlacementHandle) -> Vec<Call> {
    let heat = [1.0, 20.0];
    let mut ticket = blank_ticket();
    let mut snapshot = blank_snapshot();
    let mut executors = [0u8; 2];
    let owners = [0u8, 1];
    let mut applied = Applied {
        abi_version: ABI_VERSION,
        struct_size: mem::size_of::<Applied>() as u32,
        epoch: 0,
        quarantined_rungs: 0,
        outcome_count: 0,
    };
    let mut calls: Vec<Call> = Vec::new();
    let mut note = |entry: &'static str, rc: i32| calls.push((entry, rc, diag::last()));
    unsafe {
        note(
            "seabraid_placement_plan",
            seabraid_placement_plan(target, 1, heat.as_ptr(), 2, &mut ticket),
        );
        note(
            "seabraid_placement_apply",
            seabraid_placement_apply(target, 1, ptr::null(), ptr::null_mut(), 0, &mut applied),
        );
        note(
            "seabraid_placement_snapshot",
            seabraid_placement_snapshot(target, executors.as_mut_ptr(), 2, &mut snapshot),
        );
        note(
            "seabraid_placement_quarantine",
            seabraid_placement_quarantine(target, 1, 1),
        );
        note(
            "seabraid_placement_update",
            seabraid_placement_update(target, 1, UPDATE_ADOPT, owners.as_ptr(), 2),
        );
        note("seabraid_placement_close", seabraid_placement_close(target));
    }
    calls
}

#[test]
fn a_zeroed_or_foreign_handle_is_invalid_for_every_call_not_a_crash() {
    let _capture = Capture::start(Level::Off);
    // A zeroed block the size of a placement: what a handle looks like once its cookie is gone, and what a pointer
    // to unrelated zeroed memory looks like. (Reading a really freed block is the documented residual UB, so the
    // test builds the state instead of freeing and reading.)
    let mut block = vec![0u64; mem::size_of::<PlacementHandle>().div_ceil(8)];
    assert!(mem::align_of::<PlacementHandle>() <= 8);
    let foreign = block.as_mut_ptr().cast::<PlacementHandle>();
    for call in every_call(foreign) {
        refused(&call, INVALID, DEAD);
    }
}

#[test]
fn a_second_close_of_the_same_handle_is_invalid() {
    let _capture = Capture::start(Level::Off);
    let placement = TestPlacement::new();
    let raw = placement.0;
    // WHY forget: TestPlacement::drop would close a handle this test retires by hand, and a failed assertion
    // below must not turn into a double free.
    mem::forget(placement);
    // Everything `seabraid_placement_close` does to the cookie, without freeing, so the second call reads live
    // memory (a real double free is the documented residual UB).
    let first = unsafe { crate::ffi::lease::retire(raw) }.expect("first close retires");
    for call in every_call(raw) {
        refused(&call, INVALID, DEAD);
    }
    drop(first);
}

#[test]
fn a_null_handle_is_invalid_and_a_live_one_still_closes_once() {
    let _capture = Capture::start(Level::Off);
    let rc = unsafe { seabraid_placement_close(ptr::null_mut()) };
    refused(
        &("seabraid_placement_close", rc, diag::last()),
        INVALID,
        "placement is null",
    );
    let placement = TestPlacement::new();
    assert_eq!(plan(&placement, 1, &[1.0, 20.0]).0, 0);
    // TestPlacement::drop asserts the one real close returns OK.
}

static ENTERED: AtomicBool = AtomicBool::new(false);
static RELEASE: AtomicBool = AtomicBool::new(false);
/// The status the provider callback got when it called back into its own handle.
static REENTRANT_RC: AtomicI32 = AtomicI32::new(-1);

/// Stage callback that calls the handle's own `plan` (re-entry), then holds the apply until released.
unsafe extern "C" fn hold_and_reenter(
    context: *mut c_void,
    _: u8,
    _: u32,
    _: u32,
    bytes: *mut u8,
    size: usize,
    written: *mut usize,
) -> i32 {
    let handle = context.cast::<PlacementHandle>();
    let heat = [1.0, 20.0];
    let mut ticket = blank_ticket();
    let rc = unsafe { seabraid_placement_plan(handle, 1, heat.as_ptr(), 2, &mut ticket) };
    REENTRANT_RC.store(rc, Ordering::SeqCst);
    ENTERED.store(true, Ordering::SeqCst);
    let deadline = Instant::now() + Duration::from_secs(10);
    while !RELEASE.load(Ordering::SeqCst) && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(1));
    }
    unsafe {
        ptr::write_bytes(bytes, 7, size);
        *written = size;
    }
    0
}

#[test]
fn a_call_from_the_provider_callback_or_a_second_thread_is_busy_or_state() {
    let _capture = Capture::start(Level::Off);
    ENTERED.store(false, Ordering::SeqCst);
    RELEASE.store(false, Ordering::SeqCst);
    REENTRANT_RC.store(-1, Ordering::SeqCst);
    let placement = TestPlacement::new();
    let ticket = plan(&placement, 1, &[1.0, 20.0]).1.ticket;
    let slot = [4usize];
    let io = RawIo {
        abi_version: ABI_VERSION,
        context: placement.0.cast(),
        layer_slot_bytes: slot.as_ptr(),
        layer_count: 1,
        stage: Some(hold_and_reenter),
        replace: Some(replace_ok),
    };
    let handle = placement.0 as usize;
    let io_addr = ptr::from_ref(&io) as usize;
    // Thread A sits inside apply (the callback is held), which keeps the handle leased.
    let applier = std::thread::spawn(move || {
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
        unsafe {
            seabraid_placement_apply(
                handle as *mut PlacementHandle,
                ticket,
                io_addr as *const Provider,
                &mut outcome,
                1,
                &mut applied,
            )
        }
    });
    let deadline = Instant::now() + Duration::from_secs(10);
    while !ENTERED.load(Ordering::SeqCst) && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(ENTERED.load(Ordering::SeqCst), "the callback never ran");
    // The second thread (this one) is refused for every call, close included, while A is inside.
    for (entry, rc, message) in every_call(placement.0) {
        assert_eq!(rc, BUSY, "{entry}");
        assert!(
            message.starts_with(&format!("{entry}: placement is busy")),
            "{message}"
        );
    }
    RELEASE.store(true, Ordering::SeqCst);
    assert_eq!(applier.join().expect("applier thread"), 0);
    assert_eq!(
        REENTRANT_RC.load(Ordering::SeqCst),
        BUSY,
        "the callback calling into its own placement must be refused"
    );
    // The refusals changed nothing: the applied move is the only change, so epoch 2 is current.
    assert_eq!(plan(&placement, 2, &[1.0, 20.0]).0, 0);
}

#[test]
fn two_threads_planning_on_one_handle_only_ever_get_ok_or_busy_or_state() {
    let _capture = Capture::start(Level::Off);
    let placement = TestPlacement::new();
    let handle = placement.0 as usize;
    let worker = move || {
        let heat = [1.0, 20.0];
        (0..400)
            .map(|_| {
                let mut ticket = blank_ticket();
                unsafe {
                    seabraid_placement_plan(
                        handle as *mut PlacementHandle,
                        1,
                        heat.as_ptr(),
                        2,
                        &mut ticket,
                    )
                }
            })
            .collect::<Vec<i32>>()
    };
    let other = std::thread::spawn(worker);
    let mine = worker();
    let theirs = other.join().expect("second planner");
    for rc in mine.iter().chain(&theirs) {
        assert!(*rc == 0 || *rc == BUSY, "unexpected status {rc}");
    }
    assert!(
        mine.contains(&0) || theirs.contains(&0),
        "nobody ever got in"
    );
    // Whatever interleaving happened, the handle is intact and usable.
    assert_eq!(plan(&placement, 1, &[1.0, 20.0]).0, 0);
}
