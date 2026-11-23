// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! Handle misuse that is detectable: a dead or foreign pointer (the state a second close sees), a call from
//! inside the provider callback, and two threads on one session. Each refusal is a status, never a crash.
//! The residual undefined behavior (reading a freed or unrelated allocation) is documented in gate.rs.

use super::v2::{batch, open_v3, provider, submit, ExecuteFn};
use super::*;
use crate::ffi::batch_v2::BatchV2;
use crate::ffi::v2::seabraid_session_submit;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

const DEAD: &str = "session handle is not live (already closed, or not a session)";

fn status_code(session: *mut SessionHandle) -> i32 {
    let mut out = SessionStatus {
        abi_version: 1,
        struct_size: core::mem::size_of::<SessionStatus>() as u32,
        ..SessionStatus::default()
    };
    unsafe { seabraid_session_status(session, &mut out) }
}

fn is_busy_refusal() -> bool {
    diag::last().contains("session is busy")
}

/// The text a call made on the provider worker thread is refused with (see fi::lease::refuse_on_worker).
pub(super) fn is_worker_refusal() -> bool {
    diag::last().contains("session call refused") && diag::last().contains("provider callback")
}

/// Collects `ticket`, retrying while the gate itself (not the ticket) refuses: the other thread or the callback
/// may be inside a short call at that instant.
fn collect_through_contention(session: *mut SessionHandle, ticket: u64) -> i32 {
    let mut out = [0.0f32; 4];
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let rc = unsafe { seabraid_collect(session, ticket, 5000, out.as_mut_ptr(), 4) };
        if !(rc == Code::BusyOrState as i32 && is_busy_refusal() && Instant::now() < deadline) {
            return rc;
        }
        std::thread::yield_now();
    }
}

#[test]
fn a_dead_or_foreign_handle_is_invalid_not_a_crash() {
    let _capture = Capture::start(Level::Off);
    // A zeroed block the size of a session: what a handle looks like once its cookie is gone, and what a
    // pointer to unrelated zeroed memory looks like. (Reading a really freed block is the documented residual UB,
    // so the test builds the state instead of freeing and reading.)
    let mut block = vec![0u64; core::mem::size_of::<SessionHandle>().div_ceil(8)];
    assert!(core::mem::align_of::<SessionHandle>() <= 8);
    let foreign = block.as_mut_ptr().cast::<SessionHandle>();
    expect(
        "seabraid_session_status",
        status_code(foreign),
        Code::Invalid,
        DEAD,
    );
    expect(
        "seabraid_close",
        unsafe { seabraid_close(foreign) },
        Code::Invalid,
        DEAD,
    );
    expect(
        "seabraid_close_timeout",
        unsafe { seabraid_close_timeout(foreign, 5) },
        Code::Invalid,
        DEAD,
    );
    expect(
        "seabraid_prepare",
        unsafe { seabraid_prepare(foreign, &config()) },
        Code::Invalid,
        DEAD,
    );
    let (rc, _) = submit(foreign, &batch());
    expect("seabraid_session_submit", rc, Code::Invalid, DEAD);
}

#[test]
fn a_second_close_of_the_same_handle_is_refused() {
    let _capture = Capture::start(Level::Off);
    let session = open_session(ptr::null());
    // Everything `seabraid_close` does to the cookie, without freeing, so the second call reads live memory.
    let first = unsafe { crate::ffi::lease::retire(session) }.expect("first close retires");
    expect(
        "seabraid_close",
        unsafe { seabraid_close(session) },
        Code::Invalid,
        DEAD,
    );
    expect(
        "seabraid_session_status",
        status_code(session),
        Code::Invalid,
        DEAD,
    );
    assert_eq!(
        first.close_with(Session::close).ok(),
        Some(()),
        "the owner still closes cleanly"
    );
}

#[repr(C)]
struct Probe {
    handle: AtomicUsize,
    inside: AtomicI32,
    busy_seen: AtomicBool,
}

/// Runs on the worker thread: calls back into its own session exactly once and records what it was told. WHY no
/// retry loop: the refusal must not depend on whether another call happens to be inside the session at that instant.
unsafe extern "C" fn reenter(
    context: *mut c_void,
    _: *const Config,
    _: *const BatchV2,
    output: *mut f32,
    count: usize,
) -> i32 {
    let probe = unsafe { &*context.cast::<Probe>() };
    let handle = probe.handle.load(Ordering::SeqCst) as *mut SessionHandle;
    let rc = status_code(handle);
    probe.inside.store(rc, Ordering::SeqCst);
    if rc == Code::BusyOrState as i32 && is_worker_refusal() {
        probe.busy_seen.store(true, Ordering::SeqCst);
    }
    unsafe { ptr::write_bytes(output, 0, count) };
    0
}

#[test]
fn a_provider_callback_calling_into_its_own_session_gets_busy_or_state() {
    let _capture = Capture::start(Level::Off);
    let probe = Probe {
        handle: AtomicUsize::new(0),
        inside: AtomicI32::new(-1),
        busy_seen: AtomicBool::new(false),
    };
    let mut descriptor = provider(0, &super::v2::CONFIG, Some(reenter as ExecuteFn));
    descriptor.context = ptr::from_ref(&probe).cast_mut().cast();
    let (rc, session) = open_v3(&descriptor);
    assert_eq!(rc, 0);
    probe.handle.store(session as usize, Ordering::SeqCst);
    let (rc, ticket) = submit(session, &batch());
    assert_eq!(rc, 0);
    // The caller's collect only waits for the completion; the callback's refusal does not depend on it.
    let mut out = [0.0f32; 4];
    assert_eq!(
        unsafe { seabraid_collect(session, ticket, 5000, out.as_mut_ptr(), 4) },
        0
    );
    assert!(
        probe.busy_seen.load(Ordering::SeqCst),
        "last status seen inside the callback: {}",
        probe.inside.load(Ordering::SeqCst)
    );
    assert_eq!(unsafe { seabraid_close(session) }, 0);
}
static RELEASE: AtomicBool = AtomicBool::new(false);

unsafe extern "C" fn held_until_released(
    _: *mut c_void,
    _: *const Config,
    _: *const BatchV2,
    output: *mut f32,
    count: usize,
) -> i32 {
    while !RELEASE.load(Ordering::SeqCst) {
        std::thread::sleep(Duration::from_millis(1));
    }
    unsafe { ptr::write_bytes(output, 0, count) };
    0
}

#[test]
fn a_submit_from_a_second_thread_is_refused_while_the_first_is_inside_the_session() {
    let _capture = Capture::start(Level::Off);
    RELEASE.store(false, Ordering::SeqCst);
    let descriptor = provider(
        0,
        &super::v2::CONFIG,
        Some(held_until_released as ExecuteFn),
    );
    let (rc, session) = open_v3(&descriptor);
    assert_eq!(rc, 0);
    let (rc, ticket) = submit(session, &batch());
    assert_eq!(rc, 0);
    let handle = session as usize;
    // Thread A sits in collect (the provider is held), which keeps the session leased.
    let waiter = std::thread::spawn(move || {
        collect_through_contention(handle as *mut SessionHandle, ticket)
    });
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut refused = false;
    while !refused && Instant::now() < deadline {
        let mut second = 0u64;
        let rc = unsafe { seabraid_session_submit(session, &batch(), &mut second) };
        // A refusal for the outstanding ticket is also BUSY_OR_STATE; only the gate's own text counts.
        refused = rc == Code::BusyOrState as i32 && is_busy_refusal();
        assert_eq!(second, 0, "a refused submit leaves out_ticket untouched");
        std::thread::sleep(Duration::from_millis(1));
    }
    RELEASE.store(true, Ordering::SeqCst);
    assert_eq!(
        waiter.join().unwrap(),
        0,
        "the first thread's collect completes"
    );
    assert!(refused, "the second thread was never refused as busy");
    assert_eq!(unsafe { seabraid_close(session) }, 0);
}

#[test]
fn two_threads_hammering_one_session_only_ever_see_ok_or_busy() {
    let _capture = Capture::start(Level::Off);
    let session = open_session(ptr::null());
    let handle = session as usize;
    let hammer = move || {
        (0..2000)
            .map(|_| status_code(handle as *mut SessionHandle))
            .collect::<Vec<_>>()
    };
    let (a, b) = (std::thread::spawn(hammer), std::thread::spawn(hammer));
    let codes: Vec<i32> = [a.join().unwrap(), b.join().unwrap()].concat();
    let busy = Code::BusyOrState as i32;
    assert!(codes.iter().all(|&rc| rc == 0 || rc == busy), "{codes:?}");
    assert!(codes.contains(&0));
    assert_eq!(
        status_code(session),
        0,
        "the gate is released after every call"
    );
    assert_eq!(unsafe { seabraid_close(session) }, 0);
}

static CLOSE_RC: AtomicI32 = AtomicI32::new(-1);
static CLOSE_TIMEOUT_RC: AtomicI32 = AtomicI32::new(-1);
static CLOSE_TEXT_OK: AtomicBool = AtomicBool::new(false);

/// Runs on the worker thread: tries to close its own session, both ways, and records what it was told.
unsafe extern "C" fn close_from_callback(
    context: *mut c_void,
    _: *const Config,
    _: *const BatchV2,
    output: *mut f32,
    count: usize,
) -> i32 {
    let probe = unsafe { &*context.cast::<Probe>() };
    let handle = probe.handle.load(Ordering::SeqCst) as *mut SessionHandle;
    let first = unsafe { seabraid_close(handle) };
    let first_text = diag::last();
    let second = unsafe { seabraid_close_timeout(handle, 5) };
    let second_text = diag::last();
    CLOSE_RC.store(first, Ordering::SeqCst);
    CLOSE_TIMEOUT_RC.store(second, Ordering::SeqCst);
    let named = |entry: &str, text: &str| {
        text.starts_with(&format!("{entry}: session close refused"))
            && text.contains("provider callback")
    };
    CLOSE_TEXT_OK.store(
        named("seabraid_close", &first_text) && named("seabraid_close_timeout", &second_text),
        Ordering::SeqCst,
    );
    unsafe { ptr::write_bytes(output, 0, count) };
    0
}

#[test]
fn closing_a_session_from_its_own_provider_callback_is_refused_not_a_deadlock() {
    let _capture = Capture::start(Level::Off);
    let probe = Probe {
        handle: AtomicUsize::new(0),
        inside: AtomicI32::new(-1),
        busy_seen: AtomicBool::new(false),
    };
    let mut descriptor = provider(
        0,
        &super::v2::CONFIG,
        Some(close_from_callback as ExecuteFn),
    );
    descriptor.context = ptr::from_ref(&probe).cast_mut().cast();
    let (rc, session) = open_v3(&descriptor);
    assert_eq!(rc, 0);
    probe.handle.store(session as usize, Ordering::SeqCst);
    let (rc, ticket) = submit(session, &batch());
    assert_eq!(rc, 0);
    assert_eq!(collect_through_contention(session, ticket), 0);
    let busy = Code::BusyOrState as i32;
    assert_eq!(CLOSE_RC.load(Ordering::SeqCst), busy, "close on the worker");
    assert_eq!(
        CLOSE_TIMEOUT_RC.load(Ordering::SeqCst),
        busy,
        "close_timeout on the worker"
    );
    assert!(
        CLOSE_TEXT_OK.load(Ordering::SeqCst),
        "the refusal names its reason"
    );
    // The refused closes changed nothing: the session is live and closes from another thread.
    assert_eq!(status_code(session), 0);
    assert_eq!(unsafe { seabraid_close(session) }, 0);
}
