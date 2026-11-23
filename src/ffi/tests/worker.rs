// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! A call made from the session's own provider callback (the worker thread) is refused up front, whatever the
//! caller thread is doing: no waiting for the caller's timeout, no retry loop. Deterministic: the caller thread
//! stays outside every call until the callback has finished, so the busy flag cannot be what refuses it.

use super::handle::is_worker_refusal;
use super::v2::{batch, open_v3, provider, submit, ExecuteFn};
use super::*;
use crate::ffi::batch_v2::BatchV2;
use crate::ffi::v2::seabraid_session_submit;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

struct Calls {
    handle: AtomicUsize,
    ticket: AtomicU64,
    ticket_known: AtomicBool,
    done: AtomicBool,
    collect: AtomicI32,
    submit: AtomicI32,
    status: AtomicI32,
    texts_ok: AtomicBool,
}

/// Waits for `flag`; a handoff between two threads, not a retry of a session call.
fn wait_for(flag: &AtomicBool) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !flag.load(Ordering::SeqCst) && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// Runs on the worker thread: collect (the call that used to wait for a completion only this thread can send),
/// submit and status on its own session, once each.
unsafe extern "C" fn call_everything(
    context: *mut c_void,
    _: *const Config,
    _: *const BatchV2,
    output: *mut f32,
    count: usize,
) -> i32 {
    let calls = unsafe { &*context.cast::<Calls>() };
    wait_for(&calls.ticket_known);
    let handle = calls.handle.load(Ordering::SeqCst) as *mut SessionHandle;
    let mut out = [0.0f32; 4];
    let collect = unsafe {
        seabraid_collect(
            handle,
            calls.ticket.load(Ordering::SeqCst),
            200,
            out.as_mut_ptr(),
            4,
        )
    };
    let mut ok = is_worker_refusal();
    let mut ticket = 0u64;
    let submit = unsafe { seabraid_session_submit(handle, &batch(), &mut ticket) };
    ok &= is_worker_refusal();
    let mut report = SessionStatus {
        abi_version: 1,
        struct_size: core::mem::size_of::<SessionStatus>() as u32,
        ..SessionStatus::default()
    };
    let status = unsafe { seabraid_session_status(handle, &mut report) };
    ok &= is_worker_refusal();
    calls.collect.store(collect, Ordering::SeqCst);
    calls.submit.store(submit, Ordering::SeqCst);
    calls.status.store(status, Ordering::SeqCst);
    calls.texts_ok.store(ok, Ordering::SeqCst);
    calls.done.store(true, Ordering::SeqCst);
    unsafe { ptr::write_bytes(output, 0, count) };
    0
}

#[test]
fn collect_submit_and_status_from_the_provider_callback_are_refused_while_the_caller_is_idle() {
    let _capture = Capture::start(Level::Off);
    let calls = Calls {
        handle: AtomicUsize::new(0),
        ticket: AtomicU64::new(0),
        ticket_known: AtomicBool::new(false),
        done: AtomicBool::new(false),
        collect: AtomicI32::new(-1),
        submit: AtomicI32::new(-1),
        status: AtomicI32::new(-1),
        texts_ok: AtomicBool::new(false),
    };
    let mut descriptor = provider(0, &super::v2::CONFIG, Some(call_everything as ExecuteFn));
    descriptor.context = ptr::from_ref(&calls).cast_mut().cast();
    let (rc, session) = open_v3(&descriptor);
    assert_eq!(rc, 0);
    calls.handle.store(session as usize, Ordering::SeqCst);
    let (rc, ticket) = submit(session, &batch());
    assert_eq!(rc, 0);
    calls.ticket.store(ticket, Ordering::SeqCst);
    calls.ticket_known.store(true, Ordering::SeqCst);
    // This thread is inside no session call now, so only the worker-thread rule can refuse the callback.
    wait_for(&calls.done);
    assert!(
        calls.done.load(Ordering::SeqCst),
        "the callback never finished"
    );
    let busy = Code::BusyOrState as i32;
    let seen = |what: &AtomicI32| what.load(Ordering::SeqCst);
    assert_eq!(seen(&calls.collect), busy, "collect from the callback");
    assert_eq!(seen(&calls.submit), busy, "submit from the callback");
    assert_eq!(seen(&calls.status), busy, "status from the callback");
    assert!(
        calls.texts_ok.load(Ordering::SeqCst),
        "each refusal names its reason"
    );
    // The refusals changed nothing: the caller still collects the batch the callback served.
    let mut out = [0.0f32; 4];
    let rc = unsafe { seabraid_collect(session, ticket, 5000, out.as_mut_ptr(), 4) };
    assert_eq!(rc, 0);
    assert_eq!(unsafe { seabraid_close(session) }, 0);
}
