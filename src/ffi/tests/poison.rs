// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! A panic inside a leased call poisons the handle before the lease is released: a second thread calling on the
//! handle gets BUSY_OR_STATE while the panic unwinds and INTERNAL afterwards, never a successful call on the
//! half-updated session.
//!
//! CHECKER for the lease's poison-before-release (`gate::Entered::drop`) and for `status_on`'s fallback. One
//! invariant ("no caller enters a session a panic left half-updated"), three tests:
//! - `..._while_a_panic_unwinds` is deterministic: it stops the panicking thread in the window after the lease is
//!   released and before `status_on` could poison, and makes the second thread call from exactly there (channels,
//!   no sleeps, no scheduling assumption). It also reads the `gate::observe` hook, which records on the dropping thread
//!   whether the poison bit was already set when the busy flag was released: THIS is what pins the order of the two
//!   stores inside `Entered::drop`, every run (swapping them fails it every time; no outside observer can stop a thread
//!   between two atomic stores, so no timing test can pin it).
//! - `a_call_that_does_not_panic_leaves...` pins the hook's other side (a normal drop records "not poisoned"), so the
//!   hook cannot be satisfied by always answering true.
//! - `..._under_a_racing_spinner` is a stress check that can only fail when the invariant is broken (it never asserts
//!   that a particular interleaving happened). It is NOT the checker of the store order: a swapped order was caught in
//!   1 of 30 debug runs and 0 of 60 release runs when measured, which is why the hook exists.

use super::v2::{open_v3, reference};
use super::*;
use crate::ffi::lease::session as lease_session;
use crate::ffi::lease::Lease;
use crate::gate::observe;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::time::{Duration, Instant};

const BUSY: i32 = Code::BusyOrState as i32;
const INTERNAL: i32 = Code::Internal as i32;
/// What a probe reports when the second thread never answered. WHY a code no call returns: a hang is a failure.
const NO_ANSWER: i32 = -1;

/// The session's gate, for the `observe` hook (which is keyed by gate: see `gate::observe`).
fn gate_of(session: *mut SessionHandle) -> *const crate::gate::Gate {
    ptr::from_ref(unsafe { crate::ffi::lease::gate(session) })
}

fn status_code(session: *mut SessionHandle) -> i32 {
    let mut out = SessionStatus {
        abi_version: 1,
        struct_size: core::mem::size_of::<SessionStatus>() as u32,
        ..SessionStatus::default()
    };
    unsafe { seabraid_session_status(session, &mut out) }
}

/// A second thread that makes one `status` call on the handle each time it is asked, and answers with the code.
/// WHY a thread that only acts on request: the test, not the scheduler, decides at which point of the panic the
/// call is made.
struct Prober {
    ask: Option<Sender<()>>,
    answer: Receiver<i32>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Prober {
    fn start(handle: usize) -> Self {
        let (ask, asked) = channel::<()>();
        let (reply, answer) = channel::<i32>();
        let thread = std::thread::spawn(move || {
            // Ends when `ask` is dropped (recv fails) or the test is gone (send fails).
            while asked.recv().is_ok() {
                if reply
                    .send(status_code(handle as *mut SessionHandle))
                    .is_err()
                {
                    return;
                }
            }
        });
        Self {
            ask: Some(ask),
            answer,
            thread: Some(thread),
        }
    }

    /// Makes the second thread call now and waits for its status.
    fn call(&self) -> i32 {
        let sent = self.ask.as_ref().is_some_and(|ask| ask.send(()).is_ok());
        if !sent {
            return NO_ANSWER;
        }
        self.answer
            .recv_timeout(Duration::from_secs(10))
            .unwrap_or(NO_ANSWER)
    }
}

impl Drop for Prober {
    fn drop(&mut self) {
        self.ask = None;
        if let Some(thread) = self.thread.take() {
            let joined = thread.join();
            assert!(
                joined.is_ok() || std::thread::panicking(),
                "the prober thread panicked"
            );
        }
    }
}

/// Runs the second thread's call from its `Drop`. WHY a guard declared before the lease: locals drop in reverse
/// order, so during the unwind this runs after the lease drop has released the busy flag and before `status_on`
/// (the fallback poison write) gets control back, which is the one window where an unpoisoned session is enterable.
/// It records instead of asserting: a panic inside a drop during unwinding aborts the process.
struct CallAfterRelease<'a> {
    prober: &'a Prober,
    seen: &'a AtomicI32,
}

impl Drop for CallAfterRelease<'_> {
    fn drop(&mut self) {
        self.seen.store(self.prober.call(), Ordering::SeqCst);
    }
}

#[test]
fn a_second_caller_never_enters_a_session_a_panic_left_half_updated_while_a_panic_unwinds() {
    let _capture = Capture::start(Level::Off);
    let (rc, session) = open_v3(&reference());
    assert_eq!(rc, 0);
    let prober = Prober::start(session as usize);
    let during = AtomicI32::new(NO_ANSWER);
    let after_release = AtomicI32::new(NO_ANSWER);
    observe::clear();

    let rc = unsafe {
        status_on("seabraid_test_panic", session, || -> Result<()> {
            let _after = CallAfterRelease {
                prober: &prober,
                seen: &after_release,
            };
            let _lease = lease_session(session)?;
            // Holding the lease: the second thread must be refused as busy.
            during.store(prober.call(), Ordering::SeqCst);
            panic!("injected panic while leased")
        })
    };

    assert_eq!(rc, INTERNAL, "the panicking call reports INTERNAL");
    assert_eq!(
        observe::poisoned_at_last_leave(gate_of(session)),
        Some(true),
        "the lease drop must set the poison bit BEFORE it releases the busy flag (None: no lease was released)"
    );
    assert_eq!(
        during.load(Ordering::SeqCst),
        BUSY,
        "a call made while the lease is held is refused as busy"
    );
    assert_eq!(
        after_release.load(Ordering::SeqCst),
        INTERNAL,
        "a call made after the lease was released and before status_on ran must already see the poison"
    );
    assert_eq!(
        prober.call(),
        INTERNAL,
        "every later call is refused as poisoned"
    );
    drop(prober);
    assert_eq!(
        unsafe { seabraid_close(session) },
        0,
        "close frees a poisoned session"
    );
}

#[test]
fn a_call_that_does_not_panic_leaves_the_session_unpoisoned_when_it_releases_the_flag() {
    let _capture = Capture::start(Level::Off);
    let (rc, session) = open_v3(&reference());
    assert_eq!(rc, 0);
    observe::clear();
    assert_eq!(status_code(session), 0);
    assert_eq!(
        observe::poisoned_at_last_leave(gate_of(session)),
        Some(false),
        "a normal release must not poison (None: the hook did not run)"
    );
    assert_eq!(unsafe { seabraid_close(session) }, 0);
}

/// How long a round may wait for a lease the spinner's own call holds for microseconds. WHY a deadline: a busy flag
/// that never clears used to hang `cargo test` forever instead of failing it.
const LEASE_RETRY_LIMIT: Duration = Duration::from_secs(10);

/// Thread B of a round: spins on `status` until the handle reports poisoned and says whether it ever got a success
/// after A had taken the lease (it must not: A holds it, then the poison is set before it is released). Ends at once
/// when `stop` is set (A gave up), and panics if the handle is never reported poisoned within its own deadline.
fn spin(handle: usize, holding: &AtomicBool, stop: &AtomicBool) -> bool {
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut entered_unpoisoned = false;
    while !stop.load(Ordering::SeqCst) && Instant::now() < deadline {
        // WHY read before the call: A sets the flag only while it holds the lease, so a success after
        // seeing it can only be a call that entered once A had released it.
        let a_holds = holding.load(Ordering::SeqCst);
        let rc = status_code(handle as *mut SessionHandle);
        if rc == INTERNAL {
            return entered_unpoisoned;
        }
        entered_unpoisoned |= a_holds && rc == 0;
    }
    assert!(
        stop.load(Ordering::SeqCst),
        "the handle was never reported poisoned"
    );
    entered_unpoisoned
}

/// A lease on `session`, retried while the spinner's own call holds it, until `LEASE_RETRY_LIMIT`. WHY a retry: that call
/// holds the lease for a moment, and losing the race says nothing about poisoning; A must hold the lease before it
/// panics whatever the scheduling. Any other refusal, or the deadline, is returned as the error.
fn lease_with_deadline<'a>(session: *mut SessionHandle) -> Result<Lease<'a, Session>> {
    let retry_until = Instant::now() + LEASE_RETRY_LIMIT;
    loop {
        // SAFETY: `session` is the live handle this round opened; the round is the only caller besides the spinner.
        match unsafe { lease_session(session) } {
            Err(error) if error.code() == Code::BusyOrState && Instant::now() < retry_until => {
                std::thread::yield_now();
            }
            other => return other,
        }
    }
}

/// One round: thread A panics inside a leased call while thread B spins (see `spin`).
fn racing_round() -> bool {
    let (rc, session) = open_v3(&reference());
    assert_eq!(rc, 0);
    let handle = session as usize;
    let holding = Arc::new(AtomicBool::new(false));
    let stop = Arc::new(AtomicBool::new(false)); // set when A gave up, so the spinner ends at once, not at its deadline
    let spinner = {
        let (holding, stop) = (Arc::clone(&holding), Arc::clone(&stop));
        std::thread::spawn(move || spin(handle, &holding, &stop))
    };
    let gave_up = AtomicBool::new(false);
    let rc = unsafe {
        status_on("seabraid_test_panic", session, || -> Result<()> {
            let _lease = lease_with_deadline(session)
                .inspect_err(|_| gave_up.store(true, Ordering::SeqCst))?;
            holding.store(true, Ordering::SeqCst);
            panic!("injected panic while leased")
        })
    };
    if gave_up.load(Ordering::SeqCst) {
        stop.store(true, Ordering::SeqCst);
        spinner.join().expect("the spinner thread panicked");
        // The busy flag is stuck, so close is not attempted (it would be refused as busy too); the round fails here.
        panic!("the round got no lease within {LEASE_RETRY_LIMIT:?} (status {rc}): the busy flag did not clear");
    }
    assert_eq!(rc, INTERNAL);
    let entered_unpoisoned = spinner.join().unwrap();
    assert_eq!(
        unsafe { seabraid_close(session) },
        0,
        "close frees a poisoned session"
    );
    entered_unpoisoned
}

#[test]
fn a_racing_caller_never_enters_a_session_a_panic_left_half_updated_under_a_racing_spinner() {
    let _capture = Capture::start(Level::Off);
    let entered = (0..40).filter(|_| racing_round()).count();
    assert_eq!(
        entered, 0,
        "rounds in which a second thread entered an unpoisoned session"
    );
}
