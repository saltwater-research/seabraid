// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! The FFI handles' misuse detector (sessions and placements): a liveness cookie, a busy flag and a poison bit.
//!
//! OWNER: this module is the only place that knows what "this handle is live", "a call is in progress on it" and
//! "a panic left it half-updated" mean; the FFI accessor (`ffi::lease`) is its one caller. WHY at the crate root and
//! not under `ffi`: it is pure atomics and safe code, and everything under `ffi` is `allow(unsafe_code)`, so only
//! here does the crate's `deny(unsafe_code)` cover it.
//! CONTRACT: `Gate::new`, `is_live` and `enter` open every call on a handle; the `Entered` token `enter` returns is
//! the call in progress: dropping it ends the call (poisoning first if a panic is unwinding), and `Entered::retire`
//! ends the handle's life instead. `poison` is the fallback for a panic outside any call. Everything takes `&self`
//! (atomics), so a second thread can ask while the first is inside the handle.
//! SIZE: under 300 lines with the test hook and its tests. FAILURE: pure predicates; the caller turns a refusal into a status.
//! CHECKER: src/ffi/tests/handle.rs, src/ffi/tests/poison.rs, src/ffi/placement/tests/handle.rs, and the Miri job
//! (Stacked Borrows and Tree Borrows) in .github/workflows/ci.yml.
//!
//! HONEST LIMIT: this makes the common misuse (double close, a stale or foreign pointer, two overlapping
//! calls, a handle of the other kind) fail with a status. It cannot make it safe: reading the cookie of an already
//! freed or unrelated allocation is itself undefined behavior, and a freed block that was reused by another live
//! session passes.
//!
//! WHY the gate is a header in front of the value and never inside it (Miri found the aliasing violation): a call holds `&mut`
//! to the value for its whole length while another thread's refused call reads this gate, and a `&mut` whose bytes
//! include the gate makes that read an aliasing violation (Stacked Borrows) and a data race with the `&mut`'s retag
//! (Tree Borrows), atomics or not. `ffi::lease::Slot` puts the gate and the value side by side, so the only
//! references ever formed to the gate's bytes are shared ones, and the only `&mut` covers the value's bytes alone.
//!
//! WHY the gate is the FIRST thing in every handle, and of `Gate` its cookie: the one read a caller's
//! pointer gets before anything proves it is a handle is the cookie, at offset 0. Both handle kinds start with the
//! same `ffi::lease::Slot` header (const-asserted there), so a session pointer handed to a placement call, or the
//! reverse, reads a gate that exists in either allocation, finds the other kind's cookie, and is refused as INVALID.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread::{self, ThreadId};

/// What a live session's gate holds, and what a live placement's gate holds. Each is zeroed by `retire` before
/// the memory is freed. WHY a 64-bit pattern: freed memory, zeroed memory and most pointers will not equal it by
/// accident. WHY two: one per handle kind, so a pointer of the other kind is "not live" instead of passing as live.
pub(crate) const SESSION_COOKIE: u64 = 0x5EAB_4A1D_C0DE_F00D;
pub(crate) const PLACEMENT_COOKIE: u64 = 0x5EAB_91AC_E4E7_F00D;

/// `repr(C)` so the cookie is at offset 0 (see the module note); the other fields keep their order for no reason
/// but to read the same way.
#[repr(C)]
pub(crate) struct Gate {
    cookie: AtomicU64,
    busy: AtomicBool,
    /// Set after a panic unwound through a call. WHY an atomic in the gate and not a field of the value: it is
    /// written by the panicking call and read by the next caller's `acquire`, and the write must land before the
    /// busy flag is released (see `Entered::drop`), so it has to be touchable without `&mut` to the value.
    poisoned: AtomicBool,
    /// The provider worker's thread, fixed at open. WHY here: a close called on that thread would wait for the
    /// thread it is running on, and the gate is the one place that already decides "may this call proceed".
    worker: Option<ThreadId>,
}

impl Gate {
    /// A live gate for a handle of the kind `cookie` names (`SESSION_COOKIE` or `PLACEMENT_COOKIE`) whose provider
    /// callbacks run on `worker` (None: a handle with no worker thread).
    pub(crate) fn new(cookie: u64, worker: Option<ThreadId>) -> Self {
        Self {
            cookie: AtomicU64::new(cookie),
            busy: AtomicBool::new(false),
            poisoned: AtomicBool::new(false),
            worker,
        }
    }

    /// True when the caller is running on the recorded worker thread, i.e. inside a provider callback.
    pub(crate) fn on_worker_thread(&self) -> bool {
        self.worker == Some(thread::current().id())
    }

    /// True until `retire`, for a handle of the kind `expected` names.
    pub(crate) fn is_live(&self, expected: u64) -> bool {
        self.cookie.load(Ordering::Acquire) == expected
    }

    /// Claims the handle for one call; None when another call (another thread, or the provider callback
    /// calling back into its own session) already holds it.
    pub(crate) fn enter(&self) -> Option<Entered<'_>> {
        // WHY `then` and not `then_some`: an eagerly built token would be dropped on a refusal and release the flag
        // the other call holds.
        self.busy
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
            .then(|| Entered { gate: self })
    }

    /// Marks the handle unusable for every later call except close.
    pub(crate) fn poison(&self) {
        self.poisoned.store(true, Ordering::Release);
    }

    pub(crate) fn is_poisoned(&self) -> bool {
        self.poisoned.load(Ordering::Acquire)
    }
}

/// One call in progress on a gate: the busy flag is held until this is dropped, also while unwinding.
#[must_use = "dropping it at once ends the call it just began"]
pub(crate) struct Entered<'a> {
    gate: &'a Gate,
}

impl Entered<'_> {
    /// Marks the handle dead from inside a call; false (and the call ended) when it was not live as `expected`.
    /// WHY a compare-exchange: two racing closes cannot both pass. WHY the busy flag is never released on success:
    /// the handle is about to be freed, and a racing call that already saw it live must be refused as busy, not
    /// enter it.
    pub(crate) fn retire(self, expected: u64) -> bool {
        let retired = self
            .gate
            .cookie
            .compare_exchange(expected, 0, Ordering::AcqRel, Ordering::Relaxed)
            .is_ok();
        if retired {
            std::mem::forget(self);
        }
        retired
    }
}

impl Drop for Entered<'_> {
    /// Poisons first when a panic is unwinding, then releases the flag. WHY that order: once the flag drops another
    /// thread may enter, and a value a panic left half-updated must already read as poisoned by then. WHY stores on
    /// the fields and not calls to `&self` methods: once `busy` is false another thread may close and free the
    /// handle, and a `&Gate` argument must stay valid until its function returns (the rule `Arc`'s drop follows), so
    /// the last touch is the store through the `&AtomicBool` alone.
    fn drop(&mut self) {
        let gate = self.gate;
        if thread::panicking() {
            gate.poisoned.store(true, Ordering::Release);
        }
        // Test builds only: record whether the poison bit was already set at the instant the flag is released (see
        // `observe`); no non-test build contains this line.
        #[cfg(test)]
        observe::record(
            std::ptr::from_ref(gate) as usize,
            gate.poisoned.load(Ordering::Acquire),
        );
        // The Release store publishes everything the call wrote, including the poison bit, to the next `enter`.
        gate.busy.store(false, Ordering::Release);
    }
}

// The checker for the module note: the cookie is the first thing in the gate. (That the gate is the first thing in
// every handle is asserted beside `ffi::lease::Slot`.)
const _: () = assert!(core::mem::offset_of!(Gate, cookie) == 0);

/// Test-only window into the order of `Entered::drop`'s two writes (poison, then release the busy flag). WHY a hook
/// and not a stress test: no outside observer can stop a thread between two atomic stores, so the racing test caught a
/// swapped order in 1 of 30 debug runs and 0 of 60 release runs; this records the answer on the thread that ran the
/// drop, so the order is asserted every time. WHY a thread-local: the Rust test harness runs tests concurrently and
/// each one drops its lease on its own thread. WHY keyed by gate: a log callback another test installed runs on the
/// thread that emits, and may release a DIFFERENT session's lease there (seen: the unkeyed hook read `false` from it).
/// Compiled only under `cfg(test)`: a non-test build has no trace of it.
#[cfg(test)]
pub(crate) mod observe {
    use super::Gate;
    use std::cell::RefCell;

    /// WHY a bound: a thread that outlives many sessions (a test-harness worker running test after test) would otherwise
    /// keep one entry per gate address it ever released; the newest entries are the ones a test asks about.
    pub(super) const MAX_ENTRIES: usize = 64;

    thread_local! {
        /// (gate address, poison bit seen at that gate's latest release on this thread); at most `MAX_ENTRIES`.
        static AT_LEAVE: RefCell<Vec<(usize, bool)>> = const { RefCell::new(Vec::new()) };
    }

    /// `key` is the gate's address. WHY an address and not `&Gate`: see `Entered::drop`.
    pub(super) fn record(key: usize, poisoned: bool) {
        // WHY try_with and not with: a lease can be dropped from another thread-local's destructor, after this
        // one was destroyed; `with` would panic there (an abort inside a TLS destructor). There is then nothing to
        // record into and nobody left to ask, so the call is a no-op.
        let store = |seen: &RefCell<Vec<(usize, bool)>>| {
            let mut seen = seen.borrow_mut();
            match seen.iter_mut().find(|entry| entry.0 == key) {
                Some(entry) => entry.1 = poisoned,
                None => {
                    if seen.len() >= MAX_ENTRIES {
                        seen.remove(0); // the oldest
                    }
                    seen.push((key, poisoned));
                }
            }
        };
        let _ = AT_LEAVE.try_with(store); // allow-error-path: cfg(test) hook, no thread-local left to record into
    }

    /// How many gates this thread currently remembers (at most `MAX_ENTRIES`).
    pub(super) fn len() -> usize {
        AT_LEAVE.with(|seen| seen.borrow().len())
    }

    /// Forgets every release seen on this thread, so a later `None` means "no call on that gate ended since".
    pub(crate) fn clear() {
        AT_LEAVE.with(|seen| seen.borrow_mut().clear());
    }

    /// Whether the poison bit was set when the latest call on `gate` on this thread released the flag.
    pub(crate) fn poisoned_at_last_leave(gate: *const Gate) -> Option<bool> {
        let key = gate as usize;
        AT_LEAVE.with(|seen| {
            seen.borrow()
                .iter()
                .find(|entry| entry.0 == key)
                .map(|entry| entry.1)
        })
    }
}

#[cfg(test)]
mod observe_tests {
    use super::{observe, Gate, SESSION_COOKIE};
    use std::cell::RefCell;

    #[test]
    fn the_record_stays_bounded_over_many_gates() {
        observe::clear();
        let gates: Vec<Gate> = (0..observe::MAX_ENTRIES * 3)
            .map(|_| Gate::new(SESSION_COOKIE, None))
            .collect();
        for gate in &gates {
            drop(gate.enter());
        }
        let last = &gates[gates.len() - 1];
        assert_eq!(
            observe::poisoned_at_last_leave(last),
            Some(false),
            "the newest entry is kept"
        );
        assert_eq!(
            observe::poisoned_at_last_leave(&gates[0]),
            None,
            "the oldest entries are dropped"
        );
        assert!(
            observe::len() <= observe::MAX_ENTRIES,
            "{} entries kept",
            observe::len()
        );
    }

    /// A call ended from another thread-local's destructor, after `AT_LEAVE` was destroyed, must not panic (a panic
    /// in a TLS destructor aborts the whole test binary). Destructors run in reverse order of first use on glibc, so
    /// `Late` is registered BEFORE `AT_LEAVE` is first used and therefore dropped AFTER it.
    #[test]
    fn leave_during_thread_local_destruction_does_not_panic() {
        struct Late(Gate);
        impl Drop for Late {
            fn drop(&mut self) {
                drop(self.0.enter());
            }
        }
        thread_local! { static LATE: RefCell<Option<Late>> = const { RefCell::new(None) }; }
        std::thread::spawn(|| {
            LATE.with(|late| *late.borrow_mut() = Some(Late(Gate::new(SESSION_COOKIE, None)))); // registers LATE's destructor first
            observe::clear(); // first use of AT_LEAVE: registered second, destroyed first
        })
        .join()
        .expect("the thread ended without a panic");
    }
}
