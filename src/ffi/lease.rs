// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! How an exported call gets at a handle (a session or a placement): the handle's memory layout, the one accessor,
//! and the one way a handle is retired and freed.
//!
//! OWNER: this module is the only place that allocates a handle, turns a caller's handle pointer into a `&mut T`,
//! and frees one; it owns the handle layout (`Slot`). What "live", "busy" and "poisoned" mean is owned by
//! `crate::gate::Gate`. WHY generic over `Handle`: sessions and placements share one detector instead of each copying
//! it, so the two cannot drift apart.
//! CONTRACT: `open` makes a handle; `acquire` (or `session`) for every call except close; `retire`, then
//! `Retired::close_with` or a drop, for the `*_close*` calls; `gate` for the panic fallback in `ffi::guard`.
//! SIZE: under 300 lines.
//! FAILURE: fail-fast, each refusal logged once by `status_on`/`status`: INVALID for a pointer that is not a live
//! handle, BUSY_OR_STATE for a handle another call (or the provider callback) is inside, INTERNAL for a poisoned one.
//! CHECKER: src/ffi/tests/handle.rs (sessions), src/ffi/placement/tests/handle.rs (placements), and the CI Miri job
//! under Stacked Borrows and Tree Borrows (the aliasing rules below are what it checks).
//!
//! WHY a `Slot` (the gate, then the value in an `UnsafeCell`) and not a gate field inside the value: a call
//! holds `&mut` to the value for its whole length while another thread's refused call reads the gate, so the two
//! must not share bytes; side by side, each reference covers only its own. The rules every function here keeps:
//! (1) the only `&mut` ever formed covers `Slot::body`, never the gate; (2) the gate is reached only as `&Gate`
//! (shared, atomics); (3) no reference to a whole `Slot` is ever formed, so neither can cover the other's bytes;
//! (4) close frees the slot last, after the value's own close returned, and touches it no more afterwards.

use super::{check_pointer, refuse_poisoned};
use crate::diag::{fail, Code};
use crate::gate::{Entered, Gate, SESSION_COOKIE};
use crate::session::Session;
use crate::Result;
use std::cell::UnsafeCell;
use std::mem::{self, ManuallyDrop};
use std::ops::{Deref, DerefMut};
use std::ptr::addr_of;
use std::thread::ThreadId;

/// A value an exported handle holds: a name for messages, its kind's cookie, and the thread its callbacks run on.
pub(crate) trait Handle: Sized {
    /// What the caller's pointer is called in a refusal, e.g. "session".
    const NAME: &'static str;
    /// The cookie a live handle of this kind holds (`gate::SESSION_COOKIE` or `PLACEMENT_COOKIE`).
    const COOKIE: u64;
    /// The provider worker thread, recorded in the gate at open (see `Gate::on_worker_thread`); None: no worker.
    fn worker_thread(&self) -> Option<ThreadId> {
        None
    }
}

impl Handle for Session {
    const NAME: &'static str = "session";
    const COOKIE: u64 = SESSION_COOKIE;
    fn worker_thread(&self) -> Option<ThreadId> {
        self.worker_id()
    }
}

/// What an exported handle pointer points at. `repr(C)` with the gate first: the cookie read before anything
/// proves the pointer is a handle is at offset 0 of every kind (see crate::gate).
#[repr(C)]
pub struct Slot<T> {
    gate: Gate,
    /// WHY an `UnsafeCell`: the value is changed through a pointer the caller shares (the handle), which is what
    /// interior mutability means. WHY `ManuallyDrop`: `Retired::close_with` moves the value out before the slot's
    /// memory is freed, so freeing the memory must not drop it a second time.
    body: UnsafeCell<ManuallyDrop<T>>,
}

impl<T> Slot<T> {
    /// Where the gate sits in a handle of this kind, for the layout asserts (0: see the type note).
    pub(crate) const GATE_OFFSET: usize = mem::offset_of!(Slot<T>, gate);
}

/// The memory a `seabraid_session*` points at.
pub type SessionHandle = Slot<Session>;

// The checker for the type note on the session side (the placement asserts the same beside its definition).
const _: () = assert!(SessionHandle::GATE_OFFSET == 0);

/// Boxes `value` behind a live gate of its kind and returns the pointer the caller will hold.
pub(crate) fn open<T: Handle>(value: T) -> *mut Slot<T> {
    let gate = Gate::new(T::COOKIE, value.worker_thread());
    let body = UnsafeCell::new(ManuallyDrop::new(value));
    Box::into_raw(Box::new(Slot { gate, body }))
}

/// The gate of a handle that passed the pointer check, as the one kind of reference anyone forms to it (rule 2).
///
/// # Safety
/// `handle` is aligned and non-null; see the HONEST LIMIT in gate.rs for what a garbage or freed pointer does.
pub(crate) unsafe fn gate<'a, T>(handle: *const Slot<T>) -> &'a Gate {
    // SAFETY: `gate`'s contract: `handle` is non-null and aligned. `addr_of!` projects to the `gate` field without
    // forming a reference to the whole slot, and the `&Gate` made here is the only kind of reference this module
    // forms to those bytes (rules 2 and 3 above), so it never overlaps the `&mut` to the value.
    unsafe { &*addr_of!((*handle).gate) }
}

/// The value's bytes alone, as a raw pointer (rule 1 and 3: no reference to the slot is formed on the way).
///
/// # Safety
/// As for `gate`.
unsafe fn body<T>(handle: *mut Slot<T>) -> *mut T {
    // `ManuallyDrop<T>` is `repr(transparent)`, so the cast names the same bytes.
    // SAFETY: `body`'s contract is `gate`'s: `handle` is non-null and aligned. `addr_of!` and `UnsafeCell::raw_get`
    // give the value's address without a reference to the slot, and the `ManuallyDrop<T>` to `T` cast names the same
    // bytes (`repr(transparent)`).
    unsafe { UnsafeCell::raw_get(addr_of!((*handle).body)).cast::<T>() }
}

/// Exclusive use of a handle's value for the length of one call; the call ends (the busy flag is released, after
/// poisoning when a panic unwinds) when this is dropped.
pub(crate) struct Lease<'a, T: Handle> {
    value: &'a mut T,
    /// Held for the lease's length and dropped with it: see `gate::Entered`.
    _call: Entered<'a>,
}

impl<T: Handle> Deref for Lease<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.value
    }
}

impl<T: Handle> DerefMut for Lease<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        self.value
    }
}

/// Refuses a call made on the provider worker thread, i.e. from inside the provider callback. WHY one helper for
/// `acquire` and `retire`: the callback runs on the thread the session's completion is sent from, so any call that
/// waits (collect, close) would wait for itself for the caller's whole timeout, and naming that cause beats the
/// generic busy text; the one rule must read the same for every call. `what` says which call ("call", "close").
fn refuse_on_worker<T: Handle>(gate: &Gate, what: &str) -> Result<()> {
    if gate.on_worker_thread() {
        return Err(fail!(
            Code::BusyOrState,
            "{} {} refused: it was called from the provider callback (the worker thread); call it from another thread",
            T::NAME,
            what
        ));
    }
    Ok(())
}

/// The refusal for a pointer whose cookie is not this kind's live one.
fn not_live<T: Handle>() -> crate::Error {
    fail!(
        Code::Invalid,
        "{0} handle is not live (already closed, or not a {0})",
        T::NAME
    )
}

/// Claims a live, unpoisoned handle for one call. WHY the order (pointer, live, busy, poisoned): the cheap
/// refusals that say "this is not a handle" come before the one that touches shared state, and a poisoned
/// handle is still leased so two threads cannot both observe it.
///
/// # Safety
/// `handle` is null or a pointer returned by an open call; the caller serializes its calls (this detects, it
/// does not make a violation safe).
pub(crate) unsafe fn acquire<'a, T: Handle>(handle: *mut Slot<T>) -> Result<Lease<'a, T>> {
    check_pointer(handle, T::NAME)?;
    // SAFETY: `check_pointer` above proved `handle` non-null and aligned, which is `gate`'s whole contract; only a
    // shared `&Gate` is formed (rule 2).
    let gate = unsafe { gate(handle) };
    if !gate.is_live(T::COOKIE) {
        return Err(not_live::<T>());
    }
    refuse_on_worker::<T>(gate, "call")?;
    let Some(call) = gate.enter() else {
        return Err(fail!(
            Code::BusyOrState,
            "{} is busy: another call, or the provider callback, is inside it",
            T::NAME
        ));
    };
    let poisoned = gate.is_poisoned();
    let lease = Lease {
        // SAFETY: the busy flag `call` holds makes this the only `&mut` to the value for the lease's length, and
        // `body` covers the value's bytes alone, never the gate (rules 1 and 3); `is_live` already refused a dead
        // or foreign pointer, so `handle` points at a slot of this kind.
        value: unsafe { &mut *body(handle) },
        _call: call,
    };
    // WHY after the lease exists: a poisoned handle is still leased so two threads cannot both observe it, and a
    // refusal here must release the flag through `Drop`.
    refuse_poisoned(poisoned, T::NAME)?;
    Ok(lease)
}

/// `acquire` for a session, the name its many call sites use.
///
/// # Safety
/// As for `acquire`.
pub(crate) unsafe fn session<'a>(handle: *mut SessionHandle) -> Result<Lease<'a, Session>> {
    // SAFETY: `session`'s contract is `acquire`'s, passed on unchanged; the checks all happen inside `acquire`.
    unsafe { acquire(handle) }
}

/// Ends a handle's life for `close`: refuses a dead or busy handle, then kills the cookie so a second close of the
/// same pointer is refused. Works on a poisoned handle so it can still be freed.
///
/// # Safety
/// As for `acquire`; on `Ok` the caller owns the handle through the returned value and must not use the pointer
/// again.
pub(crate) unsafe fn retire<T: Handle>(handle: *mut Slot<T>) -> Result<Retired<T>> {
    check_pointer(handle, T::NAME)?;
    // SAFETY: `check_pointer` above proved `handle` non-null and aligned, which is `gate`'s whole contract; only a
    // shared `&Gate` is formed (rule 2).
    let gate = unsafe { gate(handle) };
    if !gate.is_live(T::COOKIE) {
        return Err(not_live::<T>());
    }
    // WHY before `enter`: closing from the provider callback would join the worker thread from itself and never
    // return (see `refuse_on_worker`).
    refuse_on_worker::<T>(gate, "close")?;
    let Some(call) = gate.enter() else {
        return Err(fail!(
            Code::BusyOrState,
            "{} is busy: close was not performed while another call is inside it",
            T::NAME
        ));
    };
    if !call.retire(T::COOKIE) {
        return Err(fail!(
            Code::Invalid,
            "{} handle is not live (closed by another thread)",
            T::NAME
        ));
    }
    Ok(Retired { slot: handle })
}

/// A handle whose cookie is dead and whose busy flag stays held: its one owner closes and frees it through this.
pub(crate) struct Retired<T> {
    slot: *mut Slot<T>,
}

impl<T> Retired<T> {
    /// Moves the value out, runs `close` on it, and frees the handle's memory after `close` returns (also when it
    /// unwinds). WHY free last (rule 4): a racing call made while `close` runs (a worker join can take seconds) reads a
    /// dead cookie in memory that still exists and is refused as INVALID, instead of reading freed memory.
    pub(crate) fn close_with<R>(self, close: impl FnOnce(T) -> R) -> R {
        let slot = ManuallyDrop::new(self).slot;
        let _free = Free(slot);
        // SAFETY: the value is read out exactly once (`Retired::drop` is skipped above) and `Free` never drops it.
        let value = unsafe { body(slot).read() };
        close(value)
    }
}

impl<T> Drop for Retired<T> {
    /// Drops the value in place, then frees the memory: a placement close, or a retired handle nobody closed.
    fn drop(&mut self) {
        let _free = Free(self.slot);
        // SAFETY: the value is still in the slot (`close_with` skips this drop) and is dropped exactly once.
        unsafe { body(self.slot).drop_in_place() };
    }
}

/// Frees a slot's memory without dropping its value (the `ManuallyDrop`); on drop, so an unwinding close still frees.
struct Free<T>(*mut Slot<T>);

impl<T> Drop for Free<T> {
    fn drop(&mut self) {
        // SAFETY: the pointer came from `open`'s `Box::into_raw` and is freed exactly once, after every other use.
        drop(unsafe { Box::from_raw(self.0) });
    }
}
