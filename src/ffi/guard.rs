// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! The ABI edge guards shared by every exported function: panic containment with handle poisoning,
//! and the pointer and version checks.
//!
//! OWNER: this module owns how a call's outcome becomes a status. Exported functions own nothing of this.
//! The `struct_size` protocol (caller-filled and library-filled structs) is owned by `sized`.
//! CONTRACT: `status` for calls with no handle, `status_on` for calls on a handle (a caught panic
//! poisons it), `check_pointer` and `check_version` for the header every struct starts with.
//! SIZE: under 200 lines.
//! FAILURE: every error is logged once here via `diag`; a panic returns `SEABRAID_INTERNAL`.

use crate::diag::{self, fail, Code, Level};
use crate::ffi::lease::{gate, Slot};
use crate::Result;
use std::panic::{catch_unwind, AssertUnwindSafe};

/// Refuses every call on a handle a panic has touched. WHY: the panic may have stopped halfway through a
/// state change, so continuing could act on a half-updated handle; only `close` stays allowed, to free it.
pub(crate) fn refuse_poisoned(poisoned: bool, what: &str) -> Result<()> {
    if poisoned {
        return Err(fail!(
            Code::Internal,
            "{} handle poisoned by earlier panic; close it",
            what
        ));
    }
    Ok(())
}

/// The single choke point where an ABI call's outcome is logged (once) and remembered for
/// `seabraid_last_error`; `entry` names the exported function. A panic must not cross the ABI.
pub(crate) fn status(entry: &'static str, call: impl FnOnce() -> Result<()>) -> i32 {
    match catch_unwind(AssertUnwindSafe(call)) {
        Ok(Ok(())) => 0,
        Ok(Err(error)) => {
            diag::report(entry, &error);
            error.code() as i32
        }
        Err(payload) => {
            diag::report_panic(entry, &*payload);
            Code::Internal as i32
        }
    }
}

/// `status` for a call on `handle`: a caught panic also poisons the handle. WHY this is the fallback and not the
/// rule: a call that holds a lease poisons as the lease drops (`gate::Entered`), before the busy flag is released, so
/// no racing caller can enter the half-updated handle unpoisoned; this write covers a panic outside any lease and is
/// an atomic store on the gate, so it never races a reader.
///
/// # Safety
/// `handle` is null or points to a live handle that the caller serializes.
pub(crate) unsafe fn status_on<T>(
    entry: &'static str,
    handle: *mut Slot<T>,
    call: impl FnOnce() -> Result<()>,
) -> i32 {
    let mut panicked = false;
    let code = status(entry, || {
        // WHY a flag set by a guard: `status` swallows the panic, and only this frame knows one happened.
        let guard = PanicFlag(&mut panicked);
        let result = call();
        guard.disarm();
        result
    });
    if panicked && !handle.is_null() && handle.is_aligned() {
        // Only the gate's atomics are touched; no reference to the value is formed.
        // SAFETY: `handle` was just checked non-null and aligned; `gate` forms only a shared `&Gate`, whose atomics
        // are all `poison` touches. A pointer that is not a live handle is no worse here than for the cookie read
        // every call already made (see HONEST LIMIT in gate.rs).
        unsafe { gate(handle) }.poison();
        diag::emit(
            Level::Error,
            Code::Internal as i32,
            &format!("{entry}: handle poisoned by panic; every later call except close will fail"),
        );
    }
    code
}

/// Sets its flag when dropped without `disarm`, which only happens while unwinding.
struct PanicFlag<'a>(&'a mut bool);

impl PanicFlag<'_> {
    fn disarm(self) {
        std::mem::forget(self);
    }
}

impl Drop for PanicFlag<'_> {
    fn drop(&mut self) {
        *self.0 = true;
    }
}

pub(crate) fn check_pointer<T>(pointer: *const T, name: &str) -> Result<()> {
    if pointer.is_null() {
        return Err(fail!(Code::Invalid, "{} is null", name));
    }
    if !pointer.is_aligned() {
        return Err(fail!(Code::Invalid, "{} is misaligned", name));
    }
    Ok(())
}

pub(crate) fn check_version(found: u32, expected: u32, what: &str) -> Result<()> {
    if found != expected {
        return Err(fail!(
            Code::Invalid,
            "{} abi_version {} != {}",
            what,
            found,
            expected
        ));
    }
    Ok(())
}
