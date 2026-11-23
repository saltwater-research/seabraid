// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! C ABI and provider bridge. Caller pointers are never stored with their batch.
//! FAILURE: fail-fast. Every exported call returns a `SEABRAID_*` status, logs one record at its choke point
//! (`guard::status`/`status_on`) and sets the thread's last error; a panic is caught and returns INTERNAL, poisoning
//! the handle it ran on.
//! UNSAFE: this module and its children are the only code in the crate that may use `unsafe` (the crate root denies
//! it); the one `allow` is the next line, so `grep -rn 'allow(unsafe_code)' src` names only files under `ffi`.
#![allow(unsafe_code)]

pub(crate) mod batch_v2;
mod batch_view;
mod guard;
pub(crate) mod lease;
mod logging;
pub(crate) mod placement;
pub(crate) mod provider_v3;
pub(crate) mod providers;
pub(crate) mod sized;
#[cfg(test)]
mod tests;
pub(crate) mod v2;

pub use self::batch_view::BatchView;
pub use self::lease::SessionHandle;
use self::lease::{retire, session};
use self::providers::{MultiProvider, Provider};
use crate::batch::{self, Config};
use crate::capabilities::{
    validate_origins, Capabilities, FORMAT_PARTIAL_F32, FORMAT_ROUTED_F32, OPERATION_ROUTED_MOE,
    ORIGINS,
};
use crate::diag::{fail, Code};
use crate::session::{Session, SessionStatus};
use crate::submission::Submission;
use crate::Result;
pub(crate) use guard::refuse_poisoned;
pub(crate) use guard::{check_pointer, check_version, status, status_on};
use sized::{base, out_size, write_out};
use std::ptr;
use std::slice;
use std::time::Duration;

/// The ABI generation every versioned struct is checked against. WHY here: the C ABI owns what the wire calls
/// version 1; core code never learns it, because the ffi checks `abi_version` before a struct reaches the core.
pub(crate) const ABI_VERSION: u32 = 1;

/// What `seabraid_query_capabilities` reports. WHY here and not in capabilities.rs: it embeds `ABI_VERSION`, and
/// the core must not import the ffi that owns it.
pub(crate) const CAPABILITIES: Capabilities = Capabilities {
    abi_version: ABI_VERSION,
    struct_size: core::mem::size_of::<Capabilities>() as u32,
    operation: OPERATION_ROUTED_MOE,
    input_format: FORMAT_ROUTED_F32,
    output_format: FORMAT_PARTIAL_F32,
    source_modalities: ORIGINS,
};

/// Layout epoch. Rule (one sentence, owned by docs/ABI.md "Layout epoch"): the epoch changes only for a
/// non-append change; appends ride on struct_size. A binding built against another epoch refuses to load.
pub(crate) const ABI_EPOCH: u32 = 1;

#[no_mangle]
pub extern "C" fn seabraid_abi_epoch() -> u32 {
    ABI_EPOCH
}

#[no_mangle]
pub unsafe extern "C" fn seabraid_query_capabilities(out: *mut Capabilities) -> i32 {
    status("seabraid_query_capabilities", || {
        // SAFETY: `out_size` checks `out` non-null and this ABI's version before it reads two `u32`s; the caller
        // promises `out` is readable and writable for its declared `struct_size`.
        let size = unsafe { out_size(out, "out", base::CAPABILITIES) }?;
        // SAFETY: `out_size` just proved the header and returned the caller's declared size; `write_out` writes at
        // most that many bytes.
        unsafe { write_out(out, CAPABILITIES, size) };
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn seabraid_session_status(
    handle: *mut SessionHandle,
    out: *mut SessionStatus,
) -> i32 {
    // SAFETY: `session` leases the handle (live, unpoisoned, not busy); `out_size` checks `out`, and `write_out`
    // writes at most the caller's declared size.
    unsafe {
        status_on("seabraid_session_status", handle, || {
            let session = session(handle)?;
            let size = out_size(out, "out", base::SESSION_STATUS)?;
            // struct_size in the caller's struct is set by write_out (sized.rs), the one owner of that rule.
            // The version header is the ABI's, so it is stamped here and not by the core.
            let status = SessionStatus {
                abi_version: ABI_VERSION,
                ..session.status()
            };
            write_out(out, status, size);
            Ok(())
        })
    }
}

unsafe fn open(
    entry: &'static str,
    out: *mut *mut SessionHandle,
    start: impl FnOnce() -> Result<Session>,
) -> i32 {
    status(entry, || {
        check_pointer(out, "out")?;
        // SAFETY: `check_pointer` above proved `out` non-null and aligned for a pointer write; the caller promises it
        // is writable.
        unsafe { out.write(ptr::null_mut()) };
        let session = start()?;
        // SAFETY: `out` is the pointer already written above, so it is still writable.
        unsafe { out.write(lease::open(session)) };
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn seabraid_open(
    provider: *const Provider,
    out: *mut *mut SessionHandle,
) -> i32 {
    // SAFETY: `Provider::load` checks the pointer for null and the version before reading (the caller promises a
    // readable descriptor), and `open` checks `out` before writing it.
    unsafe {
        open("seabraid_open", out, || {
            // WHY an error and not a fallback: a C caller who forgot the provider would otherwise run the
            // reference transform believing it is model math. The transform is opt-in via seabraid_open_reference.
            if provider.is_null() {
                return Err(fail!(
                    Code::Invalid,
                    "provider is null (use seabraid_open_reference for the built-in reference transform)"
                ));
            }
            let provider = Provider::load(provider)?;
            Session::new(1, false, move |_, config, batch, output| {
                provider.run(config, batch, output)
            })
        })
    }
}

/// Opens a session on the built-in reference transform (an API demonstration, not model math).
#[no_mangle]
pub unsafe extern "C" fn seabraid_open_reference(out: *mut *mut SessionHandle) -> i32 {
    // SAFETY: `open` writes `out` only after `check_pointer`; the reference transform reads no caller memory.
    unsafe {
        open("seabraid_open_reference", out, || {
            Session::new(1, false, |_, config, batch, output| {
                batch::reference(config, batch, output)
            })
        })
    }
}

#[no_mangle]
pub unsafe extern "C" fn seabraid_open_multi(
    provider: *const MultiProvider,
    out: *mut *mut SessionHandle,
) -> i32 {
    // SAFETY: `MultiProvider::load` checks the pointer and version before reading (the caller promises a readable
    // descriptor), and `open` checks `out` before writing it.
    unsafe {
        open("seabraid_open_multi", out, || {
            let provider = MultiProvider::load(provider)?;
            Session::with_features(
                provider.operation_count(),
                provider.features(),
                move |operation_id, config, batch, output| {
                    provider.run(operation_id, config, batch, output)
                },
            )
        })
    }
}

#[no_mangle]
pub unsafe extern "C" fn seabraid_prepare(
    handle: *mut SessionHandle,
    config: *const Config,
) -> i32 {
    // SAFETY: `session` leases the handle; `config` is checked non-null, only its first `u32` is read before the
    // version matches, then one `Config` is copied out; the caller promises a readable `seabraid_config_v1`.
    unsafe {
        status_on("seabraid_prepare", handle, || {
            let mut session = session(handle)?;
            check_pointer(config, "config")?;
            // Only the version prefix may be read before it matches this ABI generation.
            let found = config.cast::<u32>().read();
            check_version(found, ABI_VERSION, "config")?;
            session.prepare(config.read())
        })
    }
}

#[no_mangle]
pub unsafe extern "C" fn seabraid_submit(
    handle: *mut SessionHandle,
    input: *const BatchView,
    out: *mut u64,
) -> i32 {
    // SAFETY: A forward to the exported entry with the arguments unchanged; every check happens in
    // `seabraid_submit_operation`.
    unsafe { seabraid_submit_routed(handle, input, 0, out) }
}

#[no_mangle]
pub unsafe extern "C" fn seabraid_submit_routed(
    handle: *mut SessionHandle,
    input: *const BatchView,
    source_modalities: u32,
    out: *mut u64,
) -> i32 {
    // SAFETY: A forward to the exported entry with the arguments unchanged; every check happens in
    // `seabraid_submit_operation`.
    unsafe { seabraid_submit_operation(handle, 0, input, source_modalities, out) }
}

#[no_mangle]
pub unsafe extern "C" fn seabraid_submit_operation(
    handle: *mut SessionHandle,
    operation_id: u32,
    input: *const BatchView,
    source_modalities: u32,
    out: *mut u64,
) -> i32 {
    // SAFETY: The handle is leased first and `input` and `out` are checked non-null; only the version word is read
    // before it matches, then `&*input` is formed on the caller's `BatchView`, which the caller promises is readable
    // and unmodified for the call. `submit_view` forms slices only after the counts match the config, and `out` is
    // written once, after success.
    unsafe {
        status_on("seabraid_submit_operation", handle, || {
            // WHY the lease first, as in seabraid_session_submit: a refusal about the handle (not live,
            // busy, poisoned) outranks one about an argument on both entries, so a poisoned session answers
            // INTERNAL whatever else is wrong with the call, and the two entries cannot disagree about it.
            let mut session = session(handle)?;
            validate_origins(source_modalities)?;
            check_pointer(input, "batch")?;
            check_pointer(out, "out_ticket")?;
            let config = session.ready_config(operation_id)?;
            // WHY a raw read of the version word first: `&*input` asserts a whole, valid BatchView, and a
            // caller on another version (or a bad pointer) may not have one; only the prefix may be read until
            // it matches this ABI generation, as `seabraid_prepare` does.
            check_version(input.cast::<u32>().read(), ABI_VERSION, "batch")?;
            let view = &*input;
            let ticket = submit_view(&mut session, operation_id, config, view, source_modalities)?;
            out.write(ticket);
            Ok(())
        })
    }
}

// Counts and pointers come from the C caller; both are checked before any slice is formed.
unsafe fn submit_view(
    session: &mut Session,
    operation_id: u32,
    config: Config,
    input: &BatchView,
    source_modalities: u32,
) -> Result<u64> {
    let lengths = config.lengths(input.tokens)?;
    let counts = [
        input.activation_count,
        input.expert_id_count,
        input.weight_count,
    ];
    lengths.check(input.tokens, config, counts)?;
    check_pointer(input.activations, "batch.activations")?;
    check_pointer(input.expert_ids, "batch.expert_ids")?;
    check_pointer(input.weights, "batch.weights")?;
    // Validated lengths, with storage and aliasing guaranteed by the C caller for this call.
    // v1 is a v2 submission with F32 data and placement epoch 0 (see Submission), so it takes the one validation path.
    // SAFETY: `check_pointer` proved `input.activations` non-null and `lengths.check` proved the caller's count
    // equals `lengths.rows` (and that the byte extent is addressable); the caller promises that many readable,
    // unmodified `f32`s for the call.
    let activations = unsafe { slice::from_raw_parts(input.activations, lengths.rows) };
    session.submit(Submission {
        operation_id,
        tokens: input.tokens,
        format: FORMAT_ROUTED_F32,
        activations: f32_bytes(activations),
        // SAFETY: `check_pointer` proved `input.expert_ids` non-null and `lengths.check` proved the caller's count
        // equals `lengths.picks`; the caller promises that many readable, unmodified `i32`s for the call.
        expert_ids: unsafe { slice::from_raw_parts(input.expert_ids, lengths.picks) },
        // SAFETY: `check_pointer` proved `input.weights` non-null and `lengths.check` proved the caller's count
        // equals `lengths.picks`; the caller promises that many readable, unmodified `f32`s for the call.
        weights: unsafe { slice::from_raw_parts(input.weights, lengths.picks) },
        source_modalities,
        placement_epoch: 0,
    })
}

/// The same memory as bytes. WHY unsafe: a `u8` view of `f32`s is always valid (no padding, align 1).
fn f32_bytes(values: &[f32]) -> &[u8] {
    // SAFETY: A `u8` view of initialised `f32`s: same address, exact byte length, no alignment or validity
    // requirement, and the borrow of `values` keeps it alive.
    unsafe { slice::from_raw_parts(values.as_ptr().cast::<u8>(), core::mem::size_of_val(values)) }
}

#[no_mangle]
pub unsafe extern "C" fn seabraid_collect(
    handle: *mut SessionHandle,
    ticket: u64,
    timeout_ms: u32,
    output: *mut f32,
    output_count: usize,
) -> i32 {
    // SAFETY: `session` leases the handle; `output_count` was checked equal to the ticket's element count and
    // `check_pointer` proved `output` non-null, and the caller promises that many writable `f32`s; `result` is the
    // session's own storage, disjoint from the caller's.
    unsafe {
        status_on("seabraid_collect", handle, || {
            let mut session = session(handle)?;
            let expected = session.output_count(ticket)?;
            if expected != output_count {
                return Err(fail!(
                    Code::Invalid,
                    "output_count {} != tokens * hidden {}",
                    output_count,
                    expected
                ));
            }
            check_pointer(output, "output")?;
            let result = session.collect(ticket, Duration::from_millis(u64::from(timeout_ms)))?;
            // The internal result has disjoint storage from the caller's output.
            ptr::copy_nonoverlapping(result.as_ptr(), output, result.len());
            Ok(())
        })
    }
}

/// Drains and consumes the session, waiting as long as the provider callback takes.
#[no_mangle]
pub unsafe extern "C" fn seabraid_close(handle: *mut SessionHandle) -> i32 {
    status("seabraid_close", || {
        // Only close consumes the handle, and it works on a poisoned one so the handle can be freed. `retire`
        // refuses a dead or busy handle (the double-close case) before anything is freed.
        // Session::close drains the provider before its callback context is returned to the caller.
        // SAFETY: `retire` checks the pointer for null and alignment and the cookie before anything is freed; on `Ok`
        // this call owns the handle and consumes it exactly once.
        unsafe { retire(handle) }?.close_with(Session::close)
    })
}

/// Like `seabraid_close` but gives up on a stuck provider callback after `timeout_ms`: the worker thread is
/// detached (leaked, logged at ERROR) and `SEABRAID_TIMEOUT` is returned. The handle is consumed either way.
#[no_mangle]
pub unsafe extern "C" fn seabraid_close_timeout(
    handle: *mut SessionHandle,
    timeout_ms: u32,
) -> i32 {
    status("seabraid_close_timeout", || {
        let limit = Duration::from_millis(u64::from(timeout_ms));
        // SAFETY: `retire` checks the pointer for null and alignment and the cookie before anything is freed; on `Ok`
        // this call owns the handle and consumes it exactly once.
        unsafe { retire(handle) }?.close_with(|session| session.close_timeout(limit))
    })
}
