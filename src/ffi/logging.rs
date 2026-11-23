// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! C entry points for the log sink, the per-thread last error and the library version.
//! All state lives in `crate::diag`; this file only translates pointers.

use super::{check_pointer, status};
use crate::diag::{self, fail, Code, Deliver, Level};
use crate::Result;
use std::ffi::{c_char, c_void, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use std::sync::Arc;

const VERSION: &str = concat!(env!("CARGO_PKG_VERSION"), "\0");

/// A caller's log sink: `(context, level, status, message)`, as `seabraid_set_log_sink` documents it.
pub(crate) type LogFn = unsafe extern "C" fn(*mut c_void, u32, i32, *const c_char);

/// WHY here and not in `diag`: calling a C function pointer is the one `unsafe` act of the logging path, and the
/// crate keeps every `unsafe` under `ffi/` so `diag` (which owns the sink's state) stays safe code. The message
/// loses interior NULs (a C string cannot hold them); the caller guarantees the sink and its context stay valid
/// until the sink is replaced.
fn wrap_sink(function: LogFn, context: usize) -> Deliver {
    Arc::new(move |level, status, message| {
        let bytes: Vec<u8> = message.bytes().filter(|byte| *byte != 0).collect();
        // Interior NULs were removed above, so this cannot fail.
        let text = CString::new(bytes).unwrap_or_default();
        // SAFETY: `seabraid_set_log_sink`'s contract (seabraid.h) is that `function` and `context` stay callable
        // until the sink is replaced, and `text` is a NUL-terminated buffer that lives past this call.
        unsafe { function(context as *mut c_void, level as u32, status, text.as_ptr()) }
    })
}

#[no_mangle]
pub unsafe extern "C" fn seabraid_set_log_sink(
    sink: Option<LogFn>,
    context: *mut c_void,
    min_level: u32,
) -> i32 {
    status("seabraid_set_log_sink", || {
        let min = Level::from_raw(min_level).ok_or_else(|| {
            fail!(
                Code::Invalid,
                "min_level {} is not a seabraid_log_level",
                min_level
            )
        })?;
        // WHY the context as an integer: the closure must be Send + Sync; the C caller owns the pointee.
        diag::set_sink(
            sink.map(|function| wrap_sink(function, context as usize)),
            min,
        );
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn seabraid_log(level: u32, status_code: i32, message: *const c_char) -> i32 {
    status("seabraid_log", || {
        // SEABRAID_LOG_OFF is a threshold, not a record level.
        let level = Level::from_raw(level)
            .filter(|level| *level != Level::Off)
            .ok_or_else(|| fail!(Code::Invalid, "level {} is not DEBUG..ERROR", level))?;
        check_pointer(message, "message")?;
        // Lossy on purpose: a record with bad UTF-8 is still worth keeping.
        // SAFETY: `check_pointer` above proved `message` non-null; the C caller promises (seabraid.h) it is a NUL-
        // terminated string readable for this call, and the borrow in `text` ends with this call.
        let text = unsafe { CStr::from_ptr(message) }.to_string_lossy();
        diag::emit(level, status_code, &text);
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn seabraid_last_error(
    buffer: *mut c_char,
    capacity: usize,
    needed: *mut usize,
) -> i32 {
    // WHY not `status`: that records every failure as this thread's last error, so a bad argument here would
    // overwrite the very message the caller is asking for. Its own argument errors are logged and returned only.
    // SAFETY: `copy_last` is unsafe only for its pointer arguments: it checks `needed` and `buffer` for null before
    // use, and the C caller promises `buffer` holds `capacity` writable bytes and `needed` is writable
    // (`seabraid_last_error` in seabraid.h).
    match catch_unwind(AssertUnwindSafe(|| unsafe {
        copy_last(buffer, capacity, needed)
    })) {
        Ok(Ok(())) => 0,
        Ok(Err(error)) => {
            diag::note("seabraid_last_error", &error);
            error.code() as i32
        }
        Err(payload) => {
            diag::report_panic("seabraid_last_error", &*payload);
            Code::Internal as i32
        }
    }
}

unsafe fn copy_last(buffer: *mut c_char, capacity: usize, needed: *mut usize) -> Result<()> {
    let text = diag::last();
    if !needed.is_null() {
        check_pointer(needed, "needed")?;
        // SAFETY: `check_pointer` on the line above proved `needed` non-null and aligned for a `usize`; the caller
        // promises it is writable.
        unsafe { needed.write(text.len() + 1) };
    }
    if capacity == 0 {
        return Ok(());
    }
    check_pointer(buffer, "buffer")?;
    let mut length = text.len().min(capacity - 1);
    while !text.is_char_boundary(length) {
        length -= 1; // never cut a multi-byte character in half
    }
    // The destination holds `capacity` bytes and length + 1 <= capacity.
    // SAFETY: `length + 1 <= capacity` by the `min` and the character-boundary walk above; `buffer` is non-null
    // (checked above) and the caller promises `capacity` writable bytes; `text` is this thread's own `String`, so the
    // ranges cannot overlap.
    unsafe {
        ptr::copy_nonoverlapping(text.as_ptr(), buffer.cast::<u8>(), length);
        buffer.add(length).write(0);
    }
    Ok(())
}

#[no_mangle]
pub extern "C" fn seabraid_version() -> *const c_char {
    VERSION.as_ptr().cast()
}
