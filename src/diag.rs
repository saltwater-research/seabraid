// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! Failure vocabulary of the runtime: error construction, log levels, the
//! process-wide sink and the per-thread last-error text.
//!
//! * OWNER: this module is the only place that builds an [`Error`], decides a log
//!   level, or stores the sink and last error. Nothing else keeps a copy.
//! * CONTRACT: `fail!(Code::X, "fmt {}", n)` builds an error; `report`/`note`/`emit`
//!   deliver it. Imports only `std`; it never names the C ABI, a session or a placement.
//!   Messages carry codes, counts, ids and sizes, never paths, env values, host names
//!   or tensor data.
//! * SIZE BUDGET: at most 200 lines here; the sink (state, lock, delivery) lives in `diag/sink.rs`, split out
//!   when this file reached its budget.
//! * FAILURE CONTRACT: fail-open-loud. Logging cannot fail the caller; a poisoned lock is
//!   recovered, and if even stderr is unwritable there is nowhere left to report.
//! * CHECKER: `diag::tests` (levels, sink capture, re-entrancy, last error, panic text) and
//!   `ffi::tests` (every entry point logs exactly one path-free record per bad input).
//! * FINISH LINE: every error is logged once at its choke point, with a specific message.

use std::any::Any;
use std::borrow::Cow;
use std::cell::RefCell;

mod sink;
use sink::enabled;
pub(crate) use sink::{emit, set_sink, Deliver};

/// ABI status values; the header is the source of truth (see `header_parity`).
#[repr(i32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Code {
    Invalid = 1,
    BusyOrState = 2,
    Timeout = 3,
    Provider = 4,
    Internal = 5,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) enum Level {
    Debug = 0,
    Info = 1,
    Warn = 2,
    Error = 3,
    Off = 4,
}

impl Level {
    pub(crate) fn from_raw(raw: u32) -> Option<Self> {
        [Self::Debug, Self::Info, Self::Warn, Self::Error, Self::Off]
            .into_iter()
            .find(|level| *level as u32 == raw)
    }

    fn name(self) -> &'static str {
        ["DEBUG", "INFO", "WARN", "ERROR", "OFF"][self as usize]
    }
}

/// Fields are private and `__new` is hidden so that the only way to write an error is
/// `fail!`, which cannot omit the message.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Error {
    code: Code,
    message: Cow<'static, str>,
}

impl Error {
    #[doc(hidden)]
    pub(crate) fn __new(code: Code, message: Cow<'static, str>) -> Self {
        Self { code, message }
    }

    pub(crate) fn code(&self) -> Code {
        self.code
    }

    pub(crate) fn message(&self) -> &str {
        &self.message
    }
}

/// `fail!(code, "literal")` or `fail!(code, "format {}", args)`. A literal with `{}`
/// and no arguments is not formatted, so pass at least one argument when interpolating.
macro_rules! fail {
    ($code:expr, $text:literal) => {
        $crate::diag::Error::__new($code, ::std::borrow::Cow::Borrowed($text))
    };
    ($code:expr, $text:literal, $($arg:tt)+) => {
        $crate::diag::Error::__new($code, ::std::borrow::Cow::Owned(format!($text, $($arg)+)))
    };
}
pub(crate) use fail;

thread_local! {
    static LAST: RefCell<String> = const { RefCell::new(String::new()) };
}

fn level_of(code: Code) -> Level {
    match code {
        Code::Timeout => Level::Debug, // a poll that expires is routine, not a fault
        Code::Invalid | Code::BusyOrState => Level::Warn,
        Code::Provider | Code::Internal => Level::Error,
    }
}

/// Logs an error that is not an entry point's return value (worker threads, Drop).
pub(crate) fn note(context: &str, error: &Error) {
    let level = level_of(error.code);
    // WHY the early check: a zero-timeout poll that times out is routine at Debug, and formatting
    // (String allocations) for a filtered record would be pure waste. `note` sets no last-error text
    // (`report` does, before calling here), and `emit` re-reads the sink itself, so no lock is held here.
    if !enabled(level) {
        return;
    }
    emit(
        level,
        error.code as i32,
        &format!("{context}: {}", error.message()),
    );
}

/// Logs an error returned to a C caller and remembers its text on this thread.
pub(crate) fn report(entry: &str, error: &Error) {
    set_last(&format!("{entry}: {}", error.message()));
    note(entry, error);
}

/// Logs a panic caught at the ABI edge; the caller then returns `Code::Internal`.
pub(crate) fn report_panic(entry: &str, payload: &(dyn Any + Send)) {
    let text = format!("panic in {entry}: {}", payload_text(payload));
    set_last(&text);
    emit(Level::Error, Code::Internal as i32, &text);
}

pub(crate) fn payload_text(payload: &(dyn Any + Send)) -> String {
    if let Some(text) = payload.downcast_ref::<&str>() {
        (*text).to_owned()
    } else if let Some(text) = payload.downcast_ref::<String>() {
        text.clone()
    } else {
        "non-string payload".to_owned()
    }
}

fn set_last(text: &str) {
    // Err means this thread is already exiting, so nobody can read the text any more.
    LAST.try_with(|last| *last.borrow_mut() = text.to_owned())
        .ok();
}

/// This thread's most recent failure text; empty if none (or the thread is exiting).
pub(crate) fn last() -> String {
    LAST.try_with(|last| last.borrow().clone())
        .unwrap_or_default()
}

#[cfg(test)]
pub(crate) mod testing;
#[cfg(test)]
mod tests;
