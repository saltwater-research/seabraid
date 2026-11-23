// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! The process-wide log sink: where a record goes and the threshold it must pass.
//!
//! OWNER: this file owns the sink's state (`SINK`) and the one delivery path (`emit`); `diag` owns what a record
//! is (`Level`, `Code`, `Error`). Split out of `diag.rs` at its size budget (the sink was the part with its own
//! state and its own lock).
//! CONTRACT: `set_sink`, `emit`, `enabled`; imports only `std` and `Level`, never the C ABI.
//! FAILURE: fail-open-loud. A poisoned lock is recovered, and if stderr (the sink of last resort) is unwritable
//! there is nowhere left to report.
//! CHECKER: `diag::tests` (threshold, capture, re-entrancy) and `ffi::tests::logging` (the C side of the sink).

use super::Level;
use std::io::Write;
use std::sync::{Arc, Mutex, PoisonError};

/// Where a record goes once it passes the threshold. WHY a safe closure and not a C function pointer: this module
/// never names the C ABI, so the one `unsafe` call of a caller's sink lives in `ffi/logging.rs`, which wraps the
/// pointer and its context in one of these. An `Arc` so `emit` can copy it out of the lock (a sink may log).
pub(crate) type Deliver = Arc<dyn Fn(Level, i32, &str) + Send + Sync>;

#[derive(Clone)]
struct Sink {
    deliver: Option<Deliver>,
    min: Level,
}

static SINK: Mutex<Sink> = Mutex::new(Sink {
    deliver: None,
    min: Level::Warn,
});

/// `None` restores the default stderr sink; either way `min` is the new threshold.
pub(crate) fn set_sink(deliver: Option<Deliver>, min: Level) {
    *SINK.lock().unwrap_or_else(PoisonError::into_inner) = Sink { deliver, min };
}

/// Delivers one record. The sink is copied out of the lock first so a sink may log; WHY that means a replaced sink
/// can still be running on another thread (documented in seabraid.h), as holding the lock across the call would
/// deadlock any sink that logs.
pub(crate) fn emit(level: Level, status: i32, message: &str) {
    let sink = SINK.lock().unwrap_or_else(PoisonError::into_inner).clone();
    if level == Level::Off || level < sink.min {
        return;
    }
    match sink.deliver {
        Some(deliver) => deliver(level, status, message),
        None => {
            // stderr is the sink of last resort: if it is closed there is no other place to report.
            let line = format!("seabraid: {}: {message}\n", level.name());
            std::io::stderr().write_all(line.as_bytes()).ok();
        }
    }
}

/// Whether a record at `level` would reach the sink right now. Advisory: the sink may change before `emit`
/// re-checks, which is harmless as `emit` still filters authoritatively.
pub(super) fn enabled(level: Level) -> bool {
    let min = SINK.lock().unwrap_or_else(PoisonError::into_inner).min;
    level != Level::Off && level >= min
}
