// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! Test-only capture of the process-wide sink. The sink is global, so captures are
//! serialized by a lock and records are tagged by thread: a test sees only what its own
//! thread logged, even while other tests fail on purpose in parallel.
//! FAILURE: fail-open-loud like `diag`: a poisoned lock is recovered, so one failed test cannot hide the next.

use super::{set_sink, Level};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread::{self, ThreadId};

#[derive(Clone, Debug)]
pub(crate) struct Record {
    pub level: u32,
    pub status: i32,
    pub message: String,
}

static RECORDS: Mutex<Vec<(ThreadId, Record)>> = Mutex::new(Vec::new());
static EXCLUSIVE: Mutex<()> = Mutex::new(());

pub(crate) struct Capture {
    _exclusive: MutexGuard<'static, ()>,
}

/// The capturing sink: a safe closure (see `diag::Deliver`), so this test helper needs no `unsafe`.
fn record(level: Level, status: i32, message: &str) {
    let entry = (
        thread::current().id(),
        Record {
            level: level as u32,
            status,
            message: message.to_owned(),
        },
    );
    RECORDS
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .push(entry);
}

impl Capture {
    pub(crate) fn start(min: Level) -> Self {
        let guard = EXCLUSIVE.lock().unwrap_or_else(PoisonError::into_inner);
        RECORDS
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clear();
        set_sink(Some(Arc::new(record)), min);
        Self { _exclusive: guard }
    }

    /// Records emitted by the calling thread since the capture started or was last drained.
    pub(crate) fn mine(&self) -> Vec<Record> {
        let me = thread::current().id();
        let mut all = RECORDS.lock().unwrap_or_else(PoisonError::into_inner);
        let (mine, others): (Vec<_>, Vec<_>) = all.drain(..).partition(|(id, _)| *id == me);
        *all = others;
        mine.into_iter().map(|(_, record)| record).collect()
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        set_sink(None, Level::Warn);
    }
}

/// Prints one failure-table row and enforces the privacy rule on its text.
pub(crate) fn table_row(entry: &str, rc: i32, message: &str) {
    println!("TABLE|{entry}|{rc}|{message}");
    assert!(message.starts_with(entry), "{message}");
    assert!(
        !message.contains('/') && !message.contains('\\'),
        "path-like text: {message}"
    );
}
