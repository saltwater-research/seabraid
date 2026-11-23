// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! The logging entry points and the version string: `seabraid_set_log_sink`, `seabraid_log`, `seabraid_last_error`
//! and `seabraid_version`.
//! WHY its own file: they take no handle and no provider, so unlike the failure-table rows for sessions in tests.rs
//! they share none of its session helpers beyond `expect` and `open_session`.

use super::*;
use crate::diag::testing::Capture;
use crate::diag::{self, Level};
use crate::ffi::logging::{
    seabraid_last_error, seabraid_log, seabraid_set_log_sink, seabraid_version,
};
use std::ffi::{c_char, c_void, CStr};

#[test]
fn failure_table_logging_entry_points() {
    let _capture = Capture::start(Level::Off);
    let rc = unsafe { seabraid_set_log_sink(None, ptr::null_mut(), 9) };
    expect(
        "seabraid_set_log_sink",
        rc,
        Code::Invalid,
        "min_level 9 is not a seabraid_log_level",
    );
    let rc = unsafe { seabraid_log(7, 0, c"x".as_ptr()) };
    expect(
        "seabraid_log",
        rc,
        Code::Invalid,
        "level 7 is not DEBUG..ERROR",
    );
    let rc = unsafe { seabraid_log(4, 0, c"x".as_ptr()) };
    expect(
        "seabraid_log",
        rc,
        Code::Invalid,
        "level 4 is not DEBUG..ERROR",
    );
    let rc = unsafe { seabraid_log(2, 0, ptr::null()) };
    expect("seabraid_log", rc, Code::Invalid, "message is null");
    // N1: seabraid_last_error's own argument errors are returned but never replace the thread's last error.
    let before = diag::last();
    assert!(!before.is_empty());
    let rc = unsafe { seabraid_last_error(ptr::null_mut(), 5, ptr::null_mut()) };
    assert_eq!(rc, Code::Invalid as i32, "buffer is null");
    assert_eq!(
        diag::last(),
        before,
        "the failing call must not record itself"
    );
}

#[test]
fn one_invalid_call_logs_exactly_one_warn_record_and_sets_the_last_error() {
    let capture = Capture::start(Level::Debug);
    let session = open_session(ptr::null());
    // The explicit reference open logs nothing, so "exactly one" below counts only the call under test.
    let opened = capture.mine();
    assert!(
        opened.is_empty(),
        "the explicit reference open must be silent: {opened:?}"
    );
    let mut stale = config();
    stale.abi_version = 9;
    let rc = unsafe { seabraid_prepare(session, &stale) };
    assert_eq!(rc, Code::Invalid as i32);
    let records = capture.mine();
    assert_eq!(records.len(), 1, "{records:?}");
    assert_eq!(
        (records[0].level, records[0].status),
        (Level::Warn as u32, 1)
    );
    assert_eq!(records[0].message, diag::last());

    let mut needed = 0usize;
    let mut small = [0 as c_char; 8];
    assert_eq!(
        unsafe { seabraid_last_error(small.as_mut_ptr(), small.len(), &mut needed) },
        0
    );
    assert_eq!(needed, records[0].message.len() + 1);
    let cut = unsafe { CStr::from_ptr(small.as_ptr()) }.to_str().unwrap();
    assert_eq!(cut, &records[0].message[..7]);
    assert_eq!(
        unsafe { seabraid_last_error(ptr::null_mut(), 0, &mut needed) },
        0
    );
    assert_eq!(needed, records[0].message.len() + 1);
    assert_eq!(unsafe { seabraid_close(session) }, 0);
}

#[test]
fn seabraid_log_reports_notices_and_replaces_invalid_utf8() {
    let capture = Capture::start(Level::Debug);
    assert_eq!(unsafe { seabraid_log(1, 0, c"adapter ready".as_ptr()) }, 0);
    assert_eq!(unsafe { seabraid_log(3, 4, c"bad \xff byte".as_ptr()) }, 0);
    let records = capture.mine();
    assert_eq!(
        (
            records[0].level,
            records[0].status,
            records[0].message.as_str()
        ),
        (1, 0, "adapter ready")
    );
    assert_eq!((records[1].level, records[1].status), (3, 4));
    assert_eq!(records[1].message, "bad \u{fffd} byte");
}

#[test]
fn the_c_sink_receives_its_context_drops_interior_nuls_and_null_restores_the_default() {
    thread_local! {
        // Per thread, because other tests may log while this sink is installed; the sink runs on the
        // logging thread, so each thread counts only its own records.
        static HITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
        static LAST_TEXT: std::cell::RefCell<String> = const { std::cell::RefCell::new(String::new()) };
    }
    unsafe extern "C" fn count(context: *mut c_void, _: u32, _: i32, message: *const c_char) {
        assert_eq!(context as usize, 0x77);
        HITS.with(|hits| hits.set(hits.get() + 1));
        let text = unsafe { CStr::from_ptr(message) }
            .to_string_lossy()
            .into_owned();
        LAST_TEXT.with(|last| *last.borrow_mut() = text);
    }
    let _capture = Capture::start(Level::Off);
    assert_eq!(
        unsafe { seabraid_set_log_sink(Some(count), 0x77 as *mut c_void, 2) },
        0
    );
    assert_eq!(unsafe { seabraid_log(2, 0, c"seen".as_ptr()) }, 0);
    assert_eq!(HITS.with(std::cell::Cell::get), 1);
    // A Rust-side record with an interior NUL (a C string cannot hold one) reaches the C sink without it.
    crate::diag::emit(Level::Warn, 0, "a\0b");
    assert_eq!(LAST_TEXT.with(|last| last.borrow().clone()), "ab");
    assert_eq!(
        unsafe { seabraid_set_log_sink(None, ptr::null_mut(), 4) },
        0
    );
}

#[test]
fn the_version_is_the_version_file() {
    let text = unsafe { CStr::from_ptr(seabraid_version()) }
        .to_str()
        .unwrap();
    assert_eq!(text, include_str!("../../../VERSION").trim());
    assert_eq!(text, env!("CARGO_PKG_VERSION"));
}
