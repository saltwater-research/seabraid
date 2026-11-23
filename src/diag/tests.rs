// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0

use super::testing::Capture;
use super::*;
use crate::ffi::status;
use std::cell::Cell;
use std::sync::Arc;

#[test]
fn records_below_the_threshold_are_dropped_and_off_is_not_a_record_level() {
    let capture = Capture::start(Level::Warn);
    emit(Level::Debug, 0, "debug");
    emit(Level::Info, 0, "info");
    emit(Level::Warn, 1, "warn");
    emit(Level::Error, 5, "error");
    emit(Level::Off, 0, "off");
    let seen: Vec<_> = capture
        .mine()
        .into_iter()
        .map(|r| (r.level, r.status))
        .collect();
    assert_eq!(seen, [(2, 1), (3, 5)]);

    drop(capture); // the capture lock is not re-entrant
    let capture = Capture::start(Level::Off);
    emit(Level::Error, 5, "silenced");
    assert!(capture.mine().is_empty());
}

#[test]
fn an_error_is_logged_at_the_level_of_its_code_and_remembered() {
    let capture = Capture::start(Level::Debug);
    let cases = [
        (Code::Timeout, Level::Debug),
        (Code::Invalid, Level::Warn),
        (Code::BusyOrState, Level::Warn),
        (Code::Provider, Level::Error),
        (Code::Internal, Level::Error),
    ];
    for (code, level) in cases {
        report("entry", &fail!(code, "detail {}", 7));
        let records = capture.mine();
        assert_eq!(records.len(), 1, "{code:?}");
        assert_eq!(records[0].level, level as u32);
        assert_eq!(records[0].status, code as i32);
        assert_eq!(records[0].message, "entry: detail 7");
        assert_eq!(last(), "entry: detail 7");
    }
}

#[test]
fn a_panic_at_the_abi_edge_is_logged_with_its_payload_and_returns_internal() {
    let capture = Capture::start(Level::Debug);
    let rc = status("seabraid_probe", || panic!("boom {}", 7));
    assert_eq!(rc, Code::Internal as i32);
    assert_eq!(status("seabraid_probe", || panic!("static text")), 5);
    assert_eq!(status("seabraid_probe", || std::panic::panic_any(3u8)), 5);
    let messages: Vec<_> = capture.mine().into_iter().map(|r| r.message).collect();
    assert_eq!(
        messages,
        [
            "panic in seabraid_probe: boom 7",
            "panic in seabraid_probe: static text",
            "panic in seabraid_probe: non-string payload"
        ]
    );
    assert_eq!(last(), "panic in seabraid_probe: non-string payload");
}

#[test]
fn the_last_error_is_per_thread_and_empty_before_any_failure() {
    report("here", &fail!(Code::Invalid, "x"));
    let other = std::thread::spawn(last).join().unwrap();
    assert_eq!(other, "");
    assert_eq!(last(), "here: x");
}

thread_local! {
    static DEPTH: Cell<u32> = const { Cell::new(0) };
}

fn reentrant(_: Level, _: i32, _: &str) {
    // A sink that logs while being called would deadlock if the lock were still held.
    if DEPTH.with(|depth| depth.replace(1)) == 0 {
        emit(Level::Warn, 0, "inner");
    }
}

#[test]
fn a_sink_may_log_from_inside_itself() {
    let _capture = Capture::start(Level::Debug);
    set_sink(Some(Arc::new(reentrant)), Level::Debug);
    emit(Level::Warn, 0, "outer");
    assert_eq!(DEPTH.with(Cell::get), 1);
}

// The C side of the sink (interior NULs dropped, the context handed back) is tested where it lives:
// src/ffi/tests/logging.rs, `the_c_sink_receives_its_context_drops_interior_nuls_and_null_restores_the_default`.
