// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! Overnight hardening O1, rows L* and C*: last_error, the log sink and the close family (see edge.rs).

use super::edge::good_view;
use super::*;
use std::sync::atomic::{AtomicI32, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

fn last_error_of(capacity: usize, fill: u8) -> (i32, usize, Vec<u8>) {
    let mut buffer = vec![fill; capacity.max(1) + 4];
    let mut needed = usize::MAX;
    let rc = unsafe { seabraid_last_error(buffer.as_mut_ptr().cast(), capacity, &mut needed) };
    (rc, needed, buffer)
}

fn fail_with(text: &str) {
    let error = crate::diag::fail!(Code::Invalid, "{}", text);
    diag::report("edge", &error);
}

// case L1-L9: seabraid_last_error truncation, byte counts and the UTF-8 boundary.
#[test]
fn last_error_truncates_at_a_character_boundary_and_always_reports_the_full_size() {
    let _capture = Capture::start(Level::Off);
    // "edge: " (6 bytes) then three 4-byte characters: 18 bytes of text.
    fail_with("\u{1F980}\u{1F980}\u{1F980}");
    let full = diag::last();
    assert_eq!(full.len(), 6 + 12);
    for (capacity, kept) in [
        (1, 0),
        (2, 1),
        (6, 5),
        (7, 6),
        (8, 6),
        (9, 6),
        (10, 6),
        (11, 10),
        (13, 10),
        (15, 14),
        (18, 14),
        (19, 18),
        (20, 18),
        (4096, 18),
    ] {
        let (rc, needed, buffer) = last_error_of(capacity, b'#');
        assert_eq!(rc, 0);
        assert_eq!(
            needed, 19,
            "needed is the full length + 1 even when truncated (capacity {capacity})"
        );
        let text = std::ffi::CStr::from_bytes_until_nul(&buffer)
            .unwrap()
            .to_str()
            .unwrap();
        assert_eq!(text.len(), kept, "capacity {capacity}");
        assert_eq!(text, &full[..text.len()]);
        assert!(
            buffer[capacity..].iter().all(|&b| b == b'#'),
            "nothing written past the capacity {capacity}"
        );
    }
    // capacity 0 writes nothing, with or without a buffer; needed is optional.
    let mut needed = 0usize;
    assert_eq!(
        unsafe { seabraid_last_error(ptr::null_mut(), 0, &mut needed) },
        0
    );
    assert_eq!(needed, 19);
    assert_eq!(
        unsafe { seabraid_last_error(ptr::null_mut(), 0, ptr::null_mut()) },
        0
    );
    // a non-empty capacity with a null buffer is the one argument error, and it must not erase the last error.
    let rc = unsafe { seabraid_last_error(ptr::null_mut(), 5, ptr::null_mut()) };
    assert_eq!(rc, Code::Invalid as i32);
    assert_eq!(diag::last(), full);
}

// case L10-L12: a megabyte of text survives the thread-local, the size report and a sink, uncut.
#[test]
#[cfg_attr(
    miri,
    ignore = "a megabyte through the interpreter takes over 6 minutes and 1 GB; no unsafe code here is size-dependent"
)]
fn a_megabyte_error_is_remembered_whole_reported_whole_and_delivered_to_the_sink_whole() {
    let capture = Capture::start(Level::Debug);
    let text = "\u{e9}tude ".repeat(150_000); // multi-byte, 1.2 MB
    fail_with(&text);
    let (rc, needed, buffer) = last_error_of(1 << 21, 0);
    assert_eq!(rc, 0);
    assert_eq!(needed, 6 + text.len() + 1);
    let held = std::ffi::CStr::from_bytes_until_nul(&buffer)
        .unwrap()
        .to_str()
        .unwrap();
    assert_eq!(held.len(), needed - 1);
    let records = capture.mine();
    assert_eq!(records.len(), 1);
    assert_eq!(
        records[0].message.len(),
        needed - 1,
        "the sink gets every byte"
    );
    // seabraid_log with a megabyte, and with a message that is invalid UTF-8 across the cut.
    let long = std::ffi::CString::new(text.as_bytes()).unwrap();
    assert_eq!(unsafe { seabraid_log(3, 0, long.as_ptr()) }, 0);
    let mut raw = vec![b'a'; 1 << 20];
    raw.extend_from_slice(&[0xE2, 0x82]); // half of a three-byte character at the very end
    let raw = std::ffi::CString::new(raw).unwrap();
    assert_eq!(unsafe { seabraid_log(3, 0, raw.as_ptr()) }, 0);
    let records = capture.mine();
    assert_eq!(records[0].message, text);
    assert_eq!(
        records[1].message.len(),
        (1 << 20) + 3,
        "the dangling lead bytes become one U+FFFD"
    );
    assert!(records[1].message.ends_with('\u{fffd}'));
}

// case L13: a fresh thread has no last error: size 1, empty string.
#[test]
fn a_thread_that_never_failed_reads_an_empty_last_error() {
    std::thread::spawn(|| {
        let (rc, needed, buffer) = last_error_of(4, b'#');
        assert_eq!((rc, needed, buffer[0]), (0, 1, 0));
    })
    .join()
    .unwrap();
}

static HANDLE: AtomicUsize = AtomicUsize::new(0);
static CLOSE_RESULTS: [AtomicI32; 2] = [AtomicI32::new(-1), AtomicI32::new(-1)];

/// Runs on the worker thread while the main thread sits in `seabraid_collect` holding the handle. It waits
/// until the gate reports busy (so the collect is certainly inside), then tries both close calls: each must be
/// refused, never consume the handle. WHY from the callback: it is the only point where "a call is inside the
/// session" is certain, so the probe cannot race a real close.
unsafe extern "C" fn probe_closes(
    _: *mut c_void,
    _: *const Config,
    _: *const BatchView,
    output: *mut f32,
    count: usize,
) -> i32 {
    let handle = HANDLE.load(Ordering::SeqCst) as *mut SessionHandle;
    let mut probe = SessionStatus {
        abi_version: 1,
        struct_size: 64,
        ..SessionStatus::default()
    };
    let deadline = Instant::now() + Duration::from_secs(10);
    while unsafe { seabraid_session_status(handle, &mut probe) } == 0 && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(1));
    }
    CLOSE_RESULTS[0].store(unsafe { seabraid_close(handle) }, Ordering::SeqCst);
    CLOSE_RESULTS[1].store(
        unsafe { seabraid_close_timeout(handle, 1) },
        Ordering::SeqCst,
    );
    unsafe { ptr::write_bytes(output, 0, count) };
    0
}

// case C5-C8: closing while a collect holds the handle is BUSY_OR_STATE, the handle survives, then closes clean.
#[test]
fn close_and_close_timeout_during_a_collect_are_busy_and_leave_the_handle_usable() {
    let _capture = Capture::start(Level::Off);
    let raw = RawProvider {
        abi_version: 1,
        context: ptr::null_mut(),
        execute: Some(probe_closes),
    };
    let session = open_session(ptr::from_ref(&raw).cast());
    HANDLE.store(session as usize, Ordering::SeqCst);
    let (a, i, w) = ([1.0f32, 2.0], [0i32, 2], [0.25f32, 0.75]);
    let mut ticket = 0u64;
    assert_eq!(
        unsafe { seabraid_submit(session, &good_view(&a, &i, &w), &mut ticket) },
        0
    );
    let mut out = [1f32; 2];
    assert_eq!(
        unsafe { seabraid_collect(session, ticket, 20_000, out.as_mut_ptr(), 2) },
        0
    );
    let busy = Code::BusyOrState as i32;
    assert_eq!(
        CLOSE_RESULTS[0].load(Ordering::SeqCst),
        busy,
        "close during a collect"
    );
    assert_eq!(
        CLOSE_RESULTS[1].load(Ordering::SeqCst),
        busy,
        "close_timeout during a collect"
    );
    assert_eq!(
        unsafe { seabraid_close_timeout(session, u32::MAX) },
        0,
        "an idle worker closes at once even with u32::MAX ms"
    );
    // WHY no second close here: the handle is freed now, and reading a freed handle is the documented residual UB
    // (crate::gate, HONEST LIMIT; Miri reported it here). That a closed handle is refused is checked without a free
    // by handle.rs `a_second_close_of_the_same_handle_is_refused`.
}

// case C9: seabraid_close_timeout with a zero bound on an idle session is OK and raises no leak record.
#[test]
fn close_timeout_zero_on_an_idle_session_is_ok_and_logs_nothing_loud() {
    let capture = Capture::start(Level::Debug);
    let session = open_session(ptr::null());
    assert_eq!(unsafe { seabraid_close_timeout(session, 0) }, 0);
    let loud: Vec<_> = capture
        .mine()
        .into_iter()
        .filter(|r| r.level >= Level::Warn as u32)
        .collect();
    assert!(loud.is_empty(), "{loud:?}");
}
