// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! Layout epoch, versioned output structs, poisoning and bounded close, through the C entry points.

use super::*;

mod edge;
mod edge_config;
mod edge_sink;

#[test]
fn the_epoch_is_exported_and_matches_the_header() {
    assert_eq!(seabraid_abi_epoch(), ABI_EPOCH);
    assert_eq!(seabraid_abi_epoch(), 1);
}

fn status_of(session: *mut SessionHandle) -> (i32, SessionStatus) {
    let mut out = SessionStatus {
        abi_version: 1,
        struct_size: core::mem::size_of::<SessionStatus>() as u32,
        ..SessionStatus::default()
    };
    (unsafe { seabraid_session_status(session, &mut out) }, out)
}

#[test]
fn a_caller_struct_larger_than_v1_keeps_its_unknown_tail() {
    let _capture = Capture::start(Level::Off);
    let session = open_session(ptr::null());
    // A newer caller: a 72-byte struct whose bytes past the 64 the library knows must stay untouched.
    #[repr(C)]
    struct Bigger {
        known: SessionStatus,
        tail: [u8; 8],
    }
    let mut big = Bigger {
        known: SessionStatus {
            abi_version: 1,
            struct_size: 72,
            ..SessionStatus::default()
        },
        tail: [0xAB; 8],
    };
    let rc = unsafe { seabraid_session_status(session, &mut big.known) };
    assert_eq!(rc, 0);
    assert_eq!(big.tail, [0xAB; 8]);
    assert_eq!(
        big.known.struct_size, 64,
        "the library reports what it wrote"
    );
    assert_eq!(unsafe { seabraid_close(session) }, 0);
}

/// A caller built against the 48-byte v1 status: the library must write exactly 48 bytes and never the two
/// row counters appended after them.
#[test]
fn a_48_byte_status_caller_still_works_and_is_not_overrun() {
    let _capture = Capture::start(Level::Off);
    let session = open_session(ptr::null());
    let (activations, ids, weights) = ([1.0f32, 2.0], [0i32, 2], [0.25f32, 0.75]);
    let good = BatchView {
        activations: activations.as_ptr(),
        expert_ids: ids.as_ptr(),
        weights: weights.as_ptr(),
        ..view(1, [2, 2, 2])
    };
    let mut ticket = 0u64;
    assert_eq!(unsafe { seabraid_submit(session, &good, &mut ticket) }, 0);
    #[repr(C)]
    struct V1 {
        abi_version: u32,
        struct_size: u32,
        counters: [u64; 4],
        last_source_modalities: u32,
        outstanding: u32,
        guard: [u8; 16],
    }
    let mut old = V1 {
        abi_version: 1,
        struct_size: 48,
        counters: [0; 4],
        last_source_modalities: 0,
        outstanding: 0,
        guard: [0xCD; 16],
    };
    let rc = unsafe { seabraid_session_status(session, ptr::from_mut(&mut old).cast()) };
    assert_eq!(rc, 0);
    assert_eq!(old.guard, [0xCD; 16], "bytes 48.. belong to the caller");
    assert_eq!(
        (old.struct_size, old.counters[0], old.outstanding),
        (48, 1, 1)
    );
    // A caller of the current 64-byte struct also sees the row counters.
    let (rc, now) = status_of(session);
    assert_eq!(rc, 0);
    assert_eq!(
        (now.struct_size, now.rows_submitted, now.rows_shipped),
        (64, 1, 1)
    );
    // Below the v1 base is still refused.
    let mut tiny = SessionStatus {
        abi_version: 1,
        struct_size: 40,
        ..SessionStatus::default()
    };
    assert_eq!(
        unsafe { seabraid_session_status(session, &mut tiny) },
        Code::Invalid as i32
    );
    assert_eq!(unsafe { seabraid_close(session) }, 0);
}

#[test]
fn a_v2_descriptor_refuses_compact_rows_even_with_partial_routes() {
    let _capture = Capture::start(Level::Off);
    type Execute = unsafe extern "C" fn(
        *mut c_void,
        u32,
        *const Config,
        *const BatchView,
        *mut f32,
        usize,
    ) -> i32;
    unsafe extern "C" fn nothing(
        _: *mut c_void,
        _: u32,
        _: *const Config,
        _: *const BatchView,
        _: *mut f32,
        _: usize,
    ) -> i32 {
        0
    }
    #[repr(C)]
    struct RawMulti {
        abi_version: u32,
        operation_count: u32,
        flags: u32,
        context: *mut c_void,
        execute: Option<Execute>,
    }
    // WHY refused: provider v2 is frozen with PARTIAL_ROUTES only; compaction is a provider v3 feature (I6).
    let compact = crate::ffi::provider_v3::COMPACT_ROWS;
    let mut descriptor = RawMulti {
        abi_version: 2,
        operation_count: 1,
        flags: compact,
        context: ptr::null_mut(),
        execute: Some(nothing),
    };
    for flags in [compact, compact | crate::ffi::providers::PARTIAL_ROUTES] {
        descriptor.flags = flags;
        let mut session = ptr::NonNull::<SessionHandle>::dangling().as_ptr();
        let rc = unsafe { seabraid_open_multi(ptr::from_ref(&descriptor).cast(), &mut session) };
        assert_eq!(rc, Code::Invalid as i32, "flags {flags:#x}");
        assert!(session.is_null(), "*out is NULL on a refused open");
        assert!(diag::last().ends_with(&format!(
            "provider.flags {flags:#x} sets unknown flags {compact:#x}"
        )));
    }
    // PARTIAL_ROUTES alone is still the v2 contract.
    descriptor.flags = crate::ffi::providers::PARTIAL_ROUTES;
    let mut session = ptr::null_mut();
    let rc = unsafe { seabraid_open_multi(ptr::from_ref(&descriptor).cast(), &mut session) };
    assert_eq!(rc, 0);
    assert_eq!(unsafe { seabraid_close(session) }, 0);
}

#[test]
fn a_caught_panic_poisons_the_handle_and_close_still_frees_it() {
    let capture = Capture::start(Level::Debug);
    let session = open_session(ptr::null());
    let rc = unsafe {
        status_on("seabraid_test_panic", session, || -> Result<()> {
            panic!("injected panic")
        })
    };
    assert_eq!(rc, Code::Internal as i32);
    capture.mine();
    // Every later call is refused with one logged record, and does nothing.
    let (rc, _) = status_of(session);
    expect(
        "seabraid_session_status",
        rc,
        Code::Internal,
        "session handle poisoned by earlier panic; close it",
    );
    let mut ticket = 0u64;
    let rc = unsafe { seabraid_submit(session, &view(1, [2, 2, 2]), &mut ticket) };
    expect(
        "seabraid_submit_operation",
        rc,
        Code::Internal,
        "session handle poisoned by earlier panic; close it",
    );
    assert_eq!(ticket, 0);
    assert_eq!(unsafe { seabraid_close(session) }, 0);
}

#[test]
fn close_timeout_leaks_a_stuck_worker_loudly_and_closes_a_finished_one() {
    use std::sync::atomic::{AtomicBool, Ordering};
    static RELEASE: AtomicBool = AtomicBool::new(false);
    unsafe extern "C" fn stuck(
        _: *mut c_void,
        _: *const Config,
        _: *const BatchView,
        output: *mut f32,
        count: usize,
    ) -> i32 {
        while !RELEASE.load(Ordering::SeqCst) {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        unsafe { ptr::write_bytes(output, 0, count) };
        0
    }
    let capture = Capture::start(Level::Debug);
    let raw = RawProvider {
        abi_version: 1,
        context: ptr::null_mut(),
        execute: Some(stuck),
    };
    let session = open_session(ptr::from_ref(&raw).cast());
    let (activations, ids, weights) = ([1.0f32, 2.0], [0i32, 2], [0.25f32, 0.75]);
    let good = BatchView {
        activations: activations.as_ptr(),
        expert_ids: ids.as_ptr(),
        weights: weights.as_ptr(),
        ..view(1, [2, 2, 2])
    };
    let mut ticket = 0u64;
    assert_eq!(unsafe { seabraid_submit(session, &good, &mut ticket) }, 0);
    capture.mine();
    let rc = unsafe { seabraid_close_timeout(session, 30) };
    assert_eq!(rc, Code::Timeout as i32);
    let records = capture.mine();
    let loud = records
        .iter()
        .find(|r| r.level == Level::Error as u32)
        .expect("an ERROR record");
    assert!(
        loud.message.contains(&format!("ticket {ticket}")) && loud.message.contains("leaked"),
        "{}",
        loud.message
    );
    assert_eq!(
        diag::last(),
        "seabraid_close_timeout: close timed out after 30 ms"
    );
    // The leaked worker can still run, so let it finish before this test ends.
    RELEASE.store(true, Ordering::SeqCst);

    // A worker that is already idle closes within the bound and reports success.
    let idle = open_session(ptr::null());
    assert_eq!(unsafe { seabraid_close_timeout(idle, 5000) }, 0);
    let rc = unsafe { seabraid_close_timeout(ptr::null_mut(), 5) };
    expect(
        "seabraid_close_timeout",
        rc,
        Code::Invalid,
        "session is null",
    );
}
