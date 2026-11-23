// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! The failure table: one invalid input per exported function, with the exact text it logs.
//! Run `cargo test failure_table -- --nocapture` to print the TABLE rows.

mod abi;
mod guard_page;
mod handle;
mod kinds;
mod logging;
mod poison;
mod v2;
mod v2_failures;
mod v2_session;
mod worker;

use super::logging::{seabraid_last_error, seabraid_log};
use super::*;
use crate::diag::testing::{table_row, Capture};
use crate::diag::{self, Level};
use std::ffi::c_void;

fn expect(entry: &str, rc: i32, code: Code, text: &str) {
    assert_eq!(rc, code as i32, "{entry}: {text}");
    let message = diag::last();
    table_row(entry, rc, &message);
    assert_eq!(message, format!("{entry}: {text}"));
}

fn config() -> Config {
    Config {
        abi_version: 1,
        hidden: 2,
        experts: 3,
        max_tokens: 4,
        top_k: 2,
    }
}

fn view(tokens: u32, counts: [usize; 3]) -> BatchView {
    BatchView {
        abi_version: 1,
        tokens,
        activations: ptr::null(),
        activation_count: counts[0],
        expert_ids: ptr::null(),
        expert_id_count: counts[1],
        weights: ptr::null(),
        weight_count: counts[2],
    }
}

#[repr(C)]
struct RawProvider {
    abi_version: u32,
    context: *mut c_void,
    execute: Option<
        unsafe extern "C" fn(*mut c_void, *const Config, *const BatchView, *mut f32, usize) -> i32,
    >,
}

#[repr(C)]
struct RawMulti {
    abi_version: u32,
    operation_count: u32,
    flags: u32,
    context: *mut c_void,
    execute: Option<unsafe extern "C" fn()>,
}

unsafe extern "C" fn failing(
    _: *mut c_void,
    _: *const Config,
    _: *const BatchView,
    _: *mut f32,
    _: usize,
) -> i32 {
    7
}

fn open_session(provider: *const Provider) -> *mut SessionHandle {
    let mut session = ptr::null_mut();
    // A null provider here means "the explicit reference transform": seabraid_open itself refuses null.
    let rc = if provider.is_null() {
        unsafe { seabraid_open_reference(&mut session) }
    } else {
        unsafe { seabraid_open(provider, &mut session) }
    };
    assert_eq!(rc, 0);
    assert_eq!(unsafe { seabraid_prepare(session, &config()) }, 0);
    session
}

#[test]
fn failure_table_open_and_prepare() {
    let _capture = Capture::start(Level::Off);
    let mut caps = CAPABILITIES;
    caps.abi_version = 2;
    let rc = unsafe { seabraid_query_capabilities(&mut caps) };
    expect(
        "seabraid_query_capabilities",
        rc,
        Code::Invalid,
        "out abi_version 2 != 1",
    );
    caps.abi_version = 1;
    caps.struct_size = 8;
    let rc = unsafe { seabraid_query_capabilities(&mut caps) };
    expect(
        "seabraid_query_capabilities",
        rc,
        Code::Invalid,
        "out struct_size 8 < the v1 size 24",
    );
    let rc = unsafe { seabraid_query_capabilities(ptr::null_mut()) };
    expect(
        "seabraid_query_capabilities",
        rc,
        Code::Invalid,
        "out is null",
    );
    let rc = unsafe { seabraid_session_status(ptr::null_mut(), ptr::null_mut()) };
    expect(
        "seabraid_session_status",
        rc,
        Code::Invalid,
        "session is null",
    );

    let mut session = ptr::null_mut();
    let rc = unsafe { seabraid_open(ptr::null(), ptr::null_mut()) };
    expect("seabraid_open", rc, Code::Invalid, "out is null");
    let rc = unsafe { seabraid_open(ptr::null(), &mut session) };
    expect(
        "seabraid_open",
        rc,
        Code::Invalid,
        "provider is null (use seabraid_open_reference for the built-in reference transform)",
    );
    assert!(session.is_null());
    let old = RawProvider {
        abi_version: 9,
        context: ptr::null_mut(),
        execute: None,
    };
    let rc = unsafe { seabraid_open(ptr::from_ref(&old).cast(), &mut session) };
    expect(
        "seabraid_open",
        rc,
        Code::Invalid,
        "provider abi_version 9 != 1",
    );
    let hollow = RawProvider {
        abi_version: 1,
        context: ptr::null_mut(),
        execute: None,
    };
    let rc = unsafe { seabraid_open(ptr::from_ref(&hollow).cast(), &mut session) };
    expect(
        "seabraid_open",
        rc,
        Code::Invalid,
        "provider.execute is null",
    );
    assert!(session.is_null());

    let rc = unsafe { seabraid_open_multi(ptr::null(), &mut session) };
    expect("seabraid_open_multi", rc, Code::Invalid, "provider is null");
    let v1 = RawMulti {
        abi_version: 1,
        operation_count: 1,
        flags: 0,
        context: ptr::null_mut(),
        execute: None,
    };
    let rc = unsafe { seabraid_open_multi(ptr::from_ref(&v1).cast(), &mut session) };
    expect(
        "seabraid_open_multi",
        rc,
        Code::Invalid,
        "provider abi_version 1 != 2",
    );
    let flags = RawMulti {
        abi_version: 2,
        flags: 4,
        execute: Some(noop),
        ..v1
    };
    let rc = unsafe { seabraid_open_multi(ptr::from_ref(&flags).cast(), &mut session) };
    expect(
        "seabraid_open_multi",
        rc,
        Code::Invalid,
        "provider.flags 0x4 sets unknown flags 0x4",
    );
}

#[test]
fn failure_table_prepare_and_close() {
    let _capture = Capture::start(Level::Off);
    let session = open_session(ptr::null());
    let mut stale = config();
    stale.abi_version = 9;
    let rc = unsafe { seabraid_prepare(session, &stale) };
    expect(
        "seabraid_prepare",
        rc,
        Code::Invalid,
        "config abi_version 9 != 1",
    );
    let rc = unsafe { seabraid_prepare(session, &config()) };
    expect(
        "seabraid_prepare",
        rc,
        Code::BusyOrState,
        "session is already prepared",
    );
    assert_eq!(unsafe { seabraid_close(session) }, 0);
    let rc = unsafe { seabraid_close(ptr::null_mut()) };
    expect("seabraid_close", rc, Code::Invalid, "session is null");
}

unsafe extern "C" fn noop() {}

#[test]
fn failure_table_submit_and_collect() {
    let _capture = Capture::start(Level::Off);
    let session = open_session(ptr::null());
    let mut ticket = 0u64;
    let empty = view(1, [2, 2, 2]);
    let rc = unsafe { seabraid_submit_routed(session, &empty, 1 << 20, &mut ticket) };
    expect(
        "seabraid_submit_operation",
        rc,
        Code::Invalid,
        "source_modalities 0x100000 sets unknown flags 0x100000",
    );
    let rc = unsafe { seabraid_submit_operation(session, 5, &empty, 0, &mut ticket) };
    expect(
        "seabraid_submit_operation",
        rc,
        Code::Invalid,
        "operation_id 5 >= operation_count 1",
    );
    let wrong = view(3, [12, 6, 6]);
    let rc = unsafe { seabraid_submit(session, &wrong, &mut ticket) };
    expect(
        "seabraid_submit_operation",
        rc,
        Code::Invalid,
        "activation_count 12 != tokens 3 * hidden 2",
    );
    let rc = unsafe { seabraid_submit(session, &empty, &mut ticket) };
    expect(
        "seabraid_submit_operation",
        rc,
        Code::Invalid,
        "batch.activations is null",
    );
    let mut output = [0f32; 2];
    let rc = unsafe { seabraid_collect(session, 999, 0, output.as_mut_ptr(), 2) };
    expect(
        "seabraid_collect",
        rc,
        Code::BusyOrState,
        "ticket 999 is not outstanding",
    );

    let (activations, ids, weights) = ([1.0f32, 2.0], [0i32, 2], [0.25f32, 0.75]);
    let good = BatchView {
        activations: activations.as_ptr(),
        expert_ids: ids.as_ptr(),
        weights: weights.as_ptr(),
        ..view(1, [2, 2, 2])
    };
    assert_eq!(unsafe { seabraid_submit(session, &good, &mut ticket) }, 0);
    let rc = unsafe { seabraid_submit(session, &good, &mut ticket) };
    expect(
        "seabraid_submit_operation",
        rc,
        Code::BusyOrState,
        &format!("ticket {ticket} is still outstanding"),
    );
    let rc = unsafe { seabraid_collect(session, ticket, 1000, output.as_mut_ptr(), 3) };
    expect(
        "seabraid_collect",
        rc,
        Code::Invalid,
        "output_count 3 != tokens * hidden 2",
    );
    assert_eq!(
        unsafe { seabraid_collect(session, ticket, 1000, output.as_mut_ptr(), 2) },
        0
    );
    assert_eq!(output, [2.5, 5.0]);
    assert_eq!(unsafe { seabraid_close(session) }, 0);
}

#[test]
fn failure_table_provider_status_is_carried() {
    let _capture = Capture::start(Level::Off);
    let raw = RawProvider {
        abi_version: 1,
        context: ptr::null_mut(),
        execute: Some(failing),
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
    let mut output = [0f32; 2];
    let rc = unsafe { seabraid_collect(session, ticket, 2000, output.as_mut_ptr(), 2) };
    expect(
        "seabraid_collect",
        rc,
        Code::Provider,
        "provider callback returned status 7",
    );
    assert_eq!(unsafe { seabraid_close(session) }, 0);
}
