// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! The v2 entry points: bit-identity with the v1 path, the v3 callback view, and every validation branch of
//! `seabraid_session_open` and `seabraid_session_submit` with the reason it logs.

use super::*;
use crate::capabilities::FORMAT_ROUTED_F16;
use crate::ffi::batch_v2::{BatchV2, BATCH_V2_VERSION};
use crate::ffi::provider_v3::{ProviderV3, PROVIDER_V3_VERSION, REFERENCE};
use crate::ffi::v2::{seabraid_session_open, seabraid_session_submit};

const ACTIVATIONS: [f32; 4] = [0.5, -1.25, 3.0, 0.125];
const IDS: [i32; 4] = [0, 2, 1, 2];
const WEIGHTS: [f32; 4] = [0.25, 0.75, -0.5, 1.5];

pub(super) fn provider(flags: u32, config: &Config, execute: Option<ExecuteFn>) -> ProviderV3 {
    ProviderV3 {
        abi_version: PROVIDER_V3_VERSION,
        struct_size: ProviderV3::BASE_SIZE as u32,
        operation_count: 1,
        flags,
        config,
        context: ptr::null_mut(),
        execute,
    }
}

pub(super) type ExecuteFn =
    unsafe extern "C" fn(*mut c_void, *const Config, *const BatchV2, *mut f32, usize) -> i32;

pub(super) fn reference() -> ProviderV3 {
    provider(REFERENCE, &CONFIG, None)
}

pub(super) static CONFIG: Config = Config {
    abi_version: 1,
    hidden: 2,
    experts: 3,
    max_tokens: 4,
    top_k: 2,
};

pub(super) fn batch() -> BatchV2 {
    BatchV2 {
        abi_version: BATCH_V2_VERSION,
        struct_size: BatchV2::BASE_SIZE as u32,
        operation_id: 0,
        source_modalities: 0,
        placement_epoch: 0,
        tokens: 2,
        activation_format: 1,
        activations: ACTIVATIONS.as_ptr().cast(),
        activation_count: 4,
        expert_ids: IDS.as_ptr(),
        expert_id_count: 4,
        weights: WEIGHTS.as_ptr(),
        weight_count: 4,
    }
}

pub(super) fn open_v3(provider: &ProviderV3) -> (i32, *mut SessionHandle) {
    let mut session = ptr::NonNull::<SessionHandle>::dangling().as_ptr();
    (
        unsafe { seabraid_session_open(provider, &mut session) },
        session,
    )
}

pub(super) fn submit(session: *mut SessionHandle, batch: &BatchV2) -> (i32, u64) {
    let mut ticket = 77;
    (
        unsafe { seabraid_session_submit(session, batch, &mut ticket) },
        ticket,
    )
}

pub(super) fn collect(session: *mut SessionHandle, ticket: u64) -> Vec<f32> {
    let mut out = vec![0.0; 4];
    assert_eq!(
        unsafe { seabraid_collect(session, ticket, 5000, out.as_mut_ptr(), 4) },
        0
    );
    out
}

#[test]
fn reference_output_through_v2_is_bit_identical_to_v1() {
    let _capture = Capture::start(Level::Off);
    let (rc, v2) = open_v3(&reference());
    assert_eq!(rc, 0);
    let (rc, ticket) = submit(v2, &batch());
    assert_eq!(rc, 0);
    let from_v2 = collect(v2, ticket);

    let v1 = open_session(ptr::null());
    let input = BatchView {
        activations: ACTIVATIONS.as_ptr(),
        expert_ids: IDS.as_ptr(),
        weights: WEIGHTS.as_ptr(),
        ..view(2, [4, 4, 4])
    };
    let mut ticket = 0;
    assert_eq!(unsafe { seabraid_submit(v1, &input, &mut ticket) }, 0);
    let from_v1 = collect(v1, ticket);
    let bits = |values: &[f32]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&from_v2), bits(&from_v1));
    assert!(from_v1.iter().any(|&v| v != 0.0));
    assert_eq!(
        (unsafe { seabraid_close(v1) }, unsafe { seabraid_close(v2) }),
        (0, 0)
    );
}

#[repr(C)]
#[derive(Default)]
struct Seen {
    version: u32,
    size: u32,
    operation: u32,
    tokens: u32,
    format: u32,
    called: u32,
}

pub(super) unsafe extern "C" fn record(
    context: *mut c_void,
    _: *const Config,
    batch: *const BatchV2,
    output: *mut f32,
    count: usize,
) -> i32 {
    let (seen, batch) = unsafe { (&mut *context.cast::<Seen>(), &*batch) };
    *seen = Seen {
        version: batch.abi_version,
        size: batch.struct_size,
        operation: batch.operation_id,
        tokens: batch.tokens,
        format: batch.activation_format,
        called: 1,
    };
    for i in 0..count {
        unsafe { *output.add(i) = *batch.activations.cast::<f32>().add(i) * 2.0 };
    }
    0
}

#[test]
fn a_v3_callback_sees_the_library_view_of_the_batch() {
    let _capture = Capture::start(Level::Off);
    let mut seen = Seen::default();
    let mut callback = provider(0, &CONFIG, Some(record));
    callback.context = ptr::from_mut(&mut seen).cast();
    callback.operation_count = 2;
    let (rc, session) = open_v3(&callback);
    assert_eq!(rc, 0);
    let (rc, ticket) = submit(
        session,
        &BatchV2 {
            operation_id: 1,
            ..batch()
        },
    );
    assert_eq!(rc, 0);
    assert_eq!(collect(session, ticket), [1.0, -2.5, 6.0, 0.25]);
    assert_eq!(unsafe { seabraid_close(session) }, 0);
    assert_eq!(
        (seen.version, seen.size, seen.operation, seen.tokens),
        (2, 80, 1, 2)
    );
    assert_eq!((seen.format, seen.called), (1, 1));
}

#[test]
fn a_larger_struct_with_a_zero_tail_is_accepted_and_a_nonzero_tail_is_refused() {
    let _capture = Capture::start(Level::Off);
    #[repr(C)]
    struct WideProvider(ProviderV3, u64);
    let mut wide = WideProvider(
        ProviderV3 {
            struct_size: 48,
            ..reference()
        },
        0,
    );
    // WHY a pointer to the whole wide struct and not `&wide.0`: a C caller's pointer covers its whole struct, and the
    // library reads the tail past the known prefix; `&wide.0` would only grant the prefix (Miri reported the read).
    let wide_ptr = |wide: &WideProvider| ptr::from_ref(wide).cast::<ProviderV3>();
    let mut session = ptr::null_mut();
    assert_eq!(
        unsafe { seabraid_session_open(wide_ptr(&wide), &mut session) },
        0
    );

    #[repr(C)]
    struct WideBatch(BatchV2, u64);
    let mut big = WideBatch(
        BatchV2 {
            struct_size: 88,
            ..batch()
        },
        0,
    );
    let big_ptr = |big: &WideBatch| ptr::from_ref(big).cast::<BatchV2>();
    let mut ticket = 0;
    assert_eq!(
        unsafe { seabraid_session_submit(session, big_ptr(&big), &mut ticket) },
        0
    );
    assert_eq!(collect(session, ticket).len(), 4);

    big.1 = 1;
    let rc = unsafe { seabraid_session_submit(session, big_ptr(&big), &mut ticket) };
    expect(
        "seabraid_session_submit",
        rc,
        Code::Invalid,
        "batch sets bytes at offset 80 that this library does not know (its struct is 80 bytes)",
    );
    wide.1 = 1;
    let rc = unsafe { seabraid_session_open(wide_ptr(&wide), &mut session) };
    expect(
        "seabraid_session_open",
        rc,
        Code::Invalid,
        "provider sets bytes at offset 40 that this library does not know (its struct is 40 bytes)",
    );
    assert!(session.is_null());
}

#[test]
fn f16_is_refused_for_a_provider_that_did_not_opt_in() {
    let _capture = Capture::start(Level::Off);
    let (rc, session) = open_v3(&reference());
    assert_eq!(rc, 0);
    let half = [0u16; 4];
    let input = BatchV2 {
        activation_format: FORMAT_ROUTED_F16,
        activations: half.as_ptr().cast(),
        ..batch()
    };
    let (rc, _) = submit(session, &input);
    // WHY not `expect`: its table logger rejects the "/" in "ACCEPTS_F16 / ACCEPTS_BF16" as path-like text.
    assert_eq!(rc, Code::Invalid as i32);
    assert!(
        diag::last().starts_with("seabraid_session_submit: activation format 3 was not declared"),
        "{}",
        diag::last()
    );
    assert_eq!(unsafe { seabraid_close(session) }, 0);
}
