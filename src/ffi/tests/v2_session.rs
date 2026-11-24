// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! What the v2 entry points hand a v3 provider and what the session enforces on them: F16 bytes, compacted rows,
//! the placement epoch and the declared origins, each through `seabraid_session_submit` and a real callback.

use super::v2::{batch, collect, open_v3, provider, reference, submit, CONFIG};
use super::*;
use crate::capabilities::{FORMAT_ROUTED_BF16, FORMAT_ROUTED_F16};
use crate::ffi::batch_v2::BatchV2;
use crate::ffi::provider_v3::{ProviderV3, ACCEPTS_BF16, ACCEPTS_F16, COMPACT_ROWS, REFERENCE};
use crate::ffi::providers::PARTIAL_ROUTES;

/// What the callback saw. `halves` holds the first four 16-bit elements of the activations, exactly as given.
#[repr(C)]
#[derive(Default)]
struct Seen {
    format: u32,
    tokens: u32,
    count: usize,
    epoch: u64,
    origins: u32,
    called: u32,
    halves: [u16; 4],
}

unsafe extern "C" fn record_half(
    context: *mut c_void,
    _: *const Config,
    batch: *const BatchV2,
    output: *mut f32,
    count: usize,
) -> i32 {
    let (seen, batch) = unsafe { (&mut *context.cast::<Seen>(), &*batch) };
    let halves = batch.activations.cast::<u16>();
    seen.format = batch.activation_format;
    seen.tokens = batch.tokens;
    seen.count = batch.activation_count;
    seen.epoch = batch.placement_epoch;
    seen.origins = batch.source_modalities;
    seen.called = 1;
    let given = batch.activation_count;
    for (i, half) in seen.halves.iter_mut().enumerate().take(given) {
        *half = unsafe { *halves.add(i) };
    }
    // The output carries the raw bits so the test also checks the bytes survive a round trip to the caller.
    for (i, half) in seen.halves.iter().enumerate().take(count) {
        unsafe { *output.add(i) = f32::from(*half) };
    }
    0
}

fn callback(flags: u32, seen: &mut Seen) -> ProviderV3 {
    let mut p = provider(flags, &CONFIG, Some(record_half));
    p.context = ptr::from_mut(seen).cast();
    p
}

#[test]
fn an_accepts_f16_provider_receives_the_f16_bytes_exactly() {
    let _capture = Capture::start(Level::Off);
    let mut seen = Seen::default();
    let (rc, session) = open_v3(&callback(ACCEPTS_F16, &mut seen));
    assert_eq!(rc, 0);
    // 1.0, -2.0, the smallest subnormal and a NaN payload: a converting library would change the last two.
    let half: [u16; 4] = [0x3C00, 0xC000, 0x0001, 0x7E01];
    let input = BatchV2 {
        activation_format: FORMAT_ROUTED_F16,
        activations: half.as_ptr().cast(),
        source_modalities: 1 << 1,
        placement_epoch: 7,
        ..batch()
    };
    let (rc, ticket) = submit(session, &input);
    assert_eq!(rc, 0);
    assert_eq!(collect(session, ticket), [15360.0, 49152.0, 1.0, 32257.0]);
    assert_eq!(unsafe { seabraid_close(session) }, 0);
    assert_eq!(seen.halves, half);
    assert_eq!(
        (seen.format, seen.tokens, seen.count, seen.called),
        (FORMAT_ROUTED_F16, 2, 4, 1)
    );
    assert_eq!((seen.epoch, seen.origins), (7, 1 << 1));
}

#[test]
fn bf16_needs_its_own_flag_not_the_f16_one() {
    let _capture = Capture::start(Level::Off);
    let raw = [0u16; 4];
    let input = BatchV2 {
        activation_format: FORMAT_ROUTED_BF16,
        activations: raw.as_ptr().cast(),
        ..batch()
    };
    let mut seen = Seen::default();
    let (rc, session) = open_v3(&callback(ACCEPTS_F16, &mut seen));
    assert_eq!(rc, 0);
    assert_eq!(submit(session, &input).0, Code::Invalid as i32);
    assert_eq!(unsafe { seabraid_close(session) }, 0);
    assert_eq!(
        seen.called, 0,
        "a refused batch must not reach the provider"
    );

    let (rc, session) = open_v3(&callback(ACCEPTS_BF16, &mut seen));
    assert_eq!(rc, 0);
    let (rc, ticket) = submit(session, &input);
    assert_eq!(rc, 0);
    collect(session, ticket);
    assert_eq!(unsafe { seabraid_close(session) }, 0);
    assert_eq!((seen.format, seen.called), (FORMAT_ROUTED_BF16, 1));
}

// Rows 1 and 3 are fully skipped. Their activations are positive so the reference sum is +0.0, which is
// what scatter leaves in a skipped row; a negative one would make "bit-identical" depend on the sign of zero.
const FOUR_ACTIVATIONS: [f32; 8] = [0.5, 1.5, 2.0, 3.0, 4.5, 0.25, 6.0, 7.0];
const FOUR_IDS: [i32; 8] = [0, 2, -1, -1, 1, -1, -1, -1];
const FOUR_WEIGHTS: [f32; 8] = [0.25, 0.75, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0];

fn four_rows() -> BatchV2 {
    BatchV2 {
        tokens: 4,
        activations: FOUR_ACTIVATIONS.as_ptr().cast(),
        activation_count: 8,
        expert_ids: FOUR_IDS.as_ptr(),
        expert_id_count: 8,
        weights: FOUR_WEIGHTS.as_ptr(),
        weight_count: 8,
        ..batch()
    }
}

/// Opens, submits the four-row batch, collects the eight outputs and closes; returns the output and the status.
fn run_four_rows(provider: &ProviderV3) -> (Vec<f32>, SessionStatus) {
    let (rc, session) = open_v3(provider);
    assert_eq!(rc, 0);
    let (rc, ticket) = submit(session, &four_rows());
    assert_eq!(rc, 0);
    let mut out = vec![-1.0; 8];
    assert_eq!(
        unsafe { seabraid_collect(session, ticket, 5000, out.as_mut_ptr(), 8) },
        0
    );
    let mut status = SessionStatus {
        abi_version: 1,
        struct_size: core::mem::size_of::<SessionStatus>() as u32,
        ..SessionStatus::default()
    };
    assert_eq!(unsafe { seabraid_session_status(session, &mut status) }, 0);
    assert_eq!(unsafe { seabraid_close(session) }, 0);
    (out, status)
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn compacted_output_is_bit_identical_to_uncompacted_output() {
    let _capture = Capture::start(Level::Off);
    let plain = provider(REFERENCE | PARTIAL_ROUTES, &CONFIG, None);
    let compact = provider(REFERENCE | PARTIAL_ROUTES | COMPACT_ROWS, &CONFIG, None);
    let (plain_out, plain_status) = run_four_rows(&plain);
    let (compact_out, compact_status) = run_four_rows(&compact);
    assert_eq!(bits(&compact_out), bits(&plain_out));
    assert!(
        plain_out[0] != 0.0 && plain_out[4] != 0.0,
        "real work in rows 0 and 2"
    );
    assert_eq!(
        bits(&[plain_out[2], plain_out[3], plain_out[6], plain_out[7]]),
        [0; 4],
        "skipped rows are exactly +0.0"
    );
    // The counters are what tell a compacting session from one that ignored the flag.
    assert_eq!(
        (plain_status.rows_submitted, plain_status.rows_shipped),
        (4, 4)
    );
    assert_eq!(
        (compact_status.rows_submitted, compact_status.rows_shipped),
        (4, 2)
    );
}

#[test]
fn a_compact_rows_provider_sees_tokens_equal_to_the_kept_rows() {
    let _capture = Capture::start(Level::Off);
    let mut seen = Seen::default();
    let (_, status) = run_four_rows(&callback(PARTIAL_ROUTES | COMPACT_ROWS, &mut seen));
    assert_eq!((seen.tokens, seen.count), (2, 4), "kept rows 0 and 2 only");
    assert_eq!(status.rows_shipped, 2);
}

#[test]
fn a_stale_epoch_is_busy_or_state_and_zero_after_nonzero_is_invalid() {
    let _capture = Capture::start(Level::Off);
    let (rc, session) = open_v3(&reference());
    assert_eq!(rc, 0);
    let at = |placement_epoch| BatchV2 {
        placement_epoch,
        ..batch()
    };
    let (rc, ticket) = submit(session, &at(5));
    assert_eq!(rc, 0);
    collect(session, ticket);
    let (rc, _) = submit(session, &at(3));
    expect(
        "seabraid_session_submit",
        rc,
        Code::BusyOrState,
        "placement_epoch 3 is stale: the session already accepted 5",
    );
    let (rc, _) = submit(session, &at(0));
    expect(
        "seabraid_session_submit",
        rc,
        Code::Invalid,
        "placement_epoch 0 after epoch 5: a versioning engine must version every batch",
    );
    // Refusals changed nothing: the same epoch is still accepted and no ticket is outstanding.
    let (rc, ticket) = submit(session, &at(5));
    assert_eq!(rc, 0);
    collect(session, ticket);
    assert_eq!(unsafe { seabraid_close(session) }, 0);
}
