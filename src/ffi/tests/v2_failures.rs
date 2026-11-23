// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! Every validation branch of `seabraid_session_open` and `seabraid_session_submit` with the reason it logs.

use super::v2::{batch, collect, open_v3, provider, record, reference, submit, CONFIG};
use super::*;
use crate::ffi::batch_v2::BatchV2;
use crate::ffi::provider_v3::{ProviderV3, REFERENCE};
use crate::ffi::v2::{seabraid_session_open, seabraid_session_submit};

#[test]
fn open_failure_table() {
    let _capture = Capture::start(Level::Off);
    let e = "seabraid_session_open";
    let bad_config = Config { top_k: 4, ..CONFIG };
    let cases: Vec<(ProviderV3, &str)> = vec![
        (
            ProviderV3 {
                abi_version: 9,
                ..reference()
            },
            "provider abi_version 9 != 3",
        ),
        (
            ProviderV3 {
                struct_size: 8,
                ..reference()
            },
            "provider struct_size 8 must be in 40..=1024 and a multiple of 8",
        ),
        (
            ProviderV3 {
                operation_count: 0,
                ..reference()
            },
            "provider.operation_count must be at least 1",
        ),
        (
            ProviderV3 {
                flags: REFERENCE | 1 << 9,
                ..reference()
            },
            "provider.flags 0x210 sets unknown flags 0x200",
        ),
        (
            provider(2, &CONFIG, Some(record)),
            "provider.flags COMPACT_ROWS requires PARTIAL_ROUTES",
        ),
        (
            provider(REFERENCE, &CONFIG, Some(record)),
            "provider.flags REFERENCE requires null execute and context and no ACCEPTS_* flag",
        ),
        (
            provider(REFERENCE | 4, &CONFIG, None),
            "provider.flags REFERENCE requires null execute and context and no ACCEPTS_* flag",
        ),
        (
            provider(0, &CONFIG, None),
            "provider.execute is null (set REFERENCE for the built-in reference transform)",
        ),
        (
            ProviderV3 {
                config: ptr::null(),
                ..reference()
            },
            "provider.config is null",
        ),
        (
            provider(REFERENCE, &bad_config, None),
            "config top_k 4 must not exceed experts 3 (experts >= 1)",
        ),
    ];
    for (descriptor, text) in cases {
        let (rc, session) = open_v3(&descriptor);
        expect(e, rc, Code::Invalid, text);
        assert!(session.is_null(), "{text}");
    }
    let old = Config {
        abi_version: 2,
        ..CONFIG
    };
    let (rc, session) = open_v3(&provider(REFERENCE, &old, None));
    expect(e, rc, Code::Invalid, "provider.config abi_version 2 != 1");
    assert!(session.is_null());
    let mut out = ptr::null_mut();
    expect(
        e,
        unsafe { seabraid_session_open(ptr::null(), &mut out) },
        Code::Invalid,
        "provider is null",
    );
    expect(
        e,
        unsafe { seabraid_session_open(&reference(), ptr::null_mut()) },
        Code::Invalid,
        "out is null",
    );
}

#[test]
fn submit_failure_table() {
    let _capture = Capture::start(Level::Off);
    let e = "seabraid_session_submit";
    let (rc, session) = open_v3(&ProviderV3 {
        operation_count: 2,
        ..reference()
    });
    assert_eq!(rc, 0);
    // WHY u32 storage: one byte past a 4-aligned base is misaligned for every allocator and under Miri; one byte past a
    // u8 array's base is misaligned only by luck (Miri placed one where the +1 was 4-aligned, and the case failed).
    let odd = [0u32; 5];
    let misaligned = unsafe { odd.as_ptr().cast::<u8>().add(1) };
    let cases: Vec<(BatchV2, Code, &str)> = vec![
        (
            BatchV2 {
                abi_version: 1,
                ..batch()
            },
            Code::Invalid,
            "batch abi_version 1 != 2",
        ),
        (
            BatchV2 {
                struct_size: 72,
                ..batch()
            },
            Code::Invalid,
            "batch struct_size 72 must be in 80..=1024 and a multiple of 8",
        ),
        (
            BatchV2 {
                operation_id: 2,
                ..batch()
            },
            Code::Invalid,
            "operation_id 2 >= operation_count 2",
        ),
        (
            BatchV2 {
                source_modalities: 1 << 9,
                ..batch()
            },
            Code::Invalid,
            "source_modalities 0x200 sets unknown flags 0x200",
        ),
        (
            BatchV2 {
                activation_format: 2,
                ..batch()
            },
            Code::Invalid,
            "batch.activation_format 2 is not an activation format (1 F32, 3 F16, 4 BF16)",
        ),
        (
            BatchV2 {
                tokens: 0,
                ..batch()
            },
            Code::Invalid,
            "tokens 0 outside 1..=4",
        ),
        (
            BatchV2 {
                tokens: 5,
                ..batch()
            },
            Code::Invalid,
            "tokens 5 outside 1..=4",
        ),
        (
            BatchV2 {
                activation_count: 3,
                ..batch()
            },
            Code::Invalid,
            "activation_count 3 != tokens 2 * hidden 2",
        ),
        (
            BatchV2 {
                weight_count: 3,
                ..batch()
            },
            Code::Invalid,
            "expert_id_count 4 and weight_count 3 must both equal tokens 2 * top_k 2",
        ),
        (
            BatchV2 {
                activations: ptr::null(),
                ..batch()
            },
            Code::Invalid,
            "batch.activations is null",
        ),
        (
            BatchV2 {
                activations: misaligned.cast(),
                ..batch()
            },
            Code::Invalid,
            "batch.activations is misaligned for 4-byte elements",
        ),
        (
            BatchV2 {
                expert_ids: ptr::null(),
                ..batch()
            },
            Code::Invalid,
            "batch.expert_ids is null",
        ),
        (
            BatchV2 {
                weights: ptr::null(),
                ..batch()
            },
            Code::Invalid,
            "batch.weights is null",
        ),
    ];
    for (input, code, text) in cases {
        let mut ticket = 77;
        let rc = unsafe { seabraid_session_submit(session, &input, &mut ticket) };
        expect(e, rc, code, text);
        assert_eq!(ticket, 77, "{text}");
    }
    let bad_pick = [0, 5, 0, 0];
    let rc = unsafe {
        seabraid_session_submit(
            session,
            &BatchV2 {
                expert_ids: bad_pick.as_ptr(),
                ..batch()
            },
            &mut 0,
        )
    };
    assert_eq!(rc, Code::Invalid as i32);
    assert!(
        diag::last().starts_with(&format!("{e}: pick 1: expert id 5 outside 0..3")),
        "{}",
        diag::last()
    );
    let rc = unsafe { seabraid_session_submit(session, ptr::null(), &mut 0) };
    expect(e, rc, Code::Invalid, "batch is null");
    let rc = unsafe { seabraid_session_submit(session, &batch(), ptr::null_mut()) };
    expect(e, rc, Code::Invalid, "out_ticket is null");
    let rc = unsafe { seabraid_session_submit(ptr::null_mut(), &batch(), &mut 0) };
    expect(e, rc, Code::Invalid, "session is null");
    let (rc, ticket) = submit(session, &batch());
    assert_eq!(rc, 0);
    let rc = unsafe { seabraid_session_submit(session, &batch(), &mut 0) };
    assert_eq!(rc, Code::BusyOrState as i32);
    assert!(
        diag::last().contains("still outstanding"),
        "{}",
        diag::last()
    );
    collect(session, ticket);
    assert_eq!(unsafe { seabraid_close(session) }, 0);
}
