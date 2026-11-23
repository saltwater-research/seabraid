// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! Overnight hardening O1, rows G* and T* at the C edge: extreme and degenerate configs at open/prepare, and
//! extreme token counts at submit, none of which may allocate by the caller's numbers or touch their pointers.

use super::super::v2::{open_v3, provider, submit};
use super::*;
use crate::ffi::batch_v2::BatchV2;
use crate::ffi::provider_v3::REFERENCE;

const INVALID: i32 = Code::Invalid as i32;

fn cfg(hidden: u32, experts: u32, max_tokens: u32, top_k: u32) -> Config {
    Config {
        abi_version: 1,
        hidden,
        experts,
        max_tokens,
        top_k,
    }
}

// case G7-G14: the same config refusals through seabraid_session_open and seabraid_prepare; *out stays NULL.
#[test]
fn degenerate_configs_are_invalid_at_open_and_at_prepare_and_extreme_ones_open() {
    let _capture = Capture::start(Level::Off);
    let bad = [
        cfg(0, 3, 4, 2),
        cfg(2, 3, 0, 2),
        cfg(2, 3, 4, 0),
        cfg(2, 0, 4, 0),
        cfg(2, 3, 4, 4),
        cfg(u32::MAX, 3, 1 << 30, 2),
        cfg(2, u32::MAX, 1 << 30, u32::MAX),
        cfg(u32::MAX, u32::MAX, u32::MAX, u32::MAX),
    ];
    for config in &bad {
        let descriptor = provider(REFERENCE, config, None);
        let (rc, session) = open_v3(&descriptor);
        assert_eq!((rc, session.is_null()), (INVALID, true), "{config:?}");
        let mut v1 = ptr::null_mut();
        assert_eq!(unsafe { seabraid_open_reference(&mut v1) }, 0);
        assert_eq!(
            unsafe { seabraid_prepare(v1, config) },
            INVALID,
            "{config:?}"
        );
        assert_eq!(unsafe { seabraid_close(v1) }, 0);
    }
    for config in [
        cfg(1, 1, 1, 1),
        cfg(u32::MAX, 3, 1 << 29, 2),
        cfg(2, u32::MAX, 1, u32::MAX),
    ] {
        let descriptor = provider(REFERENCE, &config, None);
        let (rc, session) = open_v3(&descriptor);
        assert_eq!((rc, session.is_null()), (0, false), "{config:?}");
        assert_eq!(unsafe { seabraid_close(session) }, 0);
    }
}

// case T6-T12: tokens 0, max + 1 and u32::MAX with counts that match them, and null arrays. The refusal must come
// from the token range, before any pointer is looked at and before anything is sized by the count.
#[test]
fn tokens_out_of_range_are_refused_before_any_array_is_touched() {
    let _capture = Capture::start(Level::Off);
    let config = cfg(u32::MAX, 3, 1, 2);
    let descriptor = provider(REFERENCE, &config, None);
    let (rc, session) = open_v3(&descriptor);
    assert_eq!(rc, 0);
    for tokens in [0, 2, u32::MAX] {
        let input = BatchV2 {
            tokens,
            activation_count: tokens as usize * u32::MAX as usize,
            expert_id_count: tokens as usize * 2,
            weight_count: tokens as usize * 2,
            activations: ptr::null(),
            expert_ids: ptr::null(),
            weights: ptr::null(),
            ..super::super::v2::batch()
        };
        let (rc, ticket) = submit(session, &input);
        assert_eq!((rc, ticket), (INVALID, 77), "tokens {tokens}");
        assert!(diag::last().contains("tokens"), "{}", diag::last());
    }
    // tokens == max with honest counts but null arrays: now the pointer is what is refused, still without sizing.
    let input = BatchV2 {
        tokens: 1,
        activation_count: u32::MAX as usize,
        expert_id_count: 2,
        weight_count: 2,
        activations: ptr::null(),
        expert_ids: ptr::null(),
        weights: ptr::null(),
        ..super::super::v2::batch()
    };
    let (rc, _) = submit(session, &input);
    assert_eq!(rc, INVALID);
    assert!(diag::last().contains("null"), "{}", diag::last());
    assert_eq!(unsafe { seabraid_close(session) }, 0);
}
