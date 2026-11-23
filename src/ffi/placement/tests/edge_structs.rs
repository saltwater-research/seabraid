// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! Library-filled struct layout through the placement entry points: padding bytes, a `struct_size` that splits a
//! field, and an outcome array reused with a wider stride (R6a Z1, Z2, Z3).
//! WHY its own file: these tests share one set of buffer helpers (`filled`, `as_bytes`, `scribble`) and none of
//! edge_slots.rs's scripted-provider machinery, so the seam is "what the library writes", not "what the provider does".

use super::edge::INVALID;
use super::*;
use crate::diag::testing::Capture;
use crate::diag::Level;

/// 16 KiB of recognisable junk on the stack, so a library that copies uninitialised padding shows it.
#[inline(never)]
fn scribble() {
    let mut junk = [0u8; 16384];
    for (index, byte) in junk.iter_mut().enumerate() {
        *byte = 0xC0 | (index as u8 & 7);
    }
    std::hint::black_box(&junk);
}

const FILL: u64 = 0xAAAA_AAAA_AAAA_AAAA;

fn filled(words: usize, version: u32, size: u32) -> Vec<u64> {
    let mut buffer = vec![FILL; words];
    buffer[0] = u64::from(version) | (u64::from(size) << 32);
    buffer
}

fn as_bytes(words: &[u64]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(words.as_ptr().cast(), words.len() * 8) }
}

/// R6a Z1: `write_out` copied the library's own struct, padding included, into the caller's memory: stack garbage
/// (measured: c1 c2 .. c7 in the 7 tail bytes of a snapshot). Every padding byte of every library-filled struct
/// the placement calls write must read zero, while the caller's fill beyond `struct_size` is untouched.
#[test]
fn padding_bytes_of_library_filled_structs_are_zero_not_stack_garbage() {
    let _capture = Capture::start(Level::Off);
    // two layers so the plan holds two swaps and the outcome array has two elements
    let owners = [0u8, 1, 0, 1];
    let layout = Layout {
        abi_version: ABI_VERSION,
        layers: 2,
        experts: 2,
        cards: 2,
        owners: owners.as_ptr(),
        owner_count: 4,
    };
    let mut handle = ptr::null_mut();
    assert_eq!(
        unsafe { seabraid_placement_open(&layout, ptr::null(), &mut handle) },
        0
    );

    scribble();
    let mut ticket = filled(4, ABI_VERSION, 32);
    let heat = [1.0, 20.0, 1.0, 20.0];
    let rc =
        unsafe { seabraid_placement_plan(handle, 1, heat.as_ptr(), 4, ticket.as_mut_ptr().cast()) };
    assert_eq!(rc, 0);
    assert_eq!(ticket[1], 1, "ticket number");
    assert_eq!(ticket[3], 2, "two promotions");

    let mut snapshot = filled(6, ABI_VERSION, 40);
    let mut executors = [9u8; 4];
    scribble();
    let rc = unsafe {
        seabraid_placement_snapshot(
            handle,
            executors.as_mut_ptr(),
            4,
            snapshot.as_mut_ptr().cast(),
        )
    };
    assert_eq!(rc, 0);
    assert!(
        as_bytes(&snapshot)[33..40].iter().all(|&b| b == 0),
        "snapshot tail padding: {:02x?}",
        &as_bytes(&snapshot)[32..40]
    );
    assert_eq!(snapshot[5], FILL, "past struct_size is the caller's");

    let slot = [8usize, 8];
    let io = RawIo {
        abi_version: ABI_VERSION,
        context: ptr::null_mut(),
        layer_slot_bytes: slot.as_ptr(),
        layer_count: 2,
        stage: Some(stage_ok),
        replace: Some(replace_ok),
    };
    let mut outcomes = vec![FILL; 16]; // two 64-byte elements
    outcomes[0] = u64::from(ABI_VERSION) | (64u64 << 32);
    outcomes[8] = u64::from(ABI_VERSION) | (64u64 << 32);
    let mut applied = filled(4, ABI_VERSION, 32);
    scribble();
    let rc = unsafe {
        seabraid_placement_apply(
            handle,
            ticket[1],
            ptr::from_ref(&io).cast(),
            outcomes.as_mut_ptr().cast(),
            2,
            applied.as_mut_ptr().cast(),
        )
    };
    assert_eq!(rc, 0);
    assert_eq!(applied[3], 2, "two outcomes");
    for element in 0..2 {
        let bytes = &as_bytes(&outcomes)[element * 64..(element + 1) * 64];
        // initial.rung at 40, restore.rung at 52, upper_rung at 56 are fields; everything after each is padding
        for pad in [41..44, 53..56, 57..64] {
            assert!(
                bytes[pad.clone()].iter().all(|&b| b == 0),
                "outcome {element} bytes {pad:?}: {:02x?}",
                &bytes[pad.clone()]
            );
        }
    }
    assert_eq!(unsafe { seabraid_placement_close(handle) }, 0);
}

/// R6a Z2: a `struct_size` that is not a multiple of the struct's alignment is refused for a library-filled struct,
/// as it is for a caller-filled one: 50 bytes of a 64-byte status would have been half of a `u64`.
#[test]
fn a_struct_size_that_splits_a_field_is_refused_for_every_library_filled_struct() {
    let _capture = Capture::start(Level::Off);
    let placement = TestPlacement::new();
    let heat = [1.0, 20.0];
    for size in [33u32, 36, 39, 41, 44, 50] {
        let mut ticket = filled(4, ABI_VERSION, size);
        let before = ticket.clone();
        let rc = unsafe {
            seabraid_placement_plan(placement.0, 1, heat.as_ptr(), 2, ticket.as_mut_ptr().cast())
        };
        assert_eq!(rc, INVALID, "ticket struct_size {size}");
        assert_eq!(
            diag::last(),
            format!("seabraid_placement_plan: out struct_size {size} is not a multiple of 8")
        );
        assert_eq!(ticket, before, "a refused call writes nothing");
    }
}

/// R6a Z3: in an outcome array the stride is element 0's `struct_size`, and the library rewrites that field (like
/// every library-filled struct) with the bytes it wrote per element. A caller whose struct is wider than the
/// library's therefore sees 64, not its own 72, after a call and must set it again before reusing the array; the
/// bytes past 64 of each element are never touched. This test pins that documented rule (placement.h, ABI.md).
#[test]
fn an_outcome_array_with_a_wider_stride_has_its_struct_size_reset_to_what_was_written() {
    let _capture = Capture::start(Level::Off);
    let owners = [0u8, 1, 0, 1];
    let layout = Layout {
        abi_version: ABI_VERSION,
        layers: 2,
        experts: 2,
        cards: 2,
        owners: owners.as_ptr(),
        owner_count: 4,
    };
    let mut handle = ptr::null_mut();
    assert_eq!(
        unsafe { seabraid_placement_open(&layout, ptr::null(), &mut handle) },
        0
    );
    let mut ticket = filled(4, ABI_VERSION, 32);
    let heat = [1.0, 20.0, 1.0, 20.0];
    assert_eq!(
        unsafe { seabraid_placement_plan(handle, 1, heat.as_ptr(), 4, ticket.as_mut_ptr().cast()) },
        0
    );
    let slot = [8usize, 8];
    let io = RawIo {
        abi_version: ABI_VERSION,
        context: ptr::null_mut(),
        layer_slot_bytes: slot.as_ptr(),
        layer_count: 2,
        stage: Some(stage_ok),
        replace: Some(replace_ok),
    };
    let mut array = vec![0xEEEE_EEEE_EEEE_EEEEu64; 18]; // two 72-byte elements
    array[0] = u64::from(ABI_VERSION) | (72u64 << 32);
    array[9] = u64::from(ABI_VERSION) | (72u64 << 32);
    let mut applied = filled(4, ABI_VERSION, 32);
    let rc = unsafe {
        seabraid_placement_apply(
            handle,
            ticket[1],
            ptr::from_ref(&io).cast(),
            array.as_mut_ptr().cast(),
            2,
            applied.as_mut_ptr().cast(),
        )
    };
    assert_eq!(rc, 0);
    let size_of = |element: usize| {
        (as_bytes(&array)[element * 72 + 4..element * 72 + 8])
            .iter()
            .rev()
            .fold(0u32, |acc, &b| acc << 8 | u32::from(b))
    };
    assert_eq!(
        (size_of(0), size_of(1)),
        (64, 64),
        "both elements now say what the library wrote"
    );
    for element in 0..2 {
        assert!(
            as_bytes(&array)[element * 72 + 64..(element + 1) * 72]
                .iter()
                .all(|&b| b == 0xEE),
            "bytes past 64 of element {element} are the caller's"
        );
    }
    assert_eq!(unsafe { seabraid_placement_close(handle) }, 0);
}
