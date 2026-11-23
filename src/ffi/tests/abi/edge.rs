// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! Overnight hardening O1 at the C edge: the struct_size matrix, collect argument errors, last_error and the log
//! sink under long and multi-byte text, and the close family under contention. Case ids refer to
//! scratch/2026-11-19-overnight/O1-cases.md.

use super::super::v2::{batch, open_v3};
use super::*;
use crate::ffi::batch_v2::BatchV2;
use crate::ffi::v2::seabraid_session_submit;

const SENTINEL: u8 = 0xA5;

/// 8-aligned scratch of `words` u64s filled with a sentinel, the first two u32s set to `(version, size)`.
fn arena(words: usize, version: u32, size: u32) -> Vec<u64> {
    let mut buffer = vec![u64::from_ne_bytes([SENTINEL; 8]); words];
    buffer[0] = u64::from(version) | (u64::from(size) << 32);
    buffer
}

fn size_field(buffer: &[u64]) -> u32 {
    (buffer[0] >> 32) as u32
}

fn bytes(buffer: &[u64]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(buffer.as_ptr().cast(), buffer.len() * 8) }
}

/// Runs a library-filled call over every `struct_size` of interest and checks the one rule in seabraid.h: below
/// the v1 `base`, or not a multiple of the struct's alignment `align` (R6a Z2), is INVALID with the buffer
/// untouched; otherwise `min(size, own)` bytes are written, the sentinel beyond them survives, and the
/// `struct_size` field reads back as the byte count written.
fn check_out_matrix(call: impl Fn(*mut u64) -> i32, base: u32, own: u32, align: u32) {
    for size in [
        0,
        4,
        8,
        base - 1,
        base,
        base + 1,
        base + 4,
        base + 8,
        own,
        own + 1,
        own + 8,
        1023,
        1024,
        1025,
        4096,
        u32::MAX,
    ] {
        let mut buffer = arena(300, 1, size);
        let before = buffer.clone();
        let rc = call(buffer.as_mut_ptr());
        if size < base || size % align != 0 {
            assert_eq!(rc, Code::Invalid as i32, "struct_size {size}");
            assert_eq!(
                buffer, before,
                "a refused call writes nothing (size {size})"
            );
            continue;
        }
        assert_eq!(rc, 0, "struct_size {size}");
        let wrote = size.min(own) as usize;
        assert_eq!(
            size_field(&buffer) as usize,
            wrote,
            "struct_size {size} reads back what was written"
        );
        assert_eq!(
            &bytes(&buffer)[wrote..],
            &bytes(&before)[wrote..],
            "bytes past {wrote} are the caller's (size {size})"
        );
    }
}

// case Z1-Z17: struct_size of a library-filled struct, session status (v1 base 48, now 64).
#[test]
fn session_status_struct_size_matrix() {
    let _capture = Capture::start(Level::Off);
    let session = open_session(ptr::null());
    check_out_matrix(
        |out| unsafe { seabraid_session_status(session, out.cast()) },
        48,
        64,
        8,
    );
    assert_eq!(unsafe { seabraid_close(session) }, 0);
}

// case Z18-Z33: the same for capabilities (base 24 = current size, so nothing is ever appended yet).
#[test]
fn capabilities_struct_size_matrix() {
    let _capture = Capture::start(Level::Off);
    check_out_matrix(
        |out| unsafe { seabraid_query_capabilities(out.cast()) },
        24,
        24,
        4,
    );
}

// case Z34-Z38: a wrong abi_version, a null or misaligned out pointer, on both library-filled entry points.
#[test]
fn library_filled_structs_refuse_a_wrong_version_and_a_null_or_misaligned_pointer() {
    let _capture = Capture::start(Level::Off);
    let session = open_session(ptr::null());
    for version in [0, 2, u32::MAX] {
        let mut buffer = arena(16, version, 64);
        let before = buffer.clone();
        let rc = unsafe { seabraid_session_status(session, buffer.as_mut_ptr().cast()) };
        assert_eq!(rc, Code::Invalid as i32, "abi_version {version}");
        assert_eq!(buffer, before);
        let rc = unsafe { seabraid_query_capabilities(buffer.as_mut_ptr().cast()) };
        assert_eq!(rc, Code::Invalid as i32, "abi_version {version}");
    }
    assert_eq!(
        unsafe { seabraid_session_status(session, ptr::null_mut()) },
        Code::Invalid as i32
    );
    let mut buffer = arena(16, 1, 64);
    let skewed = unsafe { buffer.as_mut_ptr().cast::<u8>().add(4) };
    assert_eq!(
        unsafe { seabraid_session_status(session, skewed.cast()) },
        Code::Invalid as i32
    );
    assert_eq!(
        unsafe { seabraid_query_capabilities(skewed.cast()) },
        Code::Invalid as i32
    );
    assert_eq!(unsafe { seabraid_close(session) }, 0);
}

/// A caller-filled `seabraid_batch_v2` at an exact size. The 80-byte struct is followed by guard space so a
/// read past `struct_size` would be visible to a sanitizer; the library must not read it.
fn v2_with_size(size: u32, tail: Option<(usize, u8)>) -> (i32, u64) {
    let (rc, session) = open_v3(&super::super::v2::reference());
    assert_eq!(rc, 0);
    let mut words = vec![0u64; 160];
    let input = BatchV2 {
        struct_size: size,
        ..batch()
    };
    unsafe {
        ptr::copy_nonoverlapping(
            ptr::from_ref(&input).cast::<u8>(),
            words.as_mut_ptr().cast(),
            80,
        )
    };
    if let Some((at, value)) = tail {
        unsafe { words.as_mut_ptr().cast::<u8>().add(at).write(value) };
    }
    let mut ticket = 77;
    let rc = unsafe { seabraid_session_submit(session, words.as_ptr().cast(), &mut ticket) };
    assert_eq!(unsafe { seabraid_close(session) }, 0);
    (rc, ticket)
}

// case Z39-Z52: struct_size of a caller-filled struct, batch v2 (base 80, own 80, cap 1024, multiple of 8).
#[test]
fn batch_v2_struct_size_matrix() {
    let _capture = Capture::start(Level::Off);
    let invalid = Code::Invalid as i32;
    for size in [0, 4, 79, 81, 84, 87, 89, 1025, 1032, 4096, u32::MAX] {
        assert_eq!(
            v2_with_size(size, None),
            (invalid, 77),
            "struct_size {size}"
        );
    }
    for size in [80, 88, 96, 504, 1024] {
        let (rc, ticket) = v2_with_size(size, None);
        assert_eq!(
            (rc, ticket == 77),
            (0, false),
            "struct_size {size} with a zero tail"
        );
    }
    assert_eq!(
        v2_with_size(88, Some((80, 1))),
        (invalid, 77),
        "first unknown byte"
    );
    assert_eq!(
        v2_with_size(88, Some((87, 1))),
        (invalid, 77),
        "last unknown byte"
    );
    assert_eq!(
        v2_with_size(1024, Some((1023, 1))),
        (invalid, 77),
        "last byte of the cap"
    );
    assert_eq!(
        v2_with_size(88, Some((88, 1))).0,
        0,
        "a byte past struct_size is not read"
    );
}

pub(super) fn good_view<'a>(a: &'a [f32; 2], i: &'a [i32; 2], w: &'a [f32; 2]) -> BatchView {
    BatchView {
        activations: a.as_ptr(),
        expert_ids: i.as_ptr(),
        weights: w.as_ptr(),
        ..view(1, [2, 2, 2])
    }
}

// case K7-K16: output_count and pointer arguments of collect; the ticket stays collectable after every refusal.
#[test]
fn collect_with_a_wrong_count_a_null_output_or_a_wrong_ticket_keeps_the_ticket_alive() {
    let _capture = Capture::start(Level::Off);
    let session = open_session(ptr::null());
    let (a, i, w) = ([1.0f32, 2.0], [0i32, 2], [0.25f32, 0.75]);
    let mut ticket = 0u64;
    assert_eq!(
        unsafe { seabraid_submit(session, &good_view(&a, &i, &w), &mut ticket) },
        0
    );
    let mut out = [f32::from_bits(0xDEAD_BEEF); 4];
    let invalid = Code::Invalid as i32;
    for count in [0, 1, 3, 4, usize::MAX] {
        let rc = unsafe { seabraid_collect(session, ticket, 5000, out.as_mut_ptr(), count) };
        assert_eq!(rc, invalid, "output_count {count} of 2");
    }
    assert_eq!(
        unsafe { seabraid_collect(session, ticket, 5000, ptr::null_mut(), 2) },
        invalid
    );
    assert_eq!(
        unsafe {
            seabraid_collect(
                session,
                ticket,
                5000,
                out.as_mut_ptr().cast::<u8>().add(1).cast(),
                2,
            )
        },
        invalid
    );
    for wrong in [0, ticket + 1, ticket.wrapping_sub(1), u64::MAX] {
        let rc = unsafe { seabraid_collect(session, wrong, 5000, out.as_mut_ptr(), 2) };
        assert_eq!(rc, invalid, "ticket {wrong}");
    }
    assert!(
        out.iter().all(|v| v.to_bits() == 0xDEAD_BEEF),
        "a refused collect writes nothing"
    );
    assert_eq!(
        unsafe { seabraid_collect(session, ticket, 5000, out.as_mut_ptr(), 2) },
        0
    );
    assert_eq!(
        &out[..2],
        &[0.25 * 1.0 + 0.75 * 3.0, 2.0 * 0.25 + 2.0 * 0.75 * 3.0]
    );
    let busy = Code::BusyOrState as i32;
    assert_eq!(
        unsafe { seabraid_collect(session, ticket, 0, out.as_mut_ptr(), 2) },
        busy,
        "already collected"
    );
    assert_eq!(
        unsafe { seabraid_collect(session, 0, 0, out.as_mut_ptr(), 2) },
        busy,
        "nothing outstanding"
    );
    assert_eq!(unsafe { seabraid_close(session) }, 0);
}

// case K17-K18: a ticket from another session is refused, whichever session has work in flight.
#[test]
fn a_ticket_from_another_session_is_refused_on_both_sides() {
    let _capture = Capture::start(Level::Off);
    let (one, two) = (open_session(ptr::null()), open_session(ptr::null()));
    let (a, i, w) = ([1.0f32, 2.0], [0i32, 2], [0.25f32, 0.75]);
    let (mut t1, mut t2) = (0u64, 0u64);
    assert_eq!(
        unsafe { seabraid_submit(one, &good_view(&a, &i, &w), &mut t1) },
        0
    );
    let mut out = [0f32; 2];
    let rc = unsafe { seabraid_collect(two, t1, 0, out.as_mut_ptr(), 2) };
    assert_eq!(
        rc,
        Code::BusyOrState as i32,
        "session two has nothing outstanding"
    );
    assert_eq!(
        unsafe { seabraid_submit(two, &good_view(&a, &i, &w), &mut t2) },
        0
    );
    let rc = unsafe { seabraid_collect(two, t1, 0, out.as_mut_ptr(), 2) };
    assert_eq!(
        rc,
        Code::Invalid as i32,
        "session two's own ticket is another number"
    );
    assert_ne!(t1, t2);
    for (session, ticket) in [(one, t1), (two, t2)] {
        assert_eq!(
            unsafe { seabraid_collect(session, ticket, 5000, out.as_mut_ptr(), 2) },
            0
        );
        assert_eq!(unsafe { seabraid_close(session) }, 0);
    }
}
