// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! The `struct_size` protocol: caller-filled reads and library-filled sizes.

use super::*;

/// A caller-filled struct at epoch 1 plus one appended field: 24 bytes, base size 16.
// WHY allow: the header fields exist for the layout; the tests write them through the raw buffer.
#[allow(dead_code)]
#[repr(C)]
#[derive(Clone, Copy)]
struct Wire {
    abi_version: u32,
    struct_size: u32,
    a: u64,
    b: u64,
}

const BASE: usize = 16;
const SIZED: Layout = Layout::Sized { base: BASE };

/// 8-aligned zeroed space for a caller struct, with the header written.
fn caller(words: usize, size: u32) -> Vec<u64> {
    let mut buffer = vec![0u64; words];
    buffer[0] = u64::from(ABI_VERSION) | (u64::from(size) << 32);
    buffer
}

fn read(buffer: &[u64]) -> Result<Wire> {
    unsafe { read_in(buffer.as_ptr().cast::<Wire>(), "wire", ABI_VERSION, SIZED) }
}

fn refusal(result: Result<Wire>) -> crate::Error {
    result.err().expect("expected a refusal")
}

#[test]
fn below_base_is_invalid() {
    assert_eq!(refusal(read(&caller(8, 8))).code(), Code::Invalid);
    assert_eq!(refusal(read(&caller(8, 0))).code(), Code::Invalid);
}

#[test]
fn exact_size_and_smaller_than_known_read_what_is_there() {
    let mut buffer = caller(4, 24);
    (buffer[1], buffer[2]) = (7, 9);
    let wire = read(&buffer).unwrap();
    assert_eq!((wire.a, wire.b), (7, 9));
    // struct_size 16 covers `a` only: `b` is in the caller's memory but outside its struct, so it reads zero.
    buffer[0] = u64::from(ABI_VERSION) | (16u64 << 32);
    let wire = read(&buffer).unwrap();
    assert_eq!((wire.a, wire.b), (7, 0));
}

#[test]
fn larger_caller_with_zero_tail_is_accepted() {
    let mut buffer = caller(8, 64);
    buffer[1] = 5;
    assert_eq!(read(&buffer).unwrap().a, 5);
}

#[test]
fn nonzero_tail_is_invalid_and_names_the_offset() {
    let mut buffer = caller(8, 64);
    buffer[5] = 1; // byte offset 40, past the known 24
    let error = refusal(read(&buffer));
    assert_eq!(error.code(), Code::Invalid);
    assert!(error.message().contains("offset 40"), "{}", error.message());
}

#[test]
fn oversize_is_invalid_even_with_a_zero_tail() {
    assert_eq!(refusal(read(&caller(160, 1032))).code(), Code::Invalid);
    assert!(read(&caller(160, 1024)).is_ok());
}

#[test]
fn struct_size_not_a_multiple_of_alignment_is_invalid() {
    assert_eq!(refusal(read(&caller(8, 28))).code(), Code::Invalid);
}

#[test]
fn misaligned_null_and_wrong_version_pointers_are_invalid() {
    let buffer = caller(8, 24);
    let bytes = buffer.as_ptr().cast::<u8>();
    let misaligned = unsafe { read_in(bytes.add(4).cast::<Wire>(), "wire", ABI_VERSION, SIZED) };
    assert_eq!(refusal(misaligned).code(), Code::Invalid);
    let null = unsafe { read_in(ptr::null::<Wire>(), "wire", ABI_VERSION, SIZED) };
    assert_eq!(refusal(null).code(), Code::Invalid);
    let wrong = unsafe {
        read_in(
            buffer.as_ptr().cast::<Wire>(),
            "wire",
            ABI_VERSION + 1,
            SIZED,
        )
    };
    assert_eq!(refusal(wrong).code(), Code::Invalid);
}

#[test]
fn fixed_layout_reads_the_whole_struct_without_a_size() {
    let mut buffer = caller(4, 0xdead); // the second u32 is padding in a v1 struct and is ignored
    buffer[2] = 3;
    let wire = unsafe {
        read_in(
            buffer.as_ptr().cast::<Wire>(),
            "wire",
            ABI_VERSION,
            Layout::Fixed,
        )
    };
    assert_eq!(wire.ok().unwrap().b, 3);
}

/// What the library fills: the 16-byte v1 struct (`BASE`) has grown by one appended field.
#[allow(dead_code)]
#[repr(C)]
#[derive(Clone, Copy)]
struct Grown {
    abi_version: u32,
    struct_size: u32,
    a: u64,
    appended: u64,
}

#[test]
fn out_size_accepts_an_old_caller_after_the_struct_grows() {
    // An old caller allocated the 16-byte v1 struct; Grown is 24 bytes now.
    let mut old = [0u64; 3];
    old[0] = u64::from(ABI_VERSION) | (16u64 << 32);
    old[2] = 0xAAAA; // memory past the caller's struct: must stay untouched
    let out = old.as_mut_ptr().cast::<Grown>();
    let size = unsafe { out_size(out, "out", BASE) }.unwrap();
    assert_eq!(size, 16);
    let value = Grown {
        abi_version: ABI_VERSION,
        struct_size: 24,
        a: 11,
        appended: 22,
    };
    unsafe { write_out(out, value, size) };
    assert_eq!((old[1], old[2]), (11, 0xAAAA));
    // The library's own size (24) must not be claimed in a 16-byte caller struct: struct_size is clamped.
    assert_eq!(
        old[0] >> 32,
        16,
        "struct_size written back to the old caller"
    );
}

#[test]
fn out_size_still_refuses_below_base_and_foreign_versions() {
    let mut small = [0u64; 3];
    small[0] = u64::from(ABI_VERSION) | (8u64 << 32);
    assert!(unsafe { out_size(small.as_ptr().cast::<Grown>(), "out", BASE) }.is_err());
    small[0] = u64::from(ABI_VERSION + 1) | (24u64 << 32);
    assert!(unsafe { out_size(small.as_ptr().cast::<Grown>(), "out", BASE) }.is_err());
    assert!(unsafe { out_size(ptr::null::<Grown>(), "out", BASE) }.is_err());
}

#[test]
fn a_newer_larger_caller_gets_only_the_known_bytes() {
    let mut big = [0u64; 6];
    big[0] = u64::from(ABI_VERSION) | (48u64 << 32);
    big[5] = 0xBBBB;
    let out = big.as_mut_ptr().cast::<Grown>();
    let size = unsafe { out_size(out, "out", BASE) }.unwrap();
    unsafe {
        write_out(
            out,
            Grown {
                abi_version: ABI_VERSION,
                struct_size: 24,
                a: 1,
                appended: 2,
            },
            size,
        )
    };
    assert_eq!((big[2], big[5]), (2, 0xBBBB));
    // A newer caller reads struct_size back to learn how much of its struct was filled: the known 24 bytes.
    assert_eq!(
        big[0] >> 32,
        24,
        "struct_size written back to the newer caller"
    );
}

#[test]
fn base_sizes_match_the_epoch_1_snapshot() {
    let snapshot = include_str!("../../layout/abi_layout.epoch1.txt");
    let size_of_row = |name: &str| -> usize {
        let row = snapshot
            .lines()
            .find_map(|line| line.strip_prefix(name)?.strip_prefix(' '));
        row.unwrap_or_else(|| panic!("epoch1 snapshot has no row {name}"))
            .trim()
            .parse()
            .unwrap()
    };
    assert_eq!(base::CAPABILITIES, size_of_row("seabraid_capabilities_v1"));
    // WHY an offset, not the struct size: session status grew by appends inside unreleased epoch 1, and its base
    // stays the size before the first appended field so a caller built against that size keeps working.
    assert_eq!(
        base::SESSION_STATUS,
        size_of_row("seabraid_session_status_v1.rows_submitted")
    );
    assert_eq!(base::TICKET, size_of_row("seabraid_placement_ticket_v1"));
    assert_eq!(base::OUTCOME, size_of_row("seabraid_placement_outcome_v1"));
    assert_eq!(base::APPLIED, size_of_row("seabraid_placement_applied_v1"));
    assert_eq!(
        base::SNAPSHOT,
        size_of_row("seabraid_placement_snapshot_v1")
    );
}
