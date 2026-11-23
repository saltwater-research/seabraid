// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! Compile-time proof that every `repr(C)` ABI struct matches `src/layout/abi_layout.txt`.
//!
//! OWNER: the table file owns every size and offset. This module only reads it at compile time, so a
//! second copy of a number cannot exist in Rust. The C compiler checks the same table against the
//! headers (CMake generates `_Static_assert`s from it) and the Python tests check it against ctypes.
//! WHY a const parser and not a build script: a build script would have to emit the numbers again,
//! which is a second copy; `include_bytes!` plus a small const fn keeps the table the only copy.
//! FAILURE: a mismatch, a missing row, or a row for a field the macro does not list stops the build.
//! The asserts are 64-bit only: the table records LP64/LLP64 layouts and 32-bit targets differ.
//! SIZE: under 120 lines.

// WHY: rustc before 1.80 does not count a use inside a `const _` item, so the helpers look dead on the MSRV.
#![allow(dead_code)]

#[cfg(target_pointer_width = "64")]
const TABLE: &[u8] = include_bytes!("layout/abi_layout.txt");

/// True when the line `table[start..end]` begins with `key` followed by the byte `after`.
#[cfg(target_pointer_width = "64")]
const fn has_key(table: &[u8], start: usize, end: usize, key: &[u8], after: u8) -> bool {
    if end - start <= key.len() || table[start + key.len()] != after {
        return false;
    }
    let mut i = 0;
    while i < key.len() {
        if table[start + i] != key[i] {
            return false;
        }
        i += 1;
    }
    true
}

/// Parses the decimal digits in `table[from..end]`; anything else stops the build.
#[cfg(target_pointer_width = "64")]
const fn number(table: &[u8], from: usize, end: usize) -> usize {
    assert!(from < end, "abi_layout.txt row has no number");
    let (mut value, mut i) = (0, from);
    while i < end {
        let digit = table[i];
        assert!(
            digit >= b'0' && digit <= b'9',
            "abi_layout.txt number is not decimal"
        );
        value = value * 10 + (digit - b'0') as usize;
        i += 1;
    }
    value
}

/// Index just past the end of the line that starts at `start`.
#[cfg(target_pointer_width = "64")]
const fn line_end(table: &[u8], start: usize) -> usize {
    let mut end = start;
    // WHY '\r' too: a Windows checkout may hold CRLF, and a stray CR must not look like a bad digit.
    while end < table.len() && table[end] != b'\n' && table[end] != b'\r' {
        end += 1;
    }
    end
}

/// The number on the row `key value` (a size) or `key.field value` (an offset). A missing row stops the build.
#[cfg(target_pointer_width = "64")]
pub(crate) const fn lookup(key: &str) -> usize {
    let (key, table) = (key.as_bytes(), TABLE);
    let mut start = 0;
    while start < table.len() {
        let end = line_end(table, start);
        if table[start] != b'#' && has_key(table, start, end, key, b' ') {
            return number(table, start + key.len() + 1, end);
        }
        start = end + 1;
    }
    panic!("abi_layout.txt has no row for a struct or field the code asserts");
}

/// How many `prefix.field` rows the table holds, so a field the macro forgot is caught too.
#[cfg(target_pointer_width = "64")]
pub(crate) const fn field_count(prefix: &str) -> usize {
    let (prefix, table) = (prefix.as_bytes(), TABLE);
    let (mut start, mut count) = (0, 0);
    while start < table.len() {
        let end = line_end(table, start);
        if table[start] != b'#' && has_key(table, start, end, prefix, b'.') {
            count += 1;
        }
        start = end + 1;
    }
    count
}

/// `assert_layout!(RustType, "c_name", [field, ...])` asserts the size, every offset, and that the
/// table lists exactly these fields. Costs nothing at run time.
macro_rules! assert_layout {
    ($rust:ty, $c:literal, [$($field:ident),+ $(,)?]) => {
        #[cfg(target_pointer_width = "64")]
        const _: () = {
            assert!(
                core::mem::size_of::<$rust>() == $crate::layout::lookup($c),
                concat!($c, ": size differs from src/layout/abi_layout.txt")
            );
            $(
                assert!(
                    core::mem::offset_of!($rust, $field)
                        == $crate::layout::lookup(concat!($c, ".", stringify!($field))),
                    concat!($c, ".", stringify!($field), ": offset differs from src/layout/abi_layout.txt")
                );
            )+
            assert!(
                $crate::layout::field_count($c) == [$(stringify!($field)),+].len(),
                concat!($c, ": the table and the Rust struct list different fields")
            );
        };
    };
}
pub(crate) use assert_layout;
