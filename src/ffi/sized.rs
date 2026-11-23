// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! The `struct_size` protocol, both directions.
//!
//! OWNER: this module is the only place that decides how a struct that crosses the ABI is sized: the
//! structs a caller fills (`read_in`) and the structs the library fills (`out_size`, `write_out`).
//! CONTRACT: crate-internal. `read_in` returns a copy of a caller struct; `out_size` returns how many
//! bytes the caller allocated; `write_out` writes at most that many. `base` holds each library-filled
//! struct's v1 size, the smallest `struct_size` the header promises to accept (seabraid.h).
//! SIZE: under 200 lines. WHY that many: the read half and the write half are one rule in two directions and stay in
//! one file, and about half of it is the SAFETY comments that justify each raw read and write.
//! FAILURE: fail-fast `Code::Invalid`, each reason logged through `fail!` and naming the field or offset.

use super::{check_pointer, check_version};
use crate::diag::{fail, Code};
use crate::ffi::ABI_VERSION;
use crate::Result;
use std::mem::{align_of, size_of, MaybeUninit};
use std::ptr;

/// Largest `struct_size` a caller may declare. WHY a cap: the tail check reads every extra byte, so an
/// absurd size would be an unbounded read of the caller's memory.
pub(crate) const MAX_STRUCT_SIZE: usize = 1024;

/// How a caller-filled struct says how big it is.
#[derive(Clone, Copy)]
pub(crate) enum Layout {
    /// v1 layout: `abi_version` first and no `struct_size`, so the library reads `size_of::<T>()` bytes.
    Fixed,
    /// `abi_version` then `struct_size` (the first two u32s). `base` is the smallest accepted size.
    Sized { base: usize },
}

/// The v1 size of every library-filled struct. WHY one table: `out_size` must accept a caller built against
/// v1 after a struct grows, so the minimum is the frozen v1 size, never `size_of::<T>()`. The unit test
/// `base_sizes_match_the_epoch_1_snapshot` ties each number to `src/layout/abi_layout.epoch1.txt`.
pub(crate) mod base {
    pub(crate) const CAPABILITIES: usize = 24;
    pub(crate) const SESSION_STATUS: usize = 48;
    pub(crate) const TICKET: usize = 32;
    pub(crate) const OUTCOME: usize = 64;
    pub(crate) const APPLIED: usize = 32;
    pub(crate) const SNAPSHOT: usize = 40;
}

/// Copies the caller's struct at `pointer` after checking the version and, for `Layout::Sized`, the size.
///
/// Sized rule: `struct_size` is at least `base`, at most `MAX_STRUCT_SIZE` and a multiple of `align_of::<T>()`;
/// bytes past what this library knows (`size_of::<T>()`) must all be zero, so a newer caller asking for a
/// feature this library lacks fails loudly instead of being ignored; fields a smaller struct lacks read as zero.
///
/// # Safety
/// `pointer` is readable for the struct's declared size (`size_of::<T>()` when `Fixed`, `struct_size` when
/// `Sized`). `T` has `abi_version: u32` first and is valid when every byte is zero (integers, pointers, `Option<fn>`).
pub(crate) unsafe fn read_in<T: Copy>(
    pointer: *const T,
    name: &str,
    version: u32,
    layout: Layout,
) -> Result<T> {
    check_pointer(pointer, name)?;
    // SAFETY: `check_pointer` above proved `pointer` non-null and aligned for `T`, whose first field is the `u32`
    // `abi_version`; the caller promised the struct readable, so reading that one word is in bounds.
    check_version(unsafe { pointer.cast::<u32>().read() }, version, name)?;
    match layout {
        // SAFETY: `Fixed`: the caller promised `size_of::<T>()` readable bytes, the version already matched, and `T`
        // is valid for any bit pattern (`read_in`'s contract), so one `read` of it is sound.
        Layout::Fixed => Ok(unsafe { pointer.read() }),
        // SAFETY: `read_sized` has `read_in`'s contract, and the pointer and version checks above have passed.
        Layout::Sized { base } => unsafe { read_sized(pointer, name, base) },
    }
}

unsafe fn read_sized<T: Copy>(pointer: *const T, name: &str, base: usize) -> Result<T> {
    let known = size_of::<T>();
    debug_assert!((2 * size_of::<u32>()..=known).contains(&base));
    // SAFETY: `base >= 8` (asserted above), and the caller promised at least `base` readable bytes of a `Sized`
    // struct, so its second `u32`, `struct_size`, is in bounds.
    let declared = unsafe { pointer.cast::<u32>().add(1).read() } as usize;
    if declared < base || declared > MAX_STRUCT_SIZE || declared % align_of::<T>() != 0 {
        return Err(fail!(
            Code::Invalid,
            "{} struct_size {} must be in {}..={} and a multiple of {}",
            name,
            declared,
            base,
            MAX_STRUCT_SIZE,
            align_of::<T>()
        ));
    }
    if declared > known {
        // SAFETY: `known < declared <= MAX_STRUCT_SIZE` was checked above and the caller promised `declared` readable
        // bytes (its own `struct_size`), so `[known, declared)` is in bounds; `u8` has no alignment or validity
        // requirement.
        let tail = unsafe {
            std::slice::from_raw_parts(pointer.cast::<u8>().add(known), declared - known)
        };
        if let Some(at) = tail.iter().position(|&byte| byte != 0) {
            return Err(fail!(
                Code::Invalid,
                "{} sets bytes at offset {} that this library does not know (its struct is {} bytes)",
                name,
                known + at,
                known
            ));
        }
    }
    let mut value = MaybeUninit::<T>::zeroed();
    // SAFETY: `value` is a fresh local of `size_of::<T>()` bytes and `declared.min(known)` never exceeds that, the
    // caller promised `declared` readable bytes, and a local cannot overlap the caller's struct. `assume_init` is
    // sound because the value was zeroed first and `T` is valid for any bit pattern (`read_in`'s contract).
    unsafe {
        ptr::copy_nonoverlapping(
            pointer.cast::<u8>(),
            value.as_mut_ptr().cast::<u8>(),
            declared.min(known),
        );
        Ok(value.assume_init())
    }
}

/// Validates the header (`abi_version`, `struct_size`: the first two u32s) of a caller-allocated output
/// struct and returns the caller's `struct_size`. `base` is the struct's v1 size (see `mod base`). WHY both
/// fields are read from the caller's memory: the caller allocated it, so its size is the only safe bound on
/// what the library may write.
///
/// WHY the size must be a multiple of the struct's alignment, as `read_sized` requires of a caller-filled one:
/// every field is aligned to at most that, so a multiple can only fall between fields, never inside one. A size of
/// 50 for a 64-byte struct used to write 50 bytes, half of a `u64`, and report 50 as filled. Refusing is
/// the same rule in both directions and tells the caller its struct is wrong; rounding down would answer a
/// different size than the caller declared, silently. Applies to every library-filled struct and to each element
/// of an outcome array (its stride is this number).
///
/// # Safety
/// `out` is null or points to at least `base` readable bytes.
pub(crate) unsafe fn out_size<T>(out: *const T, name: &str, base: usize) -> Result<usize> {
    debug_assert!((2 * size_of::<u32>()..=size_of::<T>()).contains(&base));
    check_pointer(out, name)?;
    // SAFETY: `check_pointer` above proved `out` non-null and aligned for `T`, whose first two fields are `u32`s;
    // `out_size`'s contract is that `out` is readable for at least `base >= 8` bytes.
    let header = unsafe { out.cast::<[u32; 2]>().read() };
    check_version(header[0], ABI_VERSION, name)?;
    let declared = header[1] as usize;
    if declared < base {
        return Err(fail!(
            Code::Invalid,
            "{} struct_size {} < the v1 size {}",
            name,
            header[1],
            base
        ));
    }
    if declared % align_of::<T>() != 0 {
        return Err(fail!(
            Code::Invalid,
            "{} struct_size {} is not a multiple of {}",
            name,
            declared,
            align_of::<T>()
        ));
    }
    Ok(declared)
}

/// Writes `min(caller_size, size_of::<T>())` bytes of `value` to `out` and sets the `struct_size` field of
/// `out` (second u32) to that same count. WHY the min: a newer caller may have a larger struct whose tail this
/// library does not know and must leave untouched, and an older one has a smaller struct that must not be
/// overrun. WHY the written `struct_size`: it is the one number that tells the caller which tail fields were
/// filled, so copying the library's own size into a smaller caller struct would claim bytes it never wrote
/// (and a caller that reuses the struct would then hand the library a size beyond its allocation).
///
/// # Safety
/// `out` has `caller_size` writable bytes (checked by `out_size`, so at least the two header u32s) and does
/// not overlap `value`; `T` has `abi_version` and `struct_size` as its first two u32s.
pub(crate) unsafe fn write_out<T: Copy>(out: *mut T, value: T, caller_size: usize) {
    let bytes = caller_size.min(size_of::<T>());
    debug_assert!(bytes >= 2 * size_of::<u32>());
    // SAFETY: `write_out`'s contract: `out` has `caller_size` writable bytes (`out_size` checked the header), `bytes
    // <= caller_size`, `value` is a local that cannot overlap the caller's struct, and the `struct_size` word at
    // offset 4 lies inside the `bytes >= 8` just written.
    unsafe {
        ptr::copy_nonoverlapping(ptr::from_ref(&value).cast::<u8>(), out.cast::<u8>(), bytes);
        out.cast::<u32>().add(1).write(bytes as u32);
    };
}

#[cfg(test)]
mod tests;
