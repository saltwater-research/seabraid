// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! Test-only: a block of memory that ENDS at an unmapped page, so any read past it is a SIGSEGV instead of a silent
//! out-of-bounds read of heap garbage.
//!
//! WHY: the handle check reads a caller's pointer before anything proves it is a handle (R6a H1). A test that only
//! asserts the returned status cannot tell "read 8 bytes at offset 0" from "read 8 bytes at offset 400 of a
//! 232-byte allocation and happened to see garbage": both answer INVALID. Backing a fake handle with this block turns
//! the second into a crash, so the test passes only if nothing past the first bytes is read. Linux only (the
//! constants are Linux's); elsewhere the tests that use it are compiled out, and the ASan native run covers them.
#![cfg(target_os = "linux")]

use std::ffi::c_void;

extern "C" {
    fn mmap(addr: *mut c_void, len: usize, prot: i32, flags: i32, fd: i32, off: i64)
        -> *mut c_void;
    fn mprotect(addr: *mut c_void, len: usize, prot: i32) -> i32;
    fn munmap(addr: *mut c_void, len: usize) -> i32;
}

const PAGE: usize = 4096;

pub(super) struct EdgeBlock {
    base: *mut c_void,
    /// The first byte of the block; `len` bytes are readable and writable, the byte after is not mapped.
    pub(super) start: *mut u8,
}

impl EdgeBlock {
    /// A readable, writable block of `len` bytes (a multiple of 8, at most a page) whose end is the end of a
    /// mapped page; the next page is PROT_NONE.
    pub(super) fn new(len: usize) -> Self {
        assert!(len % 8 == 0 && len <= PAGE);
        // PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS
        let base = unsafe { mmap(std::ptr::null_mut(), 2 * PAGE, 3, 0x2 | 0x20, -1, 0) };
        assert_ne!(base as isize, -1, "mmap failed");
        assert_eq!(unsafe { mprotect(base.add(PAGE), PAGE, 0) }, 0);
        let start = unsafe { base.cast::<u8>().add(PAGE - len) };
        Self { base, start }
    }
}

impl Drop for EdgeBlock {
    fn drop(&mut self) {
        unsafe { munmap(self.base, 2 * PAGE) };
    }
}
