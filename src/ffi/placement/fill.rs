// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! C entry point for the hottest-first spillover fill. Pure: it keeps no handle and runs
//! no provider work; `placement::fill` owns the algorithm.
//! FAILURE: fail-fast `Code::Invalid` (bad shape, count, null pointer or version), logged once through `status`;
//! the caller's `owners` array is written only after every check passed, so a refused call leaves it untouched.

use super::{input, placement_error, region, shape};
use crate::diag::{fail, Code};
use crate::ffi::sized::{read_in, Layout::Fixed};
use crate::ffi::status;
use crate::ffi::ABI_VERSION;
use crate::placement::{fill, Rung};
use std::ptr;

#[derive(Clone, Copy)]
#[repr(C)]
pub struct FillRequest {
    pub(super) abi_version: u32,
    pub(super) layers: u32,
    pub(super) experts: u32,
    pub(super) cards: Rung,
    pub(super) slots: *const u64,
    pub(super) slot_count: usize,
    pub(super) heat: *const f64,
    pub(super) heat_count: usize,
}

crate::layout::assert_layout!(
    FillRequest,
    "seabraid_placement_fill_v1",
    [
        abi_version,
        layers,
        experts,
        cards,
        slots,
        slot_count,
        heat,
        heat_count
    ]
);

#[no_mangle]
pub unsafe extern "C" fn seabraid_placement_fill(
    request: *const FillRequest,
    owners: *mut Rung,
    owner_count: usize,
) -> i32 {
    status("seabraid_placement_fill", || {
        // SAFETY: `read_in` checks `request` non-null and the version before reading; the caller promises
        // `size_of::<FillRequest>()` readable bytes (placement.h).
        let request = unsafe { read_in(request, "request", ABI_VERSION, Fixed) }?;
        let cells = shape(request.layers, request.experts, request.cards)?;
        if owner_count != cells {
            return Err(fail!(
                Code::Invalid,
                "owner_count {} != cells {}",
                owner_count,
                cells
            ));
        }
        // SAFETY: `input` refuses a count other than the expected one and a null or oversized region before it forms
        // the slice; the caller promises `slot_count` readable elements at `request.slots` for this call.
        let slots = unsafe {
            input(
                request.slots,
                request.slot_count,
                usize::from(request.cards),
                "slots",
            )
        }?;
        // SAFETY: `input` refuses a count other than the cell count and a null or oversized region before it forms
        // the slice; the caller promises `heat_count` readable `f64`s at `request.heat` for this call.
        let heat = unsafe { input(request.heat, request.heat_count, cells, "heat") }?;
        region(owners.cast_const(), owner_count, "owners")?;
        let layout = fill(request.layers, request.experts, request.cards, slots, heat)
            .map_err(placement_error)?;
        // `fill` returns exactly `cells` owners and the output holds `owner_count == cells`.
        // SAFETY: `region` proved `owners` non-null and `owner_count == cells` was checked above; `layout` holds
        // exactly `cells` owners and is the library's own `Vec`, so it cannot overlap the caller's array, which the
        // caller promises is writable for `owner_count` elements.
        unsafe { ptr::copy_nonoverlapping(layout.as_ptr(), owners, cells) };
        Ok(())
    })
}
