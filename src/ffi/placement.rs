// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! The placement C ABI: handle and storage checks. Bad foreign inputs fail before policy or I/O;
//! the map and planner (`crate::placement`) remain the sole owners of their respective state.
//! WHY it lives here: every `extern "C"` function and every `#[allow(unsafe_code)]` is under `ffi/`, so the
//! unsafe surface has one owner and the placement core is plain safe Rust.

mod apply;
mod fill;
mod open;
pub(crate) mod provider;
pub(crate) mod types;

#[cfg(test)]
mod tests;

use crate::diag::{fail, Code};
use crate::ffi::lease::{acquire, retire, Handle, Lease, Slot};
use crate::ffi::sized::{base, out_size, write_out};
#[cfg(test)]
use crate::ffi::ABI_VERSION;
use crate::ffi::{check_pointer, status, status_on};
use crate::gate::PLACEMENT_COOKIE;
use crate::placement::{
    layout_size, OwnershipMap, PlacementError, PlacementPlan, PlacementPlanner, Rung,
};
use crate::{Error, Result};
use provider::Provider;
use std::{mem, ptr, slice};
// WHY test-only: the tests reach the entries and the input structs through `use super::*`.
#[cfg(test)]
use open::{seabraid_placement_open, seabraid_placement_open_v2};
#[cfg(test)]
use types::{Layout, Policy, PolicyV2};
use types::{Snapshot, Ticket};

pub(crate) const UPDATE_ADOPT: u32 = 0;
pub(crate) const UPDATE_REBUILD: u32 = 1;

/// One placement's state. The handle a caller holds is `PlacementHandle`, which puts the same gate sessions use
/// (ffi::lease) in front of it: a closed or foreign pointer is INVALID and an overlapping call is BUSY_OR_STATE where
/// that is detectable (see crate::gate), and a panic's poison is recorded there.
pub struct Placement {
    map: OwnershipMap,
    planner: PlacementPlanner,
    pending: Option<PlacementPlan>,
}

/// The memory a `seabraid_placement*` points at.
pub type PlacementHandle = Slot<Placement>;

// A pointer of the wrong kind must find a cookie at offset 0 (see crate::gate); sessions assert the same in ffi::lease.
const _: () = assert!(PlacementHandle::GATE_OFFSET == 0);

impl Handle for Placement {
    const NAME: &'static str = "placement";
    const COOKIE: u64 = PLACEMENT_COOKIE;
}

/// Copies foreign input into owned storage, failing as a logged OOM instead of aborting.
fn owned<T: Clone>(items: &[T], what: &str) -> Result<Vec<T>> {
    let mut copy = Vec::new();
    copy.try_reserve_exact(items.len()).map_err(|source| {
        fail!(
            Code::Internal,
            "cannot reserve {} {} elements ({})",
            items.len(),
            what,
            source
        )
    })?;
    copy.extend_from_slice(items);
    Ok(copy)
}

impl Placement {
    fn check_epoch(&self, epoch: u64) -> Result<()> {
        self.map.check_epoch(epoch).map_err(placement_error)
    }
}

fn placement_error(error: PlacementError) -> Error {
    let code = match error {
        PlacementError::StaleEpoch { .. } | PlacementError::EpochExhausted => Code::BusyOrState,
        _ => Code::Invalid,
    };
    fail!(code, "{}", error)
}

/// The cell count of a layout, with the offending shape named when it is rejected.
fn shape(layers: u32, experts: u32, cards: Rung) -> Result<usize> {
    layout_size(layers, experts, cards).map_err(|source| {
        fail!(
            Code::Invalid,
            "layout {} layers x {} experts with {} cards is outside 1..=2^26 cells and 1..=63 cards ({})",
            layers,
            experts,
            cards,
            source
        )
    })
}

// Counts are computed from an already-checked shape before forming a slice.
unsafe fn input<'a, T>(
    pointer: *const T,
    count: usize,
    expected: usize,
    name: &str,
) -> Result<&'a [T]> {
    if count != expected {
        return Err(fail!(
            Code::Invalid,
            "{} count {} != expected {}",
            name,
            count,
            expected
        ));
    }
    region(pointer, count, name)?;
    // SAFETY: `region` just proved `pointer` non-null and `count * size_of::<T>()` within `isize::MAX`; the caller
    // promises `count` readable, initialised elements (`Rung`, `f64`, `usize`: every bit pattern is valid) that
    // nothing writes while this slice lives, which is the call.
    Ok(unsafe { slice::from_raw_parts(pointer, count) })
}

fn region<T>(pointer: *const T, count: usize, name: &str) -> Result<()> {
    check_pointer(pointer, name)?;
    if count > isize::MAX as usize / mem::size_of::<T>() {
        return Err(fail!(
            Code::Invalid,
            "{} count {} is too large",
            name,
            count
        ));
    }
    Ok(())
}

/// Leases the handle for one call. WHY the shared lease: the pointer, liveness, busy and poison refusals are
/// one rule for every handle, owned by `ffi::lease`; only `seabraid_placement_close` bypasses the poison check.
unsafe fn handle<'a>(pointer: *mut PlacementHandle) -> Result<Lease<'a, Placement>> {
    // SAFETY: `handle` passes `acquire`'s contract on unchanged: a null or open-call pointer, and calls the caller
    // serializes (a violation is detected and refused there, not made safe).
    unsafe { acquire(pointer) }
}

#[no_mangle]
pub unsafe extern "C" fn seabraid_placement_plan(
    placement: *mut PlacementHandle,
    expected_epoch: u64,
    heat: *const f64,
    heat_count: usize,
    out: *mut Ticket,
) -> i32 {
    // SAFETY: Each caller pointer is validated before use: `handle` by the lease (live, unpoisoned, not busy); `out`
    // by `out_size` (non-null, this ABI's version, a sane size) and written only through `write_out` within that
    // size; `heat` by `input` (count equal to the cell count, non-null) and only read.
    unsafe {
        status_on("seabraid_placement_plan", placement, || {
            let mut lease = handle(placement)?;
            // WHY a plain `&mut`: through the lease's DerefMut the map and the planner could not be borrowed apart.
            let placement = &mut *lease;
            placement.check_epoch(expected_epoch)?;
            let size = out_size(out, "out", base::TICKET)?;
            let snapshot = placement.map.snapshot();
            let heat = input(heat, heat_count, snapshot.owner.len(), "heat")?;
            let plan = placement
                .planner
                .plan(snapshot, heat)
                .map_err(placement_error)?;
            write_out(out, Ticket::from(&plan), size);
            // Keep just the most recent observation. Round IDs never repeat on a handle.
            placement.pending = Some(plan);
            Ok(())
        })
    }
}

#[no_mangle]
pub unsafe extern "C" fn seabraid_placement_snapshot(
    placement: *mut PlacementHandle,
    executors: *mut Rung,
    executor_count: usize,
    out: *mut Snapshot,
) -> i32 {
    // SAFETY: `handle` leases the placement. `region` proved `executors` non-null and `executor_count == cells` was
    // checked above, so the copy and the per-index writes stay inside the caller's `executor_count` `Rung`s, which it
    // promises are writable; `snapshot` is library memory and cannot overlap them. `out` is as in
    // `seabraid_placement_plan`.
    unsafe {
        status_on("seabraid_placement_snapshot", placement, || {
            let placement = handle(placement)?;
            let snapshot = placement.map.snapshot();
            if executor_count != snapshot.owner.len() {
                return Err(fail!(
                    Code::Invalid,
                    "executor_count {} != cells {}",
                    executor_count,
                    snapshot.owner.len()
                ));
            }
            let size = out_size(out, "out", base::SNAPSHOT)?;
            region(executors, executor_count, "executors")?;
            // Without quarantine, committed owners are the authoritative executors.
            if snapshot.quarantined == 0 {
                ptr::copy_nonoverlapping(snapshot.owner.as_ptr(), executors, executor_count);
            } else {
                for (index, &owner) in snapshot.owner.iter().enumerate() {
                    executors
                        .add(index)
                        .write(snapshot.executor_for_owner(owner));
                }
            }
            write_out(out, Snapshot::from(snapshot), size);
            Ok(())
        })
    }
}

#[no_mangle]
pub unsafe extern "C" fn seabraid_placement_quarantine(
    placement: *mut PlacementHandle,
    expected_epoch: u64,
    rung: Rung,
) -> i32 {
    // SAFETY: The only caller pointer is `placement`, leased by `handle` (live, unpoisoned, not busy); no other
    // caller memory is touched.
    unsafe {
        status_on("seabraid_placement_quarantine", placement, || {
            let mut placement = handle(placement)?;
            placement.check_epoch(expected_epoch)?;
            placement
                .map
                .quarantine(rung)
                .map_err(|source| match source {
                    // Only an out-of-range rung is "not a card"; any other failure (epoch exhaustion) is reported as itself.
                    PlacementError::InvalidLayout => fail!(
                        Code::Invalid,
                        "rung {} is not a card, fallback is rung {} ({})",
                        rung,
                        placement.map.snapshot().fallback(),
                        source
                    ),
                    other => placement_error(other),
                })
        })
    }
}

#[no_mangle]
pub unsafe extern "C" fn seabraid_placement_update(
    placement: *mut PlacementHandle,
    expected_epoch: u64,
    mode: u32,
    owners: *const Rung,
    owner_count: usize,
) -> i32 {
    // SAFETY: `handle` leases the placement; `input` checks the `owners` count against the cell count and the pointer
    // for null, and the caller promises that many readable `Rung`s.
    unsafe {
        status_on("seabraid_placement_update", placement, || {
            let mut placement = handle(placement)?;
            placement.check_epoch(expected_epoch)?;
            let update = match mode {
                UPDATE_ADOPT => OwnershipMap::adopt,
                UPDATE_REBUILD => OwnershipMap::rebuild,
                _ => return Err(fail!(Code::Invalid, "update mode {} is not 0 or 1", mode)),
            };
            let cells = placement.map.snapshot().owner.len();
            let owners = input(owners, owner_count, cells, "owners")?;
            update(&mut placement.map, expected_epoch, owned(owners, "owners")?)
                .map_err(placement_error)
        })
    }
}

#[no_mangle]
pub unsafe extern "C" fn seabraid_placement_close(placement: *mut PlacementHandle) -> i32 {
    status("seabraid_placement_close", || {
        // Provider state is call-scoped, so destruction cannot call a provider. `retire` refuses a dead or busy
        // handle and kills the cookie, so a second close returns INVALID instead of freeing twice.
        // SAFETY: `retire` reads only the gate, through a pointer it checks for null and alignment first; on `Ok`
        // this call owns the handle and drops it here, its one free (a second close finds the dead cookie and is
        // refused).
        drop(unsafe { retire(placement) }?);
        Ok(())
    })
}
