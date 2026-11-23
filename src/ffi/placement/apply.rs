// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! `seabraid_placement_apply` and `seabraid_placement_apply_v2`: run the pending plan through the caller's provider and
//! write the outcomes.
//! WHY its own file: apply is the only placement call that does irreversible provider I/O and writes an array of
//! caller structs, so its ordering rule (every retryable argument check before the first provider call) and its
//! outcome-stride rule live together, apart from the handle and storage checks every call shares in placement.rs.
//! CONTRACT: both entries run `apply_with`, so the order of refusals is the same on both, and the v1 entry is exactly
//! the v2 one with `from_fallback_rungs` 0 and no `replace_from_fallback`.

use super::provider::{ProviderIo, ProviderV2, PROVIDER_V2_BASE_SIZE, PROVIDER_V2_VERSION};
use super::types::{Applied, Outcome};
use super::{handle, placement_error, Placement, PlacementHandle, Provider};
use crate::diag::{self, fail, Code, Level};
use crate::ffi::sized::{base, out_size, read_in, write_out, Layout::Fixed, Layout::Sized};
use crate::ffi::status_on;
use crate::ffi::ABI_VERSION;
use crate::placement::{AppliedRound, Rung};
use crate::Result;
use std::mem;

/// Finds the pending plan for `ticket`; a plan from a stale epoch is discarded. Returns the
/// number of outcomes the caller must have room for.
fn pending_promotions(placement: &mut Placement, ticket: u64) -> Result<usize> {
    let plan = placement
        .pending
        .as_ref()
        .filter(|plan| plan.round == ticket)
        .ok_or_else(|| {
            fail!(
                Code::BusyOrState,
                "ticket {} is not the pending plan",
                ticket
            )
        })?;
    let (epoch, count) = (plan.epoch(), plan.promotions().len());
    if let Err(error) = placement.check_epoch(epoch) {
        placement.pending = None;
        return Err(error);
    }
    Ok(count)
}

/// Takes the plan, so its results are single-use, and runs it through the provider.
fn run_round(placement: &mut Placement, io: &mut ProviderIo<'_>) -> Result<AppliedRound> {
    let plan = placement
        .pending
        .take()
        .ok_or_else(|| fail!(Code::BusyOrState, "pending plan vanished before apply"))?;
    let applied = placement
        .map
        .apply_plan(plan, io)
        .map_err(placement_error)?;
    // WHY not `?`: apply_plan already did irreversible provider I/O and changed the map, so failing here
    // would hide real outcomes from a caller who must know what moved. A record failure only costs
    // cooldown bookkeeping, so it is logged loudly (and trips debug builds) but the call still succeeds.
    if let Err(source) = placement.planner.record(&applied) {
        debug_assert!(false, "planner could not record the round: {source}");
        diag::emit(
            Level::Error,
            Code::Internal as i32,
            &format!("seabraid_placement_apply: planner could not record the round: {source}"),
        );
    }
    Ok(applied)
}

/// The body both entries share. `bind` reads and validates the caller's provider and runs after the output and
/// outcome checks (it gets the layer count and the fallback rung), so a provider v1 and a v2 meet the same refusals
/// in the same order, all before the first provider call.
///
/// # Safety
/// `placement`, `outcomes`, `outcome_capacity` and `out` have `seabraid_placement_apply`'s contract.
unsafe fn apply_with<'a>(
    placement: *mut PlacementHandle,
    ticket: u64,
    outcomes: *mut Outcome,
    outcome_capacity: usize,
    out: *mut Applied,
    bind: impl FnOnce(usize, Rung) -> Result<ProviderIo<'a>>,
) -> Result<()> {
    // SAFETY: `handle` validates the pointer (live, unpoisoned, not busy) before it forms a reference.
    let mut placement = unsafe { handle(placement) }?;
    let required = pending_promotions(&mut placement, ticket)?;
    // SAFETY: `out_size` validates `out` (non-null, this ABI's version, a sane size) before anything is read or written.
    let out_bytes = unsafe { out_size(out, "out", base::APPLIED) }?;
    let stride = check_outcomes(outcomes, outcome_capacity, required)?;
    let (layers, fallback) = {
        let snapshot = placement.map.snapshot();
        (snapshot.layers as usize, snapshot.fallback())
    };
    let mut io = bind(layers, fallback)?;

    // Output and callback geometry are retryable preflight errors, so they ran first.
    let applied = run_round(&mut placement, &mut io)?;
    let entries = applied.outcomes();
    for (index, entry) in entries.iter().enumerate() {
        // WHY stride: it is the caller's struct_size, so element i lands where the caller expects it.
        // SAFETY: `check_outcomes` proved `outcomes` holds at least `required` elements at `stride` bytes each, and
        // `entries.len() <= required`.
        unsafe {
            let slot = outcomes.cast::<u8>().add(index * stride).cast::<Outcome>();
            write_out(slot, Outcome::from(*entry), stride);
        }
    }
    let summary = Applied::new(placement.map.snapshot(), entries.len());
    // SAFETY: `out_bytes` is the size the caller declared for `out`, checked by `out_size` above.
    unsafe { write_out(out, summary, out_bytes) };
    Ok(())
}

#[no_mangle]
pub unsafe extern "C" fn seabraid_placement_apply(
    placement: *mut PlacementHandle,
    ticket: u64,
    provider: *const Provider,
    outcomes: *mut Outcome,
    outcome_capacity: usize,
    out: *mut Applied,
) -> i32 {
    // SAFETY: `apply_with` has this function's contract; `read_in` checks the provider's pointer and version before
    // it reads, and `bind` forms the size slice only after `input` has checked the pointer and count.
    unsafe {
        status_on("seabraid_placement_apply", placement, || {
            apply_with(
                placement,
                ticket,
                outcomes,
                outcome_capacity,
                out,
                |layers, _| {
                    let provider = read_in(provider, "provider", ABI_VERSION, Fixed)?;
                    provider.bind(layers)
                },
            )
        })
    }
}

/// `seabraid_placement_apply` with a sized provider that may also write a promoted expert by reference.
#[no_mangle]
pub unsafe extern "C" fn seabraid_placement_apply_v2(
    placement: *mut PlacementHandle,
    ticket: u64,
    provider: *const ProviderV2,
    outcomes: *mut Outcome,
    outcome_capacity: usize,
    out: *mut Applied,
) -> i32 {
    // SAFETY: as for `seabraid_placement_apply`; `read_in` additionally checks `struct_size` and that every byte past
    // this library's struct is zero before the struct is copied.
    unsafe {
        status_on("seabraid_placement_apply_v2", placement, || {
            apply_with(
                placement,
                ticket,
                outcomes,
                outcome_capacity,
                out,
                |layers, fallback| {
                    let layout = Sized {
                        base: PROVIDER_V2_BASE_SIZE,
                    };
                    let provider = read_in(provider, "provider", PROVIDER_V2_VERSION, layout)?;
                    provider.bind(layers, fallback)
                },
            )
        })
    }
}

/// Validates the outcome array and returns its element stride (the caller struct_size, read from
/// element 0). An empty plan needs no array, so `outcomes` may then be null and the stride is unused.
fn check_outcomes(pointer: *mut Outcome, capacity: usize, required: usize) -> Result<usize> {
    if capacity < required {
        return Err(fail!(
            Code::Invalid,
            "outcome_capacity {} < plan size {}",
            capacity,
            required
        ));
    }
    if required == 0 {
        return Ok(mem::size_of::<Outcome>());
    }
    // `out_size` also refuses a stride that is not a multiple of the alignment: element i sits at i * stride, and
    // a misaligned write would be undefined.
    // SAFETY: `out_size` refuses a null `pointer`, and `required > 0` with `capacity >= required` was checked above;
    // the caller promises element 0 of `outcomes` is readable for its two header words.
    let stride = unsafe { out_size(pointer, "outcomes", base::OUTCOME) }?;
    if required > isize::MAX as usize / stride {
        return Err(fail!(
            Code::Invalid,
            "outcomes count {} is too large",
            required
        ));
    }
    Ok(stride)
}
