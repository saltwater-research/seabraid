// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! `seabraid_placement_open` and `seabraid_placement_open_v2`: copy the layout and policy into a new handle.
//! OWNER: this file owns how a placement is opened; the planner and the map own what the policy means.
//! CONTRACT: both entries run `open_with`, so the order of refusals is the same on both (out, layout, shape, policy,
//! owners). The v1 entry is exactly the v2 one with `fallback_boundary` 0 and `max_hops_per_round` 1.
//! FAILURE: fail-fast `Code::Invalid`, `*out` NULL on every failure.

use super::types::{Layout, Policy, PolicyV2, POLICY_V2_BASE_SIZE, POLICY_V2_VERSION};
use super::{input, owned, placement_error, shape, Placement, PlacementHandle};
use crate::diag::{fail, Code};
use crate::ffi::lease::open;
use crate::ffi::sized::{read_in, Layout::Fixed, Layout::Sized};
use crate::ffi::{check_pointer, status, ABI_VERSION};
use crate::placement::{OwnershipMap, PlacementPlanner, PlannerConfig};
use crate::Result;
use std::ptr;

/// A NULL policy means the planner defaults; otherwise the versioned struct is validated and converted.
unsafe fn policy_config(optional_policy: *const Policy) -> Result<PlannerConfig> {
    if optional_policy.is_null() {
        return Ok(Default::default());
    }
    // SAFETY: `optional_policy` is non-null (tested above); `read_in` checks the version, and the caller promises
    // `size_of::<Policy>()` readable bytes.
    Ok(unsafe { read_in(optional_policy, "policy", ABI_VERSION, Fixed) }?.into())
}

/// The v2 policy: NULL is the v1 defaults; otherwise `struct_size` and the zero tail are checked by `read_in`.
unsafe fn policy_v2_config(optional_policy: *const PolicyV2) -> Result<PlannerConfig> {
    if optional_policy.is_null() {
        return Ok(Default::default());
    }
    let layout = Sized {
        base: POLICY_V2_BASE_SIZE,
    };
    // SAFETY: `optional_policy` is non-null (tested above); `read_in` checks the version, the declared size and that
    // every byte past this library's struct is zero, and the caller promises `struct_size` readable bytes.
    unsafe { read_in(optional_policy, "policy", POLICY_V2_VERSION, layout) }?.config()
}

/// WHY a check of its own: with one card the fallback boundary would be the lead with the fallback, which stays the
/// engine's adaptive cache, so the policy and the layout are refused together and the message names both.
fn check_fallback_cards(config: PlannerConfig, cards: u8) -> Result<()> {
    if config.fallback_boundary && cards < 2 {
        return Err(fail!(
            Code::Invalid,
            "policy.fallback_boundary 1 needs cards >= 2 (the lead with the fallback stays the engine's), the layout has {}",
            cards
        ));
    }
    Ok(())
}

fn open_map(layout: &Layout, owners: Vec<u8>, config: PlannerConfig) -> Result<OwnershipMap> {
    let mut map = OwnershipMap::new(layout.layers, layout.experts, layout.cards, owners)
        .map_err(placement_error)?;
    if config.fallback_boundary {
        map.enable_fallback_boundary().map_err(placement_error)?;
    }
    Ok(map)
}

/// Shared body. `load_policy` runs after the layout is read and its shape checked.
unsafe fn open_with(
    layout: *const Layout,
    out: *mut *mut PlacementHandle,
    load_policy: impl FnOnce() -> Result<PlannerConfig>,
) -> Result<()> {
    check_pointer(out, "out")?;
    // Like seabraid_open: *out is NULL on every failure, so a caller cannot close a stale pointer.
    // SAFETY: `check_pointer` proved `out` non-null and aligned for a pointer write; the caller promises it is
    // writable.
    unsafe { out.write(ptr::null_mut()) };
    // SAFETY: `read_in` checks `layout` non-null and the version before reading; the caller promises
    // `size_of::<Layout>()` readable bytes.
    let layout = unsafe { read_in(layout, "layout", ABI_VERSION, Fixed) }?;
    let count = shape(layout.layers, layout.experts, layout.cards)?;
    let config = load_policy()?;
    check_fallback_cards(config, layout.cards)?;
    let planner = PlacementPlanner::new(config).map_err(placement_error)?;
    // SAFETY: `input` refuses a count other than the cell count and a null or oversized region; the caller
    // promises `owner_count` readable `Rung`s at `layout.owners`.
    let owners = unsafe { input(layout.owners, layout.owner_count, count, "owners") }?;
    let map = open_map(&layout, owned(owners, "owners")?, config)?;
    let placement = Placement {
        map,
        planner,
        pending: None,
    };
    // SAFETY: `out` was proven non-null and aligned and already written once above, so it is still writable.
    unsafe { out.write(open(placement)) };
    Ok(())
}

#[no_mangle]
pub unsafe extern "C" fn seabraid_placement_open(
    layout: *const Layout,
    optional_policy: *const Policy,
    out: *mut *mut PlacementHandle,
) -> i32 {
    status("seabraid_placement_open", || {
        // SAFETY: `policy_config` takes a null pointer (the defaults) or the caller's readable `Policy`; this passes
        // the caller's pointer on unchanged; `open_with` has this function's contract for `layout` and `out`.
        unsafe { open_with(layout, out, || policy_config(optional_policy)) }
    })
}

#[no_mangle]
pub unsafe extern "C" fn seabraid_placement_open_v2(
    layout: *const Layout,
    optional_policy: *const PolicyV2,
    out: *mut *mut PlacementHandle,
) -> i32 {
    status("seabraid_placement_open_v2", || {
        // SAFETY: as in `seabraid_placement_open`; `policy_v2_config` reads the caller's `struct_size` bytes only
        // after `read_in` checked the header.
        unsafe { open_with(layout, out, || policy_v2_config(optional_policy)) }
    })
}
