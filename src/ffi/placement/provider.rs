// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! Foreign copy bridge. Invalid stage sizes fail before any replacement;
//! unknown callback codes leave residency uncertain and trigger quarantine. It logs nothing itself: a failure's
//! one record is written by the transaction (`placement::transaction::call`) from the `IoError` and this module's
//! `take_detail`.

use super::input;
use crate::diag::{fail, Code};
use crate::ffi::ABI_VERSION;
use crate::placement::{ExpertId, IoError, PlacementIo, Rung};
use crate::Result;
use std::alloc::{alloc_zeroed, Layout};
use std::ffi::c_void;

pub(crate) const IO_ACK: i32 = 0;
pub(crate) const IO_REFUSED: i32 = 1;
pub(crate) const IO_FAILED: i32 = 2;
pub(crate) const IO_TIMEOUT: i32 = 3;

/// Largest `layer_slot_bytes` entry `seabraid_placement_apply` accepts: 2^32 - 1 bytes, `SEABRAID_PLACEMENT_MAX_SLOT_BYTES`
/// in placement.h (src/header_parity.rs holds the two equal).
/// WHY a cap: the table is caller memory the library cannot check against anything, and each staged slot is an
/// allocation of that size (two per swap). An expert slot is megabytes to a few gigabytes; a figure past 4 GiB is a
/// corrupt table, and refusing it before any I/O (retryable, like every geometry error) beats attempting a
/// multi-terabyte allocation. It replaces `isize::MAX`, which only bounded what the Rust allocator can describe.
pub(crate) const MAX_SLOT_BYTES: usize = u32::MAX as usize;

type Stage = unsafe extern "C" fn(*mut c_void, Rung, u32, u32, *mut u8, usize, *mut usize) -> i32;
type Replace = unsafe extern "C" fn(*mut c_void, Rung, u32, u32, u32, *const u8, usize) -> i32;
type ReplaceFromFallback = unsafe extern "C" fn(*mut c_void, Rung, u32, u32, u32) -> i32;

/// `seabraid_placement_provider_v2.abi_version`.
pub(crate) const PROVIDER_V2_VERSION: u32 = 2;
/// The smallest `struct_size` of a provider v2: the whole struct. WHY no shorter form: both new fields are what the
/// version is for, and a caller without them uses `seabraid_placement_apply` and provider v1.
pub(crate) const PROVIDER_V2_BASE_SIZE: usize = 64;

#[derive(Clone, Copy)]
#[repr(C)]
pub struct Provider {
    abi_version: u32,
    context: *mut c_void,
    layer_slot_bytes: *const usize,
    layer_count: usize,
    stage: Option<Stage>,
    replace: Option<Replace>,
}

/// The caller-filled, sized provider of `seabraid_placement_apply_v2`. Every v1 field means what it means there;
/// `from_fallback_rungs` and `replace_from_fallback` are the by-reference write (docs/PLACEMENT-ABI.md).
#[derive(Clone, Copy)]
#[repr(C)]
pub struct ProviderV2 {
    abi_version: u32,
    struct_size: u32,
    context: *mut c_void,
    layer_slot_bytes: *const usize,
    layer_count: usize,
    stage: Option<Stage>,
    replace: Option<Replace>,
    from_fallback_rungs: u64,
    replace_from_fallback: Option<ReplaceFromFallback>,
}

impl ProviderV2 {
    /// `fallback` is the fallback rung (= the card count). WHY the mask is checked here and not in the transaction: it
    /// is a field of a foreign struct, and a bad one must fail before the first provider call like every geometry
    /// error, keeping the ticket.
    pub(super) unsafe fn bind<'a>(&self, layers: usize, fallback: Rung) -> Result<ProviderIo<'a>> {
        if self.from_fallback_rungs >> fallback != 0 {
            return Err(fail!(
                Code::Invalid,
                "provider.from_fallback_rungs {:#x} sets the fallback rung {} or a rung above it",
                self.from_fallback_rungs,
                fallback
            ));
        }
        if self.from_fallback_rungs != 0 && self.replace_from_fallback.is_none() {
            return Err(fail!(
                Code::Invalid,
                "provider.from_fallback_rungs is non-zero but provider.replace_from_fallback is null"
            ));
        }
        let v1 = Provider {
            abi_version: ABI_VERSION,
            context: self.context,
            layer_slot_bytes: self.layer_slot_bytes,
            layer_count: self.layer_count,
            stage: self.stage,
            replace: self.replace,
        };
        // SAFETY: the same promise as for `Provider::bind`: the caller keeps `layer_slot_bytes` readable for `layer_count`
        // entries for the whole apply (placement.h).
        let mut io = unsafe { v1.bind(layers) }?;
        io.from_fallback_rungs = self.from_fallback_rungs;
        io.replace_from_fallback = self.replace_from_fallback;
        Ok(io)
    }
}

impl Provider {
    // Only apply borrows the size table and callbacks. A provider can change
    // between rounds without binding either engine context to the handle. The lifetime is the caller's promise, not
    // this struct's: the slice is read from a raw pointer, and `ProviderV2::bind` builds a local `Provider` to reach it.
    pub(super) unsafe fn bind<'a>(&self, layers: usize) -> Result<ProviderIo<'a>> {
        let stage = self
            .stage
            .ok_or_else(|| fail!(Code::Invalid, "provider.stage is null"))?;
        let replace = self
            .replace
            .ok_or_else(|| fail!(Code::Invalid, "provider.replace is null"))?;
        // SAFETY: `bind`'s contract: the caller keeps its `layer_slot_bytes` table readable for `layer_count` entries
        // for the whole apply (placement.h). `input` refuses a count other than the layer count and a null pointer
        // before it forms the slice.
        let sizes = unsafe {
            input(
                self.layer_slot_bytes,
                self.layer_count,
                layers,
                "layer_slot_bytes",
            )
        }?;
        if let Some(layer) = sizes
            .iter()
            .position(|size| !(1..=MAX_SLOT_BYTES).contains(size))
        {
            return Err(fail!(
                Code::Invalid,
                "layer_slot_bytes[{}] is {}, outside 1..={}",
                layer,
                sizes[layer],
                MAX_SLOT_BYTES
            ));
        }
        Ok(ProviderIo {
            context: self.context,
            sizes,
            stage,
            replace,
            from_fallback_rungs: 0,
            replace_from_fallback: None,
            detail: None,
        })
    }
}

pub(super) struct ProviderIo<'a> {
    context: *mut c_void,
    sizes: &'a [usize],
    stage: Stage,
    replace: Replace,
    /// Bit r set = rung r may take a promoted expert by reference; 0 for a provider v1 (never asked).
    from_fallback_rungs: u64,
    /// Non-null whenever `from_fallback_rungs` is non-zero (`ProviderV2::bind` refuses the other case).
    replace_from_fallback: Option<ReplaceFromFallback>,
    /// What the latest failure needs said beyond its `IoError` (see `PlacementIo::take_detail`).
    detail: Option<(Code, String)>,
}

/// A zero-filled buffer of `size` bytes, or none when the allocator refuses. WHY `alloc_zeroed` and not
/// `try_reserve_exact` followed by `resize`: under memory overcommit the reservation succeeds for sizes the machine
/// cannot back, and the `resize` then writes every page, so a corrupt size table became an OOM kill of the host
/// process instead of an error. A zeroed allocation maps fresh pages without touching them; only a provider that
/// really writes the slot commits that memory. The zero fill is still there, so a short or dishonest stage callback
/// cannot turn the later replace call into an uninitialised read.
fn zeroed_bytes(size: usize) -> Option<Vec<u8>> {
    if size == 0 {
        return Some(Vec::new());
    }
    let layout = Layout::array::<u8>(size).ok()?;
    // SAFETY: `layout` has a nonzero size, which `alloc_zeroed` requires.
    let pointer = unsafe { alloc_zeroed(layout) };
    if pointer.is_null() {
        return None;
    }
    // SAFETY: `pointer` came from the global allocator with the layout of `size` `u8`s (align 1) and all of its
    // `size` bytes are initialised (zero): exactly the allocation a `Vec<u8>` of length and capacity `size` owns.
    Some(unsafe { Vec::from_raw_parts(pointer, size, size) })
}

impl PlacementIo for ProviderIo<'_> {
    fn stage(&mut self, rung: Rung, cell: ExpertId) -> std::result::Result<Vec<u8>, IoError> {
        let size = self.sizes[cell.layer as usize];
        let Some(mut bytes) = zeroed_bytes(size) else {
            self.detail = Some((
                Code::Internal,
                format!("cannot allocate the {size}-byte stage buffer"),
            ));
            return Err(IoError::Failed);
        };
        let mut written = 0;
        // SAFETY: The stage callback and `context` are the caller's and stay valid for the whole apply (placement.h);
        // `bytes` is a live, `size`-byte, zero-filled buffer that nothing else touches until the callback returns,
        // and `written` is a live local.
        let code = unsafe {
            (self.stage)(
                self.context,
                rung,
                cell.layer,
                cell.expert,
                bytes.as_mut_ptr(),
                size,
                &mut written,
            )
        };
        self.io_status(code, "stage")?;
        if written != size {
            self.detail = Some((
                Code::Provider,
                format!("stage acknowledged but wrote {written} of {size} bytes"),
            ));
            return Err(IoError::MalformedBlob);
        }
        Ok(bytes)
    }

    fn replace(
        &mut self,
        rung: Rung,
        old: ExpertId,
        new: ExpertId,
        bytes: &[u8],
    ) -> std::result::Result<(), IoError> {
        // The transaction derives one layer for each swap and its restore.
        // These bytes are a private stage copy borrowed only into this call.
        // SAFETY: As for the stage call: the callback and `context` are the caller's for the whole apply, and `bytes`
        // is a live slice of this call's private copy that the callback may only read and must not keep.
        let code = unsafe {
            (self.replace)(
                self.context,
                rung,
                old.layer,
                old.expert,
                new.expert,
                bytes.as_ptr(),
                bytes.len(),
            )
        };
        self.io_status(code, "replace")
    }

    fn takes_from_fallback(&self, rung: Rung) -> bool {
        // `checked_shr` because a rung is a u8 and the mask has 64 bits: a rung past it is simply not masked.
        self.from_fallback_rungs
            .checked_shr(u32::from(rung))
            .is_some_and(|bits| bits & 1 == 1)
    }

    fn replace_from_fallback(
        &mut self,
        rung: Rung,
        old: ExpertId,
        new: ExpertId,
    ) -> std::result::Result<(), IoError> {
        // A null callback cannot be reached with a non-zero mask; refusing is the safe answer if it ever is.
        let Some(callback) = self.replace_from_fallback else {
            return Err(IoError::Refused);
        };
        // SAFETY: As for the replace call: the callback and `context` are the caller's for the whole apply, and the
        // call passes only integers, so no buffer outlives it.
        let code = unsafe { callback(self.context, rung, old.layer, old.expert, new.expert) };
        self.io_status(code, "replace_from_fallback")
    }

    fn take_detail(&mut self) -> Option<(Code, String)> {
        self.detail.take()
    }
}

impl ProviderIo<'_> {
    /// Maps a callback code. A known failure needs no more than its `IoError`; a code outside the header keeps
    /// its number as detail, because the mapping to `Timeout` would otherwise lose it. Nothing is logged here.
    fn io_status(&mut self, code: i32, operation: &str) -> std::result::Result<(), IoError> {
        match code {
            IO_ACK => Ok(()),
            IO_REFUSED => Err(IoError::Refused),
            IO_FAILED => Err(IoError::Failed),
            IO_TIMEOUT => Err(IoError::Timeout),
            _ => {
                self.detail = Some((
                    Code::Provider,
                    format!(
                        "{operation} returned unknown code {code}; treated as a timeout (completion is uncertain)"
                    ),
                ));
                // Unknown codes cannot promise that a request completed.
                Err(IoError::Timeout)
            }
        }
    }
}

crate::layout::assert_layout!(
    Provider,
    "seabraid_placement_provider_v1",
    [
        abi_version,
        context,
        layer_slot_bytes,
        layer_count,
        stage,
        replace
    ]
);
crate::layout::assert_layout!(
    ProviderV2,
    "seabraid_placement_provider_v2",
    [
        abi_version,
        struct_size,
        context,
        layer_slot_bytes,
        layer_count,
        stage,
        replace,
        from_fallback_rungs,
        replace_from_fallback
    ]
);
