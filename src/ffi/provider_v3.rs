// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! `seabraid_provider_v3`: the Rust mirror of the header struct, its open-time validation and the execute bridge.
//!
//! OWNER: include/seabraid/seabraid.h owns the layout; src/layout/abi_layout.txt owns the numbers; this module
//! owns what a v3 descriptor may contain (`ProviderV3::load`) and how its callback is invoked (`Callback`).
//! CONTRACT: `load`, `config` and `callback` for `seabraid_session_open`; no exports.
//! FAILURE: fail-fast `Code::Invalid`, each reason logged through `fail!` and naming the field.
//! CHECKER: `assert_layout!` below, src/header_parity.rs for the constants, src/ffi/tests/v2.rs for every branch.

use super::batch_v2::{BatchV2, BATCH_V2_VERSION};
use super::providers::{callback_result, PARTIAL_ROUTES};
use super::sized::{read_in, Layout};
use super::{check_pointer, check_version};
use crate::batch::{Batch, Config, Features};
use crate::diag::{fail, Code};
use crate::ffi::ABI_VERSION;
use crate::layout::assert_layout;
use crate::Result;
use core::ffi::c_void;

pub(crate) const PROVIDER_V3_VERSION: u32 = 3;
pub(crate) const COMPACT_ROWS: u32 = 1 << 1;
pub(crate) const ACCEPTS_F16: u32 = 1 << 2;
pub(crate) const ACCEPTS_BF16: u32 = 1 << 3;
pub(crate) const REFERENCE: u32 = 1 << 4;
const ACCEPTS: u32 = ACCEPTS_F16 | ACCEPTS_BF16;
const KNOWN_FLAGS: u32 = PARTIAL_ROUTES | COMPACT_ROWS | ACCEPTS | REFERENCE;

type ExecuteV3 =
    unsafe extern "C" fn(*mut c_void, *const Config, *const BatchV2, *mut f32, usize) -> i32;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ProviderV3 {
    pub abi_version: u32,
    pub struct_size: u32,
    pub operation_count: u32,
    pub flags: u32,
    pub config: *const Config,
    pub context: *mut c_void,
    pub execute: Option<ExecuteV3>,
}

impl ProviderV3 {
    /// Size of the first layout; a larger `struct_size` means fields were appended later.
    pub const BASE_SIZE: usize = 40;

    /// Copies and validates the caller's descriptor: sized rule, operation count, flags, reference rule, execute.
    ///
    /// # Safety
    /// `pointer` is readable for its declared `struct_size`.
    pub(crate) unsafe fn load(pointer: *const Self) -> Result<Self> {
        let layout = Layout::Sized {
            base: Self::BASE_SIZE,
        };
        // SAFETY: `read_in` checks `pointer` non-null and the version before reading; `load`'s contract is that the
        // caller's descriptor is readable for its declared `struct_size`.
        let provider = unsafe { read_in(pointer, "provider", PROVIDER_V3_VERSION, layout) }?;
        provider.check_flags()?;
        provider.check_execute()?;
        Ok(provider)
    }

    fn check_flags(&self) -> Result<()> {
        if self.operation_count == 0 {
            return Err(fail!(
                Code::Invalid,
                "provider.operation_count must be at least 1"
            ));
        }
        let unknown = self.flags & !KNOWN_FLAGS;
        if unknown != 0 {
            return Err(fail!(
                Code::Invalid,
                "provider.flags {:#x} sets unknown flags {:#x}",
                self.flags,
                unknown
            ));
        }
        // WHY: compaction renumbers the rows a provider sees, which only means something when rows can be skipped.
        if self.flags & COMPACT_ROWS != 0 && self.flags & PARTIAL_ROUTES == 0 {
            return Err(fail!(
                Code::Invalid,
                "provider.flags COMPACT_ROWS requires PARTIAL_ROUTES"
            ));
        }
        Ok(())
    }

    fn check_execute(&self) -> Result<()> {
        let reference = self.flags & REFERENCE != 0;
        let callback = self.execute.is_some() || !self.context.is_null();
        if reference && (callback || self.flags & ACCEPTS != 0) {
            return Err(fail!(
                Code::Invalid,
                "provider.flags REFERENCE requires null execute and context and no ACCEPTS_* flag"
            ));
        }
        if !reference && self.execute.is_none() {
            return Err(fail!(
                Code::Invalid,
                "provider.execute is null (set REFERENCE for the built-in reference transform)"
            ));
        }
        Ok(())
    }

    /// The config the session opens with, copied now because the caller's memory is not ours after open.
    ///
    /// # Safety
    /// `self.config` is null or points to a readable `Config`.
    pub(crate) unsafe fn config(&self) -> Result<Config> {
        check_pointer(self.config, "provider.config")?;
        // SAFETY: `check_pointer` above proved `self.config` non-null and aligned; `config`'s contract says it points
        // to a readable `Config`, and only its first `u32` is read before the version matches.
        let found = unsafe { self.config.cast::<u32>().read() };
        check_version(found, ABI_VERSION, "provider.config")?;
        // SAFETY: The version check just passed, so the caller's struct is this ABI's `Config`, readable per
        // `config`'s contract; it is read once and copied.
        unsafe { self.config.read() }.validate()
    }

    /// What this descriptor declares; `Session::with_features` refuses COMPACT_ROWS without PARTIAL_ROUTES.
    pub(crate) fn features(&self) -> Features {
        Features {
            partial_routes: self.flags & PARTIAL_ROUTES != 0,
            compact_rows: self.flags & COMPACT_ROWS != 0,
            accepts_f16: self.flags & ACCEPTS_F16 != 0,
            accepts_bf16: self.flags & ACCEPTS_BF16 != 0,
        }
    }

    /// The part of the descriptor the worker thread needs. WHY a separate type: it must not carry the
    /// caller's `config` pointer, which dangles after open.
    pub(crate) fn callback(&self) -> Option<Callback> {
        self.execute.map(|execute| Callback {
            context: self.context,
            execute,
        })
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Callback {
    context: *mut c_void,
    execute: ExecuteV3,
}

// The caller keeps the context alive and synchronizes shared access until close.
// SAFETY: `Callback` is two words, the caller's `execute` function and its `context`. The provider contract
// (seabraid.h) is that `execute` may be called from the worker thread and that `context` stays valid, with any shared
// state synchronized by the caller, until close; moving the pair to that thread is therefore sound.
unsafe impl Send for Callback {}

impl Callback {
    pub(crate) fn run(
        &self,
        operation_id: u32,
        config: Config,
        batch: &Batch,
        output: &mut [f32],
    ) -> Result<()> {
        // WHY every field comes from the Batch: `tokens` is the compacted row count when COMPACT_ROWS is on, and
        // `activation_count` was sized for those rows by `Batch::load`, so the view can never disagree with the data.
        let view = BatchV2 {
            abi_version: BATCH_V2_VERSION,
            struct_size: BatchV2::BASE_SIZE as u32,
            operation_id,
            source_modalities: batch.source_modalities,
            placement_epoch: batch.placement_epoch,
            tokens: batch.tokens,
            activation_format: batch.format,
            activations: batch.activations.as_ptr().cast(),
            activation_count: batch.activation_count,
            expert_ids: batch.expert_ids.as_ptr(),
            expert_id_count: batch.expert_ids.len(),
            weights: batch.weights.as_ptr(),
            weight_count: batch.weights.len(),
        };
        // The callback must finish all device accesses to these buffers before returning.
        // SAFETY: `execute` and `context` are the caller's and valid until close (the `Send` impl above); `config`
        // and `view` are live locals and `output` a live mutable slice for the whole call, and the callback must
        // finish with all of them before it returns.
        let status = unsafe {
            (self.execute)(
                self.context,
                &config,
                &view,
                output.as_mut_ptr(),
                output.len(),
            )
        };
        callback_result(status)
    }
}

assert_layout!(
    ProviderV3,
    "seabraid_provider_v3",
    [
        abi_version,
        struct_size,
        operation_count,
        flags,
        config,
        context,
        execute
    ]
);
