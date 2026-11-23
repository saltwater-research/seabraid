// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! Native callback descriptors. Both versions use the same borrowed batch view.
//! FAILURE: fail-fast `Code::Invalid` at load (null execute, unknown flags, bad version); at run a nonzero callback
//! status becomes `Code::Provider` through `callback_result`, carrying the provider's own code in the text.

use super::sized::{read_in, Layout};
use super::BatchView;
use crate::batch::{Batch, Config, Features};
use crate::diag::{fail, Code};
use crate::Result;
use core::ffi::c_void;

pub(crate) const PROVIDER_VERSION_MULTI: u32 = 2;
/// Layout generation of `Provider` (provider_v1). WHY its own const: abi_version is a per-struct
/// layout tag (docs/ABI.md), so v1 must not track the global `ABI_VERSION`; value stays 1.
pub(crate) const PROVIDER_V1_VERSION: u32 = 1;
pub(crate) const PARTIAL_ROUTES: u32 = 1 << 0;

type Execute =
    unsafe extern "C" fn(*mut c_void, *const Config, *const BatchView, *mut f32, usize) -> i32;
type ExecuteOperation =
    unsafe extern "C" fn(*mut c_void, u32, *const Config, *const BatchView, *mut f32, usize) -> i32;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Provider {
    abi_version: u32,
    context: *mut c_void,
    execute: Option<Execute>,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct MultiProvider {
    abi_version: u32,
    operation_count: u32,
    flags: u32,
    context: *mut c_void,
    execute: Option<ExecuteOperation>,
}

// The caller keeps the context alive and synchronizes shared access until close.
// SAFETY: A `Provider` holds the caller's `execute` function and `context`; the provider contract (seabraid.h) is
// that `execute` may be called from the session's worker thread and that `context` stays valid, with shared state
// synchronized by the caller, until close.
unsafe impl Send for Provider {}
// SAFETY: Same contract as `Provider` above: the caller's `execute` is callable from the worker thread and its
// `context` stays valid and synchronized until close.
unsafe impl Send for MultiProvider {}

impl Provider {
    pub unsafe fn load(pointer: *const Self) -> Result<Self> {
        // SAFETY: `read_in` checks `pointer` non-null and the version before reading; `load`'s contract is that the
        // caller's descriptor is readable for `size_of::<Provider>()` bytes.
        let provider = unsafe { read_in(pointer, "provider", PROVIDER_V1_VERSION, Layout::Fixed) }?;
        if provider.execute.is_none() {
            return Err(fail!(Code::Invalid, "provider.execute is null"));
        }
        Ok(provider)
    }

    pub fn run(&self, config: Config, batch: &Batch, output: &mut [f32]) -> Result<()> {
        let execute = self
            .execute
            .ok_or_else(|| fail!(Code::Provider, "provider.execute is null"))?;
        let view = BatchView::from(batch);
        // The callback must finish all device accesses to these buffers before returning.
        // SAFETY: `execute` and `context` are the caller's and valid until close (the `Send` impl above); `config`
        // and `view` are live locals and `output` a live mutable slice for the whole call, and the callback must
        // finish with them before it returns.
        let status = unsafe {
            execute(
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

impl MultiProvider {
    pub unsafe fn load(pointer: *const Self) -> Result<Self> {
        let provider =
            // SAFETY: `read_in` checks `pointer` non-null and the version before reading; `load`'s contract is that
            // the caller's descriptor is readable for `size_of::<MultiProvider>()` bytes.
            unsafe { read_in(pointer, "provider", PROVIDER_VERSION_MULTI, Layout::Fixed) }?;
        if provider.execute.is_none() {
            return Err(fail!(Code::Invalid, "provider.execute is null"));
        }
        // WHY only PARTIAL_ROUTES: seabraid_provider_v2 is frozen with its documented meaning; row compaction
        // changes the rows a callback sees and is a provider v3 feature (seabraid_session_open).
        let unknown = provider.flags & !PARTIAL_ROUTES;
        if unknown != 0 {
            return Err(fail!(
                Code::Invalid,
                "provider.flags {:#x} sets unknown flags {:#x}",
                provider.flags,
                unknown
            ));
        }
        Ok(provider)
    }

    /// What this descriptor declares: partial routes only (`load` refuses every other flag).
    pub(crate) fn features(&self) -> Features {
        Features {
            partial_routes: self.flags & PARTIAL_ROUTES != 0,
            ..Features::default()
        }
    }

    pub fn operation_count(&self) -> u32 {
        self.operation_count
    }

    pub fn run(
        &self,
        operation_id: u32,
        config: Config,
        batch: &Batch,
        output: &mut [f32],
    ) -> Result<()> {
        let execute = self
            .execute
            .ok_or_else(|| fail!(Code::Provider, "provider.execute is null"))?;
        let view = BatchView::from(batch);
        // The same batch and completion lease applies to every operation.
        // SAFETY: `execute` and `context` are the caller's and valid until close (the `Send` impl above); `config`
        // and `view` are live locals and `output` a live mutable slice for the whole call, and the callback must
        // finish with them before it returns.
        let status = unsafe {
            execute(
                self.context,
                operation_id,
                &config,
                &view,
                output.as_mut_ptr(),
                output.len(),
            )
        };
        callback_result(status)
    }
}

pub(super) fn callback_result(status: i32) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        // The code is the provider's own; it is reported verbatim so the vendor can match it.
        Err(fail!(
            Code::Provider,
            "provider callback returned status {}",
            status
        ))
    }
}

crate::layout::assert_layout!(
    Provider,
    "seabraid_provider_v1",
    [abi_version, context, execute]
);
crate::layout::assert_layout!(
    MultiProvider,
    "seabraid_provider_v2",
    [abi_version, operation_count, flags, context, execute]
);
