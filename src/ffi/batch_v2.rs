// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! `seabraid_batch_v2`: the Rust mirror of the header struct. Read by `seabraid_session_submit` (src/ffi/v2.rs).
//!
//! OWNER: include/seabraid/seabraid.h owns the layout; src/layout/abi_layout.txt owns the numbers.
//! CONTRACT: `#[repr(C)]` data and constants, plus `BatchV2::submission` (the one conversion to `Submission`); no exports.
//! FAILURE: fail-fast `Code::Invalid`, logged through `fail!` and naming the field.
//! CHECKER: `assert_layout!` below, src/header_parity.rs for the constants, src/ffi/tests/v2*.rs for each branch.

use super::check_pointer;
use crate::batch::Config;
use crate::capabilities::{FORMAT_ROUTED_BF16, FORMAT_ROUTED_F16, FORMAT_ROUTED_F32};
use crate::diag::{fail, Code};
use crate::layout::assert_layout;
use crate::submission::Submission;
use crate::Result;
use core::ffi::c_void;
use std::slice;

pub(crate) const BATCH_V2_VERSION: u32 = 2;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct BatchV2 {
    pub abi_version: u32,
    pub struct_size: u32,
    pub operation_id: u32,
    pub source_modalities: u32,
    pub placement_epoch: u64,
    pub tokens: u32,
    pub activation_format: u32,
    pub activations: *const c_void,
    pub activation_count: usize,
    pub expert_ids: *const i32,
    pub expert_id_count: usize,
    pub weights: *const f32,
    pub weight_count: usize,
}

impl BatchV2 {
    /// Size of the first layout; a larger `struct_size` means fields were appended later.
    pub const BASE_SIZE: usize = 80;

    /// Checks format, counts and pointers and borrows the caller's arrays as a
    /// `Submission`. Picks are checked later, by the session.
    ///
    /// # Safety
    /// Each pointer is valid for its checked count and not written during the call (the C caller's guarantee).
    pub(crate) unsafe fn submission<'a>(&self, config: Config) -> Result<Submission<'a>> {
        let size = element_size(self.activation_format)?;
        let lengths = config.lengths(self.tokens)?;
        let counts = [
            self.activation_count,
            self.expert_id_count,
            self.weight_count,
        ];
        lengths.check(self.tokens, config, counts)?;
        check_elements(self.activations, size, "batch.activations")?;
        check_pointer(self.expert_ids, "batch.expert_ids")?;
        check_pointer(self.weights, "batch.weights")?;
        // SAFETY: `submission`'s contract is that each pointer is valid for its checked count and not written during
        // the call. `check_elements` and `check_pointer` above refused null or misaligned pointers, `lengths.check`
        // fixed every caller count against the config, and `lengths.rows * size` and `lengths.picks` are exactly
        // those checked counts.
        unsafe {
            Ok(Submission {
                operation_id: self.operation_id,
                tokens: self.tokens,
                format: self.activation_format,
                activations: slice::from_raw_parts(
                    self.activations.cast::<u8>(),
                    lengths.rows * size,
                ),
                expert_ids: slice::from_raw_parts(self.expert_ids, lengths.picks),
                weights: slice::from_raw_parts(self.weights, lengths.picks),
                source_modalities: self.source_modalities,
                placement_epoch: self.placement_epoch,
            })
        }
    }
}

/// Bytes per activation element, or INVALID for a value that is not an activation format.
fn element_size(format: u32) -> Result<usize> {
    match format {
        FORMAT_ROUTED_F32 => Ok(4),
        FORMAT_ROUTED_F16 | FORMAT_ROUTED_BF16 => Ok(2),
        _ => Err(fail!(
            Code::Invalid,
            "batch.activation_format {} is not an activation format (1 F32, 3 F16, 4 BF16)",
            format
        )),
    }
}

fn check_elements(pointer: *const c_void, size: usize, name: &str) -> Result<()> {
    if pointer.is_null() {
        return Err(fail!(Code::Invalid, "{} is null", name));
    }
    if pointer as usize % size != 0 {
        return Err(fail!(
            Code::Invalid,
            "{} is misaligned for {}-byte elements",
            name,
            size
        ));
    }
    Ok(())
}

assert_layout!(
    BatchV2,
    "seabraid_batch_v2",
    [
        abi_version,
        struct_size,
        operation_id,
        source_modalities,
        placement_epoch,
        tokens,
        activation_format,
        activations,
        activation_count,
        expert_ids,
        expert_id_count,
        weights,
        weight_count
    ]
);
