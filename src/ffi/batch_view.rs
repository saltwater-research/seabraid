// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! The v1 batch descriptor a provider receives (`seabraid_batch_v1`), split out of the C ABI root to keep it under the size limit.
//! OWNER: this file owns the v1 batch layout; the offsets are proved against src/layout/abi_layout.txt by `assert_layout!` below.

use super::ABI_VERSION;
use crate::batch::Batch;
use crate::layout::assert_layout;

#[repr(C)]
pub struct BatchView {
    pub(super) abi_version: u32,
    pub(super) tokens: u32,
    pub(super) activations: *const f32,
    pub(super) activation_count: usize,
    pub(super) expert_ids: *const i32,
    pub(super) expert_id_count: usize,
    pub(super) weights: *const f32,
    pub(super) weight_count: usize,
}

impl From<&Batch> for BatchView {
    fn from(batch: &Batch) -> Self {
        Self {
            abi_version: ABI_VERSION,
            tokens: batch.tokens,
            // v1 providers only ever see F32 (the format check at submit), so the words are f32s.
            activations: batch.activations.as_ptr().cast::<f32>(),
            activation_count: batch.activation_count,
            expert_ids: batch.expert_ids.as_ptr(),
            expert_id_count: batch.expert_ids.len(),
            weights: batch.weights.as_ptr(),
            weight_count: batch.weights.len(),
        }
    }
}

assert_layout!(
    BatchView,
    "seabraid_batch_v1",
    [
        abi_version,
        tokens,
        activations,
        activation_count,
        expert_ids,
        expert_id_count,
        weights,
        weight_count
    ]
);
