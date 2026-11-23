// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! One batch on its way into `Session::submit`, whichever ABI entry point it came through.
//!
//! OWNER   : the shape of a submission between the FFI layer (v1 shims, v2 entry points) and the session.
//! CONTRACT: plain borrowed data, `Session::submit(&mut self, s: Submission<'_>) -> Result<u64>`. Validation lives
//!           in batch.rs and session.rs; conversion from the C structs lives in src/ffi/.
//! FAILURE : none here; this type cannot fail.
//! WHY one struct: v1 and v2 submits funnel into one `Session::submit`, so there is exactly one validation path and
//! a v1 caller is just a v2 submission with F32 data and placement epoch 0 (the v1 shims in src/ffi.rs build exactly that).

#[derive(Clone, Copy, Debug)]
pub(crate) struct Submission<'a> {
    pub operation_id: u32,
    pub tokens: u32,
    /// `SEABRAID_FORMAT_ROUTED_*`; `FORMAT_ROUTED_F32` for every v1 caller.
    pub format: u32,
    /// Raw activation bytes: `tokens * hidden` elements of the format's element size.
    pub activations: &'a [u8],
    pub expert_ids: &'a [i32],
    pub weights: &'a [f32],
    pub source_modalities: u32,
    /// 0 means unversioned (every v1 submit); otherwise it must not fall below the session's high-water mark.
    pub placement_epoch: u64,
}
