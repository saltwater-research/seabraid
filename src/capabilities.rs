// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! Reports the operations of this ABI and validates activation provenance.
//! FAILURE: fail-fast `Code::Invalid` from `validate_origins` for a source-modality bit this ABI does not define;
//! everything else here is constants.

use crate::diag::{fail, Code};
use crate::Result;

pub(crate) const TEXT: u32 = 1 << 0;
pub(crate) const IMAGE: u32 = 1 << 1;
pub(crate) const AUDIO: u32 = 1 << 2;
pub(crate) const VIDEO: u32 = 1 << 3;
pub(crate) const OTHER: u32 = 1 << 4;
pub(crate) const ORIGINS: u32 = TEXT | IMAGE | AUDIO | VIDEO | OTHER;

pub(crate) const OPERATION_ROUTED_MOE: u32 = 1;
pub(crate) const FORMAT_ROUTED_F32: u32 = 1;
pub(crate) const FORMAT_PARTIAL_F32: u32 = 2;
/// `SEABRAID_FORMAT_ROUTED_F16`. WHY here: the core (batch.rs) needs the format numbers and must not import ffi.
pub(crate) const FORMAT_ROUTED_F16: u32 = 3;
/// `SEABRAID_FORMAT_ROUTED_BF16`.
pub(crate) const FORMAT_ROUTED_BF16: u32 = 4;

#[derive(Clone, Copy)]
#[repr(C)]
pub struct Capabilities {
    pub abi_version: u32,
    pub struct_size: u32,
    pub operation: u32,
    pub input_format: u32,
    pub output_format: u32,
    pub source_modalities: u32,
}

crate::layout::assert_layout!(
    Capabilities,
    "seabraid_capabilities_v1",
    [
        abi_version,
        struct_size,
        operation,
        input_format,
        output_format,
        source_modalities
    ]
);

pub fn validate_origins(origins: u32) -> Result<()> {
    let unknown = origins & !ORIGINS;
    if unknown != 0 {
        return Err(fail!(
            Code::Invalid,
            "source_modalities {:#x} sets unknown flags {:#x}",
            origins,
            unknown
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capabilities_match_the_accepted_origin_mask() {
        assert!(validate_origins(0).is_ok());
        assert!(validate_origins(AUDIO | VIDEO).is_ok());
        assert!(validate_origins(ORIGINS).is_ok());
        let error = validate_origins(AUDIO | (1 << 31)).unwrap_err();
        assert_eq!(error.code(), Code::Invalid);
        assert!(error.message().contains("0x80000000"));
    }
}
