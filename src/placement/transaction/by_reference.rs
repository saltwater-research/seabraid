// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! A same-layer swap whose upper write takes no bytes from the library: the upper rung is asked to load the promoted
//! expert from the fallback's own verified store (for Strata, the model file), and only a refusal sends bytes.
//! OWNER: this file owns the by-reference rows of the failure table (docs/PLACEMENT-ABI.md, "By-reference moves");
//! every row that is not named there is the byte path's row, in `swap.rs`, unchanged.
//! FAILURE: no error escapes; the same `Exchange` as a byte swap, so outcomes, epochs and the quarantine mask are
//! computed by the same code.

use super::swap::{call, call_with, Exchange, Swap};
use crate::placement::{IoError, IoPhase, PlacementIo};

impl Swap {
    /// The displaced expert is staged first and kept to the end: it is the recovery invariant of every swap ("preserve
    /// both originals"), and the climber's bytes are not needed for it, because the upper rung already holds it in
    /// the file. WHY the call runs on the UPPER rung and never on the fallback: the fallback is only the byte source,
    /// so no row here can close it, and its stage and replace rows stay exactly as in `swap.rs`.
    pub(super) fn run_by_reference(&self, io: &mut dyn PlacementIo) -> Exchange {
        let down = match call(io, self.hi, IoPhase::Stage, self.down, |io| {
            io.stage(self.hi, self.down)
        }) {
            Ok(bytes) if !bytes.is_empty() => bytes,
            Ok(_) => {
                let issue = self.malformed(self.hi, "staged displaced slot is empty");
                return self.stage_failed(issue);
            }
            Err(issue) => return self.stage_failed(issue),
        };
        // REFUSED = "the byte path follows in this swap"; it is not a failure and is not logged as one.
        match call_with(io, self.hi, IoPhase::Replace, self.up, false, |io| {
            io.replace_from_fallback(self.hi, self.down, self.up)
        }) {
            Ok(()) => self.write_lower(io, None, &down),
            Err(issue) if issue.error == IoError::Refused => self.bytes_after_refusal(io, &down),
            // Failed restores the upper slot from `down`; Timeout and Panicked close it with no restore sent.
            Err(issue) => self.upper_failed(io, issue, &down),
        }
    }

    /// The upper rung declined and wrote nothing, so today's stage + replace runs inside the same swap.
    fn bytes_after_refusal(&self, io: &mut dyn PlacementIo, down: &[u8]) -> Exchange {
        let up = match call(io, self.lo, IoPhase::Stage, self.up, |io| {
            io.stage(self.lo, self.up)
        }) {
            Ok(bytes) => bytes,
            Err(issue) => return self.stage_failed(issue),
        };
        match self.check_sizes(&up, down) {
            Ok(()) => self.write_both(io, up, down),
            Err(issue) => self.stage_failed(issue),
        }
    }
}
