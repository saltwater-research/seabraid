// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! Host runtime for bounded MoE expert batches.
//! FAILURE: the crate is fail-fast at its one boundary (`ffi`: a status code and one log record per refusal, panics
//! contained) and fail-open-loud in `diag` (logging never fails a call); each module states its own mode.

#![deny(unsafe_code, unsafe_op_in_unsafe_fn)]
// Every `unsafe` block states why it is sound, in a `// SAFETY:` comment right above it; `missing_safety_doc`
// (on by default) asks the same of every public `unsafe fn` as a `# Safety` section.
#![deny(clippy::undocumented_unsafe_blocks, clippy::missing_safety_doc)]
#![warn(clippy::cognitive_complexity)]
// Each assert! counts as a branch, so a long straight-line test would trip a gate meant for logic.
#![cfg_attr(test, allow(clippy::cognitive_complexity))]
// WHY not in tests: the unit tests under src/ffi call the exported C functions exactly as a C caller does, often with
// invalid arguments on purpose; the contract under test is the thing each call exercises, and one comment per call
// (about 300) would restate the test's own name. The shipped code (every non-test `unsafe`) is held to the lint.
#![cfg_attr(test, allow(clippy::undocumented_unsafe_blocks))]

mod batch;
mod capabilities;
mod diag;
// The only module allowed `unsafe` (it holds the `allow` itself, in its own first lines).
mod ffi;
mod gate;
#[cfg(test)]
mod header_parity;
mod layout;
mod picks;
// WHY crate-private: nothing outside this crate uses the placement core (no test, tool or example imports it);
// the C ABI under ffi::placement is the supported surface, so the Rust types are not a promise.
mod placement;
mod rows;
mod session;
mod submission;

use diag::Error;

type Result<T> = std::result::Result<T, Error>;
