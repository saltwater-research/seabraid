// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! The two v2 entry points: `seabraid_session_open` (provider v3, config moved into open) and
//! `seabraid_session_submit` (batch v2).
//!
//! OWNER: this module owns the v2 open sequence and the submit entry (the checks' order is not a promise); the C-to-`Submission`
//! conversion lives with the struct (batch_v2.rs); the pick rule stays in picks.rs, the count rules in batch.rs,
//! the epoch and ticket rules in session.rs, and the `struct_size` protocol in sized.rs.
//! CONTRACT: exactly the two exports named above, with the status codes of include/seabraid/seabraid.h; `*out` is
//! NULL on every failed open and `*out_ticket` is untouched on every failed submit.
//! SIZE: under 150 lines. FAILURE: fail-fast; every failure is logged once by `status`/`status_on`.
//! CHECKER: src/ffi/tests/v2.rs (every branch) and native-tests/provider_smoke.cpp (from C).

use super::batch_v2::{BatchV2, BATCH_V2_VERSION};
use super::provider_v3::{ProviderV3, REFERENCE};
use super::sized::{read_in, Layout};
use super::{check_pointer, open, session, status_on, SessionHandle};
use crate::batch;
use crate::capabilities::validate_origins;
use crate::diag::{fail, Code};
use crate::session::Session;
use crate::Result;

/// Opens a session from a v3 descriptor and prepares it with the descriptor's config.
#[no_mangle]
pub unsafe extern "C" fn seabraid_session_open(
    provider: *const ProviderV3,
    out: *mut *mut SessionHandle,
) -> i32 {
    // SAFETY: `ProviderV3::load` checks the pointer and version before reading (the caller promises a descriptor
    // readable for its `struct_size`), `provider.config()` has the same duty for the `Config` it points to, and
    // `open` checks `out` non-null before writing it.
    unsafe {
        open("seabraid_session_open", out, || {
            let provider = ProviderV3::load(provider)?;
            let config = provider.config()?;
            let mut session = start(&provider)?;
            session.prepare(config)?;
            Ok(session)
        })
    }
}

fn start(provider: &ProviderV3) -> Result<Session> {
    // WHY the descriptor's own bits: the session refuses COMPACT_ROWS without PARTIAL_ROUTES at open, so the
    // v3 flag check and the session agree by construction.
    let features = provider.features();
    if provider.flags & REFERENCE != 0 {
        return Session::with_features(
            provider.operation_count,
            features,
            |_, config, batch, output| batch::reference(config, batch, output),
        );
    }
    let callback = provider
        .callback()
        .ok_or_else(|| fail!(Code::Invalid, "provider.execute is null"))?;
    Session::with_features(
        provider.operation_count,
        features,
        move |operation, config, batch, output| callback.run(operation, config, batch, output),
    )
}

/// Validates a v2 batch and submits it; copies happen only after every check passes, so a refused batch changes
/// nothing. WHY no promised order: the contract is "refused means unchanged", and a frozen validation order
/// would pin every internal reordering as an ABI break (docs/ABI.md, "Submit").
#[no_mangle]
pub unsafe extern "C" fn seabraid_session_submit(
    handle: *mut SessionHandle,
    batch: *const BatchV2,
    out_ticket: *mut u64,
) -> i32 {
    // SAFETY: Each caller pointer is validated before use: the lease (`session`) refuses a dead, busy or poisoned
    // handle, `check_pointer` covers `out_ticket`, `read_in` checks `batch` (non-null, version, `struct_size`), and
    // `submission` forms slices only after the counts match the config and the pointers are non-null; the caller
    // promises the arrays stay unwritten during the call. `out_ticket` is written once, after success.
    unsafe {
        status_on("seabraid_session_submit", handle, || {
            // The lease comes first, as in `seabraid_submit_operation`: a refusal about the handle outranks
            // one about an argument on both submit entries, so the same bad call gets the same status on each.
            let mut session = session(handle)?;
            check_pointer(out_ticket, "out_ticket")?;
            let layout = Layout::Sized {
                base: BatchV2::BASE_SIZE,
            };
            let batch = read_in(batch, "batch", BATCH_V2_VERSION, layout)?;
            let config = session.ready_config(batch.operation_id)?;
            validate_origins(batch.source_modalities)?;
            let ticket = session.submit(batch.submission(config)?)?;
            out_ticket.write(ticket);
            Ok(())
        })
    }
}
