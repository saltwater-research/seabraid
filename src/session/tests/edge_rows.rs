// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! Overnight hardening O1, rows R*: all-skip rows with and without COMPACT_ROWS (see edge.rs).

use super::edge::{cfg, echo, partial, sub};
use super::*;
use crate::capabilities::FORMAT_ROUTED_F16;
use std::sync::{Arc, Mutex};

// case R1-R8: all-skip rows with PARTIAL_ROUTES, with and without COMPACT_ROWS.
#[test]
fn all_skip_rows_without_compaction_still_reach_the_provider_in_full() {
    let (mut session, seen) = echo(cfg(2, 3, 3, 2), partial(false));
    let act = f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let ids = [-1; 6];
    let ticket = session.submit(sub(3, &act, &ids, &[0.0; 6], 0)).unwrap();
    let out = session.collect(ticket, PATIENT).unwrap().to_vec();
    assert_eq!(
        out,
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "the provider alone decides what a skipped row yields"
    );
    assert_eq!(*seen.lock().unwrap(), [(3, 6)]);
    assert_eq!(
        (
            session.status().rows_submitted,
            session.status().rows_shipped
        ),
        (3, 3)
    );
}

#[test]
fn all_skip_batch_with_compaction_at_max_tokens_collects_zeros_and_reuse_after_a_busy_batch_is_clean(
) {
    let (mut session, seen) = echo(cfg(2, 3, 3, 2), partial(true));
    let act = f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    // A busy batch first so the worker's scratch and output buffers hold stale numbers.
    let ticket = session
        .submit(sub(3, &act, &[0, 1, 1, 2, 2, 0], &[1.0; 6], 0))
        .unwrap();
    assert_eq!(session.collect(ticket, PATIENT).unwrap().len(), 6);
    let ticket = session
        .submit(sub(3, &act, &[-1; 6], &[0.0; 6], 0))
        .unwrap();
    assert_eq!(session.collect(ticket, PATIENT).unwrap(), [0.0; 6]);
    assert_eq!(
        seen.lock().unwrap().len(),
        1,
        "the provider ran for the busy batch only"
    );
    let status = session.status();
    assert_eq!(
        (status.rows_submitted, status.rows_shipped, status.collected),
        (6, 3, 2)
    );
    // One token, all skip, then a normal batch: the idle buffer is intact.
    let ticket = session
        .submit(sub(1, &act[..8], &[-1, -1], &[0.0; 2], 0))
        .unwrap();
    assert_eq!(session.collect(ticket, PATIENT).unwrap(), [0.0; 2]);
    let ticket = session
        .submit(sub(1, &act[..8], &[2, -1], &[1.0, 0.0], 0))
        .unwrap();
    assert_eq!(session.collect(ticket, PATIENT).unwrap(), [1.0, 2.0]);
}

#[test]
fn compaction_of_odd_width_f16_rows_keeps_the_kept_rows_bytes_contiguous() {
    let features = Features {
        accepts_f16: true,
        ..partial(true)
    };
    let words = Arc::new(Mutex::new((0u32, Vec::new())));
    let log = Arc::clone(&words);
    let mut session = Session::with_features(1, features, move |_, _, batch, output| {
        *log.lock().unwrap() = (batch.tokens, batch.activations.clone());
        output.fill(7.0);
        Ok(())
    })
    .unwrap();
    session.prepare(cfg(3, 2, 3, 1)).unwrap();
    // Three F16 rows of 3 elements (6 bytes): rows 0 and 2 kept, row 1 skipped.
    let bytes: Vec<u8> = (1..=9u8).flat_map(|v| [v, 0xA0]).collect();
    let s = Submission {
        format: FORMAT_ROUTED_F16,
        ..sub(3, &bytes, &[0, -1, 1], &[1.0, 0.0, 1.0], 0)
    };
    let ticket = session.submit(s).unwrap();
    let out = session.collect(ticket, PATIENT).unwrap().to_vec();
    assert_eq!(out, [7.0, 7.0, 7.0, 0.0, 0.0, 0.0, 7.0, 7.0, 7.0]);
    let (tokens, held) = words.lock().unwrap().clone();
    assert_eq!(tokens, 2);
    let flat: Vec<u8> = held.iter().flat_map(|w| w.to_ne_bytes()).collect();
    let expect: Vec<u8> = [&bytes[..6], &bytes[12..18]].concat();
    assert_eq!(&flat[..12], &expect[..]);
}
