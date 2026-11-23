// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! Epoch rules, formats, COMPACT_ROWS and the counters, through `Session::submit`.

use super::*;
use crate::capabilities::{FORMAT_ROUTED_BF16, FORMAT_ROUTED_F16};
use std::sync::{Arc, Mutex};

fn shape() -> Config {
    Config {
        abi_version: 1,
        hidden: 2,
        experts: 3,
        max_tokens: 4,
        top_k: 2,
    }
}

fn features(compact: bool) -> Features {
    Features {
        partial_routes: true,
        compact_rows: compact,
        ..Features::default()
    }
}

/// What the provider saw on each call: kept token count and activation element count.
type Seen = Arc<Mutex<Vec<(u32, usize)>>>;

/// A session on the reference transform that records what it was handed.
fn recording(features: Features) -> (Session, Seen) {
    let seen: Seen = Seen::default();
    let log = Arc::clone(&seen);
    let mut session = Session::with_features(1, features, move |_, config, batch, output| {
        log.lock()
            .unwrap()
            .push((batch.tokens, batch.activation_count));
        reference(config, batch, output)
    })
    .unwrap();
    session.prepare(shape()).unwrap();
    (session, seen)
}

struct Input {
    tokens: u32,
    activations: Vec<u8>,
    ids: Vec<i32>,
    weights: Vec<f32>,
}

impl Input {
    fn submission(&self, epoch: u64) -> Submission<'_> {
        Submission {
            operation_id: 0,
            tokens: self.tokens,
            format: FORMAT_ROUTED_F32,
            activations: &self.activations,
            expert_ids: &self.ids,
            weights: &self.weights,
            source_modalities: 0,
            placement_epoch: epoch,
        }
    }
}

/// Four tokens: rows 1 and 3 are skipped entirely, row 2 keeps one of its two picks.
fn mixed() -> Input {
    Input {
        tokens: 4,
        activations: f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        ids: vec![0, 1, -1, -1, 2, -1, -1, -1],
        weights: vec![0.5, 0.25, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    }
}

fn dense() -> Input {
    Input {
        tokens: 1,
        activations: f32_bytes(&[1.0, 2.0]),
        ids: vec![0, 2],
        weights: vec![0.25, 0.75],
    }
}

fn run(session: &mut Session, input: &Input, epoch: u64) -> Result<Vec<f32>> {
    let ticket = session.submit(input.submission(epoch))?;
    session.collect(ticket, PATIENT).map(<[f32]>::to_vec)
}

#[test]
fn a_session_that_never_versions_accepts_epoch_zero_forever() {
    let (mut session, _) = recording(Features::default());
    for _ in 0..3 {
        run(&mut session, &dense(), 0).unwrap();
    }
    assert_eq!(session.epoch_high, 0);
}

#[test]
fn a_stale_epoch_is_busy_and_zero_after_nonzero_is_invalid() {
    let (mut session, _) = recording(Features::default());
    run(&mut session, &dense(), 5).unwrap();
    assert_eq!(session.epoch_high, 5);
    assert_eq!(code(run(&mut session, &dense(), 4)), Code::BusyOrState);
    assert_eq!(code(run(&mut session, &dense(), 0)), Code::Invalid);
    assert_eq!(session.epoch_high, 5, "refused batches never move the mark");
    run(&mut session, &dense(), 5).unwrap();
    run(&mut session, &dense(), 7).unwrap();
    assert_eq!(session.epoch_high, 7);
}

#[test]
fn a_batch_rejected_after_the_epoch_check_leaves_epoch_high_and_counters() {
    let (mut session, _) = recording(Features::default());
    run(&mut session, &dense(), 5).unwrap();
    let mut bad = dense();
    bad.ids = vec![0, 9]; // expert 9 of 3
    assert_eq!(code(session.submit(bad.submission(9))), Code::Invalid);
    assert_eq!(session.epoch_high, 5);
    let status = session.status();
    assert_eq!((status.submitted, status.rows_submitted), (1, 1));
    assert_eq!(status.outstanding, 0, "no ticket was issued");
    run(&mut session, &dense(), 6).unwrap();
}

#[test]
fn compacted_output_equals_uncompacted_output() {
    let (mut plain, plain_seen) = recording(features(false));
    let (mut compact, compact_seen) = recording(features(true));
    let expected = run(&mut plain, &mixed(), 0).unwrap();
    let actual = run(&mut compact, &mixed(), 0).unwrap();
    assert_eq!(actual, expected, "the caller cannot tell");
    assert_eq!(actual.len(), 8);
    assert_eq!(actual[2..4], [0.0, 0.0], "skipped row 1 is zeros");
    assert_eq!(actual[6..], [0.0, 0.0], "skipped row 3 is zeros");
    assert_eq!(*plain_seen.lock().unwrap(), [(4, 8)]);
    assert_eq!(
        *compact_seen.lock().unwrap(),
        [(2, 4)],
        "provider saw kept rows"
    );
    let (plain, compact) = (plain.status(), compact.status());
    assert_eq!((plain.rows_submitted, plain.rows_shipped), (4, 4));
    assert_eq!((compact.rows_submitted, compact.rows_shipped), (4, 2));
}

#[test]
fn compaction_leaves_a_dense_f32_batch_unchanged() {
    let (mut plain, _) = recording(features(false));
    let (mut compact, seen) = recording(features(true));
    let input = Input {
        tokens: 2,
        activations: f32_bytes(&[1.0, -2.0, 0.5, 4.0]),
        ids: vec![0, 1, 2, 0],
        weights: vec![0.5, 0.5, 0.25, 0.75],
    };
    let expected = run(&mut plain, &input, 0).unwrap();
    assert_eq!(run(&mut compact, &input, 0).unwrap(), expected);
    assert_eq!(*seen.lock().unwrap(), [(2, 4)]);
}

#[test]
fn an_all_skipped_batch_never_calls_the_provider_and_collects_zeros() {
    let (mut session, seen) = recording(features(true));
    let skipped = Input {
        tokens: 2,
        activations: f32_bytes(&[9.0, 9.0, 9.0, 9.0]),
        ids: vec![-1; 4],
        weights: vec![0.0; 4],
    };
    assert_eq!(run(&mut session, &skipped, 0).unwrap(), [0.0; 4]);
    assert!(seen.lock().unwrap().is_empty(), "the provider was called");
    // A dense batch afterwards works: the zero-row buffer was not left in a bad state.
    assert_eq!(run(&mut session, &dense(), 0).unwrap(), [2.5, 5.0]);
    let status = session.status();
    assert_eq!((status.submitted, status.collected), (2, 2));
    assert_eq!((status.rows_submitted, status.rows_shipped), (3, 1));
}

#[test]
fn compact_rows_needs_partial_routes_at_open() {
    let both = Features {
        compact_rows: true,
        ..Features::default()
    };
    let error = Session::with_features(1, both, |_, _, _, _| Ok(()))
        .err()
        .unwrap();
    assert_eq!(error.code(), Code::Invalid);
}

fn half_input() -> Input {
    Input {
        tokens: 1,
        activations: vec![0x00, 0x3C, 0x00, 0x40], // two 16-bit elements
        ids: vec![0, 1],
        weights: vec![0.5, 0.5],
    }
}

#[test]
fn an_undeclared_format_is_invalid_and_changes_nothing() {
    let (mut session, seen) = recording(Features::default());
    for format in [FORMAT_ROUTED_F16, FORMAT_ROUTED_BF16] {
        let input = half_input();
        let mut submission = input.submission(3);
        submission.format = format;
        assert_eq!(code(session.submit(submission)), Code::Invalid);
    }
    assert_eq!(session.epoch_high, 0);
    assert_eq!(session.status().submitted, 0);
    assert!(seen.lock().unwrap().is_empty());
}

#[test]
fn a_declared_half_format_reaches_the_provider_as_the_exact_bytes() {
    type Bytes = Arc<Mutex<Vec<(u32, Vec<u8>)>>>;
    let seen: Bytes = Bytes::default();
    let log = Arc::clone(&seen);
    let declared = Features {
        accepts_f16: true,
        ..Features::default()
    };
    let mut session = Session::with_features(1, declared, move |_, _, batch, output| {
        let bytes = batch.activations.iter().flat_map(|w| w.to_ne_bytes());
        let kept = bytes.take(batch.activation_count * 2).collect();
        log.lock().unwrap().push((batch.format, kept));
        output.fill(1.0);
        Ok(())
    })
    .unwrap();
    session.prepare(shape()).unwrap();
    let input = half_input();
    let mut submission = input.submission(0);
    submission.format = FORMAT_ROUTED_F16;
    let ticket = session.submit(submission).unwrap();
    assert_eq!(session.collect(ticket, PATIENT).unwrap(), [1.0, 1.0]);
    assert_eq!(
        *seen.lock().unwrap(),
        [(FORMAT_ROUTED_F16, vec![0x00, 0x3C, 0x00, 0x40])]
    );
    // BF16 was not declared by this provider.
    let mut submission = input.submission(0);
    submission.format = FORMAT_ROUTED_BF16;
    assert_eq!(code(session.submit(submission)), Code::Invalid);
}
