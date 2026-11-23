// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! Overnight hardening O1: geometry extremes, epoch extremes, weight and activation bit patterns, skip rows with
//! and without COMPACT_ROWS, and ticket misuse, all through `Session::submit` / `collect` / `close`.
//! Each test names the row of scratch/2026-11-19-overnight/O1-cases.md it pins.

use super::*;
use crate::capabilities::FORMAT_ROUTED_F16;
use std::sync::{Arc, Mutex};

pub(super) fn cfg(hidden: u32, experts: u32, max_tokens: u32, top_k: u32) -> Config {
    Config {
        abi_version: 1,
        hidden,
        experts,
        max_tokens,
        top_k,
    }
}

pub(super) fn partial(compact: bool) -> Features {
    Features {
        partial_routes: true,
        compact_rows: compact,
        ..Features::default()
    }
}

/// A session whose provider copies activation `i` to output `i` bit for bit and logs the batch it saw.
/// (kept tokens, activation elements) per provider call.
pub(super) type Calls = Arc<Mutex<Vec<(u32, usize)>>>;

pub(super) fn echo(config: Config, features: Features) -> (Session, Calls) {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let log = Arc::clone(&seen);
    let mut session = Session::with_features(1, features, move |_, _, batch, output| {
        log.lock()
            .unwrap()
            .push((batch.tokens, batch.activation_count));
        for (index, slot) in output.iter_mut().enumerate() {
            *slot = batch.f32_at(index);
        }
        Ok(())
    })
    .unwrap();
    session.prepare(config).unwrap();
    (session, seen)
}

pub(super) fn sub<'a>(
    tokens: u32,
    act: &'a [u8],
    ids: &'a [i32],
    weights: &'a [f32],
    epoch: u64,
) -> Submission<'a> {
    Submission {
        operation_id: 0,
        tokens,
        format: FORMAT_ROUTED_F32,
        activations: act,
        expert_ids: ids,
        weights,
        source_modalities: 0,
        placement_epoch: epoch,
    }
}

pub(super) fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn prepare_code(config: Config) -> Option<Code> {
    let mut session = Session::new(1, false, |_, _, _, _| Ok(())).unwrap();
    session.prepare(config).err().map(|e| e.code())
}

// case G1-G6: degenerate and extreme geometry at prepare.
#[test]
fn prepare_refuses_zero_dimensions_and_unaddressable_extents_and_accepts_the_edge() {
    assert_eq!(
        prepare_code(cfg(0, 3, 4, 2)),
        Some(Code::Invalid),
        "hidden 0"
    );
    assert_eq!(
        prepare_code(cfg(2, 3, 0, 2)),
        Some(Code::Invalid),
        "max_tokens 0"
    );
    assert_eq!(
        prepare_code(cfg(2, 3, 4, 0)),
        Some(Code::Invalid),
        "top_k 0"
    );
    assert_eq!(
        prepare_code(cfg(2, 0, 4, 0)),
        Some(Code::Invalid),
        "experts 0"
    );
    assert_eq!(
        prepare_code(cfg(2, 3, 4, 4)),
        Some(Code::Invalid),
        "top_k > experts"
    );
    assert_eq!(
        prepare_code(cfg(1, 1, 1, 1)),
        None,
        "experts == 1, everything 1"
    );
    // tokens * width must fit isize::MAX / 4 elements; 2^29 * (2^32 - 1) is the largest power-of-two token
    // count that does, 2^30 * (2^32 - 1) is not.
    assert_eq!(prepare_code(cfg(u32::MAX, 3, 1 << 29, 2)), None);
    assert_eq!(
        prepare_code(cfg(u32::MAX, 3, 1 << 30, 2)),
        Some(Code::Invalid)
    );
    assert_eq!(prepare_code(cfg(2, u32::MAX, 1 << 29, u32::MAX)), None);
    assert_eq!(
        prepare_code(cfg(2, u32::MAX, 1 << 30, u32::MAX)),
        Some(Code::Invalid)
    );
    assert_eq!(
        prepare_code(cfg(u32::MAX, u32::MAX, u32::MAX, u32::MAX)),
        Some(Code::Invalid)
    );
}

#[test]
fn a_refused_prepare_leaves_the_session_unprepared_and_prepare_again_works() {
    let mut session = Session::new(1, false, |_, _, _, _| Ok(())).unwrap();
    assert_eq!(code(session.prepare(cfg(0, 3, 4, 2))), Code::Invalid);
    assert_eq!(
        code(submit(&mut session)),
        Code::BusyOrState,
        "not prepared"
    );
    session.prepare(config()).unwrap();
    assert_eq!(code(session.prepare(config())), Code::BusyOrState, "twice");
}

// case T1-T5: tokens 0 / max / max+1 / u32::MAX, with the arrays sized for the honest count.
#[test]
fn tokens_zero_and_over_max_are_refused_and_max_is_accepted() {
    let (mut session, _) = echo(cfg(1, 2, 3, 1), Features::default());
    let act = f32_bytes(&[1.0, 2.0, 3.0, 4.0]);
    let (ids, weights) = ([0, 1, 0, 1], [1.0; 4]);
    for tokens in [0, 4, u32::MAX] {
        let s = sub(tokens, &act[..4], &ids[..1], &weights[..1], 0);
        assert_eq!(code(session.submit(s)), Code::Invalid, "tokens {tokens}");
    }
    assert_eq!(session.status().submitted, 0);
    let ticket = session
        .submit(sub(3, &act[..12], &ids[..3], &weights[..3], 0))
        .unwrap();
    assert_eq!(
        bits(session.collect(ticket, PATIENT).unwrap()),
        bits(&[1.0, 2.0, 3.0])
    );
}

// case N1-N6: operation_id out of range, wrong array lengths, odd byte counts.
#[test]
fn operation_id_and_array_length_mismatches_are_invalid_and_change_nothing() {
    let (mut session, _) = echo(cfg(2, 3, 2, 2), Features::default());
    let act = f32_bytes(&[1.0, 2.0, 3.0, 4.0]);
    let mut s = sub(2, &act, &[0, 1, 0, 1], &[0.5; 4], 0);
    s.operation_id = u32::MAX;
    assert_eq!(code(session.submit(s)), Code::Invalid);
    s.operation_id = 1; // operation_count is 1
    assert_eq!(code(session.submit(s)), Code::Invalid);
    let s = sub(2, &act[..12], &[0, 1, 0, 1], &[0.5; 4], 0);
    assert_eq!(
        code(session.submit(s)),
        Code::Invalid,
        "activations short by one element"
    );
    let s = sub(2, &act[..15], &[0, 1, 0, 1], &[0.5; 4], 0);
    assert_eq!(
        code(session.submit(s)),
        Code::Invalid,
        "activations not a whole element"
    );
    let s = sub(2, &act, &[0, 1, 0], &[0.5; 4], 0);
    assert_eq!(code(session.submit(s)), Code::Invalid, "ids short");
    let s = sub(2, &act, &[0, 1, 0, 1], &[0.5; 3], 0);
    assert_eq!(code(session.submit(s)), Code::Invalid, "weights short");
    let status = session.status();
    assert_eq!(
        (status.submitted, status.outstanding, status.rows_submitted),
        (0, 0, 0)
    );
    let ticket = session
        .submit(sub(2, &act, &[0, 1, 0, 1], &[0.5; 4], 0))
        .unwrap();
    session.collect(ticket, PATIENT).unwrap();
}

// case E1-E7: placement_epoch at 1, u64::MAX, below it, zero after it, and after a refused batch.
#[test]
fn epoch_u64_max_is_accepted_repeats_and_nothing_can_follow_it_but_itself() {
    let (mut session, _) = echo(cfg(2, 3, 2, 2), Features::default());
    let d = (f32_bytes(&[1.0, 2.0]), [0, 2], [0.25, 0.75]);
    let run = |session: &mut Session, epoch: u64| {
        let ticket = session.submit(sub(1, &d.0, &d.1, &d.2, epoch))?;
        session.collect(ticket, PATIENT).map(drop)
    };
    run(&mut session, u64::MAX).unwrap();
    assert_eq!(session.epoch_high, u64::MAX);
    run(&mut session, u64::MAX).unwrap();
    assert_eq!(code(run(&mut session, u64::MAX - 1)), Code::BusyOrState);
    assert_eq!(code(run(&mut session, 1)), Code::BusyOrState);
    assert_eq!(code(run(&mut session, 0)), Code::Invalid);
    assert_eq!(session.epoch_high, u64::MAX);
    assert_eq!(session.status().submitted, 2);
}

#[test]
fn a_refused_batch_at_a_huge_epoch_does_not_raise_the_mark() {
    let (mut session, _) = echo(cfg(2, 3, 2, 2), Features::default());
    let act = f32_bytes(&[1.0, 2.0]);
    let bad = session.submit(sub(1, &act, &[0, 7], &[0.25, 0.75], u64::MAX));
    assert_eq!(code(bad), Code::Invalid);
    assert_eq!(session.epoch_high, 0);
    let ticket = session
        .submit(sub(1, &act, &[0, 2], &[0.25, 0.75], 3))
        .unwrap();
    session.collect(ticket, PATIENT).unwrap();
    assert_eq!(
        session.epoch_high, 3,
        "epoch 3 is still fresh after the refused u64::MAX"
    );
}

// case W1-W9: weights of every float class on a real pick, and on a skip.
#[test]
fn weights_nan_and_infinities_are_refused_zero_subnormal_and_max_are_kept_bit_exact() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let log = Arc::clone(&seen);
    let mut session = Session::new(1, false, move |_, _, batch, output| {
        log.lock().unwrap().push(bits(&batch.weights));
        output.fill(0.0);
        Ok(())
    })
    .unwrap();
    session.prepare(cfg(1, 3, 1, 2)).unwrap();
    let act = f32_bytes(&[1.0]);
    for bad in [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(0xFFC0_1234),
    ] {
        let w = [0.5, bad];
        let s = sub(1, &act, &[0, 1], &w, 0);
        assert_eq!(code(session.submit(s)), Code::Invalid, "{bad:?}");
    }
    let good = [
        (-0.0, f32::MIN_POSITIVE / 4.0),
        (f32::MAX, f32::MIN),
        (1e-45, -1e-45),
    ];
    for (a, b) in good {
        let ticket = session.submit(sub(1, &act, &[0, 1], &[a, b], 0)).unwrap();
        session.collect(ticket, PATIENT).unwrap();
        assert_eq!(seen.lock().unwrap().last().unwrap(), &bits(&[a, b]));
    }
    assert_eq!(
        session.status().submitted,
        3,
        "only the three finite batches were accepted"
    );
}

#[test]
fn a_skip_pick_takes_weight_zero_of_either_sign_and_nothing_else() {
    let (mut session, _) = echo(cfg(1, 3, 1, 2), partial(false));
    let act = f32_bytes(&[1.0]);
    for weight in [0.0, -0.0] {
        let ticket = session
            .submit(sub(1, &act, &[1, -1], &[0.5, weight], 0))
            .unwrap();
        session.collect(ticket, PATIENT).unwrap();
    }
    for weight in [f32::NAN, f32::INFINITY, 1e-45, -1e-45, 1.0] {
        let w = [0.5, weight];
        let s = sub(1, &act, &[1, -1], &w, 0);
        assert_eq!(
            code(session.submit(s)),
            Code::Invalid,
            "skip with weight {weight:?}"
        );
    }
    let s = sub(1, &act, &[1, -2], &[0.5, 0.0], 0);
    assert_eq!(code(session.submit(s)), Code::Invalid, "only -1 is a skip");
}

// case A1-A4: activation bit patterns are never interpreted, in F32 and in F16.
#[test]
fn activation_nans_infinities_subnormals_and_negative_zero_pass_through_bit_exact() {
    let (mut session, _) = echo(cfg(4, 2, 1, 1), Features::default());
    let wild = [
        f32::from_bits(0x7FC1_2345), // quiet NaN with a payload
        f32::from_bits(0x7F80_0001), // signalling NaN
        f32::from_bits(0xFF80_0000), // -inf
        f32::from_bits(0x8000_0000), // -0.0
    ];
    let ticket = session
        .submit(sub(1, &f32_bytes(&wild), &[1], &[1.0], 0))
        .unwrap();
    assert_eq!(bits(session.collect(ticket, PATIENT).unwrap()), bits(&wild));
    let tiny = [
        f32::from_bits(1),
        f32::from_bits(0x8000_0001),
        f32::MIN_POSITIVE,
        f32::MAX,
    ];
    let ticket = session
        .submit(sub(1, &f32_bytes(&tiny), &[0], &[1.0], 0))
        .unwrap();
    assert_eq!(bits(session.collect(ticket, PATIENT).unwrap()), bits(&tiny));
}

#[test]
fn f16_nan_payloads_reach_the_provider_as_the_exact_bytes_even_at_odd_widths() {
    let features = Features {
        accepts_f16: true,
        ..Features::default()
    };
    let words = Arc::new(Mutex::new(Vec::new()));
    let log = Arc::clone(&words);
    let mut session = Session::with_features(1, features, move |_, _, batch, output| {
        *log.lock().unwrap() = batch.activations.clone();
        output.fill(0.0);
        Ok(())
    })
    .unwrap();
    session.prepare(cfg(3, 2, 1, 1)).unwrap();
    // F16: quiet NaN with payload 0x7E55, signalling 0x7C01, -inf 0xFC00 -> 6 bytes, padded to two words.
    let bytes: Vec<u8> = [0x7E55_u16, 0x7C01, 0xFC00]
        .iter()
        .flat_map(|h| h.to_ne_bytes())
        .collect();
    let s = Submission {
        format: FORMAT_ROUTED_F16,
        ..sub(1, &bytes, &[0], &[1.0], 0)
    };
    let ticket = session.submit(s).unwrap();
    session.collect(ticket, PATIENT).unwrap();
    let held = words.lock().unwrap().clone();
    let flat: Vec<u8> = held.iter().flat_map(|w| w.to_ne_bytes()).collect();
    assert_eq!(&flat[..6], &bytes[..]);
    assert_eq!(&flat[6..], &[0, 0], "the last word is zero padded");
}
