// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! Overnight hardening O1: the pick rules at the extremes of top_k, experts and id values.
//! Case ids refer to scratch/2026-11-19-overnight/O1-cases.md (rows B*).

use super::*;
use std::time::{Duration, Instant};

fn cfg(experts: u32, top_k: u32) -> Config {
    Config {
        abi_version: 1,
        hidden: 1,
        experts,
        max_tokens: 2,
        top_k,
    }
}

fn check(
    config: Config,
    tokens: u32,
    ids: &[i32],
    weights: &[f32],
    features: Features,
) -> Result<()> {
    let input = bytes_of(&vec![1.0; tokens as usize]);
    Batch::check(config, &sub(tokens, 1, &input, ids, weights), features)
}

// case B1-B3: the repeat rule must stay linear in the row width. A row of 64,000 distinct picks is a legal
// batch for a config with top_k 64,000; the O(top_k^2) scan took 12.7 s on the debug build, the sort is a few ms.
#[test]
fn a_very_wide_row_is_checked_in_time_proportional_to_its_width() {
    let top_k: u32 = 64_000;
    let ids: Vec<i32> = (0..top_k as i32).collect();
    let weights = vec![0.5f32; top_k as usize];
    let start = Instant::now();
    check(cfg(top_k, top_k), 1, &ids, &weights, Features::default()).unwrap();
    let spent = start.elapsed();
    assert!(
        spent < Duration::from_secs(3),
        "a distinct 64,000-pick row took {spent:?}"
    );
}

// case B4-B8: the wide path must refuse exactly what the narrow path refuses, with the same text.
#[test]
fn a_wide_row_still_refuses_a_repeat_a_bad_id_and_a_nonfinite_weight_and_allows_repeated_skips() {
    let top_k: u32 = 200; // wider than the narrow scan's limit
    let experts = 300;
    let row = |tweak: &dyn Fn(&mut Vec<i32>, &mut Vec<f32>)| {
        let (mut ids, mut weights): (Vec<i32>, Vec<f32>) =
            ((0..top_k as i32).collect(), vec![0.5; top_k as usize]);
        tweak(&mut ids, &mut weights);
        check(cfg(experts, top_k), 1, &ids, &weights, PARTIAL)
    };
    // WHY the counter: the narrow and the wide path refuse the same text, so only it shows this test that the
    // sort path ran (a build that scans every row would pass every text assertion below).
    let before = crate::picks::wide_rows_sorted();
    assert!(row(&|_, _| {}).is_ok());
    let sorted = crate::picks::wide_rows_sorted() - before;
    assert_eq!(sorted, 1, "a 200-pick row is checked by the sort path");
    let error = row(&|ids, _| ids[150] = 7).unwrap_err();
    assert_eq!(error.code(), Code::Invalid);
    assert!(
        error
            .message()
            .contains("pick 150: expert id 7 repeats an earlier pick of the same row"),
        "{error:?}"
    );
    let error = row(&|ids, _| ids[3] = 300).unwrap_err();
    assert!(
        error
            .message()
            .contains("pick 3: expert id 300 outside 0..300"),
        "{error:?}"
    );
    let error = row(&|_, weights| weights[199] = f32::NAN).unwrap_err();
    assert!(
        error.message().contains("pick 199: weight is not finite"),
        "{error:?}"
    );
    // Many skips in one row are not repeats; neither is the same real id in two different rows.
    assert!(row(&|ids, weights| (100..200).for_each(|i| (ids[i], weights[i]) = (-1, 0.0))).is_ok());
    let rows = [(0..200).collect::<Vec<i32>>(), (0..200).collect()].concat();
    assert!(check(cfg(experts, top_k), 2, &rows, &vec![0.5; 400], PARTIAL).is_ok());
    // A repeat in the second row only, past a clean first row.
    let mut second = rows.clone();
    second[399] = 0;
    let error = check(cfg(experts, top_k), 2, &second, &vec![0.5; 400], PARTIAL).unwrap_err();
    assert!(
        error.message().contains("pick 399: expert id 0 repeats"),
        "{error:?}"
    );
}

// case B9-B14: id values at the limits of i32 and of `experts`.
#[test]
fn expert_ids_at_the_edges_of_i32_and_of_the_expert_count() {
    let one = |id: i32, experts: u32| check(cfg(experts, 1), 1, &[id], &[1.0], Features::default());
    assert!(one(0, 1).is_ok());
    assert!(
        one(i32::MAX, i32::MAX as u32 + 1).is_ok(),
        "the last id of 2^31 experts"
    );
    assert!(one(i32::MAX, i32::MAX as u32).is_err());
    assert!(one(i32::MAX - 1, i32::MAX as u32).is_ok());
    assert!(
        one(0, u32::MAX).is_ok(),
        "an expert count above i32::MAX is not an overflow"
    );
    for id in [-1, -2, i32::MIN] {
        assert!(one(id, u32::MAX).is_err(), "id {id}");
    }
    // With partial routes only -1 is a skip.
    let skip = |id: i32| check(cfg(4, 1), 1, &[id], &[0.0], PARTIAL);
    assert!(skip(-1).is_ok());
    assert!(skip(-2).is_err() && skip(i32::MIN).is_err());
}

// case B15-B17: top_k == experts with every id used, with experts == 1, and top_k 1 of many.
#[test]
fn top_k_equal_to_experts_uses_every_id_once_and_one_expert_takes_one_pick() {
    assert!(check(
        cfg(3, 3),
        2,
        &[0, 1, 2, 2, 0, 1],
        &[0.1; 6],
        Features::default()
    )
    .is_ok());
    assert!(check(cfg(3, 3), 1, &[0, 1, 0], &[0.1; 3], Features::default()).is_err());
    assert!(check(cfg(1, 1), 2, &[0, 0], &[1.0, 1.0], Features::default()).is_ok());
    assert!(check(cfg(1, 1), 1, &[1], &[1.0], Features::default()).is_err());
}

// case R1: the sign of an all-zero reference result must not depend on the toolchain's `Iterator::sum` start value.
// A zero weight times (negative activation x scale 0) is -0.0, and -0.0 + -0.0 stays -0.0 only if the fold starts there.
#[test]
fn an_all_zero_reference_result_is_negative_zero_on_every_toolchain() {
    let config = cfg(2, 1);
    let input = bytes_of(&[-1.0]);
    let mut batch = Batch::default();
    batch
        .load(config, &sub(1, 1, &input, &[-1], &[0.0]), None)
        .unwrap();
    let mut output = [1.0f32];
    reference(config, &batch, &mut output).unwrap();
    assert_eq!(output[0].to_bits(), 0x8000_0000);
}
