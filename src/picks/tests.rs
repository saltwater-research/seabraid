// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! The wide path (sort) must name exactly the pick and the text the narrow path (scan) names, whatever mix of
//! defects a batch holds. The oracle is the narrow rule itself: `pick_error` run over every pick with the whole
//! earlier part of its row, which is what a row of any width would get if no row were ever sorted.

use super::*;

mod golden;
mod layering;
mod range_hint;

/// splitmix64: a stdlib-free deterministic generator, so a failing seed replays anywhere.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// What the narrow scan says about the batch: the first refusal in pick order, as text.
fn scan(ids: &[i32], weights: &[f32], top_k: usize, rule: PickRule) -> Option<String> {
    (0..ids.len()).find_map(|index| {
        let earlier = &ids[index - index % top_k..index];
        pick_error(index, (ids[index], weights[index]), earlier, rule)
            .map(|error| error.message().to_string())
    })
}

fn verdict(ids: &[i32], weights: &[f32], top_k: usize, rule: PickRule) -> Option<String> {
    check((ids, weights), top_k, rule)
        .err()
        .map(|error| error.message().to_string())
}

/// `rows` clean rows of `top_k` distinct experts (some picks skipped on a partial provider), then up to three
/// defects of any kind at random picks. WHY three helpers called in this order: each draws from `rng` exactly as the
/// one function they were split from did, so every seed still yields the same batch.
fn batch(rng: &mut Rng, top_k: usize, rule: PickRule) -> (Vec<i32>, Vec<f32>) {
    let mut ids = clean_rows(rng, top_k, rule);
    let mut weights = vec![0.5f32; ids.len()];
    if rule.partial {
        skip_some(rng, &mut ids, &mut weights);
    }
    for _ in 0..rng.below(4) {
        let at = rng.below(ids.len());
        defect(rng, (&mut ids, &mut weights), at, top_k, rule);
    }
    (ids, weights)
}

/// One to three rows, each `top_k` distinct experts drawn by a partial Fisher-Yates shuffle.
fn clean_rows(rng: &mut Rng, top_k: usize, rule: PickRule) -> Vec<i32> {
    let rows = 1 + rng.below(3);
    let mut ids = Vec::new();
    for _ in 0..rows {
        let mut pool: Vec<i32> = (0..rule.experts as i32).collect();
        for at in 0..top_k {
            let pick = at + rng.below(pool.len() - at);
            pool.swap(at, pick);
        }
        ids.extend_from_slice(&pool[..top_k]);
    }
    ids
}

/// Skips about one pick in eight, as a partial provider may.
fn skip_some(rng: &mut Rng, ids: &mut [i32], weights: &mut [f32]) {
    for at in 0..ids.len() {
        if rng.below(8) == 0 {
            (ids[at], weights[at]) = (SKIP_EXPERT_ID, 0.0);
        }
    }
}

/// One defect of a random kind at pick `at`.
fn defect(
    rng: &mut Rng,
    (ids, weights): (&mut [i32], &mut [f32]),
    at: usize,
    top_k: usize,
    rule: PickRule,
) {
    let row_start = at - at % top_k;
    match rng.below(6) {
        0 => ids[at] = ids[row_start + rng.below(top_k)], // may copy itself: then no defect
        1 => ids[at] = rule.experts as i32 + rng.below(5) as i32,
        2 => ids[at] = -2 - rng.below(3) as i32,
        3 => weights[at] = f32::NAN,
        4 => weights[at] = f32::INFINITY,
        _ => (ids[at], weights[at]) = (SKIP_EXPERT_ID, 1.0),
    }
}

#[test]
fn the_sort_path_refuses_exactly_what_the_scan_refuses_for_seeded_multi_defect_batches() {
    let (mut refused, mut sorted_rows) = (0, 0);
    for seed in 0..1500u64 {
        for top_k in [63usize, 64, 65, 66, 130] {
            for partial in [false, true] {
                let mut rng = Rng(seed * 8 + top_k as u64);
                let experts = (top_k + 1 + rng.below(2 * top_k)) as u32;
                let rule = PickRule { experts, partial };
                let (ids, weights) = batch(&mut rng, top_k, rule);
                WIDE_ROWS.with(|rows| rows.set(0));
                let got = verdict(&ids, &weights, top_k, rule);
                let want = scan(&ids, &weights, top_k, rule);
                assert_eq!(got, want, "seed {seed} top_k {top_k} partial {partial}");
                let sorted = WIDE_ROWS.with(|rows| rows.get());
                assert_eq!(
                    sorted > 0,
                    top_k > NARROW_ROW,
                    "top_k {top_k} took the other path"
                );
                refused += usize::from(want.is_some());
                sorted_rows += sorted;
            }
        }
    }
    assert!(
        refused > 5_000 && sorted_rows > 5_000,
        "{refused} refused, {sorted_rows} sorted rows"
    );
}

#[test]
fn the_lowest_index_defect_wins_whatever_its_kind() {
    let rule = PickRule {
        experts: 300,
        partial: true,
    };
    let mut ids: Vec<i32> = (0..400).map(|i| i % 200).collect();
    let mut weights = vec![0.5f32; 400];
    // A repeat at pick 150 (row 0), a bad id at pick 180: the repeat is first.
    ids[150] = 7;
    ids[180] = 300;
    let text = verdict(&ids, &weights, 200, rule).unwrap();
    assert_eq!(
        text,
        "pick 150: expert id 7 repeats an earlier pick of the same row"
    );
    // The bad id first: it wins over the repeat after it.
    ids[150] = 150;
    ids[180] = 7;
    ids[100] = 300;
    let text = verdict(&ids, &weights, 200, rule).unwrap();
    assert!(
        text.starts_with("pick 100: expert id 300 outside"),
        "{text}"
    );
    // A repeat in row 0 against a NaN weight in row 1: the earlier row wins.
    ids[100] = 100;
    ids[180] = 180;
    ids[5] = 4;
    weights[210] = f32::NAN;
    let text = verdict(&ids, &weights, 200, rule).unwrap();
    assert_eq!(
        text,
        "pick 5: expert id 4 repeats an earlier pick of the same row"
    );
}
