// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! The pick rule against `golden.txt`, the table the ggml adapter and the Python reference provider are held to as
//! well. WHY the verdict is a kind plus a pick index: that is what `picks::check` names, and the table says which
//! pick is the first defect, so a change of order (not only of verdict) fails here.
//! FAILURE: fail-fast; a malformed golden row panics with the row, a wrong verdict names the row's name.

use super::*;

const GOLDEN: &str = include_str!("../golden.txt");

struct Case<'a> {
    name: &'a str,
    rule: PickRule,
    top_k: usize,
    ids: Vec<i32>,
    weights: Vec<f32>,
    verdict: &'a str,
}

fn weight(token: &str) -> f32 {
    match token {
        "-0" => -0.0,
        other => other
            .parse()
            .unwrap_or_else(|_| panic!("golden weight {other}")),
    }
}

fn parse_case(line: &str) -> Case<'_> {
    let fields: Vec<&str> = line.split(" | ").collect();
    assert_eq!(fields.len(), 5, "golden row: {line}");
    let shape: Vec<usize> = fields[1].split(' ').map(|t| t.parse().unwrap()).collect();
    let ids: Vec<i32> = fields[2].split(' ').map(|t| t.parse().unwrap()).collect();
    let mut weights = vec![0.5f32; ids.len()];
    for entry in fields[3].split(' ').filter(|entry| *entry != "-") {
        let (pick, value) = entry.split_once('=').expect("pick=value");
        weights[pick.parse::<usize>().unwrap()] = weight(value);
    }
    Case {
        name: fields[0].strip_prefix("case ").expect("case row"),
        rule: PickRule {
            experts: shape[0] as u32,
            partial: shape[1] == 1,
        },
        top_k: shape[2],
        ids,
        weights,
        verdict: fields[4],
    }
}

/// `kind@pick` of the refusal `check` named, from its text: the kinds are the four clauses of `pick_error`.
fn verdict_of(error: &Error) -> String {
    let text = error.message();
    let kind = if text.contains("skipped expert has a nonzero weight") {
        "skip_weight"
    } else if text.contains("outside 0..") {
        "range"
    } else if text.contains("weight is not finite") {
        "finite"
    } else if text.contains("repeats an earlier pick") {
        "repeat"
    } else {
        panic!("a refusal outside the table's kinds: {text}")
    };
    let pick = text.strip_prefix("pick ").and_then(|t| t.split(':').next());
    format!("{kind}@{}", pick.expect("every refusal names its pick"))
}

fn cases() -> Vec<Case<'static>> {
    GOLDEN
        .lines()
        .filter(|line| line.starts_with("case "))
        .map(parse_case)
        .collect()
}

#[test]
fn every_row_of_the_golden_table_gets_its_verdict_on_the_path_its_width_selects() {
    let table = cases();
    assert!(table.len() >= 30, "only {} golden rows", table.len());
    let mut wide_seen = 0;
    for case in &table {
        assert_eq!(case.ids.len() % case.top_k, 0, "{}", case.name);
        let before = wide_rows_sorted();
        let got = match check((&case.ids, &case.weights), case.top_k, case.rule) {
            Ok(()) => "ok".to_string(),
            Err(error) => verdict_of(&error),
        };
        assert_eq!(got, case.verdict, "golden row {}", case.name);
        // WHY: the sort path must really have run for a wide row and not for a narrow one, or "wide" rows would
        // pass on the scan and prove nothing about the sort.
        let sorted = wide_rows_sorted() - before;
        assert_eq!(sorted > 0, case.top_k > NARROW_ROW, "{}", case.name);
        wide_seen += usize::from(sorted > 0);
    }
    assert!(wide_seen >= 6, "only {wide_seen} wide golden rows");
}

#[test]
fn the_golden_limits_are_the_constants_this_module_uses() {
    let limit = |name: &str| -> usize {
        let row = GOLDEN
            .lines()
            .find_map(|line| line.strip_prefix(&format!("limit {name} ")))
            .unwrap_or_else(|| panic!("golden.txt has no `limit {name}` row"));
        row.trim().parse().unwrap()
    };
    assert_eq!(limit("narrow_row"), NARROW_ROW);
    // max_picks is the ggml adapter's constant; adapters/ggml/test_routes.c holds it to this row.
    assert!(limit("max_picks") >= 1);
}
