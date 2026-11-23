// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! The out-of-range refusal must not tell a partial-routes provider to become one: the skips hint belongs only to
//! a provider without PARTIAL_ROUTES that sent the skip id; every other bad id gets the valid range and the marker.

use super::*;

fn refusal(id: i32, partial: bool) -> String {
    let rule = PickRule {
        experts: 6,
        partial,
    };
    pick_error(2, (id, 0.5), &[], rule)
        .expect("the id is out of range")
        .message()
        .to_string()
}

#[test]
fn a_partial_provider_is_not_told_to_be_partial_for_an_id_past_the_range() {
    let text = refusal(6, true);
    assert!(
        text.starts_with("pick 2: expert id 6 outside 0..6"),
        "{text}"
    );
    assert!(!text.contains("need a partial-routes provider"), "{text}");
    assert!(text.contains("-1 is the skip marker"), "{text}");
}

#[test]
fn a_partial_provider_gets_the_range_for_a_negative_id_that_is_not_the_skip_id() {
    let text = refusal(-2, true);
    assert!(!text.contains("need a partial-routes provider"), "{text}");
    assert!(text.contains("-1 is the skip marker"), "{text}");
}

#[test]
fn only_the_skip_id_on_a_provider_without_partial_routes_gets_the_skips_hint() {
    let text = refusal(SKIP_EXPERT_ID, false);
    assert!(
        text.contains("skips need a partial-routes provider"),
        "{text}"
    );
    let text = refusal(6, false);
    assert!(!text.contains("need a partial-routes provider"), "{text}");
    assert!(text.contains("-1 is the skip marker"), "{text}");
}
