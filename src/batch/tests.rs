// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::capabilities::FORMAT_PARTIAL_F32;

mod edge;

fn bytes_of(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

fn sub<'a>(
    tokens: u32,
    format: u32,
    activations: &'a [u8],
    ids: &'a [i32],
    weights: &'a [f32],
) -> Submission<'a> {
    Submission {
        operation_id: 0,
        tokens,
        format,
        activations,
        expert_ids: ids,
        weights,
        source_modalities: 0,
        placement_epoch: 0,
    }
}

const PARTIAL: Features = Features {
    partial_routes: true,
    compact_rows: false,
    accepts_f16: false,
    accepts_bf16: false,
};

#[test]
fn skips_require_provider_support_and_zero_weight() {
    let config = Config {
        abi_version: 1,
        hidden: 1,
        experts: 2,
        max_tokens: 1,
        top_k: 2,
    };
    let input = bytes_of(&[2.0]);
    let check = |ids: &[i32], weights: &[f32], partial| {
        let features = Features {
            partial_routes: partial,
            ..Features::default()
        };
        Batch::check(config, &sub(1, 1, &input, ids, weights), features)
    };
    let error = check(&[0, -1], &[0.5, 0.0], false).unwrap_err();
    assert_eq!(error.code(), Code::Invalid);
    assert!(error.message().contains("partial-routes"), "{error:?}");
    let error = check(&[0, -1], &[0.5, 0.1], true).unwrap_err();
    assert!(error.message().contains("nonzero weight"), "{error:?}");
    assert!(check(&[0, -2], &[0.5, 0.0], true).is_err());
    assert!(check(&[0, -1], &[0.5, 0.0], true).is_ok());
}

#[test]
fn rejects_invalid_extents_and_expert_ids() {
    let config = Config {
        abi_version: 1,
        hidden: 2,
        experts: 3,
        max_tokens: 2,
        top_k: 2,
    };
    let error = config.lengths(3).unwrap_err();
    assert!(
        error.message().contains("tokens 3 outside 1..=2"),
        "{error:?}"
    );
    assert!(element_count(u32::MAX, u32::MAX, "hidden").is_err());
    let input = bytes_of(&[1.0, 2.0]);
    let error = Batch::check(
        config,
        &sub(1, 1, &input, &[0, 3], &[0.5, 0.5]),
        Features::default(),
    )
    .unwrap_err();
    assert_eq!(error.code(), Code::Invalid);
    assert!(
        error.message().contains("expert id 3 outside 0..3"),
        "{error:?}"
    );
}

#[test]
fn formats_need_a_declaration_and_f32_is_always_accepted() {
    let config = Config {
        abi_version: 1,
        hidden: 2,
        experts: 2,
        max_tokens: 1,
        top_k: 1,
    };
    let half = [0x00_u8, 0x3C, 0x00, 0x40]; // two F16 elements (1.0, 2.0)
    let check =
        |format, features| Batch::check(config, &sub(1, format, &half, &[0], &[1.0]), features);
    for format in [FORMAT_ROUTED_F16, FORMAT_ROUTED_BF16] {
        let error = check(format, Features::default()).unwrap_err();
        assert_eq!(error.code(), Code::Invalid);
        assert!(error.message().contains("not declared"), "{error:?}");
    }
    let f16 = Features {
        accepts_f16: true,
        ..Features::default()
    };
    assert!(check(FORMAT_ROUTED_F16, f16).is_ok());
    assert!(
        check(FORMAT_ROUTED_BF16, f16).is_err(),
        "BF16 is its own flag"
    );
    let error = check(FORMAT_PARTIAL_F32, f16).unwrap_err();
    assert!(error.message().contains("unknown"), "{error:?}");
    let error = check(99, f16).unwrap_err();
    assert!(error.message().contains("unknown"), "{error:?}");
    // F32 with the same 4 bytes is one element short of tokens * hidden, so the count rule fires.
    assert!(check(FORMAT_ROUTED_F32, f16).is_err());
}

#[test]
fn half_precision_bytes_reach_the_provider_unchanged_in_aligned_words() {
    let config = Config {
        abi_version: 1,
        hidden: 3,
        experts: 2,
        max_tokens: 2,
        top_k: 1,
    };
    // 2 tokens * 3 elements * 2 bytes = 12 bytes, but with kept = 1 row the 6 bytes need a padded last word.
    let raw: Vec<u8> = (1..=12).collect();
    let features = Features {
        accepts_bf16: true,
        ..PARTIAL
    };
    let submission = sub(2, FORMAT_ROUTED_BF16, &raw, &[-1, 1], &[0.0, 1.0]);
    Batch::check(config, &submission, features).unwrap();
    let mut batch = Batch::default();
    batch.load(config, &submission, None).unwrap();
    let back: Vec<u8> = batch
        .activations
        .iter()
        .flat_map(|w| w.to_ne_bytes())
        .collect();
    assert_eq!(&back[..12], &raw[..]);
    assert_eq!((batch.tokens, batch.activation_count), (2, 6));
    let mut map = RowMap::default();
    map.derive(submission.expert_ids, 1).unwrap();
    batch.load(config, &submission, Some(&map)).unwrap();
    let back: Vec<u8> = batch
        .activations
        .iter()
        .flat_map(|w| w.to_ne_bytes())
        .collect();
    assert_eq!(&back[..6], &raw[6..12], "only row 1 is kept");
    assert_eq!(back[6..], [0, 0], "the last word is zero padded");
    assert_eq!((batch.tokens, batch.activation_count), (1, 3));
    assert_eq!(batch.format, FORMAT_ROUTED_BF16);
}

#[test]
fn packing_odd_slices_back_to_back_is_the_concatenation() {
    let parts: [&[u8]; 4] = [&[1], &[2, 3, 4, 5, 6], &[], &[7, 8, 9]];
    let mut words = vec![0xDEAD_BEEF];
    pack_words(&mut words, 9, parts).unwrap();
    let back: Vec<u8> = words.iter().flat_map(|w| w.to_ne_bytes()).collect();
    assert_eq!(back, [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0]);
}

#[test]
fn compact_rows_without_partial_routes_is_refused() {
    let compact = Features {
        compact_rows: true,
        ..Features::default()
    };
    assert_eq!(compact.validate().unwrap_err().code(), Code::Invalid);
    let both = Features {
        partial_routes: true,
        ..compact
    };
    assert!(both.validate().is_ok());
}

#[test]
fn an_expert_id_may_not_repeat_within_one_row() {
    let config = Config {
        abi_version: 1,
        hidden: 1,
        experts: 4,
        max_tokens: 2,
        top_k: 2,
    };
    let input = bytes_of(&[1.0, 2.0]);
    let check =
        |ids: &[i32], features| Batch::check(config, &sub(2, 1, &input, ids, &[0.5; 4]), features);
    let error = check(&[1, 3, 2, 2], Features::default()).unwrap_err();
    assert_eq!(error.code(), Code::Invalid);
    assert!(
        error
            .message()
            .contains("pick 3: expert id 2 repeats an earlier pick of the same row"),
        "{error:?}"
    );
    // The same id in different rows is fine: the rule is per row, not per batch.
    assert!(check(&[1, 3, 1, 3], Features::default()).is_ok());
    // Skips are not repeats: -1 means "another provider's pick", and each row may hold several.
    let wide = Config { top_k: 3, ..config };
    let one_row = bytes_of(&[1.0]);
    let skips = sub(1, 1, &one_row, &[1, -1, -1], &[0.5, 0.0, 0.0]);
    let result = Batch::check(wide, &skips, PARTIAL);
    assert!(result.is_ok(), "{result:?}");
}
