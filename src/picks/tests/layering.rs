// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! The dependency edge of the pick rule runs one way: `batch` and `rows` import from `picks`, never the reverse.
//! WHY a source test: a module cycle compiles in Rust, so only a check of the imports keeps `SKIP_EXPERT_ID` (which
//! once lived in `batch.rs` and was imported back by `picks.rs` and `rows.rs`) from drifting home again.

#[test]
fn picks_and_rows_never_import_batch_and_batch_does_not_define_the_skip_marker() {
    let picks = include_str!("../../picks.rs");
    let rows = include_str!("../../rows.rs");
    let batch = include_str!("../../batch.rs");
    for (name, source) in [("picks.rs", picks), ("rows.rs", rows)] {
        assert!(
            !source.contains("crate::batch"),
            "{name} imports batch: the edge must run batch -> picks and rows -> picks only"
        );
    }
    assert!(
        !batch.contains("const SKIP_EXPERT_ID"),
        "SKIP_EXPERT_ID is owned by picks.rs"
    );
    assert!(picks.contains("pub(crate) const SKIP_EXPERT_ID"));
}
