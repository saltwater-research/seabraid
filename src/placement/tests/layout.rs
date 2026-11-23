// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! Construction, stale-plan and recovery checks for the authoritative map.

use super::*;

#[test]
fn bad_shape_stale_epoch_duplicate_and_fallback_moves_touch_no_provider() {
    assert!(matches!(
        OwnershipMap::new(u32::MAX, u32::MAX, 2, vec![]),
        Err(PlacementError::InvalidLayout)
    ));
    assert!(matches!(
        OwnershipMap::new(1, 1, 2, vec![3]),
        Err(PlacementError::InvalidLayout)
    ));
    let (mut place, mut io) = world();
    assert_eq!(
        place.apply(0, p(0, 2, 0, 0), &mut io),
        Err(PlacementError::StaleEpoch {
            expected: 0,
            actual: 1
        })
    );
    for exchange in [p(0, 2, 2, 0), p(0, 3, 2, 1), p(5, 2, 0, 0), p(0, 0, 2, 0)] {
        assert_eq!(
            place.apply(1, exchange, &mut io),
            Err(PlacementError::InvalidPromotion)
        );
    }
    assert_eq!(place.snapshot().epoch(), 1);
    assert!(io.stages.is_empty() && io.writes.is_empty());
    io.reference_matches(&place);
}

#[test]
fn duplicate_cells_across_pairs_reject_the_round_before_writes() {
    let (mut place, mut io) = world();
    let plan = PlacementPlan {
        epoch: 1,
        round: 1,
        promotions: vec![p(0, 2, 0, 0), p(0, 2, 1, 0)],
    };
    assert!(matches!(
        place.apply_plan(plan, &mut io),
        Err(PlacementError::InvalidPromotion)
    ));
    assert!(io.stages.is_empty() && io.writes.is_empty());
    io.reference_matches(&place);
}

#[test]
fn only_fresh_verified_rebuild_reopens_quarantined_provider() {
    let (mut place, io) = world();
    let owner = place.snapshot().owner.clone();
    place.quarantine(1).unwrap();
    place.quarantine(1).unwrap();
    place.adopt(2, owner.clone()).unwrap();
    assert!(place.snapshot().is_quarantined(1));
    assert_eq!(
        place.rebuild(1, owner.clone()),
        Err(PlacementError::StaleEpoch {
            expected: 1,
            actual: 2
        })
    );
    place.rebuild(2, owner).unwrap();
    assert!(!place.snapshot().is_quarantined(1));
    assert_eq!(place.snapshot().epoch(), 3);
    io.reference_matches(&place);
}

#[test]
fn a_panicking_provider_and_the_quarantine_it_causes_are_logged() {
    use crate::diag::testing::Capture;
    use crate::diag::Level;
    let capture = Capture::start(Level::Debug);
    let (mut place, mut io) = world();
    io.write_fault = [None, Some(IoError::Panicked)].into();
    place.apply(1, p(0, 2, 1, 0), &mut io).unwrap();
    let records = capture.mine();
    let panic = records
        .iter()
        .find(|record| record.level == Level::Error as u32)
        .expect("the panic must be logged at ERROR");
    let text = &panic.message;
    assert!(
        text.contains("Replace on rung 1 layer 0 expert 1"),
        "{text}"
    );
    assert!(text.contains("injected adapter panic"), "{text}");
    let quarantine = records
        .iter()
        .find(|record| record.message.contains("quarantined"))
        .expect("the quarantine must be logged");
    assert_eq!(quarantine.level, Level::Warn as u32);
    assert!(quarantine.message.contains("0x2"), "{}", quarantine.message);
}
