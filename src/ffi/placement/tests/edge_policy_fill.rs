// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! Overnight hardening O1, rows P60-P73: a degenerate policy at open and the fill entry point's degenerate requests.
//! WHY its own file: both are argument-shape edges of entry points that never touch a provider, unlike the apply and
//! slot-size rows that stay in edge_io.rs.

use super::edge::INVALID;
use super::*;
use crate::diag::testing::Capture;
use crate::diag::Level;

// case P60-P66: a degenerate policy is refused at open; the in-range extremes are accepted.
#[test]
fn policy_extremes_are_refused_or_accepted_at_open() {
    let _capture = Capture::start(Level::Off);
    let policy = |margin: f64, share: f64, moves: u16| Policy {
        abi_version: ABI_VERSION,
        max_moves_per_round: moves,
        ladder_boundaries: 0,
        cooldown_rounds: u64::MAX,
        promote_margin: margin,
        max_helper_layer_share: share,
    };
    let layout = Layout {
        abi_version: ABI_VERSION,
        layers: 1,
        experts: 2,
        cards: 2,
        owners: [0u8, 1].as_ptr(),
        owner_count: 2,
    };
    let open = |policy: Policy| {
        let mut handle = ptr::null_mut();
        let rc = unsafe { seabraid_placement_open(&layout, &policy, &mut handle) };
        if rc == 0 {
            assert_eq!(unsafe { seabraid_placement_close(handle) }, 0);
        }
        rc
    };
    for bad in [
        policy(f64::NAN, 0.5, 4),
        policy(f64::INFINITY, 0.5, 4),
        policy(0.99, 0.5, 4),
        policy(-1.0, 0.5, 4),
        policy(1.25, f64::NAN, 4),
        policy(1.25, -0.1, 4),
        policy(1.25, 1.0 + f64::EPSILON, 4),
        policy(1.25, 0.5, 129),
        policy(1.25, 0.5, u16::MAX),
    ] {
        assert_eq!(
            open(bad),
            INVALID,
            "{:?}",
            (
                bad.promote_margin,
                bad.max_helper_layer_share,
                bad.max_moves_per_round
            )
        );
    }
    for good in [policy(1.0, 0.0, 0), policy(f64::MAX, 1.0, 128)] {
        assert_eq!(open(good), 0);
    }
}

// case P67-P73: the fill entry point's degenerate requests.
#[test]
fn fill_refuses_zero_shapes_miscounted_slots_and_bad_heat() {
    use fill::{seabraid_placement_fill, FillRequest};
    let _capture = Capture::start(Level::Off);
    let slots = [1u64, 1];
    let heat = [1.0f64, 2.0];
    let request = |layers, experts, cards: Rung, slot_count, heat_count| FillRequest {
        abi_version: ABI_VERSION,
        layers,
        experts,
        cards,
        slots: slots.as_ptr(),
        slot_count,
        heat: heat.as_ptr(),
        heat_count,
    };
    let mut owners = [9u8; 2];
    let mut run = |request: FillRequest, count| unsafe {
        seabraid_placement_fill(&request, owners.as_mut_ptr(), count)
    };
    assert_eq!(run(request(1, 2, 2, 2, 2), 2), 0);
    assert_eq!(run(request(0, 2, 2, 2, 2), 2), INVALID, "zero layers");
    assert_eq!(run(request(1, 0, 2, 2, 0), 0), INVALID, "zero experts");
    assert_eq!(run(request(1, 2, 0, 0, 2), 2), INVALID, "zero cards");
    assert_eq!(
        run(request(1, 2, 2, 1, 2), 2),
        INVALID,
        "one slot count for two cards"
    );
    assert_eq!(
        run(request(1, 2, 2, 3, 2), 2),
        INVALID,
        "three slot counts for two cards"
    );
    assert_eq!(run(request(1, 2, 2, 2, 1), 2), INVALID, "heat count short");
    assert_eq!(run(request(1, 2, 2, 2, 2), 1), INVALID, "owner_count short");
    assert_eq!(
        run(request(1, 2, 2, 2, 2), usize::MAX),
        INVALID,
        "owner_count huge"
    );
    let nan = [f64::NAN, 1.0];
    let bad = FillRequest {
        heat: nan.as_ptr(),
        ..request(1, 2, 2, 2, 2)
    };
    assert_eq!(run(bad, 2), INVALID, "NaN heat");
    let huge = [u64::MAX, u64::MAX];
    let greedy = FillRequest {
        slots: huge.as_ptr(),
        ..request(1, 2, 2, 2, 2)
    };
    assert_eq!(
        run(greedy, 2),
        0,
        "slot counts of u64::MAX saturate, they do not overflow"
    );
    assert_eq!(owners, [0, 0]);
}
