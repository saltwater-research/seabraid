// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! Overnight hardening O1, rows X*, K*, C*: one expert, foreign tickets and closing with work outstanding (see edge.rs).

use super::edge::{cfg, echo, sub};
use super::*;

// case X1-X5: experts == 1 and top_k == experts.
#[test]
fn a_single_expert_session_accepts_only_expert_zero_and_refuses_repeats() {
    let act = f32_bytes(&[2.0]);
    let (mut session, _) = echo(cfg(1, 1, 1, 1), Features::default());
    let ticket = session.submit(sub(1, &act, &[0], &[1.0], 0)).unwrap();
    session.collect(ticket, PATIENT).unwrap();
    for id in [1, -1, i32::MIN, i32::MAX] {
        assert_eq!(
            code(session.submit(sub(1, &act, &[id], &[1.0], 0))),
            Code::Invalid,
            "{id}"
        );
    }
    let (mut full, _) = echo(cfg(1, 2, 1, 2), Features::default());
    assert_eq!(
        code(full.submit(sub(1, &act, &[1, 1], &[0.5, 0.5], 0))),
        Code::Invalid
    );
    let ticket = full.submit(sub(1, &act, &[1, 0], &[0.5, 0.5], 0)).unwrap();
    full.collect(ticket, PATIENT).unwrap();
}

// case K1-K6: tickets that are unknown, zero, already collected, or from another session.
#[test]
fn collect_of_an_unknown_zero_collected_or_foreign_ticket_never_disturbs_the_pending_one() {
    let (mut a, _) = echo(cfg(2, 3, 2, 2), Features::default());
    let (mut b, _) = echo(cfg(2, 3, 2, 2), Features::default());
    let act = f32_bytes(&[1.0, 2.0]);
    let ticket = a.submit(sub(1, &act, &[0, 2], &[0.25, 0.75], 0)).unwrap();
    let other = b.submit(sub(1, &act, &[0, 2], &[0.25, 0.75], 0)).unwrap();
    assert_ne!(ticket, other, "tickets are unique across sessions");
    for wrong in [0, 1, ticket.wrapping_add(1), other, u64::MAX] {
        assert_eq!(
            code(a.collect(wrong, Duration::ZERO)),
            Code::Invalid,
            "ticket {wrong}"
        );
        assert_eq!(
            a.output_count(wrong).err().map(|e| e.code()),
            Some(Code::Invalid)
        );
    }
    assert_eq!(a.status().outstanding, 1);
    a.collect(ticket, PATIENT).unwrap();
    assert_eq!(
        code(a.collect(ticket, Duration::ZERO)),
        Code::BusyOrState,
        "already collected"
    );
    assert_eq!(
        code(a.collect(0, Duration::ZERO)),
        Code::BusyOrState,
        "nothing outstanding"
    );
    b.collect(other, PATIENT).unwrap();
}

// case C1-C4: closing with a ticket outstanding.
#[test]
fn close_and_close_timeout_with_an_uncollected_ticket_are_clean() {
    for timeout in [None, Some(Duration::from_secs(30))] {
        let (mut session, _) = echo(cfg(2, 3, 2, 2), Features::default());
        let act = f32_bytes(&[1.0, 2.0]);
        session
            .submit(sub(1, &act, &[0, 2], &[0.25, 0.75], 0))
            .unwrap();
        assert_eq!(session.status().outstanding, 1);
        match timeout {
            None => session.close().unwrap(),
            Some(limit) => session.close_timeout(limit).unwrap(),
        }
    }
}

#[test]
fn close_timeout_on_a_stuck_provider_times_out_and_the_released_worker_ends_without_a_panic() {
    let (release, hold) = mpsc::sync_channel::<()>(1);
    let mut session = Session::new(1, false, move |_, _, _, output| {
        hold.recv().ok();
        output.fill(0.0);
        Ok(())
    })
    .unwrap();
    session.prepare(cfg(2, 3, 2, 2)).unwrap();
    let act = f32_bytes(&[1.0, 2.0]);
    session
        .submit(sub(1, &act, &[0, 2], &[0.25, 0.75], 0))
        .unwrap();
    let error = session
        .close_timeout(Duration::from_millis(20))
        .unwrap_err();
    assert_eq!(error.code(), Code::Timeout);
    release.send(()).ok(); // the detached worker finishes against a dropped session and must not panic
}

// case C9-C10: a zero bound on a session with no callback running is not a timeout.
#[test]
fn close_timeout_zero_on_an_idle_session_succeeds_and_does_not_claim_a_leak() {
    for _ in 0..50 {
        let (session, _) = echo(cfg(2, 3, 2, 2), Features::default());
        session.close_timeout(Duration::ZERO).unwrap();
    }
    let unprepared = Session::new(1, false, |_, _, _, _| Ok(())).unwrap();
    unprepared.close_timeout(Duration::ZERO).unwrap();
    // After a collected ticket the worker is idle again.
    let (mut session, _) = echo(cfg(2, 3, 2, 2), Features::default());
    let act = f32_bytes(&[1.0, 2.0]);
    let ticket = session
        .submit(sub(1, &act, &[0, 2], &[0.25, 0.75], 0))
        .unwrap();
    session.collect(ticket, PATIENT).unwrap();
    session.close_timeout(Duration::ZERO).unwrap();
}
