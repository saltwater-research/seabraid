// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! How a session treats its worker at the end of the worker's life: `close_timeout` after a finished ticket (R6a S1)
//! and a submit after a provider panic (R6a S2).

use super::edge::{cfg, sub};
use super::*;

/// A session whose provider runs `body`, then reports through `done` that the callback returned. WHY a channel:
/// the test must know the callback is over before it closes, without sleeping on a guess.
fn finishing_session(
    done: &mpsc::Sender<()>,
    body: impl Fn() -> Result<()> + Send + 'static,
) -> Session {
    let done = done.clone();
    let mut session = Session::new(1, false, move |_, config, batch, output| {
        let result = body().and_then(|()| reference(config, batch, output));
        done.send(()).ok();
        result
    })
    .unwrap();
    session.prepare(cfg(2, 3, 2, 2)).unwrap();
    session
}

/// R6a finding S1: `close_timeout(0)` after a ticket that COMPLETED but was never collected answered TIMEOUT with an
/// ERROR "worker thread leaked" every time (2000 of 2000 in the C reproducer), because "a ticket is outstanding" was
/// taken to mean "a callback is running". Here every callback has returned before the close; the completions have
/// sat in their queues for a long while (all sessions are submitted first and one 300 ms pause covers them all, so a
/// slow thread start cannot turn this into a false failure).
#[test]
fn close_timeout_zero_after_a_completed_uncollected_ticket_is_not_a_leak() {
    let capture = diag::testing::Capture::start(diag::Level::Debug);
    let (done, finished) = mpsc::channel();
    let act = f32_bytes(&[1.0, 2.0]);
    let mut sessions: Vec<Session> = (0..40)
        .map(|_| {
            let mut session = finishing_session(&done, || Ok(()));
            session
                .submit(sub(1, &act, &[0, 2], &[0.25, 0.75], 0))
                .unwrap();
            session
        })
        .collect();
    for _ in &sessions {
        finished.recv_timeout(PATIENT).unwrap();
    }
    std::thread::sleep(Duration::from_millis(300));
    for (index, session) in sessions.drain(..).enumerate() {
        assert_eq!(
            session.close_timeout(Duration::ZERO).ok(),
            Some(()),
            "session {index}"
        );
    }
    let loud: Vec<_> = capture
        .mine()
        .into_iter()
        .filter(|r| r.level >= diag::Level::Error as u32)
        .collect();
    assert!(
        loud.is_empty(),
        "no leak record for a finished worker: {loud:?}"
    );
}

/// The other half of S1's contract: the uncollected-ticket log survives the new path (the completion is read at the
/// top of `join` now), and a callback that really is still running is still a TIMEOUT, logged at ERROR.
#[test]
fn close_timeout_zero_still_logs_an_uncollected_failure_and_still_times_out_a_running_callback() {
    let capture = diag::testing::Capture::start(diag::Level::Debug);
    let (done, finished) = mpsc::channel();
    let act = f32_bytes(&[1.0, 2.0]);
    let mut lost = finishing_session(&done, || Err(fail!(Code::Provider, "lost on purpose")));
    lost.submit(sub(1, &act, &[0, 2], &[0.25, 0.75], 0))
        .unwrap();
    finished.recv_timeout(PATIENT).unwrap();
    std::thread::sleep(Duration::from_millis(300));
    assert_eq!(lost.close_timeout(Duration::ZERO).ok(), Some(()));
    let records = capture.mine();
    let uncollected: Vec<_> = records
        .iter()
        .filter(|r| r.message.contains("uncollected") && r.message.contains("lost on purpose"))
        .collect();
    assert_eq!(uncollected.len(), 1, "{records:?}");

    let (release, hold) = mpsc::sync_channel::<()>(1);
    let mut stuck = Session::new(1, false, move |_, _, _, _| {
        hold.recv().ok();
        Ok(())
    })
    .unwrap();
    stuck.prepare(cfg(2, 3, 2, 2)).unwrap();
    stuck
        .submit(sub(1, &act, &[0, 2], &[0.25, 0.75], 0))
        .unwrap();
    assert_eq!(
        stuck.close_timeout(Duration::ZERO).unwrap_err().code(),
        Code::Timeout
    );
    release.send(()).ok();
    assert!(
        capture.mine().iter().any(|r| {
            r.level == diag::Level::Error as u32 && r.message.contains("worker thread leaked")
        }),
        "the leak is still logged at ERROR"
    );
}

/// Blocks in `drop` until released. Captured by a provider closure it keeps the worker thread alive, with its end of
/// the work channel open, for as long as the test wants after the callback panicked (the closure is dropped when
/// `serve` returns, before the thread's receiver is).
struct HoldOnDrop(mpsc::Receiver<()>);

impl Drop for HoldOnDrop {
    fn drop(&mut self) {
        self.0.recv_timeout(PATIENT).ok();
    }
}

/// R6a finding S2: a submit that arrives while the panicked worker is still exiting used to be ACCEPTED (the channel
/// still had a receiver), raising `epoch_high` and the counters for a batch that could never run, and failed only at
/// collect. Now the collected completion says the worker stopped, and the next submit is refused before any change.
#[test]
fn a_submit_after_a_provider_panic_is_refused_before_it_changes_anything() {
    let (release, hold) = mpsc::channel();
    let guard = HoldOnDrop(hold);
    let mut session = Session::new(1, false, move |_, _, _, _| {
        let _keep = &guard;
        panic!("boom in the provider")
    })
    .unwrap();
    session.prepare(cfg(2, 3, 2, 2)).unwrap();
    // WHY re-bound after `session`: locals drop in reverse order, so a failing assertion drops `release` (which lets
    // the held worker go) before `session` joins that worker, instead of deadlocking until the guard's timeout.
    let release = release;
    let act = f32_bytes(&[1.0, 2.0]);
    let ticket = session
        .submit(sub(1, &act, &[0, 2], &[0.25, 0.75], 7))
        .unwrap();
    assert_eq!(code(session.collect(ticket, PATIENT)), Code::Provider);
    let seen = |s: &Session| {
        (
            s.epoch_high,
            s.status().submitted,
            s.status().rows_submitted,
        )
    };
    let before = seen(&session);
    // The worker is still alive here (the guard holds it): the window the old code accepted a batch in.
    let refused = session
        .submit(sub(1, &act, &[0, 2], &[0.25, 0.75], 9))
        .unwrap_err();
    assert_eq!(refused.code(), Code::Provider);
    assert!(refused.message().contains("worker stopped"), "{refused:?}");
    assert_eq!(seen(&session), before, "a refused submit changes nothing");
    assert_eq!(session.status().outstanding, 0);
    release.send(()).unwrap();
    session.close().unwrap();
}
