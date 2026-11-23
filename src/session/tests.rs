// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::batch::reference;
use crate::capabilities::FORMAT_ROUTED_F32;
use crate::diag;

mod edge;
mod edge_rows;
mod edge_tickets;
mod rows;
mod worker_end;

fn config() -> Config {
    Config {
        abi_version: 1,
        hidden: 2,
        experts: 3,
        max_tokens: 2,
        top_k: 2,
    }
}

/// F32 activations as the bytes a v1 caller's floats are.
pub(super) fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

fn submit_op(session: &mut Session, operation_id: u32) -> Result<u64> {
    let activations = f32_bytes(&[1.0, 2.0]);
    session.submit(Submission {
        operation_id,
        tokens: 1,
        format: FORMAT_ROUTED_F32,
        activations: &activations,
        expert_ids: &[0, 2],
        weights: &[0.25, 0.75],
        source_modalities: 0,
        placement_epoch: 0,
    })
}

fn submit(session: &mut Session) -> Result<u64> {
    submit_op(session, 0)
}

fn code<T>(result: Result<T>) -> Code {
    result.err().expect("expected a failure").code()
}

/// Ceiling for a collect that is expected to succeed. A completion returns at once, so the size costs nothing
/// when the test passes; it only bounds a hang. WHY not a few seconds: a loaded CI box can starve a worker
/// thread for longer than that, and a spurious Timeout is a flake, not a finding.
const PATIENT: Duration = Duration::from_secs(60);

/// Blocks until the worker thread has fully ended, which is when it has dropped its end of the work channel.
/// WHY: the completion reaches the caller *before* the worker thread winds down, so a `submit` issued straight
/// after the failed `collect` races the worker and may be queued into a channel that is about to close. Joining
/// is the event "the worker stopped", so no sleep and no clock is involved.
fn wait_for_worker_exit(session: &mut Session) {
    let worker = session.worker.take().expect("the worker handle is held");
    worker
        .join()
        .expect("serve catches provider panics, so the worker ends normally");
}

#[test]
fn a_panicking_provider_is_reported_with_its_payload_and_stops_the_worker() {
    let mut session = Session::new(1, false, |_, _, _, _| panic!("boom 7")).unwrap();
    session.prepare(config()).unwrap();
    let ticket = submit(&mut session).unwrap();
    let error = session
        .collect(ticket, PATIENT)
        .map(<[f32]>::len)
        .unwrap_err();
    assert_eq!(error.code(), Code::Provider);
    assert!(error.message().contains("panicked: boom 7"), "{error:?}");
    assert_eq!(session.status().provider_failures, 1);
    wait_for_worker_exit(&mut session);
    assert_eq!(code(submit(&mut session)), Code::Provider);
    session.close().unwrap();
}

#[test]
fn timeout_and_wrong_ticket_preserve_the_pending_batch() {
    let (release, gate) = mpsc::sync_channel(1);
    let mut session = Session::new(1, false, move |_, config, batch, output| {
        gate.recv_timeout(PATIENT)
            .map_err(|source| fail!(Code::Provider, "test gate never opened ({})", source))?;
        reference(config, batch, output)
    })
    .unwrap();
    session.prepare(config()).unwrap();
    let ticket = submit(&mut session).unwrap();
    assert_eq!(
        code(session.collect(ticket + 1, Duration::ZERO)),
        Code::Invalid
    );
    assert_eq!(code(session.collect(ticket, Duration::ZERO)), Code::Timeout);
    assert_eq!(code(submit(&mut session)), Code::BusyOrState);
    assert_eq!(session.status().wait_timeouts, 1);
    assert_eq!(session.status().outstanding, 1);
    release.send(()).unwrap();
    assert_eq!(session.collect(ticket, PATIENT).unwrap(), [2.5, 5.0]);
    assert_eq!(
        code(session.collect(ticket, Duration::ZERO)),
        Code::BusyOrState
    );
    assert_eq!(session.status().submitted, 1);
    assert_eq!(session.status().collected, 1);
    assert_eq!(session.status().outstanding, 0);
    session.close().unwrap();
}

#[test]
fn one_worker_runs_multiple_operations_without_leaking_the_previous_id() {
    let mut session = Session::new(2, false, |operation, _, batch, output| {
        for (index, result) in output.iter_mut().enumerate() {
            *result = batch.f32_at(index) + operation as f32;
        }
        Ok(())
    })
    .unwrap();
    session.prepare(config()).unwrap();
    let rejected = submit_op(&mut session, 2);
    assert_eq!(code(rejected), Code::Invalid);
    assert_eq!(session.status().submitted, 0);
    let ticket = submit_op(&mut session, 1).unwrap();
    assert_eq!(session.collect(ticket, PATIENT).unwrap(), [2.0, 3.0]);
    let ticket = submit(&mut session).unwrap();
    assert_eq!(session.collect(ticket, PATIENT).unwrap(), [1.0, 2.0]);
}

#[test]
fn replacement_session_rejects_an_old_ticket() {
    let mut old = Session::new(1, false, |_, config, batch, output| {
        reference(config, batch, output)
    })
    .unwrap();
    old.prepare(config()).unwrap();
    let stale = submit(&mut old).unwrap();
    old.close().unwrap();

    let mut replacement = Session::new(1, false, |_, config, batch, output| {
        reference(config, batch, output)
    })
    .unwrap();
    replacement.prepare(config()).unwrap();
    let ticket = submit(&mut replacement).unwrap();
    assert_eq!(
        code(replacement.collect(stale, Duration::ZERO)),
        Code::Invalid
    );
    assert_eq!(replacement.collect(ticket, PATIENT).unwrap(), [2.5, 5.0]);
}

fn failing_session() -> Session {
    let mut session = Session::new(1, false, |_, _, _, _| {
        Err(fail!(Code::Provider, "lost provider failure"))
    })
    .unwrap();
    session.prepare(config()).unwrap();
    submit(&mut session).unwrap();
    session
}

#[test]
fn an_uncollected_failed_ticket_is_logged_once_on_close() {
    let capture = diag::testing::Capture::start(diag::Level::Warn);
    let session = failing_session();
    session.close().unwrap();
    let lost: Vec<_> = capture
        .mine()
        .into_iter()
        .filter(|r| r.message.contains("uncollected"))
        .collect();
    assert_eq!(lost.len(), 1, "{lost:?}");
    assert!(
        lost[0].message.contains("lost provider failure"),
        "{lost:?}"
    );
}

#[test]
fn an_uncollected_failed_ticket_is_logged_once_on_drop() {
    let capture = diag::testing::Capture::start(diag::Level::Warn);
    drop(failing_session());
    let lost: Vec<_> = capture
        .mine()
        .into_iter()
        .filter(|r| r.message.contains("uncollected"))
        .collect();
    assert_eq!(lost.len(), 1, "{lost:?}");
}
