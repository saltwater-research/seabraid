// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! Owns the worker and one in-flight buffer lease per session.

use crate::batch::{Batch, Config, Features};
use crate::capabilities::validate_origins;
use crate::diag::{fail, Code};
use crate::layout::assert_layout;
use crate::submission::Submission;
use crate::Result;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, SyncSender};
use std::thread::{self, JoinHandle};
use std::time::Duration;

mod close;
mod worker;
use worker::{serve, Completion, Work};

// Tickets remain distinct across closed and reopened sessions in this process.
static NEXT_TICKET: AtomicU64 = AtomicU64::new(1);

fn next_ticket() -> Result<u64> {
    let mut ticket = NEXT_TICKET.load(Ordering::Relaxed);
    loop {
        let next = ticket
            .checked_add(1)
            .ok_or_else(|| fail!(Code::Internal, "ticket counter exhausted"))?;
        match NEXT_TICKET.compare_exchange_weak(ticket, next, Ordering::Relaxed, Ordering::Relaxed)
        {
            Ok(_) => return Ok(ticket),
            Err(current) => ticket = current,
        }
    }
}

struct Pending {
    ticket: u64,
    rows: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct SessionStatus {
    pub abi_version: u32,
    pub struct_size: u32,
    pub submitted: u64,
    pub collected: u64,
    pub provider_failures: u64,
    pub wait_timeouts: u64,
    pub last_source_modalities: u32,
    pub outstanding: u32,
    /// Rows the callers submitted: the sum of `tokens` over accepted submits.
    pub rows_submitted: u64,
    /// Rows handed to the provider; below `rows_submitted` only when COMPACT_ROWS dropped some.
    pub rows_shipped: u64,
}

assert_layout!(
    SessionStatus,
    "seabraid_session_status_v1",
    [
        abi_version,
        struct_size,
        submitted,
        collected,
        provider_failures,
        wait_timeouts,
        last_source_modalities,
        outstanding,
        rows_submitted,
        rows_shipped
    ]
);

// No implicit padding: `write_out` copies this whole struct into caller memory, so a padding byte would carry stack
// garbage out (the placement structs say the same in ffi/placement/types.rs). The sum is the checker.
const _: () = assert!(
    core::mem::size_of::<SessionStatus>() == 4 + 4 + 4 * 8 + 4 + 4 + 2 * 8,
    "SessionStatus has implicit padding: name it as a field"
);

/// One session. WHY no handle state here (cookie, busy flag): the C handle is `ffi::lease::Slot`, which keeps that
/// gate beside the session instead of inside it, so a call's `&mut Session` never covers bytes another thread reads.
pub struct Session {
    operation_count: u32,
    features: Features,
    config: Option<Config>,
    submit: Option<SyncSender<Work>>,
    complete: Receiver<Completion>,
    idle: Option<Work>,
    worker: Option<JoinHandle<()>>,
    // Disconnects when the worker closure ends, so a bounded close can wait without joining.
    exited: Receiver<()>,
    pending: Option<Pending>,
    status: SessionStatus,
    /// Highest nonzero placement epoch accepted so far; only an accepted submit raises it.
    epoch_high: u64,
    /// Set when a collected completion said the worker stopped (a provider panic) or the worker vanished: every
    /// later submit is refused up front, so it cannot raise `epoch_high` or a counter for work that cannot run.
    worker_stopped: bool,
}

impl Session {
    pub fn new(
        operation_count: u32,
        partial_routes: bool,
        execute: impl FnMut(u32, Config, &Batch, &mut [f32]) -> Result<()> + Send + 'static,
    ) -> Result<Self> {
        let features = Features {
            partial_routes,
            ..Features::default()
        };
        Self::with_features(operation_count, features, execute)
    }

    /// Opens a session for a provider with the declared `features` (partial routes, compaction, formats).
    pub(crate) fn with_features(
        operation_count: u32,
        features: Features,
        execute: impl FnMut(u32, Config, &Batch, &mut [f32]) -> Result<()> + Send + 'static,
    ) -> Result<Self> {
        let features = features.validate()?;
        if operation_count == 0 {
            return Err(fail!(Code::Invalid, "operation_count must be at least 1"));
        }
        let (submit, work) = mpsc::sync_channel::<Work>(1);
        let (finish, complete) = mpsc::sync_channel(1);
        let (alive, exited) = mpsc::channel::<()>();
        let worker = thread::Builder::new()
            .name("seabraid-provider".to_owned())
            .spawn(move || {
                let _alive = alive; // dropped when this closure ends, even by unwinding
                serve(&work, &finish, execute)
            })
            .map_err(|source| {
                fail!(
                    Code::Internal,
                    "cannot start the provider worker thread ({:?})",
                    source.kind()
                )
            })?;
        Ok(Self {
            operation_count,
            features,
            config: None,
            submit: Some(submit),
            complete,
            idle: None,
            worker: Some(worker),
            exited,
            pending: None,
            status: SessionStatus::default(),
            epoch_high: 0,
            worker_stopped: false,
        })
    }

    /// The thread provider callbacks run on, until close takes the worker away.
    pub(crate) fn worker_id(&self) -> Option<thread::ThreadId> {
        self.worker.as_ref().map(|worker| worker.thread().id())
    }

    pub fn prepare(&mut self, config: Config) -> Result<()> {
        if self.config.is_some() {
            return Err(fail!(Code::BusyOrState, "session is already prepared"));
        }
        let config = config.validate()?;
        self.idle = Some(Work::new(config));
        self.config = Some(config);
        Ok(())
    }

    pub fn ready_config(&self, operation_id: u32) -> Result<Config> {
        if self.worker_stopped {
            return Err(fail!(
                Code::Provider,
                "provider worker stopped (its callback panicked or its thread ended); the session cannot run work, close it"
            ));
        }
        if operation_id >= self.operation_count {
            return Err(fail!(
                Code::Invalid,
                "operation_id {} >= operation_count {}",
                operation_id,
                self.operation_count
            ));
        }
        if let Some(pending) = &self.pending {
            return Err(fail!(
                Code::BusyOrState,
                "ticket {} is still outstanding",
                pending.ticket
            ));
        }
        self.config
            .ok_or_else(|| fail!(Code::BusyOrState, "session is not prepared"))
    }

    /// The one submit path for every ABI entry point. Every check precedes the first copy (their relative order is
    /// not a contract), so a refused batch changes nothing: not the buffer, not `epoch_high`, not a counter.
    pub(crate) fn submit(&mut self, s: Submission<'_>) -> Result<u64> {
        let config = self.ready_config(s.operation_id)?;
        validate_origins(s.source_modalities)?;
        self.check_epoch(s.placement_epoch)?;
        Batch::check(config, &s, self.features)?;
        let shipped = self.stage(config, &s)?;
        let ticket = next_ticket()?;
        let rows = config.lengths(s.tokens)?.rows;
        self.dispatch(s.operation_id)?;
        self.pending = Some(Pending { ticket, rows });
        self.epoch_high = self.epoch_high.max(s.placement_epoch);
        let status = &mut self.status;
        status.submitted = status.submitted.saturating_add(1);
        status.rows_submitted = status.rows_submitted.saturating_add(u64::from(s.tokens));
        status.rows_shipped = status.rows_shipped.saturating_add(shipped);
        status.last_source_modalities = s.source_modalities;
        Ok(ticket)
    }

    /// Fills the idle buffer (compacted when the provider opted in) and returns how many rows it holds.
    fn stage(&mut self, config: Config, s: &Submission<'_>) -> Result<u64> {
        let compact = self.features.compact_rows;
        let rows = config.lengths(s.tokens)?.rows;
        let work = self.idle.as_mut().ok_or_else(|| {
            fail!(
                Code::Provider,
                "provider worker is gone: no idle buffer to reuse"
            )
        })?;
        if compact {
            work.rows.derive(s.expert_ids, config.top_k as usize)?;
        }
        work.batch.load(config, s, compact.then_some(&work.rows))?;
        work.compacted = compact;
        work.size_output(rows)?;
        Ok(u64::from(work.batch.tokens))
    }

    /// 0 is unversioned and fine until the first nonzero epoch; after that an engine must version every batch.
    fn check_epoch(&self, epoch: u64) -> Result<()> {
        if epoch == 0 && self.epoch_high != 0 {
            return Err(fail!(
                Code::Invalid,
                "placement_epoch 0 after epoch {}: a versioning engine must version every batch",
                self.epoch_high
            ));
        }
        if epoch != 0 && epoch < self.epoch_high {
            return Err(fail!(
                Code::BusyOrState,
                "placement_epoch {} is stale: the session already accepted {}",
                epoch,
                self.epoch_high
            ));
        }
        Ok(())
    }

    /// Hands the loaded idle buffer to the worker; it is gone from `idle` until collected.
    fn dispatch(&mut self, operation_id: u32) -> Result<()> {
        let submit = self
            .submit
            .as_ref()
            .ok_or_else(|| fail!(Code::BusyOrState, "session is closing"))?;
        let mut work = self
            .idle
            .take()
            .ok_or_else(|| fail!(Code::Internal, "idle buffer vanished after validation"))?;
        work.operation_id = operation_id;
        submit.send(work).map_err(|source| {
            fail!(
                Code::Provider,
                "provider worker has stopped; the session cannot run work ({})",
                source
            )
        })
    }

    /// The counters, with `abi_version` left as the core knows it (0): the version header is the ABI's, and only
    /// `ffi::seabraid_session_status` stamps it, so a caller outside the ffi must set it itself.
    pub fn status(&self) -> SessionStatus {
        SessionStatus {
            struct_size: core::mem::size_of::<SessionStatus>() as u32,
            outstanding: u32::from(self.pending.is_some()),
            ..self.status
        }
    }

    pub fn output_count(&self, ticket: u64) -> Result<usize> {
        let pending = self
            .pending
            .as_ref()
            .ok_or_else(|| fail!(Code::BusyOrState, "ticket {} is not outstanding", ticket))?;
        if pending.ticket != ticket {
            return Err(fail!(
                Code::Invalid,
                "ticket {} is not the outstanding ticket {}",
                ticket,
                pending.ticket
            ));
        }
        Ok(pending.rows)
    }

    pub fn collect(&mut self, ticket: u64, timeout: Duration) -> Result<&[f32]> {
        self.output_count(ticket)?;
        let Completion {
            work,
            result,
            stopped,
        } = match self.complete.recv_timeout(timeout) {
            Ok(completion) => completion,
            Err(RecvTimeoutError::Timeout) => {
                self.status.wait_timeouts = self.status.wait_timeouts.saturating_add(1);
                return Err(fail!(
                    Code::Timeout,
                    "ticket {} not complete within {} ms",
                    ticket,
                    timeout.as_millis()
                ));
            }
            Err(RecvTimeoutError::Disconnected) => {
                self.pending = None;
                self.worker_stopped = true;
                self.status.provider_failures = self.status.provider_failures.saturating_add(1);
                return Err(fail!(
                    Code::Provider,
                    "provider worker ended without a result for ticket {}",
                    ticket
                ));
            }
        };
        self.pending = None;
        self.worker_stopped = stopped;
        let output = &self.idle.insert(work).output;
        if let Err(error) = result {
            self.status.provider_failures = self.status.provider_failures.saturating_add(1);
            return Err(error);
        }
        self.status.collected = self.status.collected.saturating_add(1);
        Ok(output)
    }
}

#[cfg(test)]
mod tests;
