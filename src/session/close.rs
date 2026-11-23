// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! Closing a session: `close`, `close_timeout`, the join they share, the leaked-worker report and `Drop`.
//! WHY its own file: closing is the one lifecycle phase that touches the worker handle and the completion channel
//! after the submit/collect path is done, and it carries its own ordering rules (try_recv before a bounded wait,
//! log an uncollected failure last), so the submit/collect code in session.rs can be read without them.

use super::{Completion, Session};
use crate::diag::{self, fail, Code, Level};
use crate::{Error, Result};
use std::sync::mpsc::{RecvTimeoutError, TryRecvError};
use std::time::Duration;

impl Session {
    pub fn close(mut self) -> Result<()> {
        self.join(None)
    }

    /// Waits at most `limit` for the worker. WHY detach instead of killing: a thread cannot be stopped safely
    /// from outside, so a stuck provider callback is leaked loudly and the caller keeps its context alive.
    pub fn close_timeout(mut self, limit: Duration) -> Result<()> {
        self.join(Some(limit))
    }

    fn join(&mut self, limit: Option<Duration>) -> Result<()> {
        self.submit.take();
        let mut uncollected = None;
        if let Some(limit) = limit {
            // WHY only while a callback can still be running: a callback is running exactly when a ticket is
            // outstanding and its completion has not been queued. With no ticket, or a completed one nobody
            // collected, the worker leaves as soon as the channel closes; timing that exit would report a stuck
            // provider (and a leak claim) for a session that was merely idle or finished, always so for a zero
            // bound. The queued completion is taken here (not peeked: a channel cannot) and logged below.
            if self.pending.is_some() && self.worker.is_some() {
                match self.complete.try_recv() {
                    Err(TryRecvError::Empty) => {
                        if self.exited.recv_timeout(limit) == Err(RecvTimeoutError::Timeout) {
                            return Err(self.detach_stuck_worker(limit));
                        }
                    }
                    // A completion, or a worker already gone: no callback is running either way.
                    other => uncollected = other.ok(),
                }
            }
        }
        // At most one completion can be queued, so the worker can finish even if nobody collects.
        let joined = self.worker.take().map_or(Ok(()), |worker| {
            worker.join().map_err(|payload| {
                let text = diag::payload_text(&*payload);
                fail!(Code::Provider, "provider worker panicked: {}", text)
            })
        });
        // The worker is done, so any queued failure was never collected; this is the last place
        // that can log it (and it must run even when the join itself failed).
        if let Some(Completion {
            result: Err(lost), ..
        }) = uncollected.or_else(|| self.complete.try_recv().ok())
        {
            diag::note("session close: uncollected ticket", &lost);
        }
        joined
    }
}

impl Session {
    /// Drops the join handle (which detaches the thread) and says, at ERROR, what was leaked.
    fn detach_stuck_worker(&mut self, limit: Duration) -> Error {
        self.worker.take();
        let ticket = self
            .pending
            .as_ref()
            .map_or_else(|| "none".to_owned(), |pending| pending.ticket.to_string());
        let text = format!(
            "close timed out after {} ms: provider callback for ticket {} is still running; worker thread leaked, and the provider context must stay valid for the process lifetime",
            limit.as_millis(),
            ticket
        );
        diag::emit(Level::Error, Code::Timeout as i32, &text);
        fail!(
            Code::Timeout,
            "close timed out after {} ms",
            limit.as_millis()
        )
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        // Drop cannot return an error, so a failed join is logged here instead.
        if let Err(error) = self.join(None) {
            diag::note("session drop", &error);
        }
    }
}
