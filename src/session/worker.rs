// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! OWNER: the provider worker loop and the work item it carries.
//! CONTRACT: `serve` runs callbacks for `Work` received from the session until
//! the session closes or a callback panics; visible only inside `session`.
//! FAILURE: a provider panic is caught and delivered as `Code::Provider`, then
//! the worker stops for good; a gone receiver ends the loop silently.
//! WHY the worker is not respawned after a panic: the panicked provider's state is unknown, so calling it again is
//! exactly what stopping avoids, and a new thread would break the documented promise that every callback of a
//! session runs on one thread (a thread-bound device context, and `Gate::on_worker_thread`, name that thread).
//! The session therefore refuses later submits (`Session::ready_config`) and the caller closes it.
//! COMPACT_ROWS: the provider sees only the kept rows and writes `kept * hidden` results into `scratch`; they are
//! scattered into `output` (zeroed, full size). With no kept row the provider is not called and the zeros stand.

use crate::batch::{Batch, Config};
use crate::diag::{self, fail, Code};
use crate::rows::RowMap;
use crate::Result;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::mpsc::{Receiver, SyncSender};

pub(super) struct Work {
    pub(super) operation_id: u32,
    pub(super) config: Config,
    pub(super) batch: Batch,
    pub(super) output: Vec<f32>,
    /// Set when this submission was compacted; `rows` and `scratch` are meaningful only then.
    pub(super) compacted: bool,
    pub(super) rows: RowMap,
    pub(super) scratch: Vec<f32>,
}

impl Work {
    pub(super) fn new(config: Config) -> Self {
        Self {
            operation_id: 0,
            config,
            batch: Batch::default(),
            output: Vec::new(),
            compacted: false,
            rows: RowMap::default(),
            scratch: Vec::new(),
        }
    }

    pub(super) fn size_output(&mut self, rows: usize) -> Result<()> {
        zeroed(&mut self.output, rows)
    }

    fn run(
        &mut self,
        execute: &mut impl FnMut(u32, Config, &Batch, &mut [f32]) -> Result<()>,
    ) -> Result<()> {
        if !self.compacted {
            return execute(
                self.operation_id,
                self.config,
                &self.batch,
                &mut self.output,
            );
        }
        if self.batch.tokens == 0 {
            return Ok(()); // every row was skipped: nothing to compute, and `output` is already zeros
        }
        let hidden = self.config.hidden as usize;
        zeroed(&mut self.scratch, self.batch.tokens as usize * hidden)?;
        execute(
            self.operation_id,
            self.config,
            &self.batch,
            &mut self.scratch,
        )?;
        self.rows.scatter(&self.scratch, hidden, &mut self.output)
    }
}

fn zeroed(storage: &mut Vec<f32>, len: usize) -> Result<()> {
    storage
        .try_reserve_exact(len.saturating_sub(storage.len()))
        .map_err(|source| {
            fail!(
                Code::Internal,
                "cannot reserve {} output elements ({})",
                len,
                source
            )
        })?;
    storage.clear();
    storage.resize(len, 0.0);
    Ok(())
}

/// What the worker sends back for each item it ran.
pub(super) struct Completion {
    pub(super) work: Work,
    pub(super) result: Result<()>,
    /// True when the worker will serve nothing more (a callback panicked). WHY it travels with the completion:
    /// the caller that receives it learns, in the same message, that the next submit cannot run, so it can refuse
    /// it before changing any state, instead of racing the worker's exit (a channel send succeeds until the worker
    /// has dropped its receiver, which happens a moment after the completion is delivered).
    pub(super) stopped: bool,
}

/// Runs provider callbacks until the session closes or a callback panics.
pub(super) fn serve(
    work: &Receiver<Work>,
    finish: &SyncSender<Completion>,
    mut execute: impl FnMut(u32, Config, &Batch, &mut [f32]) -> Result<()>,
) {
    while let Ok(mut item) = work.recv() {
        let ran = catch_unwind(AssertUnwindSafe(|| item.run(&mut execute)));
        // A panicked provider has unknown state, so its error is delivered and the worker stops.
        let (result, alive) = match ran {
            Ok(result) => (result, true),
            Err(payload) => {
                let text = diag::payload_text(&*payload);
                (
                    Err(fail!(
                        Code::Provider,
                        "provider callback panicked: {}",
                        text
                    )),
                    false,
                )
            }
        };
        // A gone receiver means the session is gone; uncollected results are drained and logged by Session::join.
        let completion = Completion {
            work: item,
            result,
            stopped: !alive,
        };
        if finish.send(completion).is_err() || !alive {
            break;
        }
    }
}
