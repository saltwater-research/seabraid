// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! One same-layer swap between two adjacent rungs: stage both cells, write the upper rung, then the lower one,
//! and undo the upper write when the lower one fails.
//! OWNER: this file owns the per-swap failure table (docs/PLACEMENT-ABI.md, "Provider failures and recovery") and the
//! rule for a swap whose lower rung is the fallback; `by_reference.rs` owns the swap whose upper write takes no bytes.
//! FAILURE: no error escapes; every provider failure becomes an `Exchange::Aborted` naming the outcome and the rungs to close.

use super::{ExpertId, IoError, IoPhase, MoveOutcome, ProviderIssue};
use crate::diag::{self, Code, Level};
use crate::placement::{PlacementIo, Rung};
use std::panic::{catch_unwind, AssertUnwindSafe};

pub(super) struct Swap {
    pub(super) hi: Rung,
    pub(super) lo: Rung,
    pub(super) up: ExpertId,
    pub(super) down: ExpertId,
    pub(super) up_index: usize,
    pub(super) down_index: usize,
    /// The lower rung is the fallback (only on a map opened with the fallback boundary). WHY a flag on the swap: the
    /// fallback is never closed, so every rule below that would close `lo` asks this one question.
    pub(super) lo_fallback: bool,
}

pub(super) enum Exchange {
    /// Both writes were acknowledged. Carries the climber's bytes so a later hop of the same climb need not stage it
    /// again; `None` after a by-reference upper write, which never held them (the next hop asks its rung again).
    Acknowledged(Option<Vec<u8>>),
    Aborted {
        outcome: MoveOutcome,
        quarantine: u64,
    },
}

impl Swap {
    /// `climber` is the bytes of the cell the previous swap of this plan moved up, when it is `self.up` again.
    /// WHY safe to reuse: those bytes were what the previous swap's acknowledged write put into the climber's new rung.
    pub(super) fn run(&self, io: &mut dyn PlacementIo, climber: Option<Vec<u8>>) -> Exchange {
        // WHY only without climber bytes: bytes already in hand are the cheaper write, and asking a rung to read them
        // again from the file would undo the reuse the climb is built on (docs/PLACEMENT-ABI.md, by-reference moves).
        if climber.is_none() && io.takes_from_fallback(self.hi) {
            return self.run_by_reference(io);
        }
        let (up, down) = match self.stage(io, climber) {
            Ok(blobs) => blobs,
            Err(issue) => return self.stage_failed(issue),
        };
        self.write_both(io, up, &down)
    }

    /// The two writes from staged bytes: upper first (it takes the climber), then the lower (it takes the displaced).
    pub(super) fn write_both(
        &self,
        io: &mut dyn PlacementIo,
        up: Vec<u8>,
        down: &[u8],
    ) -> Exchange {
        // Preserve both originals before either slot is written. The upper
        // provider acknowledges before the lower can overwrite the source.
        if let Err(issue) = call(io, self.hi, IoPhase::Replace, self.up, |io| {
            io.replace(self.hi, self.down, self.up, &up)
        }) {
            return self.upper_failed(io, issue, down);
        }
        self.write_lower(io, Some(up), down)
    }

    /// The lower write, after the upper rung acknowledged. `up` is what the next hop may reuse (`None` by reference).
    pub(super) fn write_lower(
        &self,
        io: &mut dyn PlacementIo,
        up: Option<Vec<u8>>,
        down: &[u8],
    ) -> Exchange {
        let write = |io: &mut dyn PlacementIo| io.replace(self.lo, self.up, self.down, down);
        let result = if self.lo_fallback {
            self.call_fallback_lower(io, write)
        } else {
            call(io, self.lo, IoPhase::Replace, self.down, write)
        };
        match result {
            Ok(()) => Exchange::Acknowledged(up),
            Err(issue) => self.lower_failed(io, issue, down),
        }
    }

    /// The lower write when the lower rung is the fallback.  WHY not `call`: a non-REFUSED answer from the fallback
    /// breaks its all-or-nothing provider rule, and that record must say so; `call` would log the same failure
    /// first, so a failure would be two records.  REFUSED is an ordinary answer and keeps `call`'s record; the
    /// adapter's own detail, when it gave one, is kept in the breach record.
    fn call_fallback_lower(
        &self,
        io: &mut dyn PlacementIo,
        operation: impl FnOnce(&mut dyn PlacementIo) -> Result<(), IoError>,
    ) -> Result<(), ProviderIssue> {
        let Err((error, detail)) = attempt(io, operation) else {
            return Ok(());
        };
        let issue = ProviderIssue {
            rung: self.lo,
            phase: IoPhase::Replace,
            error,
        };
        if error == IoError::Refused {
            let (code, text) =
                detail.unwrap_or_else(|| (Code::Provider, default_detail(error).to_owned()));
            log_issue(issue, self.down, code, &text);
            return Err(issue);
        }
        let (code, why) =
            detail.unwrap_or_else(|| (Code::Provider, default_detail(error).to_owned()));
        let text = format!(
            "fallback rung {} answered {:?} to a replace of layer {} expert {}: its provider rule is all-or-nothing; the card swap is undone and the fallback stays open ({})",
            self.lo, error, self.up.layer, self.up.expert, why
        );
        diag::emit(Level::Error, code as i32, &text);
        Err(issue)
    }

    fn stage(
        &self,
        io: &mut dyn PlacementIo,
        climber: Option<Vec<u8>>,
    ) -> Result<(Vec<u8>, Vec<u8>), ProviderIssue> {
        let up = match climber {
            Some(bytes) => bytes,
            None => call(io, self.lo, IoPhase::Stage, self.up, |io| {
                io.stage(self.lo, self.up)
            })?,
        };
        let down = call(io, self.hi, IoPhase::Stage, self.down, |io| {
            io.stage(self.hi, self.down)
        })?;
        self.check_sizes(&up, &down)?;
        Ok((up, down))
    }

    /// Both staged blobs must be non-empty and equal in size (the layer's slot size); the issue names the rung at fault.
    pub(super) fn check_sizes(&self, up: &[u8], down: &[u8]) -> Result<(), ProviderIssue> {
        if up.is_empty() || up.len() != down.len() {
            let rung = if up.is_empty() { self.lo } else { self.hi };
            let text = format!(
                "staged slot sizes {} and {} differ or are empty",
                up.len(),
                down.len()
            );
            return Err(self.malformed(rung, &text));
        }
        Ok(())
    }

    pub(super) fn malformed(&self, rung: Rung, text: &str) -> ProviderIssue {
        let issue = ProviderIssue {
            rung,
            phase: IoPhase::Stage,
            error: IoError::MalformedBlob,
        };
        log_issue(issue, self.up, Code::Provider, text);
        issue
    }

    /// A stage writes nothing, so it only closes a rung whose completion is uncertain (a timeout or a panic), and
    /// never the fallback: WHY, the fallback is where every closed card's cells go, so a stage there is Rejected
    /// whatever it returned (a non-refused status is still logged at ERROR by `call`).
    pub(super) fn stage_failed(&self, issue: ProviderIssue) -> Exchange {
        let uncertain = matches!(issue.error, IoError::Timeout | IoError::Panicked);
        if uncertain && !(self.lo_fallback && issue.rung == self.lo) {
            return quarantined(issue, None, 1 << issue.rung);
        }
        aborted(MoveOutcome::Rejected(issue))
    }

    pub(super) fn upper_failed(
        &self,
        io: &mut dyn PlacementIo,
        issue: ProviderIssue,
        down: &[u8],
    ) -> Exchange {
        match issue.error {
            IoError::Refused => aborted(MoveOutcome::Rejected(issue)),
            IoError::Failed => self.restore_upper(io, self.down, down, issue, 0),
            // A timeout may still be writing after return; never send a restore there.
            _ => quarantined(issue, None, 1 << self.hi),
        }
    }

    fn lower_failed(
        &self,
        io: &mut dyn PlacementIo,
        issue: ProviderIssue,
        down: &[u8],
    ) -> Exchange {
        // As in Strata, a non-refused lower write disqualifies that card. The upper
        // acknowledgement is known, so undo its swap from staged bytes. The fallback is the exception: it is never
        // closed, because by its provider rule a replace that did not ACK left it serving every cell from verified
        // bytes, so restoring the upper card returns the map to a state where every cell has a correct executor.
        let closed = if self.lo_fallback || issue.error == IoError::Refused {
            0
        } else {
            1 << self.lo
        };
        self.restore_upper(io, self.up, down, issue, closed)
    }

    fn restore_upper(
        &self,
        io: &mut dyn PlacementIo,
        current: ExpertId,
        original: &[u8],
        issue: ProviderIssue,
        closed: u64,
    ) -> Exchange {
        match call(io, self.hi, IoPhase::Restore, self.down, |io| {
            io.replace(self.hi, current, self.down, original)
        }) {
            Ok(()) if closed == 0 => aborted(MoveOutcome::Restored(issue)),
            Ok(()) => quarantined(issue, None, closed),
            Err(restore) => quarantined(issue, Some(restore), closed | (1 << self.hi)),
        }
    }
}

pub(super) fn aborted(outcome: MoveOutcome) -> Exchange {
    Exchange::Aborted {
        outcome,
        quarantine: 0,
    }
}

pub(super) fn quarantined(
    initial: ProviderIssue,
    restore: Option<ProviderIssue>,
    quarantine: u64,
) -> Exchange {
    Exchange::Aborted {
        outcome: MoveOutcome::Quarantined { initial, restore },
        quarantine,
    }
}

/// One provider operation on `cell`. Every failure is logged here, once, with its context: the adapter reports
/// extra facts through `take_detail` instead of logging itself, so a failure is never two records, and the text
/// says what happened (an ACK with a short write is not "did not acknowledge").
pub(super) fn call<T>(
    io: &mut dyn PlacementIo,
    rung: Rung,
    phase: IoPhase,
    cell: ExpertId,
    operation: impl FnOnce(&mut dyn PlacementIo) -> Result<T, IoError>,
) -> Result<T, ProviderIssue> {
    call_with(io, rung, phase, cell, true, operation)
}

/// `call`, with a choice for the one answer that is not a failure. `log_refused` false is for a request the provider
/// may decline as a matter of course (a by-reference write: REFUSED means "send bytes instead"), so that answer is
/// returned without a record, while every other failure is logged exactly as in `call`.
pub(super) fn call_with<T>(
    io: &mut dyn PlacementIo,
    rung: Rung,
    phase: IoPhase,
    cell: ExpertId,
    log_refused: bool,
    operation: impl FnOnce(&mut dyn PlacementIo) -> Result<T, IoError>,
) -> Result<T, ProviderIssue> {
    let (error, detail) = match attempt(io, operation) {
        Ok(value) => return Ok(value),
        Err(failure) => failure,
    };
    let issue = ProviderIssue { rung, phase, error };
    if error == IoError::Refused && !log_refused {
        return Err(issue);
    }
    let (code, text) = detail.unwrap_or_else(|| (Code::Provider, default_detail(error).to_owned()));
    log_issue(issue, cell, code, &text);
    Err(issue)
}

/// One provider operation, unlogged: the failure and its best account (the panic text, else the adapter's own detail,
/// which is taken even after a panic so it cannot go stale).  The callers decide the one record a failure gets.
fn attempt<T>(
    io: &mut dyn PlacementIo,
    operation: impl FnOnce(&mut dyn PlacementIo) -> Result<T, IoError>,
) -> Result<T, (IoError, Option<(Code, String)>)> {
    // An unwinding Rust adapter is also an unacknowledged provider operation.
    let (error, text) = match catch_unwind(AssertUnwindSafe(|| operation(&mut *io))) {
        Ok(Ok(value)) => return Ok(value),
        Ok(Err(error)) => (error, None),
        Err(payload) => (
            IoError::Panicked,
            Some((
                Code::Provider,
                format!("callback panicked: {}", diag::payload_text(&*payload)),
            )),
        ),
    };
    let detail = io.take_detail();
    Err((error, text.or(detail)))
}

/// What an `IoError` means when the adapter said nothing more.
fn default_detail(error: IoError) -> &'static str {
    match error {
        IoError::Refused => "callback refused; no write began",
        IoError::Failed => {
            "callback reported a failure; the slot may be torn and the old index stands"
        }
        IoError::Timeout => "callback gave no acknowledgement; completion is uncertain",
        IoError::Panicked => "callback panicked",
        IoError::MalformedBlob => "staged blob was empty or the wrong size",
    }
}

/// One provider-issue record. WHY the level: ERROR for everything but a clean refusal, because a failure, a
/// timeout, a panic or a torn blob leaves provider state in doubt (and PROVIDER and INTERNAL records are ERROR
/// everywhere else, see seabraid.h); a REFUSED is the provider declining before any write began, an expected
/// outcome that `apply` reports in its outcomes with SEABRAID_OK, so it is WARN.
fn log_issue(issue: ProviderIssue, cell: ExpertId, code: Code, detail: &str) {
    let level = if issue.error == IoError::Refused {
        Level::Warn
    } else {
        Level::Error
    };
    let text = format!(
        "placement {:?} on rung {} layer {} expert {}: {:?}: {}",
        issue.phase, issue.rung, cell.layer, cell.expert, issue.error, detail
    );
    diag::emit(level, code as i32, &text);
}
