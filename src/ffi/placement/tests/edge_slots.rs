// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! R6a P2/P3 (one accurate record per provider failure, at the documented level), P4 (a huge slot size cannot make
//! the stage buffer touch memory), Z1 (no stack garbage in library-filled structs), Z2 (a struct_size that splits
//! a field) and Z3 (what an outcome array's `struct_size` does when the array is reused).

use super::*;
use crate::diag::testing::Capture;
use crate::diag::Level;
use crate::ffi::placement::types::{OUTCOME_QUARANTINED, OUTCOME_REJECTED, OUTCOME_RESTORED};
use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};

/// What the scripted provider does. `stage` is the code every stage call returns; with code 0 it writes the slot
/// except for `short` bytes. `replace[i]` is the code of the i-th replace call (the last entry repeats).
struct Script {
    stage: i32,
    short: usize,
    replace: [i32; 3],
    replaces: AtomicUsize,
}

impl Script {
    fn new(stage: i32, short: usize, replace: [i32; 3]) -> Self {
        Self {
            stage,
            short,
            replace,
            replaces: AtomicUsize::new(0),
        }
    }
}

unsafe extern "C" fn scripted_stage(
    context: *mut c_void,
    _: u8,
    _: u32,
    _: u32,
    bytes: *mut u8,
    size: usize,
    written: *mut usize,
) -> i32 {
    let script = unsafe { &*context.cast::<Script>() };
    if script.stage == 0 {
        unsafe {
            ptr::write_bytes(bytes, 7, size);
            *written = size - script.short;
        }
    }
    script.stage
}

unsafe extern "C" fn scripted_replace(
    context: *mut c_void,
    _: u8,
    _: u32,
    _: u32,
    _: u32,
    _: *const u8,
    _: usize,
) -> i32 {
    let script = unsafe { &*context.cast::<Script>() };
    let call = script.replaces.fetch_add(1, Ordering::SeqCst);
    script.replace[call.min(script.replace.len() - 1)]
}

fn blank_outcome() -> Outcome {
    Outcome {
        abi_version: ABI_VERSION,
        struct_size: mem::size_of::<Outcome>() as u32,
        ..Outcome::default()
    }
}

fn blank_applied() -> Applied {
    Applied {
        abi_version: ABI_VERSION,
        struct_size: mem::size_of::<Applied>() as u32,
        epoch: 0,
        quarantined_rungs: 0,
        outcome_count: 0,
    }
}

/// One plan and one apply of the single swap `TestPlacement` produces, against `script`.
fn apply_script(script: &Script) -> (i32, Outcome) {
    let placement = TestPlacement::new();
    let ticket = placement.plan(1).ticket;
    let slot = [8usize];
    let io = RawIo {
        abi_version: ABI_VERSION,
        context: ptr::from_ref(script).cast_mut().cast(),
        layer_slot_bytes: slot.as_ptr(),
        layer_count: 1,
        stage: Some(scripted_stage),
        replace: Some(scripted_replace),
    };
    let (mut outcome, mut applied) = (blank_outcome(), blank_applied());
    let rc = unsafe {
        seabraid_placement_apply(
            placement.0,
            ticket,
            ptr::from_ref(&io).cast(),
            &mut outcome,
            1,
            &mut applied,
        )
    };
    (rc, outcome)
}

/// The provider-issue records of one apply (not the quarantine notice that may follow them).
fn issue_records(capture: &Capture) -> Vec<crate::diag::testing::Record> {
    capture
        .mine()
        .into_iter()
        .filter(|r| r.message.starts_with("placement ") && !r.message.contains("quarantined"))
        .collect()
}

/// R6a P2: a failure was logged twice (once by the adapter, once by the transaction), and an ACK with a short write
/// was called "did not acknowledge". P3: PROVIDER and INTERNAL records are ERROR (seabraid.h), a clean refusal WARN.
#[test]
fn each_provider_failure_is_one_accurate_record_at_its_documented_level() {
    let warn = Level::Warn as u32;
    let error = Level::Error as u32;
    // (script, expected outcome kind, level, text it must carry, text it must not)
    type Case = (Script, u32, u32, &'static str, &'static str);
    let cases: [(&str, Case); 5] = [
        (
            "ACK with a short write",
            (
                Script::new(0, 1, [0; 3]),
                OUTCOME_REJECTED,
                error,
                "wrote 7 of 8 bytes",
                "did not acknowledge",
            ),
        ),
        (
            "an unknown stage code",
            (
                Script::new(17, 0, [0; 3]),
                OUTCOME_QUARANTINED,
                error,
                "unknown code 17",
                "did not acknowledge",
            ),
        ),
        (
            "a refused replace",
            (
                Script::new(0, 0, [1; 3]),
                OUTCOME_REJECTED,
                warn,
                "refused",
                "did not acknowledge",
            ),
        ),
        (
            "a failed replace, restored",
            (
                Script::new(0, 0, [2, 0, 0]),
                OUTCOME_RESTORED,
                error,
                "reported a failure",
                "did not acknowledge",
            ),
        ),
        (
            "a timed-out stage",
            (
                Script::new(3, 0, [0; 3]),
                OUTCOME_QUARANTINED,
                error,
                "no acknowledgement",
                "wrote",
            ),
        ),
    ];
    for (name, (script, kind, level, carries, lacks)) in cases {
        let capture = Capture::start(Level::Debug);
        let (rc, outcome) = apply_script(&script);
        assert_eq!(rc, 0, "{name}");
        assert_eq!(outcome.kind, kind, "{name}");
        let records = issue_records(&capture);
        assert_eq!(
            records.len(),
            1,
            "{name}: one record per failure: {records:?}"
        );
        let record = &records[0];
        assert_eq!(record.level, level, "{name}: {}", record.message);
        assert!(
            record.message.contains(carries),
            "{name}: {}",
            record.message
        );
        assert!(
            !record.message.contains(lacks),
            "{name}: {}",
            record.message
        );
        assert_eq!(record.status, Code::Provider as i32, "{name}");
    }
}

#[cfg(target_os = "linux")]
mod resident {
    use super::*;

    /// Resident set of this process in bytes, from /proc/self/statm (second field, in pages).
    pub(super) fn rss() -> usize {
        let text = std::fs::read_to_string("/proc/self/statm").unwrap();
        let pages: usize = text.split_whitespace().nth(1).unwrap().parse().unwrap();
        pages * 4096
    }

    pub(super) static DURING: AtomicUsize = AtomicUsize::new(0);

    /// Reads the resident set while the stage buffer exists and writes nothing into it (a short write of zero).
    pub(super) unsafe extern "C" fn watching_stage(
        _: *mut c_void,
        _: u8,
        _: u32,
        _: u32,
        _: *mut u8,
        _: usize,
        written: *mut usize,
    ) -> i32 {
        DURING.store(rss(), Ordering::SeqCst);
        unsafe { *written = 0 };
        0
    }
}

/// R6a P4: the stage buffer was `try_reserve_exact` then `resize`, which writes every page: with memory overcommit the
/// reservation succeeds for a size the machine cannot back, so a corrupt `layer_slot_bytes` was an OOM kill of the host
/// process, not an error. The buffer is now zero-allocated without being touched. WHY 1 GiB and a resident-set read
/// taken inside the callback: the old code commits the whole GiB before the callback runs, the new one commits
/// nothing; the 256 MiB bound is far from both and from the noise of other tests' allocations.
#[cfg(target_os = "linux")]
#[test]
#[cfg_attr(
    miri,
    ignore = "reads /proc/self/statm, which Miri's isolation refuses, and RSS is the assertion"
)]
fn a_huge_stage_buffer_is_not_touched_before_the_provider_writes_it() {
    let _capture = Capture::start(Level::Off);
    let placement = TestPlacement::new();
    let ticket = placement.plan(1).ticket;
    let slot = [1usize << 30];
    let io = RawIo {
        abi_version: ABI_VERSION,
        context: ptr::null_mut(),
        layer_slot_bytes: slot.as_ptr(),
        layer_count: 1,
        stage: Some(resident::watching_stage),
        replace: Some(replace_ok),
    };
    let (mut outcome, mut applied) = (blank_outcome(), blank_applied());
    let before = resident::rss();
    let rc = unsafe {
        seabraid_placement_apply(
            placement.0,
            ticket,
            ptr::from_ref(&io).cast(),
            &mut outcome,
            1,
            &mut applied,
        )
    };
    let during = resident::DURING.load(Ordering::SeqCst);
    assert_eq!(rc, 0);
    assert_eq!(
        outcome.kind, OUTCOME_REJECTED,
        "a short write is a rejected swap"
    );
    assert!(
        during < before + (256 << 20),
        "the 1 GiB stage buffer was committed: resident {before} -> {during}"
    );
}
