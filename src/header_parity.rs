// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! Test-only: the C headers are the ABI source of truth, so every `#define` and enum
//! value in them must equal the Rust constant that implements it, in both directions.
//! A header constant with no Rust counterpart fails here, so a new one cannot be forgotten.

use crate::capabilities as cap;
use crate::diag::{Code, Level};
use crate::ffi::batch_v2::BATCH_V2_VERSION;
use crate::ffi::placement::provider::{self as io, MAX_SLOT_BYTES};
use crate::ffi::placement::types as ty;
use crate::ffi::placement::{UPDATE_ADOPT, UPDATE_REBUILD};
use crate::ffi::provider_v3::{
    ACCEPTS_BF16, ACCEPTS_F16, COMPACT_ROWS, PROVIDER_V3_VERSION, REFERENCE,
};
use crate::ffi::providers::{PARTIAL_ROUTES, PROVIDER_VERSION_MULTI};
use crate::ffi::ABI_EPOCH;
use crate::ffi::ABI_VERSION;
use crate::picks::SKIP_EXPERT_ID;
use crate::placement::{MAX_CARDS, MAX_CELLS};
use std::collections::BTreeMap;

const HEADERS: [&str; 2] = [
    include_str!("../include/seabraid/seabraid.h"),
    include_str!("../include/seabraid/placement.h"),
];

fn strip_comments(text: &str) -> String {
    let mut out = String::new();
    let mut rest = text;
    while let Some(start) = rest.find("/*") {
        out.push_str(&rest[..start]);
        let end = rest[start..]
            .find("*/")
            .expect("unterminated comment in header");
        rest = &rest[start + end + 2..];
    }
    out.push_str(rest);
    out
}

fn eval(expression: &str) -> i64 {
    let expression = expression
        .trim()
        .trim_start_matches('(')
        .trim_end_matches(')')
        .trim();
    if let Some((left, right)) = expression.split_once("<<") {
        return eval(left) << eval(right);
    }
    let digits = expression.trim_end_matches(['u', 'U']);
    digits
        .parse()
        .unwrap_or_else(|_| panic!("unparsable header expression: {expression}"))
}

/// Header macros that are compiler attributes, not numeric constants. WHY a list and not a skip of everything
/// unparsable: an unreadable constant must still fail loudly, so only a named attribute macro is exempt.
const ATTRIBUTE_MACROS: [&str; 1] = ["SEABRAID_DEPRECATED"];

fn defines(text: &str, into: &mut BTreeMap<String, i64>) {
    for line in text
        .lines()
        .filter_map(|line| line.trim().strip_prefix("#define "))
    {
        if let Some((name, value)) = line.trim().split_once(char::is_whitespace) {
            if !ATTRIBUTE_MACROS.contains(&name) {
                into.insert(name.to_owned(), eval(value));
            }
        }
    }
}

fn enums(text: &str, into: &mut BTreeMap<String, i64>) {
    for chunk in text.split("enum ").skip(1) {
        let (Some(open), Some(close)) = (chunk.find('{'), chunk.find('}')) else {
            continue;
        };
        let mut next = 0;
        for item in chunk[open + 1..close]
            .split(',')
            .map(str::trim)
            .filter(|i| !i.is_empty())
        {
            let (name, value) = match item.split_once('=') {
                Some((name, value)) => (name.trim(), eval(value)),
                None => (item, next),
            };
            into.insert(name.to_owned(), value);
            next = value + 1;
        }
    }
}

fn header_values() -> BTreeMap<String, i64> {
    let mut values = BTreeMap::new();
    for header in HEADERS {
        let text = strip_comments(header);
        defines(&text, &mut values);
        enums(&text, &mut values);
    }
    values
}

fn rust_values() -> Vec<(&'static str, i64)> {
    let code = |code: Code| code as i64;
    vec![
        ("SEABRAID_ABI_VERSION", i64::from(ABI_VERSION)),
        (
            "SEABRAID_MULTI_PROVIDER_VERSION",
            i64::from(PROVIDER_VERSION_MULTI),
        ),
        (
            "SEABRAID_PROVIDER_PARTIAL_ROUTES",
            i64::from(PARTIAL_ROUTES),
        ),
        ("SEABRAID_SKIP_EXPERT_ID", i64::from(SKIP_EXPERT_ID)),
        ("SEABRAID_BATCH_V2_VERSION", i64::from(BATCH_V2_VERSION)),
        (
            "SEABRAID_PROVIDER_V3_VERSION",
            i64::from(PROVIDER_V3_VERSION),
        ),
        ("SEABRAID_PROVIDER_COMPACT_ROWS", i64::from(COMPACT_ROWS)),
        ("SEABRAID_PROVIDER_ACCEPTS_F16", i64::from(ACCEPTS_F16)),
        ("SEABRAID_PROVIDER_ACCEPTS_BF16", i64::from(ACCEPTS_BF16)),
        ("SEABRAID_PROVIDER_REFERENCE", i64::from(REFERENCE)),
        ("SEABRAID_ABI_EPOCH", i64::from(ABI_EPOCH)),
        ("SEABRAID_OK", 0),
        ("SEABRAID_INVALID", code(Code::Invalid)),
        ("SEABRAID_BUSY_OR_STATE", code(Code::BusyOrState)),
        ("SEABRAID_TIMEOUT", code(Code::Timeout)),
        ("SEABRAID_PROVIDER", code(Code::Provider)),
        ("SEABRAID_INTERNAL", code(Code::Internal)),
        ("SEABRAID_SOURCE_TEXT", i64::from(cap::TEXT)),
        ("SEABRAID_SOURCE_IMAGE", i64::from(cap::IMAGE)),
        ("SEABRAID_SOURCE_AUDIO", i64::from(cap::AUDIO)),
        ("SEABRAID_SOURCE_VIDEO", i64::from(cap::VIDEO)),
        ("SEABRAID_SOURCE_OTHER", i64::from(cap::OTHER)),
        (
            "SEABRAID_OPERATION_ROUTED_MOE",
            i64::from(cap::OPERATION_ROUTED_MOE),
        ),
        (
            "SEABRAID_FORMAT_ROUTED_F32",
            i64::from(cap::FORMAT_ROUTED_F32),
        ),
        (
            "SEABRAID_FORMAT_PARTIAL_F32",
            i64::from(cap::FORMAT_PARTIAL_F32),
        ),
        (
            "SEABRAID_FORMAT_ROUTED_F16",
            i64::from(cap::FORMAT_ROUTED_F16),
        ),
        (
            "SEABRAID_FORMAT_ROUTED_BF16",
            i64::from(cap::FORMAT_ROUTED_BF16),
        ),
        ("SEABRAID_LOG_DEBUG", Level::Debug as i64),
        ("SEABRAID_LOG_INFO", Level::Info as i64),
        ("SEABRAID_LOG_WARN", Level::Warn as i64),
        ("SEABRAID_LOG_ERROR", Level::Error as i64),
        ("SEABRAID_LOG_OFF", Level::Off as i64),
        ("SEABRAID_PLACEMENT_IO_ACK", i64::from(io::IO_ACK)),
        ("SEABRAID_PLACEMENT_IO_REFUSED", i64::from(io::IO_REFUSED)),
        ("SEABRAID_PLACEMENT_IO_FAILED", i64::from(io::IO_FAILED)),
        ("SEABRAID_PLACEMENT_IO_TIMEOUT", i64::from(io::IO_TIMEOUT)),
        (
            "SEABRAID_PLACEMENT_COMMITTED",
            i64::from(ty::OUTCOME_COMMITTED),
        ),
        (
            "SEABRAID_PLACEMENT_REJECTED",
            i64::from(ty::OUTCOME_REJECTED),
        ),
        (
            "SEABRAID_PLACEMENT_RESTORED",
            i64::from(ty::OUTCOME_RESTORED),
        ),
        (
            "SEABRAID_PLACEMENT_QUARANTINED",
            i64::from(ty::OUTCOME_QUARANTINED),
        ),
        ("SEABRAID_PLACEMENT_SKIPPED", i64::from(ty::OUTCOME_SKIPPED)),
        (
            "SEABRAID_PLACEMENT_POLICY_V2_VERSION",
            i64::from(ty::POLICY_V2_VERSION),
        ),
        (
            "SEABRAID_PLACEMENT_PROVIDER_V2_VERSION",
            i64::from(io::PROVIDER_V2_VERSION),
        ),
        ("SEABRAID_PLACEMENT_STAGE", i64::from(ty::PHASE_STAGE)),
        ("SEABRAID_PLACEMENT_REPLACE", i64::from(ty::PHASE_REPLACE)),
        ("SEABRAID_PLACEMENT_RESTORE", i64::from(ty::PHASE_RESTORE)),
        ("SEABRAID_PLACEMENT_REFUSED", i64::from(ty::ISSUE_REFUSED)),
        ("SEABRAID_PLACEMENT_FAILED", i64::from(ty::ISSUE_FAILED)),
        (
            "SEABRAID_PLACEMENT_UNACKNOWLEDGED",
            i64::from(ty::ISSUE_UNACKNOWLEDGED),
        ),
        ("SEABRAID_PLACEMENT_PANICKED", i64::from(ty::ISSUE_PANICKED)),
        (
            "SEABRAID_PLACEMENT_MALFORMED_BLOB",
            i64::from(ty::ISSUE_MALFORMED_BLOB),
        ),
        ("SEABRAID_PLACEMENT_MAX_CARDS", i64::from(MAX_CARDS)),
        ("SEABRAID_PLACEMENT_MAX_CELLS", MAX_CELLS as i64),
        ("SEABRAID_PLACEMENT_MAX_SLOT_BYTES", MAX_SLOT_BYTES as i64),
        ("SEABRAID_PLACEMENT_ADOPT", i64::from(UPDATE_ADOPT)),
        ("SEABRAID_PLACEMENT_REBUILD", i64::from(UPDATE_REBUILD)),
    ]
}

#[test]
fn header_constants_equal_the_rust_constants() {
    let header = header_values();
    for (name, value) in rust_values() {
        assert_eq!(header.get(name), Some(&value), "{name}");
    }
    let rust: Vec<_> = rust_values().into_iter().map(|(name, _)| name).collect();
    for name in header.keys() {
        // Only SEABRAID_ constants are ABI values; guards and similar carry no value.
        assert!(
            rust.contains(&name.as_str()),
            "{name} is in a header but not in the parity table"
        );
    }
}

#[test]
fn the_parser_reads_what_the_headers_say() {
    let header = header_values();
    assert_eq!(header["SEABRAID_SOURCE_OTHER"], 16);
    assert_eq!(header["SEABRAID_SKIP_EXPERT_ID"], -1);
    assert!(header.len() > 35, "parsed only {} constants", header.len());
}

/// Every struct row of the layout table must be a typedef in a header, so the table cannot keep rows for
/// structs that were renamed or removed (the C compile and the Rust asserts catch the opposite drift).
#[test]
fn every_layout_table_struct_is_a_header_typedef() {
    let table = include_str!("layout/abi_layout.txt");
    let headers = HEADERS.join(
        "
",
    );
    let names: Vec<_> = table
        .lines()
        .filter(|line| !line.starts_with('#') && !line.contains('.'))
        .filter_map(|line| line.split_whitespace().next())
        .collect();
    assert!(names.len() >= 15, "parsed only {} structs", names.len());
    for name in names {
        assert!(
            headers.contains(&format!("}} {name};")),
            "{name} is not a typedef in a header"
        );
    }
}

/// `seabraid.h` and the exports behind it name the same functions, so a new entry point (the v2 pair included)
/// cannot ship undeclared and a declaration cannot outlive its export. Placement has its own header.
#[test]
fn seabraid_h_declares_exactly_the_exported_functions() {
    let sources = [
        include_str!("ffi.rs"),
        include_str!("ffi/v2.rs"),
        include_str!("ffi/logging.rs"),
    ];
    let mut exported: Vec<&str> = sources
        .iter()
        .flat_map(|text| text.split("extern \"C\" fn ").skip(1))
        .filter_map(|rest| rest.split('(').next())
        .filter(|name| name.starts_with("seabraid_"))
        .collect();
    let header = strip_comments(HEADERS[0]);
    let mut declared: Vec<&str> = header
        .match_indices("seabraid_")
        .filter_map(|(start, _)| {
            let rest = &header[start..];
            let end = rest.find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))?;
            rest[end..].starts_with('(').then_some(&rest[..end])
        })
        .collect();
    exported.sort_unstable();
    declared.sort_unstable();
    assert!(
        exported.len() > 15,
        "parsed only {} exports",
        exported.len()
    );
    assert!(exported.contains(&"seabraid_session_open"));
    assert!(exported.contains(&"seabraid_session_submit"));
    assert_eq!(declared, exported);
}

/// `placement.h` and the placement exports name the same functions, so an entry point (`seabraid_placement_open_v2`
/// included) cannot ship undeclared and a declaration cannot outlive its export. The same rule as for `seabraid.h`.
#[test]
fn placement_h_declares_exactly_the_exported_placement_functions() {
    let sources = [
        include_str!("ffi/placement.rs"),
        include_str!("ffi/placement/apply.rs"),
        include_str!("ffi/placement/fill.rs"),
        include_str!("ffi/placement/open.rs"),
    ];
    let mut exported: Vec<&str> = sources
        .iter()
        .flat_map(|text| text.split("extern \"C\" fn ").skip(1))
        .filter_map(|rest| rest.split('(').next())
        .filter(|name| name.starts_with("seabraid_placement_"))
        .collect();
    let header = strip_comments(HEADERS[1]);
    let mut declared: Vec<&str> = header
        .match_indices("seabraid_placement_")
        .filter_map(|(start, _)| {
            let rest = &header[start..];
            let end = rest.find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))?;
            rest[end..].starts_with('(').then_some(&rest[..end])
        })
        .collect();
    exported.sort_unstable();
    declared.sort_unstable();
    assert!(exported.contains(&"seabraid_placement_open_v2"));
    assert_eq!(declared, exported);
}
