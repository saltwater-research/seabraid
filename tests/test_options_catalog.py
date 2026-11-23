# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""The options catalog (docs/OPTIONS.md) stays true: every switch it lists exists, every row is complete.

OWNER   : what "a valid catalog row" means.  docs/OPTIONS.md owns the facts; this file owns the rules about them.
CONTRACT: parse every table whose header starts with "| Option |"; per row (columns Option, What it does, Owner, Switch,
          Default, Engines today, Evidence): (a) every `backticked` switch token is a tracked file path (when it has a
          "/") or appears in tracked source outside docs/; (b) owner, default and evidence are non-empty and the
          evidence is "not measured" or a link to a README or RESEARCH-LOG that exists; (c) a default that does not start
          with "off" contains "correctness fix"; (d) a switch absent from source is allowed ONLY when the row's Engines today cell
          holds "pending: track FR-<letter>" (the track that will add it); a bare "pending" or no track name is not enough.
FAILURE : fail-fast, one message per problem naming the row.
CHECKER : this file.  test_catalog_check_bites_* run the same check on a scratch copy with one planted fault each, so the
          gate is shown to fail (the sabotage record is in checks/SABOTAGE.md's style: plant one fault, expect one failure).
"""

from __future__ import annotations

import os
import re

from conftest import REPO

import _common

CATALOG = "docs/OPTIONS.md"
COLUMNS = ["Option", "What it does", "Owner", "Switch", "Default", "Engines today", "Evidence"]


def parse_rows(text: str) -> list[dict[str, str]]:
    """Every data row of every catalog table, as a dict keyed by column name."""
    rows: list[dict[str, str]] = []
    in_table = False
    for line in text.splitlines():
        if line.startswith("| Option |"):
            in_table = True
            header = [c.strip() for c in line.strip().strip("|").split("|")]
            assert header == COLUMNS, f"catalog table header is {header}, expected {COLUMNS}"
            continue
        if not in_table:
            continue
        if not line.startswith("|"):
            in_table = False
            continue
        if re.match(r"^\|[-| ]+\|$", line):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        assert len(cells) == len(COLUMNS), f"row has {len(cells)} cells, expected {len(COLUMNS)}: {line[:80]}"
        rows.append(dict(zip(COLUMNS, cells)))
    return rows


def switch_tokens(cell: str) -> list[str]:
    """The name part of each `backticked` token: `--pcie-frac N` -> --pcie-frac, `A=<dir>` -> A, a path stays whole."""
    out = []
    for span in re.findall(r"`([^`]+)`", cell):
        out.append(re.split(r"[ =]", span.strip(), maxsplit=1)[0])
    return out


def source_files(root: str) -> list[str]:
    """Tracked files outside docs/ and compat/strata/docs/ and outside this test (it quotes switches in its own text)."""
    skip = {"tests/test_options_catalog.py"}
    return [r for r in _common.tracked(root) if not r.startswith(("docs/", "compat/strata/docs/")) and r not in skip]


# WHY a pending marker at all: a switch whose implementing track has not landed would otherwise force the row out of the catalog or the
# checker off.  The marker must name the track, so a stale "pending" is greppable and one bare word cannot excuse a missing switch.
PENDING = re.compile(r"pending: track FR-[A-Z]\b")


def switch_problems(name: str, cell: str, fileset: set[str], blobs: list[str], pending: bool = False) -> list[str]:
    tokens = switch_tokens(cell)
    bad = [] if tokens else [f"{name}: no backticked switch"]
    for tok in tokens:
        if "/" in tok:
            if tok not in fileset:
                bad.append(f"{name}: switch path {tok} is not a tracked file")
        elif not pending and not any(tok in blob for blob in blobs):
            bad.append(f"{name}: switch {tok} not found in tracked source outside docs/")
    return bad


def evidence_problems(name: str, root: str, evidence: str) -> list[str]:
    if not evidence or evidence.startswith("not measured"):
        return []
    link = re.search(r"\]\(([^)#]+)(#[^)]*)?\)", evidence)
    if not link or os.path.basename(link.group(1)) not in ("README.md", "RESEARCH-LOG.md"):
        return [f"{name}: evidence must be 'not measured' or a README / RESEARCH-LOG link"]
    if not os.path.exists(os.path.normpath(os.path.join(root, "docs", link.group(1)))):
        return [f"{name}: evidence link {link.group(1)} does not exist"]
    return []


def row_problems(row: dict[str, str], root: str, fileset: set[str], blobs: list[str]) -> list[str]:
    name = row["Option"]
    bad = switch_problems(name, row["Switch"], fileset, blobs, bool(PENDING.search(row["Engines today"])))
    bad += [f"{name}: empty {col}" for col in ("Owner", "Default", "Evidence") if not row[col]]
    default = row["Default"]
    if default and not default.startswith("off") and "correctness fix" not in default:
        bad.append(f"{name}: default {default!r} is not off and is not marked correctness fix")
    return bad + evidence_problems(name, root, row["Evidence"])


def problems(root: str) -> list[str]:
    with open(os.path.join(root, CATALOG), encoding="utf-8") as fh:
        rows = parse_rows(fh.read())
    if not rows:
        return [f"{CATALOG}: no catalog rows found"]
    files = source_files(root)
    blobs = [t for t in (_common.read_text(root, r) for r in files) if t]
    return [p for row in rows for p in row_problems(row, root, set(files), blobs)]


def test_catalog_is_true():
    assert problems(REPO) == []


def test_catalog_covers_the_required_tweaks():
    with open(os.path.join(REPO, CATALOG), encoding="utf-8") as fh:
        text = fh.read()
    for needed in ("promote_margin", "cooldown_rounds", "max_moves_per_round", "max_helper_layer_share",
                   "ladder_boundaries", "fallback_boundary", "max_hops_per_round", "compact_rows", "SEABRAID_STRATA_HOT_SHARE",
                   "--retier", "--retier-promote", "STRATA_PREFILL_REMOTE_COMPUTE", "--prefill-tokens",
                   "--remote-expert-opt", "STRATA_ROUTE_HIST", "--expert-cache-device1", "--adapt-every",
                   "--adapt-swaps", "--pcie-frac", "SEABRAID_STRATA_EXCLUDE_HELPER_RAM", "SEABRAID_QWEN_HELPER",
                   "SEABRAID_LIBRARY"):
        assert f"`{needed}" in text, f"catalog has no row for {needed}"


def _scratch(tree, old: str, new: str) -> str:
    path = os.path.join(tree, CATALOG)
    with open(path, encoding="utf-8", newline="") as fh:
        text = fh.read()
    assert old in text, f"setup text not found: {old!r}"
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(text.replace(old, new, 1))
    return path


def test_catalog_check_bites_on_a_renamed_switch(tree):
    assert problems(tree) == []
    _scratch(tree, "`--adapt-swaps`", "`--adapt-swaps-renamed`")
    assert any("--adapt-swaps-renamed" in p for p in problems(tree))


PLANTED = "| Planted | x. | engine flag | `--planted-switch-xyz` | off | Strata%s | not measured |\n"
MARK = "\n\n## Not found in code"


def _plant(tree, engines: str) -> None:
    """Add one row to the last table (a row after the blank line would not be part of it)."""
    _scratch(tree, MARK, "\n" + PLANTED % engines + MARK[1:])


def test_a_switch_not_in_source_is_allowed_only_by_an_explicit_pending_track_cell(tree):
    _plant(tree, "; pending: track FR-E")
    assert problems(tree) == []


def test_catalog_check_bites_on_an_unmarked_missing_switch(tree):
    _plant(tree, "")
    assert any("--planted-switch-xyz" in p for p in problems(tree))


def test_catalog_check_bites_on_a_vague_pending(tree):
    for vague in ("; pending", "; pending: track", "; pending: track FR-", "; pending: track FR-ab"):
        _plant(tree, vague)
        assert any("--planted-switch-xyz" in p for p in problems(tree)), vague
        _scratch(tree, PLANTED % vague, "")


def test_catalog_check_bites_on_a_missing_owner(tree):
    _scratch(tree, "| engine flag | `--adapt-swaps` |", "|  | `--adapt-swaps` |")
    assert any("empty Owner" in p for p in problems(tree))


def test_catalog_check_bites_on_an_unmarked_on_default(tree):
    _scratch(tree, "| on (correctness fix) | Strata | not measured |\n| Channel hardening", "| on | Strata | not measured |\n| Channel hardening")
    assert any("not marked correctness fix" in p for p in problems(tree))
