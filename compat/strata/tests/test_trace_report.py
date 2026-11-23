# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Checker for tools/trace_report.py on a synthetic trace in the format adapters/strata/placement_trace.cpp writes.

The numbers asserted by name below were worked out by hand from tests/trace_report/synthetic.jsonl and engine.log, so a
wrong formula changes a number this file knows; the golden files then pin the whole rendered text and both CSVs.
Regenerate a golden only after checking the change by hand: REGEN=1 pytest tests/test_trace_report.py.  CPU only."""
import csv
import os
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tools"))
import trace_report as T  # noqa: E402

FIX = HERE / "trace_report"
TRACE, LOG = FIX / "synthetic.jsonl", FIX / "engine.log"


@pytest.fixture(scope="module")
def loaded():
    records, bad = T.load(str(TRACE))
    return records, bad


def golden(name, text):
    path = FIX / name
    if os.environ.get("REGEN"):
        path.write_text(text, newline="\n")
    assert text == path.read_text().replace("\r\n", "\n"), name


def test_percentile_is_nearest_rank_like_the_writer():
    assert T.percentile(list(range(100, 0, -1)), 0.50) == 50
    assert T.percentile(list(range(1, 101)), 0.99) == 99
    assert T.percentile([40, 10, 30, 20], 0.50) == 20 and T.percentile([40, 10, 30, 20], 0.99) == 40
    assert T.percentile([], 0.5) == 0


def test_unreadable_lines_are_counted_not_fatal(loaded):
    records, bad = loaded
    assert len(records) == 11 and bad == 1


def test_journey_routes_and_hops(loaded):
    rows = T.journeys(loaded[0])
    assert len(rows) == 7                      # 3 committed swaps x 2 experts, plus the rejected one
    assert T.paths(rows) == {(0, 2): "0>1>0", (0, 3): "1>2", (0, 5): "1>0>1", (0, 7): "2>1"}
    five = [r for r in rows if r["expert"] == 5]
    assert [(r["hop"], r["direction"], r["from_rung"], r["to_rung"]) for r in five] == [(1, "up", 1, 0), (2, "down", 0, 1)]


def test_journey_source_and_timing(loaded):
    fb = [r for r in T.journeys(loaded[0]) if r["expert"] == 7][0]
    assert fb["src_or_dst"] == "fallback" and fb["stage_us"] == 2000 and fb["write_us"] == 500


def test_journey_failures(loaded):
    rows = T.journeys(loaded[0])
    failed = [r for r in rows if r["status"] == "rejected"]
    assert len(failed) == 1 and failed[0]["expert"] == 4 and failed[0]["issue"] == "issue stage/failed@rung1"
    assert not [r for r in rows if r["expert"] == 6]   # an uncommitted swap moved nobody down


def test_latency_numbers_by_hand(loaded):
    rows = {r["phase"]: r for r in T.latency(loaded[0], LOG.read_text())}
    sub = rows["helper submit->collect"]
    assert (sub["n"], sub["mean_us"], sub["p50_us"], sub["p99_us"]) == (8, "200.0", "90", "900")
    blk = rows["helper blocked in collect"]
    assert (blk["mean_us"], blk["p50_us"], blk["p99_us"]) == ("30.0", "10", "100")
    rnd = rows["round (fence to publish)"]
    assert (rnd["n"], rnd["mean_us"], rnd["p50_us"], rnd["p99_us"]) == (2, "6000.0", "3000", "9000")
    assert (rows["move stage"]["mean_us"], rows["move stage"]["p99_us"]) == ("750.0", "2000")
    assert (rows["move write"]["mean_us"], rows["move write"]["p50_us"]) == ("383.3", "350")
    ssd = rows["SSD wait (decode waited)"]
    assert (ssd["n"], ssd["mean_us"], ssd["p50_us"], ssd["p99_us"]) == (2, "10000.0", "7500", "12500")
    assert rows["engine decode window"]["mean_us"] == "58000.0"


def test_without_a_log_there_are_no_engine_rows(loaded):
    assert not [r for r in T.latency(loaded[0]) if r["source"] == "engine log"]


def test_ram_vs_nvme_sums_and_names_the_missing_patch_when_absent(loaded):
    assert T.ram_vs_nvme(loaded[0]) == (1, 3, 2)
    bare = [dict(r, fallback=None) if r["rec"] == "round" else r for r in loaded[0]]
    assert T.ram_vs_nvme(bare) is None
    text = T.render(bare, 0)[0]
    assert "fallback_counters" in text and "ram_hits()" in text and "generate.cpp" in text


def test_golden_text_and_csvs(tmp_path, capsys):
    assert T.main([str(TRACE), "--log", str(LOG), "--out", str(tmp_path)]) == 0
    golden("golden_report.txt", capsys.readouterr().out)
    golden("golden_journeys.csv", (tmp_path / "journeys.csv").read_text().replace("\r\n", "\n"))
    golden("golden_latency.csv", (tmp_path / "latency.csv").read_text().replace("\r\n", "\n"))
    with open(tmp_path / "latency.csv", newline="") as fh:
        assert [r["phase"] for r in csv.DictReader(fh)][0] == "helper submit->collect"


def test_missing_or_empty_trace_is_exit_2(tmp_path):
    assert T.main([str(tmp_path / "nope.jsonl")]) == 2
    empty = tmp_path / "e.jsonl"
    empty.write_text("not json\n")
    assert T.main([str(empty)]) == 2
