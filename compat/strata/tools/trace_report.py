#!/usr/bin/env python3
# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""trace_report.py TRACE.jsonl [--log ENGINE_LOG] [--out DIR]: read the file Seabraid writes when SEABRAID_TRACE=<path>
(adapters/strata/placement_trace.hpp) and answer two questions with data:

  1. The expert journey table: every expert that moved, each hop with its time, from -> to rung, source (card or
     fallback), bytes and the stage and write time of the swap it was part of.  A swap moves TWO experts (one up, one
     down), so each committed swap is two rows; a swap that did not commit is one row with its kind (rejected,
     restored, quarantined, skipped) and the issue that stopped it.
  2. The latency table: where the time goes, per phase, with n, mean, p50, p99 in microseconds: helper submit->collect,
     time blocked in collect, the placement round, a move's stage and write time and (with --log) the SSD wait and the
     engine's ms per window, read from Strata's own lines ("decode waited X ms in N fetches", "strata decode timing:").

With --out DIR it also writes journeys.csv and latency.csv.  Stdlib only.

WINDOW ROWS ARE SUMMARIES: a window record carries that window's mean, p50 and p99, not its samples.  The report's mean
is the exact sample-weighted mean; its p50 is the median of the windows' p50 and its p99 the largest window p99 (an
upper bound).  Round, stage, write and log rows are computed from the raw values and are exact.

FAILURE: a missing or empty trace is exit 2; a line that is not a JSON object is counted and skipped, never fatal.
"""
import argparse
import csv
import json
import math
import os
import re
import sys

DECODE_WAIT = re.compile(r"decode waited ([\d.]+) ms in (\d+) fetches")
FILE_IO = re.compile(r"the OS read ([\d.]+) MB from storage \((\d+) major faults\)")
DECODE_TIMING = re.compile(r"strata decode timing: (\d+) windows, avg T [\d.]+, [\d.]+ tokens/window, ([\d.]+) ms/window")
PATCH_NOTE = ("RAM-vs-NVMe split NOT AVAILABLE: src=fallback only says the lower rung is the fallback, not whether the "
              "expert came from the RAM copy or was read from the pack on NVMe. FallbackRung::ram_hits()/file_reads()/"
              "ram_installs() exist in adapters/strata/strata-v7-seabraid.patch but nothing reads them. "
              "PATCH CHANGE NEEDED: in src/program/generate.cpp, right after `placement_fallback = std::make_unique<"
              "strata::core::FallbackRung>(...)` and before PlacementDriver::open, add "
              "placement_cfg.fallback_counters = [fb = placement_fallback.get()](int64_t& ram, int64_t& file) "
              "{ ram = fb->ram_hits(); file = fb->file_reads(); }; (the adapter already reads it per round). Also "
              "FallbackRung::replace returns kOk silently when replace_resident_from fails: add a counter "
              "(e.g. ram_install_misses_) next to ram_installs_ and report it the same way.")
JOURNEY_COLS = ["layer", "expert", "hop", "ts_us", "time", "round", "epoch", "direction", "status", "from_rung",
                "to_rung", "src_or_dst", "bytes", "stage_us", "write_us", "issue"]
LATENCY_COLS = ["phase", "source", "n", "mean_us", "p50_us", "p99_us", "note"]


def percentile(values, p):
    """Nearest rank, the same rule as the writer (placement_trace.cpp): the smallest value with p of the samples below."""
    ordered = sorted(values)
    return ordered[max(1, math.ceil(p * len(ordered))) - 1] if ordered else 0


def load(path):
    """(records, bad_lines).  FAILURE: unreadable file raises OSError; the caller turns it into exit 2."""
    records, bad = [], 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                bad += 1
                continue
            if isinstance(rec, dict) and "rec" in rec:
                records.append(rec)
            else:
                bad += 1
    return records, bad


def _issue_text(rec):
    parts = []
    for key in ("issue", "restore_issue"):
        if key in rec:
            parts.append("%s %s/%s@rung%d" % (key.split("_")[0], rec[key]["stage"], rec[key]["result"], rec[key]["rung"]))
    return "; ".join(parts)


def journeys(records):
    """Rows for JOURNEY_COLS, ordered by (layer, expert, time); hop counts that expert's rows within its layer."""
    rows = []
    for rec in (r for r in records if r["rec"] == "move"):
        base = {"ts_us": rec["ts_us"], "time": rec["time"], "round": rec["round"], "epoch": rec["epoch"],
                "layer": rec["layer"], "bytes": rec["bytes"], "stage_us": rec["stage_us"], "write_us": rec["write_us"],
                "issue": _issue_text(rec), "status": rec["kind"]}
        ends = [("up", rec["up"], rec["src"])]
        if rec["kind"] == "committed":
            ends.append(("down", rec["down"], rec["dst"]))   # a swap that did not commit moved nobody down
        for direction, end, where in ends:
            rows.append(dict(base, direction=direction, expert=end["expert"], from_rung=end["from_rung"],
                             to_rung=end["to_rung"], src_or_dst=where))
    rows.sort(key=lambda r: (r["layer"], r["expert"], r["ts_us"]))
    seen = {}
    for row in rows:
        key = (row["layer"], row["expert"])
        seen[key] = seen.get(key, 0) + 1
        row["hop"] = seen[key]
    return rows


def paths(rows):
    """{(layer, expert): 'rung>rung>rung'} over committed hops: the expert's route through the rungs."""
    out = {}
    for row in rows:
        if row["status"] != "committed":
            continue
        chain = out.setdefault((row["layer"], row["expert"]), [str(row["from_rung"])])
        chain.append(str(row["to_rung"]))
    return {key: ">".join(chain) for key, chain in out.items()}


def _row(phase, source, values, note=""):
    values = list(values)
    mean = sum(values) / len(values) if values else 0.0
    return {"phase": phase, "source": source, "n": len(values), "mean_us": "%.1f" % mean,
            "p50_us": "%d" % percentile(values, 0.50), "p99_us": "%d" % percentile(values, 0.99), "note": note}


def _window_row(phase, source, windows, key):
    """A summary row over window records: exact weighted mean, median of p50, max of p99 (see the module docstring)."""
    total = sum(w[key]["n"] for w in windows)
    mean = sum(w[key]["mean"] * w[key]["n"] for w in windows) / total if total else 0.0
    p50s = sorted(w[key]["p50"] for w in windows)
    return {"phase": phase, "source": source, "n": total, "mean_us": "%.1f" % mean,
            "p50_us": "%d" % (p50s[(len(p50s) - 1) // 2] if p50s else 0),
            "p99_us": "%d" % max([w[key]["p99"] for w in windows] or [0]), "note": "window summaries"}


def _helper_rows(records):
    rows = []
    for rung in sorted({r["rung"] for r in records if r["rec"] == "window"}):
        wins = [r for r in records if r["rec"] == "window" and r["rung"] == rung]
        rows.append(_window_row("helper submit->collect", "helper %d" % rung, wins, "submit_to_collect_us"))
        rows.append(_window_row("helper blocked in collect", "helper %d" % rung, wins, "blocked_us"))
    return rows


def _placement_rows(records):
    moves = [r for r in records if r["rec"] == "move" and r["kind"] == "committed" and r["timed"]]
    return [_row("round (fence to publish)", "trace", [r["round_us"] for r in records if r["rec"] == "round"]),
            _row("move stage", "trace", [m["stage_us"] for m in moves], "per swap"),
            _row("move write", "trace", [m["write_us"] for m in moves], "per swap")]


def _log_rows(log_text):
    waits = [float(m.group(1)) * 1000.0 for m in DECODE_WAIT.finditer(log_text)]
    windows = [float(m.group(2)) * 1000.0 for m in DECODE_TIMING.finditer(log_text)]
    return [_row("SSD wait (decode waited)", "engine log", waits, "per request"),
            _row("engine decode window", "engine log", windows, "ms/window x 1000")]


def latency(records, log_text=""):
    return _helper_rows(records) + _placement_rows(records) + (_log_rows(log_text) if log_text else [])


def ram_vs_nvme(records):
    """(ram_hits, file_reads, counted rounds), or None when no round carried the counters."""
    counted = [r["fallback"] for r in records if r["rec"] == "round" and r.get("fallback")]
    if not counted:
        return None
    return sum(c["ram_hits"] for c in counted), sum(c["file_reads"] for c in counted), len(counted)


def _table(cols, rows):
    cells = [[str(r.get(c, "")) for c in cols] for r in rows]
    width = [max([len(c)] + [len(row[i]) for row in cells]) for i, c in enumerate(cols)]
    line = lambda vals: "  ".join(v.ljust(w) for v, w in zip(vals, width)).rstrip()
    return "\n".join([line(cols)] + [line(row) for row in cells])


def render(records, bad, log_text=""):
    jrows, lrows = journeys(records), latency(records, log_text)
    out = ["SEABRAID TRACE REPORT", "records: %d (%d unreadable lines skipped)" % (len(records), bad), ""]
    out += ["EXPERT JOURNEYS (one row per expert hop; stage_us and write_us are the whole swap's, shared by its two rows)",
            _table(["layer", "expert", "hop", "round", "direction", "status", "from_rung", "to_rung", "src_or_dst", "bytes",
                    "stage_us", "write_us", "time", "issue"], jrows) if jrows else "(no moves recorded)", ""]
    route = paths(jrows)
    if route:
        out += ["ROUTES (committed hops, rung>rung)"] + ["layer %d expert %d: %s" % (k[0], k[1], v)
                                                         for k, v in sorted(route.items())] + [""]
    out += ["LATENCY (microseconds)", _table(LATENCY_COLS, lrows), ""]
    split = ram_vs_nvme(records)
    out.append("FALLBACK STAGES: " + ("RAM %d, NVMe %d over %d rounds" % split if split else PATCH_NOTE))
    events = [r for r in records if r["rec"] == "helper" and r["event"] in ("down", "closed", "served_after_loss")]
    if events:
        out += ["", "HELPER EVENTS"] + ["rung %d %s (%s) after_loss_us=%d" % (e["rung"], e["event"], e["reason"], e["after_loss_us"])
                                        for e in events]
    return "\n".join(out) + "\n", jrows, lrows


def write_csv(path, cols, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("trace")
    ap.add_argument("--log", help="engine log with Strata's file-tier and decode-timing lines")
    ap.add_argument("--out", help="directory for journeys.csv and latency.csv")
    args = ap.parse_args(argv)
    try:
        records, bad = load(args.trace)
        log_text = open(args.log, encoding="utf-8", errors="replace").read() if args.log else ""
    except OSError as exc:
        print("trace_report: %s" % exc, file=sys.stderr)
        return 2
    if not records:
        print("trace_report: no trace records in %s (%d unreadable lines)" % (args.trace, bad), file=sys.stderr)
        return 2
    text, jrows, lrows = render(records, bad, log_text)
    sys.stdout.write(text)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        write_csv(os.path.join(args.out, "journeys.csv"), JOURNEY_COLS, jrows)
        write_csv(os.path.join(args.out, "latency.csv"), LATENCY_COLS, lrows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
