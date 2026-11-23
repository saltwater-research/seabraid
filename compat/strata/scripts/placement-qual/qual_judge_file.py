# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""F1, file-reference moves: every live F1 start must show file bytes moved by the helper and ZERO bad sums; the blob arm must show no file read.

OWNER: this module owns the F1 decision for the `file-moves` persona (workloads tagged ``"persona": "file-moves"`` in arms.json).  The
counters are parsed from the engine log by qual_stats.parse_io with the ``io_patterns`` file_* / bad_sum regexes in arms.json, which are
PROVISIONAL until tracks FR-E / FR-H define the real log line (the expected line is documented once, in arms.json
``io_patterns_file_note``).  The speed bars of FR-A1..A3 (hop within +10%, tok/s within -1%) are judged by the maintainers from the
recorded numbers, not here: the design makes file moves default-on only on that evidence, so F1 gates correctness, not speed.
CONTRACT: ``f1(cfg, results) -> verdict`` or ``None`` when the persona is not in the selected workloads (a default run never shows it).
PASS needs, for every required start (each persona workload x arm F1 x start 0..n-1): a result, the four counters present, file_reads > 0
and direct + buffered bytes > 0, and bad_sum == 0; and every F0 start with counters shows file_reads == 0 (the reference really is the
blob path).  Any bad_sum > 0, or a file read in F0, is FAIL.  A missing start or counter is WARN: absence of evidence never passes.
FAILURE: pure, no I/O; counters are cumulative (Header), so the last value in the log is the figure (max of the parsed list).
"""

from __future__ import annotations

import qual_judge as qj

TITLE = "File-reference moves"
COUNTERS = ("file_direct_bytes", "file_buffered_bytes", "file_reads", "bad_sum")
PERSONA = "file-moves"


def _last(io: dict, name: str) -> float | None:
    xs = io.get(name) or []
    return max(xs) if xs else None


def _counters(r: dict) -> dict | None:
    io = r.get("io", {})
    got = {k: _last(io, k) for k in COUNTERS}
    return None if None in got.values() else got


def _live(r: dict, c: dict) -> tuple[str, str]:
    """(status, note) of one F1 start's counters."""
    tag = f"{r['workload']} s{r['idx']}"
    if c["bad_sum"] > 0:
        return "FAIL", f"{tag}: {c['bad_sum']:.0f} bad sum(s)"
    if c["file_reads"] <= 0 or c["file_direct_bytes"] + c["file_buffered_bytes"] <= 0:
        return "WARN", f"{tag}: no file read happened (reads {c['file_reads']:.0f}), so nothing was tested"
    return "PASS", f"{tag}: {c['file_reads']:.0f} reads, {c['file_direct_bytes'] + c['file_buffered_bytes']:.0f} B, 0 bad sums"


def _one(wl: str, key: str, i: int, r: dict | None) -> tuple[str, str]:
    """(status, note) of one required F1 start: absent result or absent counters are WARN, otherwise its counters decide."""
    if r is None:
        return "WARN", f"{wl} {key} s{i}: no start result"
    c = _counters(r)
    return ("WARN", f"{wl} {key} s{i}: counters not in the log (PROVISIONAL line)") if c is None else _live(r, c)


def _f1_starts(wls: dict, res: list[dict]) -> list[tuple[str, str]]:
    out = []
    for wl, spec in wls.items():
        have = {(r["key"], r["idx"]): r for r in res if r["workload"] == wl and r.get("ok")}
        keys = [f"F1/{v}" if v != "base" else "F1" for v in spec["arms"].get("F1", [])]
        out += [_one(wl, key, i, have.get((key, i))) for key in keys for i in range(spec["n"])]
    return out


def _f0_reads(wls: dict, res: list[dict]) -> list[tuple[str, str]]:
    bad = [r for r in res if r["workload"] in wls and r["arm"] == "F0" and (_last(r.get("io", {}), "file_reads") or 0) > 0]
    return [("FAIL", f"{r['workload']} F0 s{r['idx']}: the blob arm read {_last(r['io'], 'file_reads'):.0f} file(s)") for r in bad]


def f1(cfg: dict, res: list[dict]) -> dict | None:
    wls = {k: v for k, v in cfg["workloads"].items() if v.get("persona") == PERSONA}
    if not wls:
        return None
    got = _f1_starts(wls, res) + _f0_reads(wls, res)
    status = max((g[0] for g in got), key=qj._RANK.get)
    return qj._v("F1", TITLE, status, f"{len(got)} checks over {', '.join(wls)}: " + "; ".join(g[1] for g in got)[:900])
