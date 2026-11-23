# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""The recorded chain: every step with name, wall-clock start, seconds, rc and captured output; plus the report writers.

OWNER: this module owns the chain.tsv row format, step capture, and the chain_report.html / SUMMARY.txt layout.
Pattern reused from a chain-report HTML template (same CSS tokens, stat tiles, a rail of
step nodes with PASS/WARN/FAIL chips, ranked findings, "ideas to chain next") and from the 2026-11-17 campaign's
report script (chain.tsv = name, start epoch, seconds, rc).  CONTRACT: ``Recorder(outdir).step(name, fn)``
runs fn() -> (value, note), never lets an exception escape: it records FAIL with the traceback and returns a StepResult.
FAILURE: fail-open-loud: a failed step is a FAIL row plus a stderr line, and the run continues; the exit code of the CLI
carries the verdict.  Plain HTML, no scripts, no network.
"""

from __future__ import annotations

import html
import os
import sys
import time
import traceback
from dataclasses import dataclass

CSS = """:root{--bg:#F6F8F5;--panel:#fff;--ink:#17211B;--dim:#5E6D60;--line:#DBE3DC;--go:#1E8A4F;--warn:#B8860B;--bad:#C03D3D}
@media (prefers-color-scheme:dark){:root{--bg:#0E1411;--panel:#151C17;--ink:#E4EBE5;--dim:#8FA093;--line:#26312A;--go:#3FC37A;--warn:#D9A62E;--bad:#E06868}}
body{background:var(--bg);color:var(--ink);margin:0;font:15px/1.55 system-ui,sans-serif}.wrap{max-width:980px;margin:0 auto;padding:28px 20px 60px}
.eyebrow{font:11px monospace;letter-spacing:.18em;text-transform:uppercase;color:var(--dim)}h1{font-size:26px;margin:6px 0 4px}h2{font-size:17px;margin:32px 0 10px}
.chain{margin-left:10px;padding-left:26px;border-left:2px solid var(--line)}.node{position:relative;margin:0 0 14px}
.node:before{content:"";position:absolute;left:-33px;top:6px;width:12px;height:12px;border-radius:50%;background:var(--go)}
.node.warn:before{background:var(--warn)}.node.fail:before{background:var(--bad)}.chip{font:11px monospace;padding:2px 8px;border-radius:10px;border:1px solid var(--line);margin-left:6px}
.chip.pass{color:var(--go);border-color:var(--go)}.chip.warn{color:var(--warn);border-color:var(--warn)}.chip.fail{color:var(--bad);border-color:var(--bad)}
.note{color:var(--dim);font-size:13px;white-space:pre-wrap;margin:2px 0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px}
.stat{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px 14px}.stat .n{font:600 24px monospace}.stat .l{color:var(--dim);font-size:12px}
table{border-collapse:collapse;font-size:13px}td,th{border:1px solid var(--line);padding:3px 8px;text-align:left}pre{background:var(--panel);padding:8px;overflow:auto;font-size:12px}"""


@dataclass
class StepResult:
    ok: bool
    value: object = None
    err: str = ""


class Recorder:
    def __init__(self, outdir: str):
        self.outdir = outdir
        os.makedirs(os.path.join(outdir, "steps"), exist_ok=True)
        self.rows: list[dict] = []
        self.tsv = os.path.join(outdir, "chain.tsv")
        with open(self.tsv, "w", encoding="utf-8") as fh:
            fh.write("name\tstart_epoch\tseconds\trc\tstatus\tnote\n")

    def _add(self, name: str, t0: float, rc: int, status: str, note: str) -> None:
        secs = time.time() - t0
        n = len(self.rows) + 1
        flat = note.replace("\t", " ").replace("\n", " | ")[:300]
        self.rows.append({"name": name, "t0": t0, "secs": secs, "rc": rc, "status": status, "note": note})
        with open(os.path.join(self.outdir, "steps", f"{n:04d}-{name.replace('/', '_')}.txt"), "w", encoding="utf-8") as fh:
            fh.write(note + "\n")
        with open(self.tsv, "a", encoding="utf-8") as fh:
            fh.write(f"{name}\t{t0:.3f}\t{secs:.2f}\t{rc}\t{status}\t{flat}\n")

    def step(self, name: str, fn) -> StepResult:
        """Run fn() -> (value, note).  An exception is recorded as FAIL (traceback in the step file and on stderr)."""
        t0 = time.time()
        try:
            value, note = fn()
        except Exception as exc:  # the step boundary: record everything, never swallow
            tb = traceback.format_exc()
            print(f"FAIL {name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            self._add(name, t0, 1, "FAIL", tb)
            return StepResult(False, None, f"{type(exc).__name__}: {exc}")
        self._add(name, t0, 0, "PASS", str(note))
        return StepResult(True, value)

    def import_rows(self, tsv: str, source: str) -> None:
        """Re-record the step rows of an earlier invocation (criterion rows excluded: they are decided again), names prefixed by the source dir."""
        with open(tsv, encoding="utf-8") as fh:
            lines = fh.read().splitlines()[1:]
        for line in lines:
            name, t0, secs, rc, status, note = (line.split("\t") + [""] * 6)[:6]
            if name.startswith("criterion-"):
                continue
            self.rows.append({"name": f"{source}/{name}", "t0": float(t0), "secs": float(secs), "rc": int(rc), "status": status, "note": note})
            with open(self.tsv, "a", encoding="utf-8") as out:
                out.write(f"{source}/{name}\t{t0}\t{secs}\t{rc}\t{status}\t{note}\n")

    def mark(self, name: str, status: str, note: str) -> None:
        """A step that is a verdict, not an action (criteria rows): status PASS / WARN / FAIL."""
        self._add(name, time.time(), 1 if status == "FAIL" else 0, status, note)


def write_summary(outdir: str, header: str, verdicts: list[dict], tables: str) -> str:
    lines = [header, "", "CRITERIA (section 5.4):"]
    lines += [f"  {v['id']:<3} {v['status']:<5} {v['title']}: {v['detail']}" for v in verdicts]
    path = os.path.join(outdir, "SUMMARY.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n\n" + tables + "\n")
    return path


def _node(r: dict, n: int) -> str:
    cls = r["status"].lower()
    return (f'<div class="node {cls}"><b>{n} &middot; {html.escape(r["name"])}</b>'
            f'<span class="chip {cls}">{r["secs"]:.1f}s</span><span class="chip {cls}">{r["status"]}</span>'
            f'<div class="note">{html.escape(r["note"][:600])}</div></div>')


def findings(rows: list[dict], verdicts: list[dict]) -> list[str]:
    """Ranked: failed criteria, failed steps, then warnings (no data / provisional)."""
    out = [f"{v['id']} FAIL: {v['title']} - {v['detail']}" for v in verdicts if v["status"] == "FAIL"]
    out += [f"step FAIL: {r['name']}" for r in rows if r["status"] == "FAIL"]
    out += [f"{v['id']} WARN: {v['title']} - {v['detail']}" for v in verdicts if v["status"] == "WARN"]
    return out or ["no failures, no warnings"]


IDEAS = [
    "Re-run this chain after every S5/S7 tree change: the runner is the regression record, even when all green.",
    "Move the fence budget from provisional to a measured value once n >= 3 exclusive windows exist.",
    "Add an arm sweeping --retier-moves (2, 4, 8) once C1 holds, to see where the swap cost starts to show.",
    "Chain the W5 fault run against the named transport once a lead-card fault arm exists.",
]


def _count(items: list[dict], key: str = "status") -> dict[str, int]:
    return {k: sum(1 for x in items if x[key] == k) for k in ("PASS", "WARN", "FAIL")}


def write_html(outdir: str, title: str, verdicts: list[dict], rows: list[dict], summary_text: str) -> str:
    st, cv = _count(rows), _count(verdicts)
    tiles = "".join(f'<div class="stat"><div class="n">{a}</div><div class="l">{b}</div></div>' for a, b in (
        (len(rows), "recorded steps"), (f"{st['FAIL']} / {st['WARN']}", "failed / warned steps"),
        (f"{cv['PASS']}/{len(verdicts)}", "criteria PASS"), (f"{cv['FAIL']} / {cv['WARN']}", "criteria FAIL / WARN"),
        (f"{sum(r['secs'] for r in rows) / 60:.1f} min", "step wall time")))
    page = [f"<!doctype html><meta charset=utf-8><title>{html.escape(title)}</title><style>{CSS}</style><div class=wrap>",
            f"<div class=eyebrow>Seabraid placement qualification &middot; recorded verification chain &middot; {time.strftime('%Y-%m-%d')}</div>",
            f"<h1>{html.escape(title)}</h1><div class=grid>{tiles}</div><h2>Criteria</h2><table><tr><th>#</th><th>status</th><th>criterion</th><th>detail</th></tr>"]
    page += [f'<tr><td>{v["id"]}</td><td><span class="chip {v["status"].lower()}">{v["status"]}</span></td><td>{html.escape(v["title"])}</td><td>{html.escape(v["detail"])}</td></tr>' for v in verdicts]
    page += ["</table><h2>Findings, ranked</h2><ol>"] + [f"<li>{html.escape(f)}</li>" for f in findings(rows, verdicts)] + ["</ol>"]
    page += ["<h2>The chain</h2><div class=chain>"] + [_node(r, i) for i, r in enumerate(rows, 1)] + ["</div>"]
    page += ["<h2>Ideas to chain next</h2><ol>"] + [f"<li>{html.escape(i)}</li>" for i in IDEAS] + ["</ol>"]
    page += [f"<h2>SUMMARY.txt</h2><pre>{html.escape(summary_text)}</pre></div>"]
    path = os.path.join(outdir, "chain_report.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(page))
    return path
