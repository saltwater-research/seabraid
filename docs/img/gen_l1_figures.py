#!/usr/bin/env python3
# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Generate the live-run L1 result figures (pyramid-l1, l1-moves, l1-round-cost, l1-kl) as flat SVG.

OWNER: this file owns how the four SVGs look; docs/img/data/l1-*.csv own the numbers (small derived values of live run L1,
2026-11-19; the full record is docs/RESEARCH-LOG.md).  The TikZ twins for the paper live in docs/paper/figures/gen_live_figures.py
and read the same CSVs, so one number has one home.
CONTRACT: ``python docs/img/gen_l1_figures.py [outdir]`` writes the four files (default: next to this script); pure stdlib,
hand-written SVG, no fonts or URLs fetched.  Classic figure style: serif text, thin black strokes, white background, two
muted fills only (grey #888, dark blue #1f3a6e), no title or headline inside the image: the sentence that says what is plotted
is the caption in the document that embeds the figure.  Reruns are byte-identical.
SIZE: under 300 lines.  FAILURE: an exception (missing CSV, bad number); nothing is written half-way.
CHECKER: tests/test_l1_figures.py regenerates into a temp dir and diffs against the committed files, and fails any SVG whose
<text> is bold or 16px and up (a headline).
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else HERE
CSS = """<style>
text{font-family:'Times New Roman',Times,Georgia,serif;fill:#000;font-size:13px}
.s{font-size:12px}.e{text-anchor:end}.c{text-anchor:middle}
.rule{stroke:#bbb;stroke-width:.75;fill:none}.box{fill:#fff;stroke:#000;stroke-width:1}
.up{fill:#1f3a6e;stroke:#1f3a6e}.dn{fill:#888;stroke:#888}.ref{fill:#fff;stroke:#000;stroke-width:1}
.gate{stroke:#000;stroke-width:1;stroke-dasharray:5 4;fill:none}.ln{stroke-width:1.5;fill:none}.dash{stroke-dasharray:5 3}
</style>"""


def rows(name):
    with open(os.path.join(HERE, "data", name), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def t(x, y, s, cls="", extra=""):
    return f'<text x="{x:g}" y="{y:g}"' + (f' class="{cls}"' if cls else "") + f"{extra}>{esc(s)}</text>\n"


def rect(x, y, w, h, cls):
    return f'<rect x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" class="{cls}"/>\n'


def line(x1, y1, x2, y2, cls="rule"):
    return f'<line x1="{x1:g}" y1="{y1:g}" x2="{x2:g}" y2="{y2:g}" class="{cls}"/>\n'


def dot(x, y, r, cls):
    return f'<circle cx="{x:g}" cy="{y:g}" r="{r}" class="{cls}"/>\n'


def svg(w, h, title, desc, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" '
            f'aria-labelledby="t d">\n<title id="t">{esc(title)}</title>\n<desc id="d">{esc(desc)}</desc>\n{CSS}\n'
            f'<rect width="{w}" height="{h}" fill="#fff"/>\n{body}</svg>\n')


def arrow(x, y_from, y_to, cls):
    """A vertical arrow with an explicit head (no <marker>: renders the same everywhere); down arrows are dashed."""
    d = 1 if y_to > y_from else -1
    return (line(x, y_from, x, y_to - 8 * d, f"ln {cls}" + (" dash" if d > 0 else "")) +
            f'<polygon points="{x - 4:g},{y_to - 9 * d:g} {x + 4:g},{y_to - 9 * d:g} {x:g},{y_to:g}" class="{cls}"/>\n')


CX, Y0, BH, STEP = 290, 24, 56, 118  # ladder centre x, first rung top, rung height, rung pitch
RUNGS = [("main", "R9700 (HIP)", "2,582 slots", 240), ("helper1", "RTX 3090 (CUDA)", "1,500 slots", 340),
         ("helper2", "HIP process, same R9700", "1,000 slots", 440), ("F", "engine RAM + NVMe", "", 540)]


def rung_f(x, y, w):
    """Rung F is two tiers side by side: RAM (the small part) and the NVMe pack file (66% of the experts)."""
    rw = w * 0.34
    cn = x + rw + (w - rw) / 2
    return (line(x + rw, y, x + rw, y + BH, "ln") + t(x + rw / 2, y + 24, "RAM", "c") + t(x + rw / 2, y + 42, "6 GiB budget", "s c")
            + t(cn, y + 24, "NVMe (SSD pack file)", "c") + t(cn, y + 42, "66% of the 24,576 experts", "s c"))


def hop_arrow(h, ys, names):
    """One traced hop, drawn in the gap between the two rungs it joins: solid blue up, dashed grey down, labelled with its round."""
    up = h["direction"] == "up"
    lower = names.index(h["from"] if up else h["to"])
    top, bot = ys[names[lower - 1]] + BH, ys[names[lower]]
    x = (CX - 60 if up else CX + 100) if names[lower] == "F" else (CX - 150 if up else CX + 150)
    return (arrow(x, bot if up else top, top if up else bot, "up" if up else "dn")
            + t(x + (-9 if up else 9), (top + bot) / 2 + 4, f"round {h['round']}", "s e" if up else "s"))


def pyramid():
    b = ""
    names, ys = [r[0] for r in RUNGS], {}
    for i, (name, hw, slots, w) in enumerate(RUNGS):
        y = ys[name] = Y0 + i * STEP
        b += rect(CX - w / 2, y, w, BH, "box")
        b += rung_f(CX - w / 2, y, w) if name == "F" else t(CX, y + 24, f"{name}   {hw}", "c") + t(CX, y + 42, slots, "s c")
    b += "".join(hop_arrow(h, ys, names) for h in rows("l1-trace.csv"))
    kx = 600
    b += t(kx, 36, "Traced expert: layer 25, expert 407") + t(kx, 54, "2,329,600 bytes per hop;", "s") + t(kx, 70, "verified at every hop.", "s")
    b += arrow(kx + 8, 156, 116, "up") + t(kx + 26, 122, "up in round 7") + t(kx + 26, 140, "SSD file > helper2 >", "s") + t(kx + 26, 156, "helper1 > main", "s")
    b += arrow(kx + 8, 188, 228, "dn") + t(kx + 26, 202, "pushed back down") + t(kx + 26, 220, "main > helper1 (round 290),", "s")
    b += t(kx + 26, 236, "helper1 > helper2 (round 333),", "s") + t(kx + 26, 252, "helper2 > rung F (round 388),", "s") + t(kx + 26, 268, "landing SSD-only", "s")
    return svg(820, 440, "Placement pyramid with the traced expert",
               "Four rungs, main, helper1, helper2 and rung F (engine RAM and NVMe). Expert 407 of layer 25 climbs from the SSD pack file to the main card in round 7 and is pushed back down in rounds 290, 333 and 388.", b)


def moves():
    data = rows("l1-moves.csv")
    src = {r["source"]: int(r["moves"]) for r in rows("l1-fallback-source.csv")}
    names = {"main-helper1": "main  <->  helper1", "helper1-helper2": "helper1  <->  helper2", "helper2-F": "helper2  <->  rung F"}
    x0, sc = 190, 0.72
    b = rect(x0 - 16, 10, 10, 10, "up") + t(x0, 20, "moves up (promoted)", "s") + rect(x0 + 174, 10, 10, 10, "dn") + t(x0 + 190, 20, "moves down (displaced)", "s")
    for v in (0, 200, 400, 600):
        b += line(x0 + v * sc, 34, x0 + v * sc, 198) + t(x0 + v * sc, 214, str(v), "s c")
    b += t(x0 + 300 * sc, 232, "moves (count over the run)", "s c")
    for i, (k, nm) in enumerate(names.items()):
        y = 44 + i * 52
        b += t(x0 - 12, y + 24, nm, "e")
        for j, d in enumerate(("up", "down")):
            n = next(int(r["moves"]) for r in data if r["boundary"] == k and r["direction"] == d)
            b += rect(x0, y + j * 24, n * sc, 20, "up" if d == "up" else "dn") + t(x0 + n * sc + 8, y + j * 24 + 15, str(n))
    ram, ssd = src["ram"], src["ssd_pack_file"]
    b += t(20, 264, "Source of the 538 experts promoted from rung F")
    b += rect(20, 274, ram * 0.9 - 1, 24, "dn") + rect(20 + ram * 0.9 + 1, 274, ssd * 0.9 - 1, 24, "ref")
    b += t(20, 318, f"{ram} from the engine's RAM") + t(20 + ram * 0.9 + 1, 318, f"{ssd} read from the SSD pack file")
    return svg(760, 336, "Placement moves per boundary and source of fallback promotions",
               "Bars: 520, 481 and 538 moves up and the same counts down across the three boundaries. Of 538 promotions from rung F, 413 came from RAM and 125 from the SSD pack file.", b)


def roundcost():
    data = rows("l1-round-cost.csv")
    x0, sc, n = 120, 8.6, len(data)
    b = rect(x0, 10, 10, 10, "up") + t(x0 + 16, 20, "mean", "s") + dot(x0 + 90, 15, 5, "ref") + t(x0 + 100, 20, "99th percentile", "s")
    b += t(700, 20, "mean / median / p99 (ms)", "s e")
    for v in (0, 10, 20, 30, 40, 50):
        b += line(x0 + v * sc, 32, x0 + v * sc, 32 + 36 * n) + t(x0 + v * sc, 50 + 36 * n, str(v), "s c")
    b += t(x0 + 25 * sc, 70 + 36 * n, "time per step (ms)", "s c")
    for i, r in enumerate(data):
        y, mean, p99 = 42 + i * 36, float(r["mean_ms"]), float(r["p99_ms"])
        b += t(x0 - 10, y + 15, r["step"], "e") + rect(x0, y, max(mean * sc, 1.5), 20, "up")
        b += line(x0 + mean * sc, y + 10, x0 + p99 * sc - 5, y + 10, "ln up") + dot(x0 + p99 * sc, y + 10, 5, "ref")
        b += t(700, y + 15, f"{mean:.1f} / {float(r['p50_ms']):.1f} / {p99:.1f}", "e")
    return svg(720, 84 + 36 * n, "Placement round step costs",
               "Mean and 99th percentile per step over 513 rounds: fence 18.7 and 52.0 ms, stage 7.9 and 35.1, replace 6.0 and 20.2, plan 2.2 and 7.1, publish 0.1 and 0.5.", b)


def kl_panel(x, y, w, rowsd, key, lo, hi, ticks, gate, side, title):
    sc = w / (hi - lo)
    b = t(x, y - 26, title)
    for v in ticks:
        b += line(x + (v - lo) * sc, y - 8, x + (v - lo) * sc, y + 126) + t(x + (v - lo) * sc, y + 144, f"{v:g}", "s c")
    gx = x + (gate - lo) * sc
    b += line(gx, y - 8, gx, y + 128, "gate") + t(gx, y + 164, f"gate {'≤' if side > 0 else '≥'} {gate:g}", "s c")
    for i, r in enumerate(rowsd):
        cy, cls = y + 14 + i * 44, ("up", "dn", "ref")[i]
        v = float(r[key])
        b += dot(x + (v - lo) * sc, cy, 6, cls) + t(x + (v - lo) * sc, cy - 12, f"{v:.5f}" if key == "kl" else f"{v:.4f}", "c")
    return b


def kl():
    data = rows("l1-kl.csv")
    b = kl_panel(250, 50, 220, data, "kl", 0, 0.035, (0, 0.01, 0.02, 0.03), 0.03, 1, "KL divergence (lower is better)")
    b += kl_panel(610, 50, 240, data, "top1", 0.94, 1.0, (0.94, 0.96, 0.98, 1.0), 0.95, -1, "top-1 agreement (higher is better)")
    for i, r in enumerate(data):
        b += t(20, 50 + 14 + i * 44 + 5, r["run"])
    return svg(880, 232, "Decode-path KL and top-1 agreement against the main card alone",
               "Pyramid runs C1a and C1b score KL 0.00132 and 0.00130 with top-1 agreement 0.9817 and 0.9858; the noise floor is 0.00135 and 0.985. Gates are KL at most 0.03 and top-1 at least 0.95.", b)


def main():
    figs = {"pyramid-l1.svg": pyramid(), "l1-moves.svg": moves(), "l1-round-cost.svg": roundcost(), "l1-kl.svg": kl()}
    for name, text in figs.items():  # built first, written after: no half-written set
        with open(os.path.join(OUT, name), "w", encoding="utf-8", newline="\n") as f:
            f.write(text)


if __name__ == "__main__":
    main()
