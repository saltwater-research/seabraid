#!/usr/bin/env python3
# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Generate the live-run L1 TikZ figures fig-pyramid-live.tex and fig-l1-moves.tex (plain primitives, same style as gen_figures.py).

OWNER: this file owns the drawing of the two L1 paper figures; the numbers live in docs/img/data/l1-*.csv (one home, shared with
the SVG generator docs/img/gen_l1_figures.py).  The primitives rect/text/HEAD are imported from gen_figures.py, not copied.
CONTRACT: ``python gen_live_figures.py [outdir]`` writes the two files next to this script by default; pure text out.
SIZE: under 100 lines.  FAILURE: an exception (missing CSV); nothing is written half-way.  CHECKER: tests/test_l1_figures.py
regenerates and diffs, and checks the braces and tikzpicture environments balance (no TeX engine on the build box).
No title or bold headline inside a picture: the caption in seabraid.tex carries the sentence.
Schematic: the captions say so.
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gen_figures import HEAD, rect, text  # noqa: E402

DATA = os.path.join(HERE, "..", "..", "img", "data")
OUT = sys.argv[1] if len(sys.argv) > 1 else HERE


def rows(name):
    with open(os.path.join(DATA, name), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def arrow(x, y1, y2, color, dashed):
    d = "dashed, " if dashed else ""
    return f"\\draw[{d}-{{Stealth}}, line width=0.9pt, draw={color}] ({x:.3f},{y1:.3f}) -- ({x:.3f},{y2:.3f});\n"


def hop_tex(h, names, ys, bh):
    """One traced hop: a solid arrow up, a dashed arrow down, in the gap between the two rungs it joins, labelled with its round."""
    up, a, b = h["direction"] == "up", names.index(h["from"]), names.index(h["to"])
    top, low = names[min(a, b)], names[max(a, b)]  # the upper and the lower rung of this boundary
    xs = {"main": (-1.4, 1.4), "helper1": (-2.4, 2.4), "helper2": (-0.8, 1.8)}[top]
    x = xs[0] if up else xs[1]
    y_low, y_top = ys[low], ys[top] - bh
    o = arrow(x, y_low if up else y_top, y_top if up else y_low, "steelc" if up else "portc", not up)
    return o + text(x + (-0.15 if up else 0.15), (y_low + y_top) / 2, "round " + h["round"], size="tiny", anchor="east" if up else "west", color="dim")


def fig_pyramid():
    names = ["main", "helper1", "helper2", "F"]
    label = {"main": "main: R9700 (HIP)", "helper1": "helper 1: RTX 3090 (CUDA)", "helper2": "helper 2: second HIP process, same R9700", "F": ""}
    sub = {"main": "2,582 slots", "helper1": "1,500 slots", "helper2": "1,000 slots"}
    ys = {n: -i * 2.2 for i, n in enumerate(names)}
    bh, wid = 1.0, {"main": 4.2, "helper1": 5.8, "helper2": 7.4, "F": 9.0}
    o = "\\begin{tikzpicture}\n"
    for n in names:
        o += rect(-wid[n] / 2, ys[n], wid[n], bh, "rule", 0.4, "ink")
        if n != "F":
            o += text(0, ys[n] - 0.32, label[n], anchor="center", size="tiny") + text(0, ys[n] - 0.7, sub[n], anchor="center", size="tiny", color="dim")
    x0, wr = -wid["F"] / 2, wid["F"] * 0.34
    o += f"\\draw[line width=0.4pt] ({x0 + wr:.3f},{ys['F']:.3f}) -- ({x0 + wr:.3f},{ys['F'] - bh:.3f});\n"
    o += text(x0 + wr / 2, ys["F"] - 0.32, "RAM", anchor="center", size="tiny") + text(x0 + wr / 2, ys["F"] - 0.7, "6 GiB budget", anchor="center", size="tiny", color="dim")
    o += text(x0 + wr + (wid["F"] - wr) / 2, ys["F"] - 0.32, "NVMe (SSD pack file)", anchor="center", size="tiny")
    o += text(x0 + wr + (wid["F"] - wr) / 2, ys["F"] - 0.7, "66\\% of 24,576 experts", anchor="center", size="tiny", color="dim")
    o += "".join(hop_tex(h, names, ys, bh) for h in rows("l1-trace.csv"))
    o += text(5.3, ys["main"] - 0.1, "layer 25, expert 407", size="tiny", anchor="west")
    o += text(5.3, ys["main"] - 0.5, "2,329,600 bytes per hop,", size="tiny", color="dim") + text(5.3, ys["main"] - 0.85, "verified at every hop", size="tiny", color="dim")
    o += text(5.3, ys["helper1"] + 0.2, "solid blue: up in round 7", size="tiny", color="steel") + text(5.3, ys["helper1"] - 0.15, "dashed red: pushed down", size="tiny", color="port")
    return o + "\\end{tikzpicture}\n"


def fig_moves():
    data = rows("l1-moves.csv")
    src = {r["source"]: int(r["moves"]) for r in rows("l1-fallback-source.csv")}
    names = {"main-helper1": "main / helper 1", "helper1-helper2": "helper 1 / helper 2", "helper2-F": "helper 2 / rung F"}
    sc, x0 = 0.011, 3.2
    o = "\\begin{tikzpicture}\n" + text(0, 0.35, "moves per boundary: blue up, red down")
    for i, (k, nm) in enumerate(names.items()):
        y = -i * 1.2
        o += text(x0 - 0.15, y - 0.4, nm, anchor="east")
        for j, (d, c) in enumerate((("up", "steel"), ("down", "port"))):
            n = next(int(r["moves"]) for r in data if r["boundary"] == k and r["direction"] == d)
            o += rect(x0, y - j * 0.5, n * sc, 0.4, c, 1.0) + text(x0 + n * sc + 0.12, y - j * 0.5 - 0.2, str(n), color="dim")
    ram, ssd = src["ram"], src["ssd_pack_file"]
    y = -3.9
    o += text(0, y + 0.35, "source of the 538 promotions from rung F") + rect(0, y - 0.1, ram * sc, 0.45, "brass", 1.0) + rect(ram * sc + 0.03, y - 0.1, ssd * sc, 0.45, "brass", 0.4)
    o += text(0, y - 0.7, f"{ram} from RAM", color="dim") + text(ram * sc + 0.03, y - 0.7, f"{ssd} from the SSD pack file", color="dim")
    return o + "\\end{tikzpicture}\n"


if __name__ == "__main__":
    texts = {"fig-pyramid-live.tex": fig_pyramid(), "fig-l1-moves.tex": fig_moves()}
    for name, body in texts.items():
        with open(os.path.join(OUT, name), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(HEAD.replace("gen_figures.py", "gen_live_figures.py") + body)
