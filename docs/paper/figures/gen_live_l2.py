#!/usr/bin/env python3
# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Generate the live-run L2 TikZ figure fig-l2-walk.tex (the twin of docs/img/l2-walk.svg; plain primitives, as gen_live_figures.py).

OWNER: this file owns the drawing of the paper figure; the eight hops live in docs/img/data/l2-walk.csv (one home, shared with the
SVG generator docs/img/gen_l2_figure.py).  text/HEAD come from gen_figures.py and arrow/rows from gen_live_figures.py.
CONTRACT: ``python gen_live_l2.py [outdir]`` writes fig-l2-walk.tex next to this script by default; pure text out.
x = placement round, y = level (main on top).  Solid blue = card-to-card hop, dashed red = hop that touches RAM or the SSD.
SIZE: under 100 lines.  FAILURE: an exception (missing CSV, unknown level); nothing is written half-way.
CHECKER: tests/test_l2_figure.py regenerates and diffs, and checks braces and tikzpicture balance (no TeX engine on the build box).
No title or bold headline in the picture: the caption in seabraid.tex carries the sentence.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gen_figures import HEAD, text  # noqa: E402
from gen_live_figures import arrow, rows  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else HERE
LEVELS = ["main", "helper1", "helper2", "RAM", "SSD"]
LABEL = {"main": "main", "helper1": "helper 1", "helper2": "helper 2", "RAM": "RAM", "SSD": "SSD (NVMe)"}
SC, W, PITCH = 0.0225, 300, 1.0  # cm per round, last round drawn, cm per level


def px(r):
    return float(r) * SC


def py(level):
    return -LEVELS.index(level) * PITCH


def thin(x1, x2, y, color="black"):
    return f"\\draw[line width=0.4pt, draw={color}] ({x1:.3f},{y:.3f}) -- ({x2:.3f},{y:.3f});\n"


def axes():
    o = ""
    for lv in LEVELS:
        o += thin(0, px(W), py(lv), "black!25") + text(-0.15, py(lv), LABEL[lv], anchor="east", size="tiny")
    for v in range(0, W + 1, 50):
        o += text(px(v), py("SSD") - 0.35, str(v), anchor="center", size="tiny", color="dim")
    return o + text(px(W / 2), py("SSD") - 0.75, "placement round", anchor="center", size="tiny", color="dim")


def dwell(hops):
    """The thin line: the expert sits at its level from one hop to the next (starting on the SSD)."""
    o, at, x_prev = "", hops[0]["from"], 0.0
    for h in hops:
        o += thin(x_prev, px(h["round"]), py(at))
        x_prev, at = px(h["round"]), h["to"]
    return o + thin(x_prev, px(W), py(at))


def hop(h):
    card = h["kind"] == "card"
    return arrow(px(h["round"]), py(h["from"]), py(h["to"]), "steelc" if card else "portc", not card)


def labels(hops):
    """The round number beside the vertical of its hops (left of it for 274, which sits next to 278)."""
    o = ""
    for r in dict.fromkeys(h["round"] for h in hops):
        grp = [h for h in hops if h["round"] == r]
        ym, left = sum(py(h["from"]) + py(h["to"]) for h in grp) / (2 * len(grp)), r == "274"
        o += text(px(r) + (-0.12 if left else 0.12), ym, r, size="tiny", anchor="east" if left else "west", color="dim")
    return o


def notes():
    return (text(px(185), 0.1, "layer 39, expert 16", size="tiny") + text(px(185), -0.25, "1,971,200 bytes per hop", size="tiny", color="dim")
            + text(px(185), -0.6, "solid blue: card to card, verified", size="tiny", color="steel")
            + text(px(185), -0.95, "dashed red: touches RAM or SSD", size="tiny", color="port"))


def fig_walk():
    hops = rows("l2-walk.csv")
    return "\\begin{tikzpicture}\n" + axes() + dwell(hops) + "".join(hop(h) for h in hops) + labels(hops) + notes() + "\\end{tikzpicture}\n"


if __name__ == "__main__":
    body = fig_walk()  # built first, written after: no half-written file
    with open(os.path.join(OUT, "fig-l2-walk.tex"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(HEAD.replace("gen_figures.py", "gen_live_l2.py") + body)
