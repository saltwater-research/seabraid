#!/usr/bin/env python3
# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Generate the live-run L2 result figure l2-walk.svg: one expert (layer 39, expert 16) through all five levels.

OWNER: this file owns how the figure looks; docs/img/data/l2-walk.csv owns the eight hops (live run L2, 2026-11-20, taken from
strata.log of run S1-walk).  Style primitives (svg, t, line, dot, arrow, rows) are imported from gen_l1_figures.py, not copied.
CONTRACT: ``python docs/img/gen_l2_figure.py [outdir]`` writes l2-walk.svg (default: next to this script); pure stdlib.
x = placement round, y = level (main on top, SSD at the bottom).  Thin black line = where the expert sits; solid blue arrow =
card-to-card hop (verified=1); dashed grey arrow = hop that touches RAM or the SSD (fallback log).
No title or headline inside the image: the caption carries the sentence.  Reruns are byte-identical.
SIZE: under 100 lines.  FAILURE: an exception (missing CSV, unknown level); nothing is written half-way.
CHECKER: tests/test_l2_figure.py (regeneration diff, self-contained, no headline, TikZ twin balanced, CSV matches the claim).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_l1_figures as g  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else g.HERE
LEVELS = ["main", "helper1", "helper2", "RAM", "SSD"]
LABEL = {"main": "main", "helper1": "helper 1", "helper2": "helper 2", "RAM": "RAM", "SSD": "SSD (NVMe)"}
X0, SC, TOP, PITCH, XMAX = 130, 2.1, 50, 60, 300  # plot left edge, px per round, first level y, level pitch, last round drawn


def px(rnd):
    return X0 + int(rnd) * SC


def py(level):
    return TOP + LEVELS.index(level) * PITCH


def black(x1, y, x2):
    return f'<line x1="{x1:g}" y1="{y:g}" x2="{x2:g}" y2="{y:g}" style="stroke:#000;stroke-width:1"/>\n'


def arrow(x, y1, y2, card):
    """Vertical arrow with an explicit head.  Style follows the KIND of hop (solid blue card, dashed grey fallback), not its direction."""
    d, cls = (1 if y2 > y1 else -1), ("up" if card else "dn")
    return (g.line(x, y1, x, y2 - 8 * d, f"ln {cls}" + ("" if card else " dash"))
            + f'<polygon points="{x - 4:g},{y2 - 9 * d:g} {x + 4:g},{y2 - 9 * d:g} {x:g},{y2:g}" class="{cls}"/>\n')


def dwell(hops):
    """The thin line: the expert sits at its level from one hop to the next (starting on the SSD, ending where the last hop ends)."""
    b, at, x_prev = "", hops[0]["from"], X0
    for h in hops:
        b += black(x_prev, py(at), px(h["round"]))
        x_prev, at = px(h["round"]), h["to"]
    return b + black(x_prev, py(at), px(XMAX))


def hop(h):
    card = h["kind"] == "card"
    return arrow(px(h["round"]), py(h["from"]), py(h["to"]), card) + g.dot(px(h["round"]), py(h["to"]), 3.5, "up" if card else "dn")


def round_label(r, hops):
    """The round number beside the vertical of its hops (left of it for 274, which sits next to 278)."""
    grp = [h for h in hops if h["round"] == r]
    ym = sum(py(h["from"]) + py(h["to"]) for h in grp) / (2 * len(grp))
    left = r == "274"
    return g.t(px(r) + (-10 if left else 10), ym + 4, r, "s e" if left else "s")


def axes():
    b, yb = "", py("SSD")
    for i, lv in enumerate(LEVELS):
        b += g.line(X0, py(lv), px(XMAX), py(lv)) + g.t(X0 - 12, py(lv) + 4, LABEL[lv], "e")
    for v in range(0, XMAX + 1, 50):
        b += g.line(px(v), yb + 14, px(v), yb + 20) + g.t(px(v), yb + 36, str(v), "s c")
    return b + g.t(px(XMAX / 2), yb + 58, "placement round", "s c")


def key(y):
    b = arrow(X0 + 20, y + 6, y - 6, True) + g.t(X0 + 34, y + 4, "card to card, verified=1", "s")
    return b + arrow(X0 + 270, y - 6, y + 6, False) + g.t(X0 + 284, y + 4, "touches RAM or SSD, fallback log", "s")


def walk():
    hops = g.rows("l2-walk.csv")
    b = axes() + dwell(hops) + "".join(hop(h) for h in hops)
    b += "".join(round_label(r, hops) for r in dict.fromkeys(h["round"] for h in hops))
    b += g.t(440, 62, "Traced expert: layer 39, expert 16") + g.t(440, 80, "1,971,200 bytes per hop.", "s")
    b += g.t(440, 96, "SSD read after the page-cache drop: 2,761 us.", "s")
    ly = py("SSD") + 84
    return g.svg(780, ly + 16, "One expert through five levels",
                 "Expert 16 of layer 39 goes SSD to RAM in round 28, to helper 2, helper 1 and main in round 79, back to helper 1 in round 118, "
                 "to helper 2 in round 176, to RAM in round 274 and to the SSD in round 278.", b + key(ly))


if __name__ == "__main__":
    text = walk()  # built first, written after: no half-written file
    with open(os.path.join(OUT, "l2-walk.svg"), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
