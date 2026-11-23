# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""The live-run L2 figure is generated: regenerating must reproduce the committed SVG and TikZ files byte for byte.

OWNER: docs/img/gen_l2_figure.py (SVG) and docs/paper/figures/gen_live_l2.py (TikZ) own the pictures; docs/img/data/l2-walk.csv
owns the eight hops.  CHECKER role: a hand-edited figure, or a CSV edited without regenerating, fails here; the headline and
self-containment rules of tests/test_l1_figures.py are applied to the new SVG by importing its checkers, not copying them.
"""

from __future__ import annotations

import csv
import os
import re
import subprocess
import sys

import pytest
from conftest import REPO
from test_l1_figures import headlines

SVG, TEX = "docs/img/l2-walk.svg", "docs/paper/figures/fig-l2-walk.tex"
GENS = [("docs/img/gen_l2_figure.py", SVG), ("docs/paper/figures/gen_live_l2.py", TEX)]


@pytest.mark.parametrize("script,committed", GENS)
def test_regeneration_is_byte_identical(tmp_path, script, committed):
    subprocess.run([sys.executable, os.path.join(REPO, script), str(tmp_path)], check=True, capture_output=True)
    with open(os.path.join(REPO, committed), "rb") as a, open(tmp_path / os.path.basename(committed), "rb") as b:
        assert a.read() == b.read(), f"{committed} differs from its generator output: rerun {script}"


def test_svg_is_self_contained_and_has_no_headline():
    text = open(os.path.join(REPO, SVG), encoding="utf-8").read()
    ns = re.search(r'xmlns="([^"]+)"', text).group(1)  # the one namespace name an SVG must carry; it is never fetched
    urls = [u for u in re.findall(r"https?://[^\s\"')]+", text) if u != ns]
    assert not urls and "@import" not in text and "<image" not in text, urls
    assert text.count("<svg") == 1 and text.rstrip().endswith("</svg>")
    assert headlines(text) == [], "the image draws a headline: move the sentence to the caption"


def test_tikz_is_balanced_and_has_no_bold_title():
    text = open(os.path.join(REPO, TEX), encoding="utf-8").read()
    body = re.sub(r"\\[{}]", "", text)
    assert body.count("{") == body.count("}")
    assert text.count("\\begin{tikzpicture}") == text.count("\\end{tikzpicture}") == 1
    assert "\\bfseries" not in text


def test_csv_is_one_connected_walk_of_the_claimed_run():
    with open(os.path.join(REPO, "docs/img/data/l2-walk.csv"), newline="", encoding="utf-8") as f:
        hops = list(csv.DictReader(f))
    assert len(hops) == 8 and hops[0]["from"] == hops[-1]["to"] == "SSD"
    assert all(a["to"] == b["from"] for a, b in zip(hops, hops[1:])), "each hop must start where the last one ended"
    assert [int(h["round"]) for h in hops] == sorted(int(h["round"]) for h in hops)
    assert {h["verified"] for h in hops if h["kind"] == "card"} == {"1"}
    assert all({h["from"], h["to"]} & {"RAM", "SSD"} for h in hops if h["kind"] == "fallback")
