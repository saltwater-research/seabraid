# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""The live-run L1 figures are generated: regenerating must reproduce the committed files byte for byte.

OWNER: docs/img/gen_l1_figures.py (SVG) and docs/paper/figures/gen_live_figures.py (TikZ) own the pictures; docs/img/data/l1-*.csv
own the numbers.  CHECKER role: a hand-edited figure, or a CSV edited without regenerating, fails here.  The generators are pure
stdlib, so nothing is skipped; the TikZ files also get a balance check because no TeX engine exists on the build box.
Headline rule: no picture carries a title; any <text> that is bold or 16px and up fails (the sentence belongs in the caption),
and the TikZ twins carry no bold node.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest
from conftest import REPO

SVG = ["pyramid-l1.svg", "l1-moves.svg", "l1-round-cost.svg", "l1-kl.svg"]
TEX = ["fig-pyramid-live.tex", "fig-l1-moves.tex"]


def run(script: str, out) -> None:
    subprocess.run([sys.executable, os.path.join(REPO, script), str(out)], check=True, capture_output=True)


@pytest.mark.parametrize("script,names,home", [("docs/img/gen_l1_figures.py", SVG, "docs/img"),
                                               ("docs/paper/figures/gen_live_figures.py", TEX, "docs/paper/figures")])
def test_regeneration_is_byte_identical(tmp_path, script, names, home):
    run(script, tmp_path)
    for n in names:
        with open(os.path.join(REPO, home, n), "rb") as a, open(tmp_path / n, "rb") as b:
            assert a.read() == b.read(), f"{home}/{n} differs from its generator output: rerun {script}"


def test_svgs_are_self_contained():
    for n in SVG:
        text = open(os.path.join(REPO, "docs/img", n), encoding="utf-8").read()
        urls = [u for u in re.findall(r"https?://[^\s\"')]+", text) if u != "http://www.w3.org/2000/svg"]
        assert not urls and "@import" not in text and "<image" not in text, f"{n} references something external: {urls}"
        assert text.count("<svg") == 1 and text.rstrip().endswith("</svg>")


def test_tikz_is_balanced():
    for n in TEX:
        text = open(os.path.join(REPO, "docs/paper/figures", n), encoding="utf-8").read()
        body = re.sub(r"\\[{}]", "", text)
        assert body.count("{") == body.count("}"), n
        assert text.count("\\begin{tikzpicture}") == text.count("\\end{tikzpicture}") == 1, n


def _decls(body: str) -> dict:
    return dict(p.split(":", 1) for p in body.split(";") if ":" in p)


def _style_rules(root) -> dict:
    """Selector -> declarations from every <style> element (enough CSS for these generated figures)."""
    css = " ".join(e.text or "" for e in root.iter() if e.tag.endswith("style"))
    rules = {}
    for sel, body in re.findall(r"([^{}]+)\{([^}]*)\}", css):
        for one in sel.split(","):
            rules.setdefault(one.strip(), {}).update(_decls(body))
    return rules


def _is_headline(props: dict) -> bool:
    size = float(props.get("font-size", "16px").strip().removesuffix("px"))
    weight = props.get("font-weight", "400").strip()
    return size >= 16 or weight == "bold" or (weight.isdigit() and int(weight) >= 600)


def headlines(svg_text: str) -> list[str]:
    """Texts of an SVG that look like a headline: font-size >= 16px (the browser default when unset) or bold (weight >= 600)."""
    root = ET.fromstring(svg_text)
    rules = _style_rules(root)
    bad = []
    for e in (x for x in root.iter() if x.tag.endswith("text")):
        props = dict(rules.get("text", {}))
        for c in (e.get("class") or "").split():
            props.update(rules.get("." + c, {}))
        props.update(_decls(e.get("style") or ""))
        if _is_headline(props):
            bad.append("".join(e.itertext()))
    return bad


def test_checker_catches_a_headline():
    old = ('<svg xmlns="http://www.w3.org/2000/svg"><style>.h{font-size:15px;font-weight:600}.m{font-size:13px}</style>'
           '<text class="h">Moves up and down: all verified</text><text class="m">520</text></svg>')
    assert headlines(old) == ["Moves up and down: all verified"]
    assert headlines('<svg xmlns="http://www.w3.org/2000/svg"><text style="font-size:18px">x</text></svg>') == ["x"]


def test_svgs_have_no_headline():
    for n in SVG:
        text = open(os.path.join(REPO, "docs/img", n), encoding="utf-8").read()
        assert headlines(text) == [], f"{n} draws a headline inside the image: move the sentence to the caption"


def test_tikz_has_no_bold_title():
    for n in TEX:
        assert "\\bfseries" not in open(os.path.join(REPO, "docs/paper/figures", n), encoding="utf-8").read(), n
