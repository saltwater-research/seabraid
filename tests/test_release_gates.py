# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Preflight gates for the release files: the changelog's sections, the citation date, and a tree free of fill-in markers.

Each gate has a unit case that fires and one that stays quiet, plus a case on the real tree where the gate is a property
of the shipped files (changelog sections, .zenodo.json).  The citation date is not asserted on the real tree: it must be
today's date only on the day of the release, so the preflight stage is where it is enforced.
"""

from __future__ import annotations

import datetime
import json
import os
import sys

from conftest import REPO, run_tool

sys.path.insert(0, os.path.join(REPO, "scripts", "release"))
import rel_build  # noqa: E402
import rel_preflight  # noqa: E402

VERSION = "0.5.0-alpha.2"
GOOD = f"# C\n\n## {VERSION} - unreleased\n\nnotes\n\n## 0.5.0-alpha.1 - 2026-11-15\n\nold\n"
STRAY = f"# C\n\n## Unreleased\n\nlost\n\n## {VERSION} - unreleased\n\nnotes\n"


def test_unreleased_section_above_the_release_section_is_a_fail():
    problems = rel_preflight.changelog_problems(STRAY, VERSION)
    assert problems and "Unreleased" in problems[0]


def test_one_release_section_is_clean():
    assert rel_preflight.changelog_problems(GOOD, VERSION) == []
    assert rel_preflight.changelog_problems("# C\n", VERSION) == []  # a missing section is the separate WARN, not this FAIL


def test_the_real_changelog_has_no_unreleased_section_above_the_release():
    text = open(os.path.join(REPO, "CHANGELOG.md"), encoding="utf-8").read()
    version = open(os.path.join(REPO, "VERSION"), encoding="utf-8").read().strip()
    assert rel_preflight.changelog_problems(text, version) == []
    title, body = rel_build.release_notes(text, version)
    assert title.startswith(version) and "--file-moves" in body, "the Unreleased entries must be in the notes that get published"


def cff(tmp_path, line: str):
    (tmp_path / "CITATION.cff").write_text(f"cff-version: 1.2.0\nversion: x\n{line}\nlicense: Apache-2.0\n", encoding="utf-8")
    return tmp_path


def test_a_stale_citation_date_is_refused_and_today_passes(tmp_path):
    today = datetime.date(2026, 11, 20)
    assert rel_preflight.citation_problems(cff(tmp_path, "date-released: 2026-11-15  # alpha.1"), today)
    assert rel_preflight.citation_problems(cff(tmp_path, "date-released: '2026-11-21'"), today), "a future date is also not the tag date"
    assert rel_preflight.citation_problems(cff(tmp_path, "date-released: 2026-11-20  # tag date"), today) == []
    assert rel_preflight.citation_problems(cff(tmp_path, "title: no date here"), today)


def test_no_unfilled_marker_is_tracked():
    code, out = run_tool(REPO, "checks/check_no_placeholders.py")
    assert code == 0, out


def test_zenodo_metadata_names_no_paper_doi():
    with open(os.path.join(REPO, ".zenodo.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    assert "related_identifiers" not in meta, "no paper DOI exists yet; an entry would be published as a real citation"
