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

import pytest

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


# ---- the release identity: the runner's own git identity, never a fixed one ---------------------------------------------

def _repo_with(tmp_path, name=None, email=None):
    import subprocess
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    for key, value in (("user.name", name), ("user.email", email)):
        if value:
            subprocess.run(["git", "config", key, value], cwd=tmp_path, check=True)
    return tmp_path


@pytest.fixture()
def isolated_git(monkeypatch, tmp_path):
    """WHY: the machine's global and system git config must not supply an identity the test did not set."""
    empty = tmp_path / "empty-gitconfig"
    empty.write_text("", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(empty))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")


@pytest.mark.parametrize("name,email", [(None, None), ("t", None), (None, "1+t@users.noreply.github.com")])
def test_identity_refuses_when_name_or_email_is_unset(isolated_git, tmp_path, name, email):
    from rel_core import Refuse, release_identity
    with pytest.raises(Refuse, match="user.name and user.email are not both set"):
        release_identity(_repo_with(tmp_path / "r", name, email))


def test_identity_refuses_a_personal_address_with_the_fix(isolated_git, tmp_path):
    from rel_core import Refuse, release_identity
    personal = "someone" + "@" + "gmail" + ".com"  # WHY assembled: the tree scan must not see a literal address
    with pytest.raises(Refuse) as err:
        release_identity(_repo_with(tmp_path / "r", "t", personal))
    assert "git config user.email <id>+<login>@users.noreply.github.com" in str(err.value) and "github.com/settings/emails" in str(err.value)


def test_identity_accepts_noreply_and_the_project_contact(isolated_git, tmp_path):
    import rel_secrets
    from rel_core import release_identity
    assert release_identity(_repo_with(tmp_path / "a", "t", "1+t@users.noreply.github.com")) == ("t", "1+t@users.noreply.github.com")
    assert release_identity(_repo_with(tmp_path / "b", "t", rel_secrets.PUBLIC_CONTACT)) == ("t", rel_secrets.PUBLIC_CONTACT)


def test_author_env_gives_author_and_committer_the_same_runner_identity(isolated_git, tmp_path):
    from rel_core import author_env
    env = author_env(_repo_with(tmp_path / "r", "t", "1+t@users.noreply.github.com"))
    assert (env["GIT_AUTHOR_NAME"], env["GIT_AUTHOR_EMAIL"]) == (env["GIT_COMMITTER_NAME"], env["GIT_COMMITTER_EMAIL"]) == ("t", "1+t@users.noreply.github.com")
