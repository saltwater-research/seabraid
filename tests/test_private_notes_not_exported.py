# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""docs/private/ is dev-only: `git archive HEAD` (what scripts/release exports) must leave it out."""

from __future__ import annotations

import os
import subprocess
import tarfile
import io

import pytest

from conftest import REPO


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", REPO, *args], capture_output=True)


def test_the_private_notes_directory_is_export_ignored():
    with open(os.path.join(REPO, ".gitattributes"), encoding="utf-8") as fh:
        assert "docs/private/ export-ignore" in fh.read().splitlines()


def test_git_archive_of_head_contains_no_private_note():
    if _git("rev-parse", "--git-dir").returncode != 0:
        pytest.skip("not a git checkout (a release export has no docs/private to check)")
    tracked = _git("ls-files", "docs/private").stdout.decode().split()
    if not tracked:
        pytest.skip("no private note committed in this checkout")
    archive = _git("archive", "--format=tar", "HEAD")
    assert archive.returncode == 0, archive.stderr.decode()
    names = tarfile.open(fileobj=io.BytesIO(archive.stdout)).getnames()
    assert not [n for n in names if n.startswith("docs/private")], "docs/private leaked into git archive"


def test_the_d4_drain_patch_is_export_ignored_and_absent_from_git_archive():
    patch = "compat/strata/docs/legacy/strata-d4-drain.patch"
    with open(os.path.join(REPO, ".gitattributes"), encoding="utf-8") as fh:
        assert f"{patch} export-ignore" in fh.read().splitlines()
    if _git("rev-parse", "--git-dir").returncode != 0:
        pytest.skip("not a git checkout")
    archive = _git("archive", "--format=tar", "HEAD")
    assert archive.returncode == 0, archive.stderr.decode()
    names = tarfile.open(fileobj=io.BytesIO(archive.stdout)).getnames()
    assert patch not in names, "the D4 drain patch leaked into git archive"
