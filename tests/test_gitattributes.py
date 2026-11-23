# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Line-ending and ignore policy: every patch is byte-exact so `git am` works from a Windows checkout, text is LF."""

from __future__ import annotations

import os
import subprocess

import pytest
from conftest import REPO

SIGN = ["-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "commit.gpgsign=false"]


# WHY ls-files, not rev-parse: a plain copy of the tree inside some OTHER repo must not count as "a checkout of this one".
IN_GIT = subprocess.run(["git", "-C", REPO, "ls-files", "--error-unmatch", "VERSION"], capture_output=True).returncode == 0
needs_git = pytest.mark.skipif(not IN_GIT, reason="needs a git checkout (not a plain copy of the tree)")


def git(cwd: str, *args: str, check: bool = True) -> str:
    return subprocess.run(["git", "-C", cwd, *args], capture_output=True, text=True, check=check).stdout


def attr(name: str, path: str) -> str:
    return git(REPO, "check-attr", name, "--", path).strip().rsplit(": ", 1)[-1]


@needs_git
def test_attribute_rules():
    assert attr("text", "x/y.patch") == "unset"
    assert attr("eol", "README.md") == "lf" and attr("text", "README.md") == "auto"
    assert attr("eol", "compat/strata/tools/kl_gate.sh") == "lf" and attr("text", "compat/strata/tools/kl_gate.sh") == "auto"


@needs_git
def test_no_tracked_text_file_is_stored_with_crlf_and_patches_are_untouched():
    lines = git(REPO, "ls-files", "--eol").splitlines()
    assert len(lines) > 100
    crlf = [ln for ln in lines if ln.split()[0] in ("i/crlf", "i/mixed") and not ln.endswith(".patch")]
    assert crlf == [], crlf[:5]
    patches = [ln for ln in lines if ln.endswith(".patch")]
    assert patches and all("attr/-text" in ln for ln in patches)


def test_byte_exactness_survives_a_windows_style_checkout(tmp_path):
    """core.autocrlf=true is the Windows default; a patch (even one carrying CR bytes) must round-trip untouched."""
    repo = str(tmp_path)
    git(repo, "init", "-q")
    git(repo, "config", "core.autocrlf", "true")
    with open(os.path.join(REPO, ".gitattributes"), "rb") as src, open(os.path.join(repo, ".gitattributes"), "wb") as dst:
        dst.write(src.read())
    patch = b"diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-old\r\n+new\r\n"
    plain = b"line one\nline two\n"
    for name, data in (("0001-x.patch", patch), ("notes.txt", plain)):
        with open(os.path.join(repo, name), "wb") as fh:
            fh.write(data)
    git(repo, "add", "-A")
    git(repo, *SIGN, "commit", "-q", "-m", "t")
    for name in ("0001-x.patch", "notes.txt"):
        os.remove(os.path.join(repo, name))
    git(repo, "checkout", "--", ".")
    assert open(os.path.join(repo, "0001-x.patch"), "rb").read() == patch
    assert open(os.path.join(repo, "notes.txt"), "rb").read() == plain  # eol=lf wins over autocrlf


def test_editorconfig_and_gitignore_cover_the_basics():
    ec = open(os.path.join(REPO, ".editorconfig"), encoding="utf-8").read()
    for needle in ("root = true", "end_of_line = lf", "insert_final_newline = true", "indent_size = 4", "[*.{yml,yaml,cmake}]", "indent_size = 2"):
        assert needle in ec, needle
    ignored = open(os.path.join(REPO, ".gitignore"), encoding="utf-8").read().split()
    for pattern in ("build*/", "target/", "dist/", "*.egg-info/", "__pycache__/", ".venv*/"):
        assert pattern in ignored, pattern
