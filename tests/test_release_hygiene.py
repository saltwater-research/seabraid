# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Release hygiene: no tracked file may identify the developer, their machines or their other projects.

This test scans for machine-specific values (home directories, LAN addresses, local mounts, session paths).  The sibling
``tests/install/test_no_hardcoded_paths.py`` owns MACHINE VALUES in the installer and tools (ports, shm names,
architectures); no pattern is duplicated between the two.

Failure contract: fail-fast and loud.  A hit is reported as ``path:line: <kind>`` with the offending text, so the
fix is a rewrite in that file; there is no allowlist.
The one exemption is the ``From:`` author line in a patch file's mail header: it is the commit author, the same
identity git records on every commit of this repository, so scanning it would only re-report the commit metadata.
Everything else in a patch (subject, body, diff) is scanned like any other file.
"""

from __future__ import annotations

import os
import re

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Written with character classes so this file does not match its own patterns.
SHAPES = [
    (re.compile(r"/home/[a-z]"), "a user's home directory"),
    (re.compile(r"[A-Za-z]:\\{1,2}Users\\"), "a Windows user profile path"),
    (re.compile(r"192\.168\."), "a private LAN address"),
    (re.compile(r"/media/[A-Za-z]"), "a machine-specific mount"),
    (re.compile(r"/mnt/[a-z]/"), "a Windows drive mount"),
    (
        re.compile(r"/\S*[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"),
        "a session UUID inside a path",
    ),
]

# A git format-patch mail header's author line ("From: Name <email>"), only before the header's first blank line.
PATCH_AUTHOR = re.compile(r"^From: [^\n<]+<[^\n>]+>$")


def find_identity(text: str) -> list[tuple[int, str]]:
    """Return (line number, description) for every identity hit in ``text``; empty when clean."""
    hits: list[tuple[int, str]] = []
    for n, line in enumerate(text.splitlines(), start=1):
        for pat, why in SHAPES:
            m = pat.search(line)
            if m:
                hits.append((n, "%s: '%s'" % (why, m.group(0))))
    return hits


# WHY one owner: "what ships" is checks/_common.tracked (git's list, or without .git the walk of the shipped top-level
# entries); a second walk here scanned an in-tree logs/ directory of an unpacked archive and failed on it.
from _common import tracked  # noqa: E402  (checks/ is on sys.path via conftest)


def tracked_files() -> list[str]:
    """The files that ship: tracked plus untracked-not-ignored with a .git, the shipped top-level walk without one."""
    return tracked(REPO)


def read_text(rel: str, root: str = REPO) -> str | None:
    """File text, or None for binary or unreadable files (a NUL byte in the first 8 KiB marks binary)."""
    try:
        with open(os.path.join(root, rel), "rb") as fh:
            raw = fh.read()
    except OSError:
        return None
    return None if b"\0" in raw[:8192] else raw.decode("utf-8", "replace")


def without_patch_author(rel: str, text: str) -> str:
    """Blank the author line of a .patch file's mail header, keeping line numbers; any other file is unchanged."""
    if not rel.endswith(".patch"):
        return text
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if not line.strip():
            break
        if PATCH_AUTHOR.match(line.rstrip("\r")):
            lines[i] = ""
    return "\n".join(lines)


# WHY: this file plants every shape on purpose to prove the checker fires, so scanning it would always report itself.
PLANTED_FIXTURES = {"tests/test_no_local_refs.py"}


def scan(paths: list[str], root: str = REPO) -> list[str]:
    bad = []
    for rel in paths:
        if rel in PLANTED_FIXTURES:
            continue
        text = read_text(rel, root)
        if text is not None:
            text = without_patch_author(rel, text)
            bad += ["%s:%d: %s" % (rel, n, why) for n, why in find_identity(text)]
    return bad


class TestChecker:
    """The checker must be shown to fire; a guard nobody has watched fail is not known to work."""

    @pytest.mark.parametrize(
        "sample",
        [
            "/home/" + "x/y",
            "C:" + "\\Users\\" + "x",
            "C:" + "\\\\Users\\\\" + "x",
            "192." + "168." + "77.2",
            "/media/" + "store",
            "/mnt/" + "d/x",
            "/tmp/q/" + "0b8f3a52-5c1e-4d7a-9e21-3f4a6b7c8d9e" + "/out",
        ],
    )
    def test_each_shape_fires(self, sample):
        assert find_identity(sample), sample

    def test_clean_text_passes(self):
        assert find_identity("GPU 0: RTX 3090, 24 GiB\nport 8080\n/dev/shm/seabraid0 /path/to/models\n") == []

    def test_only_the_patch_author_header_is_skipped(self):
        w = "/home/" + "x/y"
        patch = "From 000 Mon Sep 17 00:00:00 2001\nFrom: a <x@y>\nSubject: [PATCH] a\n\nFrom: a <x@y>\n+// %s\n" % w
        kept = without_patch_author("p/0001-a.patch", patch)
        assert [n for n, _ in find_identity(kept)] == [6]
        assert without_patch_author("notes.md", patch) == patch

    def test_scan_is_not_vacuous(self):
        files = tracked_files()
        assert len(files) > 100 and "README.md" in files
        assert sum(1 for f in files if read_text(f) is not None) > 100


def test_the_no_git_walk_ignores_generated_directories_and_still_catches_a_leak(tmp_path):
    """A source tarball has no .git; after a build the walk must not scan build output (it holds this test's own samples)."""
    leak = "/home/" + "x/y"

    def put(rel: str, text: str) -> None:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text + chr(10))

    # Generated directories INSIDE a shipped top-level entry (the walk prunes them), and entries that are not shipped at all.
    for rel in ("src/build/CMakeFiles/log.txt", "src/target/debug/d.txt", "src/dist/w.txt", "src/.pytest_cache/v/c.txt",
                "src/pkg/__pycache__/m.txt", "src/.venv/lib/a.txt", "src/venv/lib/a.txt", "src/out/odd/x.txt",
                "src/.venv-wheel/lib/site-packages/m.py", "logs/run.txt", "sdk/include/h.h"):
        put(rel, "planted " + leak)
    put("src/out/odd/CMakeCache.txt", "x")  # a CMake build tree under any name
    put("src/.venv-wheel/pyvenv.cfg", "home = /usr/bin")  # the README wheel recipe's venv, found by its marker
    put("src/clean.c", "int x;")

    def hits() -> list[str]:
        return scan(tracked(str(tmp_path)), str(tmp_path))

    assert hits() == []
    put("src/leaky.c", "// " + leak)
    assert len(hits()) == 1 and hits()[0].startswith("src/leaky.c:1:")


def test_no_tracked_file_identifies_the_developer():
    bad = scan(tracked_files())
    assert not bad, "identity found in tracked files (rename or rewrite it there):\n  " + "\n  ".join(bad[:60])
