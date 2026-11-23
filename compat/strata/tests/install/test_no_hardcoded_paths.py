# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""The installer and the tools must run on someone else's machine.

This is the checker for "genericize everything".  It scans ``install/`` and
``tools/`` for values that only make sense on one machine
-- a home directory, a data mount, a fixed shared-memory name, a fixed port,
one model's layer and expert counts, and one card's architecture.

This test owns MACHINE VALUES (paths, ports, shm names, architectures).  Home
directories, LAN addresses and mounts anywhere in the repository are owned by
``tests/test_release_hygiene.py``; no pattern is duplicated between the two.

SCOPE, stated honestly: comments and triple-quoted blocks are skipped, so a
banned value may be NAMED in prose that explains it -- that is how the gfx
decoding rule is documented -- but may not appear in code or in a string the
program uses.  The stripper is a line-wise state machine over ``\"\"\"`` and
full-line ``#``; a triple quote inside a single-line string would confuse it.
That is accepted: every file in scope is one of ours, and the test is run
against a deliberate sabotage before it is believed.

Run:  python3 -m pytest tests/ -q

NOT ``unittest discover``.  It silently collects only the unittest-style classes
and MISSES every pytest-style test function in this directory -- most pytest-style
tests, so it reports a green suite while some tests never ran.  That is the
same "the check never ran" failure this file's own guards exist to catch, so the
gate is pytest.  ``unittest ... -k <name>`` is still the right tool for driving
ONE test during a sabotage, where the count is the point.
"""

from __future__ import annotations

import os
import re
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCAN_DIRS = ("install", "tools")
SCAN_EXT = (".py", ".sh", ".bash")

BANNED = [
    (re.compile(r"/home/[A-Za-z][\w.-]*"), "a user's home directory"),
    (re.compile(r"/media/[A-Za-z][\w.-]*"), "a specific removable or data mount"),
    (re.compile(r"\bseabraid\d+\b"), "a fixed shared-memory segment name"),
    (re.compile(r"(?i)(?:--port|\bport)\W{0,3}(?!8080\b)\d{4,5}\b"), "a fixed server port other than the documented default"),
    (re.compile(r"\bgfx\d{3,4}[0-9a-f]?\b"), "a literal GPU architecture"),
    (re.compile(r"\bsm_\d{2,3}\b"), "a literal GPU architecture"),
    (
        re.compile(r"(?i)(?:layers|experts)\W{1,3}(?:48|512)\b"),
        "one model's layer or expert count",
    ),
]


def strip_prose(text: str) -> list[tuple[int, str]]:
    """Return (line number, code) for every line that is not comment or prose."""
    out: list[tuple[int, str]] = []
    in_block = False
    for n, raw in enumerate(text.splitlines(), start=1):
        line = raw
        if in_block:
            if '"""' in line or "'''" in line:
                in_block = False
            continue
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        opens = stripped.count('"""') + stripped.count("'''")
        if opens == 1:
            in_block = True
            continue
        out.append((n, line))
    return out


# A line may be exempted, but only out loud and only with a reason attached, so
# every exemption in the project is one grep away and carries its own defense.
# A whole-vendor architecture TABLE is the legitimate case: it is the opposite
# of machine-specific, and relegating it to a data file would hide the taxonomy
# from the code that uses it.
PRAGMA = re.compile(r"#\s*seabraid-generic:\s*(\S.*)$")


def scan_file(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    bad: list[str] = []
    for lineno, line in strip_prose(text):
        pragma = PRAGMA.search(line)
        if pragma:
            continue
        for pattern, why in BANNED:
            m = pattern.search(line)
            if m:
                bad.append(
                    "%s:%d: %s (%s)"
                    % (os.path.relpath(path, REPO), lineno, m.group(0), why)
                )
    return bad


def files_in_scope() -> list[str]:
    found: list[str] = []
    for d in SCAN_DIRS:
        root = os.path.join(REPO, d)
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [x for x in dirnames if x not in ("__pycache__", ".git")]
            for name in filenames:
                if name.endswith(SCAN_EXT):
                    found.append(os.path.join(dirpath, name))
    return sorted(found)


class TestNoHardcodedPaths(unittest.TestCase):
    def test_scope_is_not_empty(self):
        """A checker that scans nothing passes everything -- guards that scan
        nothing are decorative."""
        self.assertTrue(files_in_scope(), "nothing was scanned: the paths are wrong")

    def test_no_machine_specific_values(self):
        bad: list[str] = []
        for path in files_in_scope():
            bad.extend(scan_file(path))
        self.assertEqual([], bad, "machine-specific values found:\n  " + "\n  ".join(bad))


class TestPortRule(unittest.TestCase):
    def _hits(self, line: str) -> list[str]:
        return [why for pat, why in BANNED if pat.search(line) and "port" in why]

    def test_a_fixed_port_other_than_the_default_is_flagged(self):
        self.assertTrue(self._hits("exec server --port 9999"))
        self.assertTrue(self._hits("PORT=${PORT:-9999}"))

    def test_the_documented_default_is_allowed(self):
        self.assertEqual([], self._hits("exec server --port 8080"))
        self.assertEqual([], self._hits("PORT=${PORT:-8080}"))


class TestStripProse(unittest.TestCase):
    """The stripper is the part that could silently let everything through, so
    it is tested on its own rather than only through the scan."""

    def test_keeps_code(self):
        self.assertEqual([(1, "x = 1")], strip_prose("x = 1"))

    def test_drops_full_line_comment(self):
        self.assertEqual([], strip_prose("# x = 1"))

    def test_drops_block(self):
        src = '"""\nsecret value\n"""\ny = 2'
        self.assertEqual([(4, "y = 2")], strip_prose(src))

    def test_keeps_code_after_one_line_docstring(self):
        self.assertEqual([(1, '"""doc."""'), (2, "z = 3")], strip_prose('"""doc."""\nz = 3'))


class TestPragma(unittest.TestCase):
    def test_a_bare_pragma_word_does_not_exempt_anything(self):
        """An exemption must carry a reason, or it is just a way to switch the
        checker off one line at a time."""
        self.assertIsNone(PRAGMA.search("ARCH = 'sm_86'  # seabraid-generic:"))

    def test_an_explained_exemption_is_honoured(self):
        self.assertTrue(PRAGMA.search("ARCH = 'sm_86'  # seabraid-generic: vendor taxonomy"))


if __name__ == "__main__":
    unittest.main()
