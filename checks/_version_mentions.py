# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Where the documents state the release version: every statement must equal the VERSION file (R8 M6).

OWNER: this module owns the list of version statements (STATEMENTS) and the sweep rule; VERSION stays the one source of the
number and checks/check_version.py is the caller.  CONTRACT: ``problems(root, want)`` -> list of ``file:line: ...`` strings;
pure over a repo root; imports only _common (one-way edge).  Stdlib only.
TWO RULES, both needed: (1) STATEMENTS names each place a document says "the version is X" by a regex with one capture group
per number; each must match at least once (a deleted statement is a failure, not a pass) and carry the form it is written in.
(2) the SWEEP: any pre-release token (``1.2.3-alpha.4`` or the PEP 440 ``1.2.3a4``) in README, SECURITY, CITATION, CONTRIBUTING,
MAP and docs/*.md must be VERSION in one of its two forms, so a new mention that no statement lists still cannot go stale.
WHY the history exceptions: CHANGELOG.md is the record of older versions, and a "Done in <version>" heading in a roadmap
names the release that finished the work, so both legitimately keep an old number (HISTORY below, one definition).
FAILURE: fail-fast and loud.  SIZE: <=150 lines.  CHECKER: tests/test_pins_and_versions.py.
"""

from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as common  # noqa: E402

# (file, regex, forms of its capture groups).  Forms: "semver" = VERSION, "pep440" = the PyPI spelling, "numeric" = x.y.z
# without the pre-release tag (what CMake and find_package see), "either" = semver or pep440.
STATEMENTS = (
    ("CITATION.cff", r"(?m)^version:\s*(\S+)", ("semver",)),
    ("README.md", r"\bVersion (\S+?)\.\s", ("semver",)),
    ("README.md", r"(?m)^seabraid (\S+)\s*$", ("pep440",)),
    ("SECURITY.md", r"alpha software \(([^)\s]+)\)", ("semver",)),
    ("docs/ROADMAP.md", r"is alpha \(([^)\s]+)\)", ("semver",)),
    ("docs/TECHNICAL-NOTE.md", r"Software version (\S+)", ("semver",)),
    ("docs/USER-GUIDE.md", r"(?m)^seabraid (\S+)\s*$", ("pep440",)),
    ("docs/ABI.md", r"`([0-9][^`\s]*)` is `([0-9.]+)` to `find_package`", ("semver", "numeric")),
)
SWEEP_TOKEN = re.compile(r"\b\d+\.\d+\.\d+(?:-(?:alpha|beta|rc)\.\d+|(?:a|b|rc)\d+)\b")
HISTORY_FILES = {"CHANGELOG.md"}
HISTORY_LINE = re.compile(r"^#+\s+Done in\b")


def pep440(semver: str) -> str:
    """0.5.0-alpha.2 -> 0.5.0a2 (also beta -> b, rc -> rc); anything else is returned unchanged."""
    m = re.fullmatch(r"(\d+\.\d+\.\d+)-(alpha|beta|rc)\.(\d+)", semver)
    return semver if not m else m.group(1) + {"alpha": "a", "beta": "b", "rc": "rc"}[m.group(2)] + m.group(3)


def _expected(form: str, want: str) -> set[str]:
    forms = {"semver": {want}, "pep440": {pep440(want)}, "numeric": {want.split("-")[0].split("+")[0]}}
    forms["either"] = forms["semver"] | forms["pep440"]
    return forms[form]


def _statement_problems(root: str, want: str) -> list[str]:
    bad = []
    for rel, rx, forms in STATEMENTS:
        if not os.path.exists(os.path.join(root, rel)):
            continue  # WHY skip: like check_version's other files, a document that does not exist yet is not a mismatch
        text = common.read_text(root, rel) or ""
        hits = list(re.finditer(rx, text))
        if not hits:
            bad.append(f"{rel}: version statement /{rx}/ not found (deleted or reworded? update checks/_version_mentions.py)")
        for m in hits:
            line = (text or "").count("\n", 0, m.start()) + 1
            bad += [f"{rel}:{line}: version {got!r}, expected {' or '.join(sorted(_expected(form, want)))}"
                    for got, form in zip(m.groups(), forms) if got not in _expected(form, want)]
    return bad


def _swept_files(root: str) -> list[str]:
    def in_docs(rel: str) -> bool:
        return rel.startswith("docs/") or "/" not in rel
    return [r for r in common.tracked(root) if r.endswith((".md", ".cff")) and in_docs(r) and r not in HISTORY_FILES]


def _sweep_problems(root: str, want: str) -> list[str]:
    ok = _expected("either", want)
    bad = []
    for rel in _swept_files(root):
        for n, line in enumerate((common.read_text(root, rel) or "").splitlines(), 1):
            stale = [t for t in SWEEP_TOKEN.findall(line) if t not in ok]
            bad += [f"{rel}:{n}: stale version {t} (VERSION is {want})" for t in stale if not HISTORY_LINE.match(line)]
    return bad


def problems(root: str, want: str) -> list[str]:
    return _statement_problems(root, want) + _sweep_problems(root, want)
