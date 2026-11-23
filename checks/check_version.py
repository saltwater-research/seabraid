#!/usr/bin/env python3
# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""One version everywhere: VERSION == Cargo.toml == CMake project() numeric part == pyproject's version source.

OWNER: the VERSION file at the repo root is the single source of truth; this checker owns the comparison, and
checks/_version_mentions.py owns the list of document statements (CITATION.cff, README, SECURITY, docs) it is compared with.
CONTRACT: VERSION holds a SemVer string, e.g. 0.5.0-alpha.1.  Cargo.toml [package] version and the seabraid entry of Cargo.lock
must equal it exactly.
CMake ``project(... VERSION x.y.z)`` must equal its numeric part (CMake cannot carry a pre-release tag).
pyproject.toml must either be ``dynamic = ["version"]`` reading VERSION (the file path is checked) or carry
``version = "<VERSION>"`` literally.  A file that does not exist yet is SKIPPED with a printed ``skip:`` note (other
tracks add pyproject and bump Cargo.toml); a file that exists but holds no version is a failure, never a skip.  The
one exception: Cargo.lock missing next to an existing Cargo.toml is a failure (builds use ``--locked``).
SIZE: <=150 lines.  FAILURE: fail-fast and loud; exit 1 and ``file: found X, expected Y``.
CHECKER: tests/test_checkers.py (planted mismatch in each file) and tests/test_pins_and_versions.py (the documents).
Run:  python3 checks/check_version.py [--root DIR]
"""

from __future__ import annotations

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as common  # noqa: E402
import _version_mentions as mentions  # noqa: E402

SEMVER = re.compile(r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$")
CARGO = re.compile(r"(?ms)^\[package\]\s.*?^version\s*=\s*\"([^\"]+)\"")
# WHY Cargo.lock: `cargo build --locked` (CMake, CI) refuses a lock file whose seabraid entry lags Cargo.toml, so a bump that
# edits VERSION and Cargo.toml but not the lock passed this check and then failed every locked build.
CARGO_LOCK = re.compile(r"(?m)^\[\[package\]\]\r?\nname = \"seabraid\"\r?\nversion = \"([^\"]+)\"")
CMAKE = re.compile(r"(?is)\bproject\s*\([^)]*?\bVERSION\s+([0-9][0-9.]*)")
# WHY: a project() VERSION taken from a variable is single-sourced only when CMake reads the VERSION file itself.
CMAKE_FROM_FILE = re.compile(r"(?is)file\s*\(\s*READ\s+\"?\$\{CMAKE_CURRENT_SOURCE_DIR\}/VERSION\"?.*\bproject\s*\([^)]*?\bVERSION\s+\$\{")
PYPROJECT_LITERAL = re.compile(r"(?m)^version\s*=\s*\"([^\"]+)\"")
PYPROJECT_DYNAMIC = re.compile(r"(?s)dynamic\s*=\s*\[[^\]]*\"version\"")
PYPROJECT_FILE = re.compile(r"(?m)^\s*(?:version\s*=\s*\{\s*file\s*=\s*\"([^\"]+)\"|file\s*=\s*\"([^\"]+)\")")


def _read(root: str, rel: str) -> str | None:
    path = os.path.join(root, rel)
    if not os.path.exists(path):
        return None
    text = common.read_text(root, rel)
    if text is None:
        raise SystemExit(f"check_version: {rel} exists but is unreadable")
    return text


def compare(rel: str, found: str | None, want: str) -> str | None:
    if found is None:
        return f"{rel}: no version found, expected {want}"
    return None if found == want else f"{rel}: found {found}, expected {want}"


def pyproject_problem(text: str, want: str) -> str | None:
    if PYPROJECT_DYNAMIC.search(text):
        m = PYPROJECT_FILE.search(text)
        pointed = (m.group(1) or m.group(2)) if m else None
        return None if pointed == "VERSION" or (pointed or "").endswith("/VERSION") else f"pyproject.toml: dynamic version must read the VERSION file, found {pointed}"
    m = PYPROJECT_LITERAL.search(text)
    return compare("pyproject.toml", m.group(1) if m else None, want)


VERSIONED_FILES = (("Cargo.toml", CARGO), ("Cargo.lock", CARGO_LOCK), ("CMakeLists.txt", CMAKE), ("pyproject.toml", None))


def _file_problem(root: str, rel: str, rx: re.Pattern | None, want: str, numeric: str) -> tuple[str | None, str | None]:
    """-> (problem, skip note) for one versioned file."""
    text = _read(root, rel)
    if text is None and rel == "Cargo.lock" and os.path.exists(os.path.join(root, "Cargo.toml")):
        # WHY a failure and not a skip: every build here runs `cargo ... --locked`, which refuses to run without the lock file.
        return "Cargo.lock: missing, but Cargo.toml exists and builds use --locked", None
    if text is None:
        return None, f"skip: {rel} does not exist yet"
    if rx is None:
        return pyproject_problem(text, want), None
    target = numeric if rel == "CMakeLists.txt" else want
    m = rx.search(text)
    found = m.group(1) if m else (target if rel == "CMakeLists.txt" and CMAKE_FROM_FILE.search(text) else None)
    return compare(rel, found, target), None


def check(root: str) -> tuple[list[str], list[str]]:
    version = _read(root, "VERSION")
    if version is None:
        return ["VERSION: file missing"], []
    want = version.strip()
    bad = [] if SEMVER.match(want) else [f"VERSION: {want!r} is not SemVer"]
    notes = [f"note: VERSION = {want}"]
    numeric = want.split("-")[0].split("+")[0]  # WHY: CMake project() cannot carry a pre-release tag
    for rel, rx in VERSIONED_FILES:
        problem, note = _file_problem(root, rel, rx, want, numeric)
        bad += [problem] if problem else []
        notes += [note] if note else []
    # WHY the documents too (R8 M6): CITATION.cff, README, SECURITY and the docs each restated the number and none was checked.
    return bad + mentions.problems(root, want), notes


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--root", default=common.default_root())
    bad, notes = check(ap.parse_args(argv).root)
    return common.finish("check_version", bad, notes)


if __name__ == "__main__":
    sys.exit(main())
