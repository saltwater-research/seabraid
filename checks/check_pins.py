#!/usr/bin/env python3
# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""One owner per pin and per floor: every other mention of it must agree (R8 M6).

OWNER: this checker owns the rule; the VALUES live in their owners: lizard / numpy / PyYAML in .github/workflows/ci.yml (what CI
installs), the Rust MSRV in Cargo.toml ``rust-version``, the CMake floor in the root CMakeLists.txt ``cmake_minimum_required``.
CONTRACT: every mention elsewhere (docs, scripts, checkers, other CMake files) is compared with its owner's value; a mention
that disagrees, or an owner value that is missing, is a failure.  History files (CHANGELOG.md) are exempt.  A sub-project that
states a LOWER cmake_minimum_required than the root is listed in checks/cmake_floor_allow.txt (``path<TAB>version<TAB>why``);
*.patch files are not scanned for it (they add files to another project's tree, whose own floor applies); a row is stale, and fails, once its file is raised to the root's floor or no longer says that version, so the list only shrinks.
WHY: the pins were copied into five files and a mismatch was only a stderr note; the MSRV and CMake floor were prose in
README and the user guide that nothing compared with the build files.
FAILURE: fail-fast and loud; exit 1 and ``file:line: <what> X, owner says Y``.  SIZE: <=150 lines.
CHECKER: tests/test_pins_and_versions.py (a planted mismatch in each kind of mention, and the allow-list rules).
Run:  python3 checks/check_pins.py [--root DIR]
"""

from __future__ import annotations

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as common  # noqa: E402

CI = ".github/workflows/ci.yml"
ALLOW = "checks/cmake_floor_allow.txt"
EXEMPT = {"CHANGELOG.md"}
# (label, regex with the tool in group 1 and the version in group 2).  The `==` form is a pip pin; the prose form is
# "lizard X.Y.Z"; LIZARD_PIN = "x" is the checker's own copy.  A bare "numpy 2" is not a pin and does not match.
PIN_PATTERNS = (re.compile(r"(?i)\b(lizard|numpy|pyyaml)==(\d+\.\d+\.\d+)\b"), re.compile(r"(?i)\b(lizard) (\d+\.\d+\.\d+)\b"),
                re.compile(r"\b(LIZARD)_PIN\s*=\s*\"(\d+\.\d+\.\d+)\""))
MSRV_PATTERNS = (re.compile(r"\bRust (\d+\.\d+)(?:\+| or newer)"), re.compile(r"\bMSRV (\d+\.\d+)"),
                 re.compile(r"rustup toolchain install (\d+\.\d+)\b"), re.compile(r"cargo \+(\d+\.\d+) "))
CMAKE_PROSE = re.compile(r"(?i)\bCMake (\d+\.\d+)(?:\+| or newer| floor)")
CMAKE_PIP = re.compile(r"\bcmake==(\d+\.\d+)\.\d+")
CMAKE_MIN = re.compile(r"cmake_minimum_required\(VERSION (\d+\.\d+)")


def _texts(root: str) -> dict[str, str]:
    out = {}
    for rel in common.tracked(root):
        text = None if rel in EXEMPT else common.read_text(root, rel)
        if text is not None and not rel.startswith(("checks/_version_mentions", "tests/test_pins_and_versions")):
            out[rel] = text
    return out


def _mentions(texts: dict[str, str], patterns, skip: tuple[str, ...] = ()) -> list[tuple[str, int, tuple[str, ...]]]:
    found = []
    for rel, text in texts.items():
        for rx in patterns:
            found += [(rel, text.count("\n", 0, m.start()) + 1, m.groups()) for m in rx.finditer(text) if rel not in skip]
    return found


def pin_problems(texts: dict[str, str]) -> list[str]:
    owner: dict[str, set[str]] = {}
    for _, _, (tool, ver) in _mentions({CI: texts.get(CI, "")}, PIN_PATTERNS[:1]):
        owner.setdefault(tool.lower(), set()).add(ver)
    bad = [f"{CI}: no pin for {t} (CI must install an exact version of it)" for t in ("lizard", "numpy", "pyyaml") if t not in owner]
    bad += [f"{CI}: {t} is pinned to several versions: {sorted(v)}" for t, v in owner.items() if len(v) > 1]
    for rel, line, (tool, ver) in _mentions(texts, PIN_PATTERNS, skip=(CI,)):
        want = owner.get(tool.lower(), set())
        bad += [f"{rel}:{line}: {tool} {ver}, but {CI} installs {sorted(want)[0]}"] if want and ver not in want else []
    return bad


def msrv_problems(root: str, texts: dict[str, str]) -> list[str]:
    m = re.search(r'(?m)^rust-version\s*=\s*"(\d+\.\d+)"', texts.get("Cargo.toml", ""))
    if not m:
        return ["Cargo.toml: no rust-version (the MSRV owner)"]
    return [f"{rel}:{line}: Rust {got}, but Cargo.toml rust-version is {m.group(1)}"
            for rel, line, (got,) in _mentions(texts, MSRV_PATTERNS) if got != m.group(1)]


def _load_allow(texts: dict[str, str]) -> tuple[dict[str, str], list[str]]:
    rows, errs = {}, []
    for n, raw in enumerate(texts.get(ALLOW, "").splitlines(), 1):
        if raw.strip() and not raw.lstrip().startswith("#"):
            parts = raw.split("\t")
            if len(parts) != 3 or len(parts[2].strip()) < 20:
                errs.append(f"{ALLOW}:{n}: want path<TAB>version<TAB>why (why >= 20 chars)")
            else:
                rows[parts[0].strip()] = parts[1].strip()
    return rows, errs


def _patch_files(texts: dict[str, str]) -> tuple[str, ...]:
    return tuple(rel for rel in texts if rel.endswith(".patch"))


def cmake_problems(texts: dict[str, str]) -> list[str]:
    root_min = CMAKE_MIN.search(texts.get("CMakeLists.txt", ""))
    if not root_min:
        return ["CMakeLists.txt: no cmake_minimum_required (the CMake floor owner)"]
    floor = root_min.group(1)
    allow, bad = _load_allow(texts)
    for rel, line, (got,) in _mentions(texts, (CMAKE_PROSE, CMAKE_PIP)):
        bad += [f"{rel}:{line}: CMake {got}, but the root CMakeLists.txt requires {floor}"] if got != floor else []
    seen = set()
    # WHY skip *.patch: a patch adds a CMake file to STRATA's tree, which Strata's own build floor governs, not ours; the
    # floor rule is about the files this repository builds.  (compat/strata/patches/strata/0001 adds one at 3.24.)
    for rel, line, (got,) in _mentions(texts, (CMAKE_MIN,), skip=_patch_files(texts)):
        seen.add((rel, got))
        if got != floor and allow.get(rel) != got:
            bad.append(f"{rel}:{line}: cmake_minimum_required {got}, root requires {floor} (raise it, or list it in {ALLOW})")
    bad += [f"{ALLOW}: stale row for {rel} {ver} (the file no longer says that; delete the row)" for rel, ver in sorted(allow.items()) if (rel, ver) not in seen or ver == floor]
    return bad


def check(root: str) -> list[str]:
    texts = _texts(root)
    return pin_problems(texts) + msrv_problems(root, texts) + cmake_problems(texts)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--root", default=common.default_root())
    return common.finish("check_pins", check(ap.parse_args(argv).root))


if __name__ == "__main__":
    sys.exit(main())
