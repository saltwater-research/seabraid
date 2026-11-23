# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""R8 M6: a version, a tool pin, the MSRV and the CMake floor each have one owner; every other mention must equal it.

Each test copies the tree, damages ONE mention, and runs the real checker over the copy: it must exit 1 naming the file.
OWNER of the rules: checks/check_version.py (+ _version_mentions.py) and checks/check_pins.py; this file is their checker.
"""

from __future__ import annotations

import os
import re

import pytest
from conftest import REPO, run_tool

VERSION = open(os.path.join(REPO, "VERSION"), encoding="utf-8").read().strip()
PEP = re.sub(r"-(alpha|beta|rc)\.(\d+)", lambda m: {"alpha": "a", "beta": "b", "rc": "rc"}[m.group(1)] + m.group(2), VERSION)


def damage(tree: str, rel: str, old: str, new: str, count: int = 1) -> None:
    path = os.path.join(tree, rel)
    text = open(path, encoding="utf-8", newline="").read()
    assert old in text, f"{rel} no longer contains {old!r}; update this test with the doc"
    open(path, "w", encoding="utf-8", newline="").write(text.replace(old, new, count))


def version(tree: str) -> tuple[int, str]:
    return run_tool(tree, "checks/check_version.py")


def pins(tree: str) -> tuple[int, str]:
    return run_tool(tree, "checks/check_pins.py")


def test_the_real_tree_is_consistent():
    assert run_tool(REPO, "checks/check_version.py")[0] == 0
    assert run_tool(REPO, "checks/check_pins.py")[0] == 0


@pytest.mark.parametrize("rel, old, new", [
    ("CITATION.cff", f"version: {VERSION}", "version: 0.0.1-alpha.9"),
    ("README.md", f"Version {VERSION}.", "Version 0.0.1-alpha.9."),
    ("README.md", f"seabraid {PEP}", "seabraid 0.0.1a9"),
    ("SECURITY.md", f"({VERSION})", "(0.0.1-alpha.9)"),
    ("docs/ROADMAP.md", f"alpha ({VERSION})", "alpha (0.0.1-alpha.9)"),
    ("docs/TECHNICAL-NOTE.md", f"Software version {VERSION}", "Software version 0.0.1-alpha.9"),
    ("docs/USER-GUIDE.md", f"seabraid {PEP}", "seabraid 0.0.1a9"),
    ("docs/ABI.md", f"`{VERSION}` is `", "`0.0.1-alpha.9` is `"),
])
def test_a_stale_version_statement_in_a_document_fails(tree, rel, old, new):
    damage(tree, rel, old, new)
    code, out = version(tree)
    assert code == 1 and f"{rel}:" in out, out


def test_a_new_stale_mention_that_no_statement_lists_still_fails_the_sweep(tree):
    with open(os.path.join(tree, "docs", "NEW-NOTE.md"), "w", encoding="utf-8") as fh:
        fh.write("# New\n\nThe fix landed in 0.0.1-alpha.9 and is unchanged.\n")
    code, out = version(tree)
    assert code == 1 and "docs/NEW-NOTE.md:3: stale version 0.0.1-alpha.9" in out, out


def test_history_is_exempt_from_the_sweep(tree):
    with open(os.path.join(tree, "CHANGELOG.md"), "a", encoding="utf-8") as fh:
        fh.write("\n## 0.0.1-alpha.9\n- older\n")
    with open(os.path.join(tree, "docs", "NEW-NOTE.md"), "w", encoding="utf-8") as fh:
        fh.write("# New\n\n## Done in 0.0.1-alpha.9\n")
    assert version(tree)[0] == 0


def test_a_deleted_statement_is_a_failure_not_a_pass(tree):
    damage(tree, "SECURITY.md", "alpha software", "pre-release software")
    code, out = version(tree)
    assert code == 1 and "SECURITY.md: version statement" in out, out


def _pin(tool: str) -> str:
    return re.search(rf"{tool}==(\d+\.\d+\.\d+)", open(os.path.join(REPO, ".github", "workflows", "ci.yml"), encoding="utf-8").read()).group(1)


@pytest.mark.parametrize("rel, tool", [
    ("CONTRIBUTING.md", "lizard"), ("CONTRIBUTING.md", "numpy"), ("CONTRIBUTING.md", "PyYAML"),
    ("scripts/journey/steps.toml", "lizard"), ("scripts/journey/steps.toml", "numpy"),
    ("checks/SABOTAGE.md", "lizard"), ("checks/check_complexity.py", "LIZARD_PIN"),
])
def test_a_pin_that_differs_from_ci_fails(tree, rel, tool):
    real = _pin("lizard" if tool == "LIZARD_PIN" else tool)
    old = f'LIZARD_PIN = "{real}"' if tool == "LIZARD_PIN" else (f"lizard {real}" if rel.endswith("SABOTAGE.md") else f"{tool}=={real}")
    new = old.replace(real, "0.0.1")
    damage(tree, rel, old, new)
    code, out = pins(tree)
    assert code == 1 and f"{rel}:" in out and "0.0.1" in out, out


def test_ci_must_still_pin_every_tool(tree):
    damage(tree, ".github/workflows/ci.yml", f"lizard=={_pin('lizard')}", "lizard")
    code, out = pins(tree)
    assert code == 1 and "no pin for lizard" in out, out


@pytest.mark.parametrize("rel, old", [("README.md", "Rust 1.79 or newer"), ("docs/USER-GUIDE.md", "Rust 1.79+"),
                                      (".github/workflows/ci.yml", "rustup toolchain install 1.79 "),
                                      (".github/workflows/ci.yml", "cargo +1.79 check")])
def test_an_msrv_mention_that_differs_from_cargo_toml_fails(tree, rel, old):
    damage(tree, rel, old, old.replace("1.79", "1.70"))
    code, out = pins(tree)
    assert code == 1 and f"{rel}:" in out and "rust-version is 1.79" in out, out


def test_a_bumped_rust_version_without_the_docs_fails(tree):
    damage(tree, "Cargo.toml", 'rust-version = "1.79"', 'rust-version = "1.80"')
    code, out = pins(tree)
    assert code == 1 and "README.md" in out, out


@pytest.mark.parametrize("rel, old", [("README.md", "CMake 3.22 or newer"), ("docs/USER-GUIDE.md", "CMake 3.22+"),
                                      ("CONTRIBUTING.md", "cmake==3.22.6"), ("docs/USER-GUIDE.md", "cmake_minimum_required(VERSION 3.22)")])
def test_a_cmake_floor_mention_that_differs_from_the_root_fails(tree, rel, old):
    damage(tree, rel, old, old.replace("3.22", "3.20"))
    code, out = pins(tree)
    assert code == 1 and f"{rel}:" in out and "requires 3.22" in out, out


def test_a_new_subproject_below_the_root_floor_fails_unless_listed(tree):
    os.makedirs(os.path.join(tree, "adapters", "newone"), exist_ok=True)
    with open(os.path.join(tree, "adapters", "newone", "CMakeLists.txt"), "w", encoding="utf-8") as fh:
        fh.write("cmake_minimum_required(VERSION 3.10)\nproject(x)\n")
    code, out = pins(tree)
    assert code == 1 and "adapters/newone/CMakeLists.txt:1: cmake_minimum_required 3.10" in out, out


def test_a_patch_that_adds_a_lower_floor_cmake_file_to_another_tree_is_not_ours_to_gate(tree):
    path = os.path.join(tree, "compat", "strata", "patches", "strata", "9999-x.patch")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("+++ b/x/CMakeLists.txt\n+cmake_minimum_required(VERSION 3.10)\n")
    code, out = pins(tree)
    assert code == 0, out


def test_a_cmake_floor_row_goes_stale_once_the_file_is_raised(tree):
    damage(tree, "adapters/strata/CMakeLists.txt", "cmake_minimum_required(VERSION 3.16)", "cmake_minimum_required(VERSION 3.22)")
    code, out = pins(tree)
    assert code == 1 and "stale row for adapters/strata/CMakeLists.txt 3.16" in out, out
