# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""R8 L12: MAP.md must list every src module, name every checker and script dependency, and say where unsafe may live.

OWNER of the rule: checks/check_map.py.  Each case damages ONE thing in a private copy of the tree and runs the real CLI.
"""

from __future__ import annotations

import os

import pytest
from conftest import REPO, run_tool

ATTR = "#[allow(" + "unsafe_code)]"


def put(tree: str, rel: str, text: str) -> None:
    path = os.path.join(tree, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def edit_map(tree: str, old: str, new: str) -> None:
    path = os.path.join(tree, "MAP.md")
    text = open(path, encoding="utf-8", newline="").read()
    assert old in text, old
    open(path, "w", encoding="utf-8", newline="").write(text.replace(old, new, 1))


def check(tree: str) -> tuple[int, str]:
    return run_tool(tree, "checks/check_map.py")


def test_the_real_tree_matches_its_map():
    assert run_tool(REPO, "checks/check_map.py")[0] == 0


def test_a_new_src_module_must_be_described(tree):
    put(tree, "src/newmod.rs", "// x\n")
    code, out = check(tree)
    assert code == 1 and "src module `src/newmod.rs`" in out, out


def test_a_module_that_is_renamed_leaves_no_ghost_row(tree):
    edit_map(tree, "`src/gate.rs`", "`src/gate_old.rs`")
    code, out = check(tree)
    assert code == 1 and "`src/gate_old.rs` is named but does not exist" in out and "src/gate.rs" in out, out


def test_the_checks_row_must_name_every_checker(tree):
    put(tree, "checks/check_zzz.py", "# x\n")
    code, out = check(tree)
    assert code == 1 and "the `checks/` row does not name check_zzz" in out, out


def test_the_scripts_row_must_name_a_new_third_party_dependency(tree):
    put(tree, "scripts/uses_numpy.py", "import numpy\n")
    code, out = check(tree)
    assert code == 1 and "the `scripts/` row does not name its dependency numpy" in out, out


def test_a_sibling_module_in_the_same_scripts_directory_is_internal_not_a_dependency(tree):
    put(tree, "scripts/pipeline/main.py", "import helper_mod\nimport other_dir_mod\n")
    put(tree, "scripts/pipeline/helper_mod.py", "X = 1\n")
    put(tree, "scripts/elsewhere/other_dir_mod.py", "X = 2\n")
    code, out = check(tree)
    # helper_mod sits beside its importer: internal.  other_dir_mod sits in ANOTHER directory: still a named dependency.
    assert code == 1 and "does not name its dependency other_dir_mod" in out and "helper_mod" not in out, out


def test_unsafe_outside_the_listed_places_fails(tree):
    put(tree, "src/sneaky.rs", f"{ATTR}\nfn f() {{}}\n")
    code, out = check(tree)
    assert code == 1 and "src/sneaky.rs: allow(unsafe_code) outside the places MAP.md lists" in out, out


def test_a_listed_place_that_no_longer_holds_unsafe_must_be_deleted(tree):
    edit_map(tree, "`src/ffi.rs` (the one inner attribute", "`src/gate.rs` and `src/ffi.rs` (the one inner attribute")
    code, out = check(tree)
    assert code == 1 and "`src/gate.rs` is listed for allow(unsafe_code) but holds none" in out, out


def test_prose_that_merely_names_the_attribute_is_not_a_holder(tree):
    put(tree, "src/prose.rs", "//! the attribute `" + ATTR + "` lives under ffi only\nfn f() {}\n")
    code, out = check(tree)
    assert "src/prose.rs: allow(unsafe_code)" not in out, out


@pytest.mark.parametrize("old", ["`src/rows.rs`", "`.github/`"])
def test_removing_a_row_is_caught(tree, old):
    edit_map(tree, old, "`gone`")
    assert check(tree)[0] == 1
