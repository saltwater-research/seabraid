# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""scripts/retire_name.py (the owner-approved map of the retired add-on name) and the gates that keep that name out.

OWNER: this file is the checker of the map: every row of the 2026-11-21 table is one case below, the refusals are planted in
throwaway git trees, and the real tree is held to ``check_names`` with the retired name forbidden.  The name is assembled from
halves because the whole-tree name gate reads this file too.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys

import pytest
from conftest import REPO

import _common

OLD = "blam" + "mer"
UP, CAP = OLD.upper(), OLD.capitalize()
TAG = "BLAM" + "MR1"
_spec = importlib.util.spec_from_file_location("retire_name_tool", os.path.join(REPO, "scripts", "retire_name.py"))
tool = importlib.util.module_from_spec(_spec)
sys.modules["retire_name_tool"] = tool
_spec.loader.exec_module(tool)

# (text in a code file, what it must become); every row of the approved map has at least one line here
CODE = [
    (f"namespace {OLD} {{", "namespace seabraid {"),
    (f"{OLD}::Channel c;", "seabraid::Channel c;"),
    (f"RemoteExperts::{OLD}_open(x)", "RemoteExperts::seabraid_open(x)"),
    (f"{OLD}_closed = 1;", "seabraid_closed = 1;"),
    (f"RemoteExperts::{OLD}_begin(", "RemoteExperts::seabraid_helper_begin("),   # the one deviation: seabraid_begin is taken
    (f"{OLD}_finish(out, err)", "seabraid_helper_finish(out, err)"),
    (f"#if defined(STRATA_HAVE_{UP}_RETIER)", "#if defined(STRATA_HAVE_SEABRAID_RETIER)"),
    (f"getenv(\"{UP}_SHM\")", "getenv(\"SEABRAID_STRATA_SHM\")"),
    (f"-D{UP}_SOURCE_DIR=x", "-DSEABRAID_STRATA_SOURCE_DIR=x"),
    (f"log(\"{UP}: card CLOSED\")", "log(\"seabraid: card CLOSED\")"),
    (f"{OLD}-helper {OLD}-sums {OLD}-server {OLD}-helper-0.service", "seabraid-helper seabraid-sums seabraid-server seabraid-helper-0.service"),
    (f"/{OLD}0 /{OLD}1", "/seabraid0 /seabraid1"),
    (f"{UP} helper: msg", "seabraid helper: msg"),
    (f"{UP}-CARD-BENCH v1", "SEABRAID-CARD-BENCH v1"),
    (f"{OLD}_helper_loss_test {OLD}_setup.py", "seabraid_helper_loss_test seabraid_setup.py"),
    (f"From: {UP} <{OLD}@example.invalid>", "From: Seabraid <seabraid@example.invalid>"),
    (f"constexpr uint64_t kMagic = 0x31524D4D414C42ull;   // \"{TAG}\"", "constexpr uint64_t kMagic = 0x31445242414553ull;   // \"SEABRD1\""),
    (f"k{CAP}[] = {{}};", "kSeabraid[] = {};"),
    (f"compat/{OLD}/install/{OLD}_setup.py", "compat/strata/install/seabraid_setup.py"),
    (f'ROOT / "compat" / "{OLD}" / "patches"', 'ROOT / "compat" / "strata" / "patches"'),
    (f'os.path.join(REPO, "compat", "{OLD}")', 'os.path.join(REPO, "compat", "strata")'),
    (f'add_subdirectory("${{X}}/strata-ext" "${{CMAKE_BINARY_DIR}}/{OLD}")', 'add_subdirectory("${X}/strata-ext" "${CMAKE_BINARY_DIR}/seabraid-strata-ext")'),
]


@pytest.mark.parametrize("src,want", CODE)
def test_every_row_of_the_map_in_code(src, want):
    assert tool.rewrite(src, "c")[0] == want


def test_documents_read_as_prose_and_quote_what_the_code_prints():
    assert tool.rewrite(f"The {UP} add-on", "d")[0] == "The Strata integration"
    assert tool.rewrite(f"{UP} lets a GPU", "d")[0] == "Strata integration lets a GPU"
    assert tool.rewrite(f"`{UP} is serving x`", "d")[0] == tool.rewrite(f"\"{UP} is serving x\"", "c")[0].replace('"', "`")


def test_a_path_never_gets_a_space():
    assert tool.rewrite(f"0005-{UP}-retier-x.patch", "p")[0] == "0005-seabraid-retier-x.patch"
    assert tool.rewrite(f"0006-{UP}_HOT_SHARE-opt-in.patch", "p")[0] == "0006-SEABRAID_STRATA_HOT_SHARE-opt-in.patch"
    assert " " not in tool.rewrite(f"{UP}", "p")[0]


def git(root, *args):
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


def mini(tmp_path, files):
    for rel, text in files.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text, encoding="utf-8", newline="\n")
    git(tmp_path, "init", "-q")
    git(tmp_path, "add", "-A")
    return tmp_path


def run(root, *args):
    p = subprocess.run([sys.executable, "-B", os.path.join(REPO, "scripts", "retire_name.py"), "--root", str(root), *args], capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def test_a_run_moves_rewrites_and_a_second_run_changes_nothing(tmp_path):
    root = mini(tmp_path, {f"compat/{OLD}/a.hpp": f"namespace {OLD} {{ int {UP}_X; }}\n", "README.md": f"The {UP} add-on.\n"})
    rc, out = run(root, "--skip-token-check")
    assert rc == 0, out
    assert (root / "compat/strata/a.hpp").read_text() == "namespace seabraid { int SEABRAID_STRATA_X; }\n"
    assert (root / "README.md").read_text() == "The Strata integration.\n" and not (root / "compat" / OLD).exists()
    again = run(root, "--skip-token-check")
    assert again[0] == 0 and "0 replacement(s) in 0 file(s), 0 path(s) moved" in again[1]


def test_two_sources_for_one_destination_refuse_before_any_write(tmp_path):
    root = mini(tmp_path, {f"src/{OLD}_a.c": "x\n", "src/seabraid_a.c": "y\n"})
    rc, out = run(root, "--skip-token-check")
    assert rc == 1 and "collision" in out and (root / f"src/{OLD}_a.c").exists()


def test_an_identifier_that_already_means_something_refuses(tmp_path):
    root = mini(tmp_path, {"src/a.c": f"int {OLD}_probe(void);\n", "src/b.c": "int seabraid_probe(void);\n"})
    rc, out = run(root)
    assert rc == 1 and f"{OLD}_probe -> seabraid_probe" in out and f"{OLD}_probe" in (root / "src/a.c").read_text()


def test_a_generated_patch_that_still_names_the_old_tool_is_refused_not_edited(tmp_path):
    root = mini(tmp_path, {f"compat/{OLD}/patches/strata/0001-x.patch": f"Subject: {UP} x\n"})
    rc, out = run(root, "--skip-token-check")
    assert rc == 1 and "regenerate" in out and (root / f"compat/{OLD}/patches/strata/0001-x.patch").exists()


def test_filter_mode_rewrites_a_commit_message():
    p = subprocess.run([sys.executable, "-B", os.path.join(REPO, "scripts", "retire_name.py"), "--filter"], input=f"{UP}: x {UP}_SHM\n", capture_output=True, text=True)
    assert p.stdout == "seabraid: x SEABRAID_STRATA_SHM\n"


@pytest.mark.parametrize("rel,text", [(f"src/{OLD}/x.c", "int x;\n"), ("src/a.cpp", f"// {CAP}\n"), ("README.md", f"{UP}\n"),
                                      ("docs/x.patch", f"+ {OLD}\n"), ("src/b.h", f"#define {TAG}\n")])
def test_a_planted_old_name_in_an_exported_file_fails_the_gate(tmp_path, rel, text):
    import check_names

    (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / rel).write_text(text)
    hits = check_names.scan(str(tmp_path), [OLD, "blam" + "mr"])
    assert hits and rel.split("/")[0] in hits[0]


def test_the_release_guard_refuses_the_retired_name_as_a_product_name():
    sys.path.insert(0, os.path.join(REPO, "scripts", "release"))
    import rel_guard

    assert OLD in rel_guard.PROVISIONAL
    assert any("provisional" in p for p in rel_guard.name_problems(OLD, __import__("pathlib").Path(REPO)))


def test_the_real_tree_names_neither_the_old_name_nor_its_wire_tag():
    import check_names

    assert check_names.scan(REPO, [OLD, "blam" + "mr"]) == []
    assert _common.tracked(REPO)
