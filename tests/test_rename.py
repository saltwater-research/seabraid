# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""scripts/rename.py, exercised ONLY on temp copies of the tree (never the real worktree): rename to a throwaway
name and prove the result, that a second run is a no-op, that English-looking compounds survive, and that bad input
is refused before any write.

OLD is read from Cargo.toml, so this file keeps working after the real rename (the tool rewrites Cargo.toml but
leaves this file alone, see SELF_EXEMPT in rename.py).  NEW is a throwaway name that must differ from the real one.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import re
import subprocess
import sys

import pytest
from conftest import REPO, copy_tree, run_tool

import _common

with open(os.path.join(REPO, "Cargo.toml"), encoding="utf-8") as _fh:
    OLD = re.search(r'(?m)^name\s*=\s*"([^"]+)"', _fh.read()).group(1)
# WHY assembled: the whole-tree name gate must not see the probe name in this file.
_PROBE = "zz" + "rig"
NEW = _PROBE if OLD != _PROBE else "yy" + "rig"
_spec = importlib.util.spec_from_file_location("rename_tool", os.path.join(REPO, "scripts", "rename.py"))
rename = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rename)

TITLE, UP = OLD.capitalize(), OLD.upper()
# The token has NO word boundary (review B8): a letter touching the name does not protect it, so compounds and CamelCase
# (k<Name>, <NAME>X, b<name>) are rewritten.  Only a whole word listed in _common.EXCLUDED_WORDS survives.
MIXED = OLD[0].upper() + OLD[1:3] + OLD[3:].upper()  # neither lower, Title nor UPPER: rewritten as Capitalised
WORDS = f"// b{OLD} k{TITLE} {UP}X {OLD}s {OLD}ed {TITLE}is {MIXED}\n// -l{OLD} lib{OLD} {UP}_X {TITLE}::runtime {OLD}.dll\n"
SELF = rename.SELF_EXEMPT


def digest(root: str) -> str:
    h = hashlib.sha256()
    for rel in _common.tracked(root):
        with open(os.path.join(root, rel), "rb") as fh:
            h.update(rel.encode() + b"\0" + fh.read() + b"\0")
    return h.hexdigest()


def plant(root: str) -> None:
    for rel, data in {
        "src/words.rs": WORDS.encode(),
        f"src/crlf_{OLD}.rs": f"fn {OLD}_a() {{}}\r\nfn b() {{}}\r\n".encode(),
        "src/data.dat": f"\0\1\2{OLD}\3".encode(),
        f"src/{OLD}_blob.dat": b"\0\1\2plain\3",
    }.items():
        with open(os.path.join(root, rel), "wb") as fh:
            fh.write(data)


def run_rename(root: str, *extra: str, new: str = NEW) -> tuple[int, str]:
    return run_tool(REPO, "scripts/rename.py", "--from", OLD, "--to", new, "--root", root, *extra)


@pytest.fixture(scope="module")
def renamed(tmp_path_factory):
    """(root, digest before, first run, second run, digest after run 1, digest after run 2)."""
    root = copy_tree(str(tmp_path_factory.mktemp("rename") / "repo"))
    plant(root)
    before = digest(root)
    first = run_rename(root)
    mid = digest(root)
    second = run_rename(root)
    return root, before, first, second, mid, digest(root)


def test_dry_run_reports_counts_and_writes_nothing(tree):
    before = digest(tree)
    code, out = run_rename(tree, "--dry-run")
    assert code == 0 and "DRY RUN" in out and "replacement(s)" in out
    assert re.search(rf"\d+ replacement\(s\)  include/{OLD}/{OLD}\.h  ->  include/{NEW}/{NEW}\.h", out)
    assert f"strata-v7-{OLD}.patch  ->  adapters/strata/strata-v7-{NEW}.patch" in out
    assert digest(tree) == before


def test_no_old_name_tokens_remain_outside_the_tool_itself(renamed):
    root, rx = renamed[0], _common.name_regex(OLD)
    left = [rel for rel in _common.tracked(root) if rel not in SELF and (rx.search(rel) or rx.search(_common.read_text(root, rel) or ""))]
    assert left == []


def test_whole_tree_passes_the_post_rename_gate_except_the_binary_it_cannot_rewrite(renamed):
    root = renamed[0]
    code, out = run_tool(root, "checks/check_names.py", "--root", root, "--forbid", OLD)
    assert code == 1 and out.count("forbidden name") == 1 and "src/data.dat: binary file contains" in out  # rename.py leaves binaries alone; the gate sees them
    allow = os.path.join(root, "checks", "names_allow.txt")
    with open(allow, encoding="utf-8") as fh:
        original = fh.read()
    try:
        with open(allow, "a", encoding="utf-8", newline="\n") as fh:
            fh.write("src/data.dat\tplanted binary the renamer leaves alone by design\n")
        code, out = run_tool(root, "checks/check_names.py", "--root", root, "--forbid", OLD)
        assert code == 0, out[-800:]
    finally:
        with open(allow, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(original)
    code, out = run_tool(root, "checks/check_names.py", "--root", root, "--forbid", NEW)
    assert code == 1 and "forbidden name" in out  # the gate is not vacuous on the renamed tree


def test_python_package_imports_under_the_new_name(renamed):
    root = renamed[0]
    probe = f"import sys; sys.path.insert(0, {os.path.join(root, 'bindings', 'python')!r}); import {NEW}; print({NEW}.__file__)"
    proc = subprocess.run([sys.executable, "-B", "-c", probe], capture_output=True, text=True, cwd=os.path.dirname(root))
    assert proc.returncode == 0, proc.stderr
    assert not os.path.exists(os.path.join(root, "bindings", "python", OLD))


def test_headers_have_no_old_macros_or_names(renamed):
    inc = os.path.join(renamed[0], "include")
    names = [os.path.join(b, f) for b, _, fs in os.walk(inc) for f in fs]
    assert names and all(f"include{os.sep}{NEW}" in n for n in names)
    for n in names:
        with open(n, encoding="utf-8") as fh:
            text = fh.read()
        assert f"{UP}_" not in text and not _common.name_regex(OLD).search(text) and f"{NEW.upper()}_" in text


def test_paths_are_renamed_everywhere(renamed):
    root = renamed[0]
    for old_rel in (f"cmake/{TITLE}Config.cmake.in", f"adapters/ggml/_{OLD}_ggml_cache.h", f"adapters/strata/strata-v7-{OLD}.patch", f"src/{OLD}_blob.dat"):
        assert os.path.isfile(os.path.join(root, rename.plan_path(old_rel, OLD, NEW))), old_rel
        assert not os.path.exists(os.path.join(root, old_rel)), old_rel
    assert os.path.isfile(os.path.join(root, "bindings", "python", NEW, "__init__.py"))
    assert not any(OLD in p.lower() for p in _common.tracked(root) if p not in SELF)


def test_second_run_is_a_no_op(renamed):
    _, _, first, second, mid, after = renamed
    assert first[0] == 0 and second[0] == 0
    assert "0 replacement(s) in 0 file(s), 0 path(s) renamed" in second[1]
    assert mid == after


def test_compounds_and_camel_case_are_rewritten(renamed):
    with open(os.path.join(renamed[0], "src", "words.rs"), encoding="utf-8") as fh:
        got = fh.read().split("\n")
    cap = NEW.capitalize()
    assert got[0] == f"// b{NEW} k{cap} {NEW.upper()}X {NEW}s {NEW}ed {cap}is {cap}"
    assert got[1] == f"// -l{NEW} lib{NEW} {NEW.upper()}_X {cap}::runtime {NEW}.dll"


def test_a_whole_word_in_excluded_words_survives(monkeypatch):
    monkeypatch.setitem(_common.EXCLUDED_WORDS, OLD, (f"b{OLD}",))
    assert rename.substitute(f"b{OLD} k{TITLE}", OLD, NEW) == (f"b{OLD} k{NEW.capitalize()}", 1)


def test_crlf_is_preserved_binary_untouched_and_reported(renamed):
    root, _, first, *_ = renamed
    with open(os.path.join(root, "src", f"crlf_{NEW}.rs"), "rb") as fh:
        assert fh.read() == f"fn {NEW}_a() {{}}\r\nfn b() {{}}\r\n".encode()
    with open(os.path.join(root, "src", "data.dat"), "rb") as fh:
        assert fh.read() == f"\0\1\2{OLD}\3".encode()
    assert "warning: src/data.dat: binary file contains the name; not rewritten" in first[1]


def test_patches_keep_line_counts_and_valid_hunks(renamed):
    root, rx = renamed[0], _common.name_regex(OLD)
    patches = [p for p in _common.tracked(REPO) if p.endswith(".patch")]
    assert patches
    for rel in patches:
        with open(os.path.join(REPO, rel), "rb") as fh:
            old = fh.read().decode("utf-8", "surrogateescape")
        with open(os.path.join(root, rename.plan_path(rel, OLD, NEW)), "rb") as fh:
            new = fh.read().decode("utf-8", "surrogateescape")
        assert old.count("\n") == new.count("\n"), rel
        assert len(rename.hunk_problems(new)) <= len(rename.hunk_problems(old)), rel
        if not rx.search(old):
            assert old == new, f"{rel} has no tokens and must be byte-identical"


def test_hunk_checker_notices_a_dropped_line():
    assert rename.hunk_problems("@@ -1,2 +1,2 @@\n a\n-b\n+c\n") == []
    assert rename.hunk_problems("@@ -1,2 +1,2 @@\n a\n+c\n")


def test_patch_that_would_break_is_refused_before_any_write(tree, monkeypatch):
    before = digest(tree)
    real = rename.substitute

    def lossy(text, old, new):
        out, n = real(text, old, new)
        at = out.find("@@ -")
        if not (n and at >= 0):
            return out, n
        cut = out.index("\n+", at)  # drop the first added line of the first hunk
        return out[:cut] + out[out.index("\n", cut + 1) :], n

    monkeypatch.setattr(rename, "substitute", lossy)
    with pytest.raises(rename.RenameError, match="hunks inconsistent"):
        rename.build_plan(tree, OLD, NEW)
    assert digest(tree) == before


@pytest.mark.parametrize("bad", ["Bad", "ab", "1abc", OLD + "y", "has-dash", "a" * 22])
def test_bad_new_names_are_refused_and_nothing_changes(tree, bad):
    before = digest(tree)
    code, out = run_rename(tree, new=bad)
    assert code == 1 and "REFUSED" in out and digest(tree) == before


def test_path_collision_is_refused_and_nothing_changes(tree):
    for rel, text in ((f"src/{NEW}_x.rs", "// existing\n"), (f"src/{OLD}_x.rs", "// to be renamed\n")):
        with open(os.path.join(tree, rel), "w") as fh:
            fh.write(text)
    before = digest(tree)
    code, out = run_rename(tree)
    assert code == 1 and "collision" in out and digest(tree) == before


# WHY assembled: the whole-tree name gate must not see the old name's story phrase in this file.
_STORY = "braided into one" + " rope"


def readme_etymology_state(text: str) -> str:
    """Return "marked" (line present), "gone" (a rename removed it, no new story yet) or "lost" (story, no marker).

    WHY three states: the tree this suite runs on may already be renamed, where the marker is absent by design until the
    maintainers write the new name's story; only a README that still tells the old story without the marker is a bug.
    """
    if rename.ETYMOLOGY in text:
        return "marked"
    return "lost" if _STORY in text else "gone"


def test_readme_etymology_state_tells_a_renamed_tree_from_a_lost_marker():
    assert readme_etymology_state(f"x\n*{_STORY}* {rename.ETYMOLOGY}\n") == "marked"
    assert readme_etymology_state(f"x\n*{_STORY}*\n") == "lost"
    assert readme_etymology_state("# Newname\n\nno story yet\n") == "gone"


def test_the_readme_name_etymology_is_marked_removed_and_reported_by_a_rename(renamed):
    """The README sentence that explains the OLD name must not survive a rename and describe a name that is gone."""
    root, _, first, *_ = renamed
    with open(os.path.join(REPO, "README.md"), encoding="utf-8") as fh:
        state = readme_etymology_state(fh.read())
    assert state != "lost", "the README's name sentence lost its marker, so a rename would keep it"
    if state == "gone":
        pytest.skip("this tree was already renamed: its README has no etymology line until the new name's story is written")
    with open(os.path.join(root, "README.md"), encoding="utf-8") as fh:
        after = fh.read()
    assert rename.ETYMOLOGY not in after and "braided into one rope" not in after
    assert "removed the name-etymology line" in first[1]


def test_drop_etymology_removes_the_line_and_one_blank_and_nothing_else():
    text = f"a\n\n*why old* {rename.ETYMOLOGY}\n\n> b\nc\n"
    assert rename.drop_etymology(text) == ("a\n\n> b\nc\n", 1)
    assert rename.drop_etymology("a\nb\n") == ("a\nb\n", 0)
