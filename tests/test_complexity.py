# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Tests for checks/check_complexity.py: each rule of the exception file, on temp trees, plus the real tree."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import subprocess
import sys

import pytest
from conftest import REPO

from check_complexity import LIZARD_PIN

pytestmark = pytest.mark.skipif(importlib.util.find_spec("lizard") is None, reason="lizard is not installed (pip install lizard==1.26.1)")

CHECKER = os.path.join(REPO, "checks", "check_complexity.py")
WHY = "flat list of independent checks; splitting scatters them"


def branches(name: str, n: int) -> str:
    """A Python function whose CCN is exactly n + 1 (n independent ifs)."""
    body = "".join(f"    if x == {i}:\n        return {i}\n" for i in range(n))
    return f"def {name}(x):\n{body}    return -1\n"


def run(tmp_path, source: str, rows: str | None = None) -> tuple[int, str]:
    (tmp_path / "src").mkdir(exist_ok=True)
    (tmp_path / "src" / "mod.py").write_text(source, encoding="utf-8")
    (tmp_path / "checks").mkdir(exist_ok=True)
    allow = tmp_path / "checks" / "complexity_allow.txt"
    allow.write_text(rows if rows is not None else "# none\n", encoding="utf-8")
    proc = subprocess.run([sys.executable, CHECKER, "--root", str(tmp_path)], capture_output=True, text=True)
    return proc.returncode, proc.stdout + proc.stderr


def row(func="f", ceiling=12, why=WHY, path="src/mod.py") -> str:
    return f"{path}\t{func}\t{ceiling}\t{why}\n"


def test_function_at_the_default_passes(tmp_path):
    assert run(tmp_path, branches("f", 9))[0] == 0


def test_over_default_without_a_row_fails_with_location(tmp_path):
    code, out = run(tmp_path, branches("f", 11))
    assert code == 1 and "src/mod.py:1: f CCN 12 > limit 10" in out


def test_valid_row_passes(tmp_path):
    assert run(tmp_path, branches("f", 11), row())[0] == 0


def test_ceiling_above_hard_max_fails(tmp_path):
    code, out = run(tmp_path, branches("f", 11), row(ceiling=16))
    assert code == 1 and "must be 11..15" in out


def test_function_over_hard_max_cannot_be_excepted(tmp_path):
    code, out = run(tmp_path, branches("f", 15), row(ceiling=15))
    assert code == 1 and "CCN 16 > limit 15" in out


def test_stale_row_fails(tmp_path):
    code, out = run(tmp_path, branches("f", 5), row())
    assert code == 1 and "stale row" in out


def test_row_for_a_missing_function_fails(tmp_path):
    code, out = run(tmp_path, branches("f", 11), row() + row(func="gone"))
    assert code == 1 and "matches no function" in out


def test_function_that_grew_past_its_ceiling_fails(tmp_path):
    code, out = run(tmp_path, branches("f", 13), row(ceiling=12))
    assert code == 1 and "CCN 14 > limit 12 (grew past its row)" in out


@pytest.mark.parametrize("why", ["", "too short"])
def test_weak_why_fails(tmp_path, why):
    code, out = run(tmp_path, branches("f", 11), row(why=why))
    assert code == 1 and "why for f" in out


def test_duplicate_row_fails(tmp_path):
    code, out = run(tmp_path, branches("f", 11), row() + row())
    assert code == 1 and "duplicate row" in out


def test_malformed_row_fails(tmp_path):
    code, out = run(tmp_path, branches("f", 11), "src/mod.py f 12 why\n")
    assert code == 1 and "want path<TAB>function" in out


def test_switch_with_twelve_cases_counts_as_one_decision(tmp_path):
    cases = "".join(f"    case {i}: return {i};\n" for i in range(12))
    src = f"int f(int x) {{\n  switch (x) {{\n{cases}    default: return -1;\n  }}\n}}\n"
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "sw.c").write_text(src, encoding="utf-8")
    proc = subprocess.run([sys.executable, CHECKER, "--root", str(tmp_path)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout


def run_tree(tmp_path, files: dict[str, str], ratchet: str | None = None) -> tuple[int, str]:
    """The checker over a tree of exactly ``files`` (repo-relative path -> text), with an optional ratchet file."""
    for rel, text in files.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text, encoding="utf-8")
    (tmp_path / "checks").mkdir(exist_ok=True)
    (tmp_path / "checks" / "complexity_allow.txt").write_text("# none\n", encoding="utf-8")
    if ratchet is not None:
        (tmp_path / "checks" / "complexity_ratchet.txt").write_text(ratchet, encoding="utf-8")
    proc = subprocess.run([sys.executable, CHECKER, "--root", str(tmp_path)], capture_output=True, text=True)
    return proc.returncode, proc.stdout + proc.stderr


@pytest.mark.parametrize("rel", ["compat/strata/strata-ext/src/a.py", "compat/strata/tests/t.py", "native-tests/b.py", "scripts/c.py",
                                 "tests/d.py", "examples/e.py", "checks/f.py", "include/g.py", "bindings/python/tests/h.py"])
def test_a_violation_in_a_formerly_unscoped_directory_fails(tmp_path, rel):
    # R8 M1: compat/strata/strata-ext, native-tests, checks, scripts, examples and tests were all outside the old scope list.
    code, out = run_tree(tmp_path, {rel: branches("f", 11)})
    assert code == 1 and f"{rel}:1: f CCN 12 > limit 10" in out, out


def test_a_code_file_outside_every_scope_fails(tmp_path):
    # WHY git init: with no .git the file walk reads only the shipped top-level names, which would hide the stray directory.
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    code, out = run_tree(tmp_path, {"newdir/x.py": branches("f", 2)})
    assert code == 1 and "newdir/x.py: code file outside SCOPES" in out


def test_every_shipped_code_directory_of_the_real_tree_is_in_scope():
    # Derived from the tree (git ls-files or the shipped-top-level walk), not from a list of names: the claim in SCOPES.
    import check_complexity

    assert check_complexity.unscoped(REPO) == []


def ratchet_row(ceiling: int, func="f", why="pending split by track X; a new violation still fails") -> str:
    return f"src/mod.py\t{func}\t{ceiling}\t{why}\n"


def test_ratchet_row_holds_a_function_above_the_hard_max_at_exactly_its_ceiling(tmp_path):
    src = {"src/mod.py": branches("f", 17)}  # CCN 18
    assert run_tree(tmp_path, src, ratchet_row(18))[0] == 0


def test_ratchet_row_that_the_function_grew_past_fails(tmp_path):
    code, out = run_tree(tmp_path, {"src/mod.py": branches("f", 18)}, ratchet_row(18))
    assert code == 1 and "CCN 19 > limit 18 (grew past its row)" in out


def test_ratchet_row_must_be_lowered_after_a_split(tmp_path):
    code, out = run_tree(tmp_path, {"src/mod.py": branches("f", 16)}, ratchet_row(20))
    assert code == 1 and "lower the row to 17" in out


def test_ratchet_row_is_deleted_once_the_function_is_down_to_the_allow_range(tmp_path):
    code, out = run_tree(tmp_path, {"src/mod.py": branches("f", 12)}, ratchet_row(20))
    assert code == 1 and "delete it" in out


def test_ratchet_ceiling_must_be_above_the_hard_max(tmp_path):
    code, out = run_tree(tmp_path, {"src/mod.py": branches("f", 12)}, ratchet_row(14))
    assert code == 1 and "must be 16..any" in out


def lizard_version() -> str | None:
    """The installed lizard version, or None. WHY a function: version() raises when lizard is missing, and evaluated at
    import time it aborted collection of every test in `pytest tests`; the module-level skip above already covers the
    no-lizard case, so this must return, not raise."""
    try:
        return importlib.metadata.version("lizard")
    except importlib.metadata.PackageNotFoundError:
        return None


@pytest.mark.skipif(lizard_version() != LIZARD_PIN, reason=f"real-tree counts are only authoritative under the CI-pinned lizard {LIZARD_PIN} (or lizard is not installed)")
def test_real_tree_passes():
    proc = subprocess.run([sys.executable, CHECKER, "--root", REPO], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout
