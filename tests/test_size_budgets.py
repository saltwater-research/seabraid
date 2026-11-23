# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""R8 L5: the SIZE budget a file writes in its header must be true; the false ones today are a shrink-only list.

OWNER of the rule: checks/check_size_budgets.py.  The planted headers build the tag from TAG so this file's own header
(which the checker reads too) never contains a budget line of its own.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
from conftest import REPO

import check_size_budgets as budgets

CHECKER = os.path.join(REPO, "checks", "check_size_budgets.py")
TAG = "SI" + "ZE"


def planted(tmp_path, rel: str, header: str, total: int, rows: str | None = None) -> tuple[int, str]:
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'"""{header}"""\n' + "x = 1\n" * (total - 1), encoding="utf-8", newline="\n")
    (tmp_path / "checks").mkdir(exist_ok=True)
    (tmp_path / "checks" / "size_budget_allow.txt").write_text(rows or "# none\n", encoding="utf-8")
    proc = subprocess.run([sys.executable, CHECKER, "--root", str(tmp_path)], capture_output=True, text=True)
    return proc.returncode, proc.stdout + proc.stderr


@pytest.mark.parametrize("text, expect", [
    (f"{TAG}: <=150 lines.  FAILURE: fail-fast", ("ok", 150)),
    (f"{TAG}    : under 280 lines.", ("ok", 280)),
    (f"{TAG}: at most 200 lines here; split the sink out at 160.", ("ok", 200)),
    (f"{TAG} BUDGET: 60 lines. FAILURE CONTRACT: nothing", ("ok", 60)),
    (f"{TAG}      : target ~330 lines, hard cap 500.", ("ok", 500)),
    (f"{TAG} BUDGET: header <=80, source <=120 lines.", ("multi", None)),
    (f"{TAG}: soon.", ("bad", None)),
    ("no budget here", ("none", None)),
])
def test_the_budget_formats_in_the_tree_are_read(text, expect):
    assert budgets.budget_of(text) == expect


def test_a_false_budget_fails_and_a_true_one_passes(tmp_path):
    code, out = planted(tmp_path, "src/a.py", f"{TAG}: under 10 lines.", 11)
    assert code == 1 and "src/a.py: 11 lines, header budget 10" in out
    assert planted(tmp_path, "src/a.py", f"{TAG}: under 10 lines.", 10)[0] == 0


def test_an_unreadable_size_line_is_a_failure(tmp_path):
    code, out = planted(tmp_path, "src/a.py", f"{TAG}: small.", 5)
    assert code == 1 and "cannot read" in out


def test_a_known_false_budget_is_ratcheted_shrink_only(tmp_path):
    row = "src/a.py\t12\tfalse today; the owner splits the module soon\n"
    assert planted(tmp_path, "src/a.py", f"{TAG}: under 10 lines.", 12, row)[0] == 0
    code, out = planted(tmp_path, "src/a.py", f"{TAG}: under 10 lines.", 13, row)
    assert code == 1 and "past its row of 12" in out
    code, out = planted(tmp_path, "src/a.py", f"{TAG}: under 10 lines.", 11, row)
    assert code == 1 and "lower the row for src/a.py to 11" in out
    code, out = planted(tmp_path, "src/a.py", f"{TAG}: under 10 lines.", 10, row)
    assert code == 1 and "stale row" in out


def test_the_real_tree_budgets_are_true_or_listed():
    proc = subprocess.run([sys.executable, CHECKER], capture_output=True, text=True, cwd=REPO)
    assert proc.returncode == 0, proc.stdout
