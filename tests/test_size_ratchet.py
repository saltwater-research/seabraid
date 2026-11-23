# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""R8 L6: a file over 400 lines FAILS unless ratcheted at its current length (shrink-only); .yml/.toml/.json are sized too.

Each case plants a file in a temp tree and runs the real checker over it.  OWNER of the rule: checks/check_size.py.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest
from conftest import REPO

CHECKER = os.path.join(REPO, "checks", "check_size.py")


def plant(root, rel: str, lines: int) -> None:
    path = os.path.join(str(root), rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("x = 1\n" * lines)


def ratchet(root, rows: dict) -> None:
    os.makedirs(os.path.join(str(root), "checks"), exist_ok=True)
    with open(os.path.join(str(root), "checks", "size_ratchet.json"), "w", encoding="utf-8") as fh:
        json.dump(rows, fh)


def run(root, *args: str) -> tuple[int, str]:
    proc = subprocess.run([sys.executable, CHECKER, "--root", str(root), *args], capture_output=True, text=True)
    return proc.returncode, proc.stdout + proc.stderr


def test_a_file_at_the_split_point_passes_and_one_line_over_fails(tmp_path):
    plant(tmp_path, "src/a.py", 400)
    assert run(tmp_path)[0] == 0
    plant(tmp_path, "src/a.py", 401)
    code, out = run(tmp_path)
    assert code == 1 and "src/a.py: 401 lines, over the 400-line split point and not in checks/size_ratchet.json" in out


def test_a_ratcheted_file_passes_at_exactly_its_length(tmp_path):
    plant(tmp_path, "src/a.py", 450)
    ratchet(tmp_path, {"src/a.py": 450})
    assert run(tmp_path)[0] == 0


def test_a_ratcheted_file_may_not_grow(tmp_path):
    plant(tmp_path, "src/a.py", 451)
    ratchet(tmp_path, {"src/a.py": 450})
    code, out = run(tmp_path)
    assert code == 1 and "past its ratchet of 450" in out


def test_a_ratcheted_file_that_shrank_must_lower_its_row(tmp_path):
    plant(tmp_path, "src/a.py", 420)
    ratchet(tmp_path, {"src/a.py": 450})
    code, out = run(tmp_path)
    assert code == 1 and "lower it" in out


def test_a_row_for_a_file_back_under_the_split_point_is_stale(tmp_path):
    plant(tmp_path, "src/a.py", 399)
    ratchet(tmp_path, {"src/a.py": 450})
    code, out = run(tmp_path)
    assert code == 1 and "ratchet row is stale" in out


def test_write_ratchet_lowers_and_drops_rows_but_never_adds_or_raises(tmp_path):
    plant(tmp_path, "src/lowered.py", 420)
    plant(tmp_path, "src/dropped.py", 100)
    plant(tmp_path, "src/raised.py", 480)
    plant(tmp_path, "src/new.py", 460)
    ratchet(tmp_path, {"src/lowered.py": 450, "src/dropped.py": 450, "src/raised.py": 430, "src/gone.py": 450})
    code, out = run(tmp_path, "--write-ratchet")
    rows = json.load(open(os.path.join(tmp_path, "checks", "size_ratchet.json"), encoding="utf-8"))
    assert rows == {"src/lowered.py": 420, "src/raised.py": 430}, rows
    assert "src/new.py: 460 lines, over the 400-line split point" in out and "src/raised.py: 480 lines, past its ratchet" in out and code == 1


@pytest.mark.parametrize("rel", [".github/workflows/ci.yml", "tools/data.toml", "tools/data.json", "tools/other.yaml"])
def test_config_and_data_files_are_sized_like_code(tmp_path, rel):
    plant(tmp_path, rel, 401)
    code, out = run(tmp_path)
    assert code == 1 and f"{rel}: 401 lines" in out


def test_the_real_tree_has_no_unratcheted_file_over_400():
    code, out = run(REPO)
    assert code == 0, out
