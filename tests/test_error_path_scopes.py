# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""R8 L7: check_error_paths reads tools/, scripts/ (strict) and native-tests/ (ratchet), not only the old runtime scopes.

OWNER of the rule: checks/check_error_paths.py.  Each case plants a swallowed error in a temp tree and runs the real CLI.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
from conftest import REPO

CHECKER = os.path.join(REPO, "checks", "check_error_paths.py")
SWALLOW_PY = "try:\n    x = 1\nexcept OSError:\n    pass\n"
SWALLOW_C = "int f() { try { g(); } catch (...) { } return 0; }\n"


def run(root, *args: str) -> tuple[int, str]:
    proc = subprocess.run([sys.executable, CHECKER, "--root", str(root), *args], capture_output=True, text=True)
    return proc.returncode, proc.stdout + proc.stderr


def plant(root, rel: str, text: str) -> None:
    path = os.path.join(str(root), rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


@pytest.mark.parametrize("rel, text", [("tools/sim/x.py", SWALLOW_PY), ("scripts/y.py", SWALLOW_PY), ("tools/sim/z.cpp", SWALLOW_C)])
def test_tools_and_scripts_are_zero_under_strict_runtime(tmp_path, rel, text):
    plant(tmp_path, rel, text)
    code, out = run(tmp_path, "--strict-runtime")
    assert code == 1 and f"{rel}:" in out, out


def test_native_tests_are_a_ratchet_not_a_blind_spot(tmp_path):
    plant(tmp_path, "native-tests/t.cpp", SWALLOW_C)
    code, out = run(tmp_path, "--strict-runtime")
    assert code == 1 and "native-tests/t.cpp: 1 hit(s), baseline 0" in out, out
    plant(tmp_path, "checks/error_paths_baseline.json", '{"native-tests/t.cpp": 1}\n')
    assert run(tmp_path, "--strict-runtime")[0] == 0, "a known site at its baseline passes; only growth fails"


def test_the_wheel_script_announces_its_fallback_instead_of_swallowing_it():
    # the line the review found (scripts/build_wheel.py: `except ...: pass`) is now a logged fallback, so scripts/ is strict-clean
    code, out = run(REPO, "--strict-runtime")
    assert code == 0 and "scripts/build_wheel.py" not in out, out
