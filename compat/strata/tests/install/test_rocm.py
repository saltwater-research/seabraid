# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""install/_rocm.py: the sysdeps probe CONFIG and BUILD share, asserted on the function itself (not through a stage)."""
from __future__ import annotations

import ast
from pathlib import Path

from install._rocm import rocm_lib_dirs, sysdeps_dirs

_ROCM_SRC = Path(__file__).resolve().parents[2] / "install" / "_rocm.py"


def test_opt_rocm_tree_without_sysdeps_returns_exactly_the_detected_dirs():
    present = {"/opt/rocm", "/opt/rocm/lib", "/opt/rocm/llvm/bin/clang++"}
    assert rocm_lib_dirs(("/opt/rocm/lib",), present.__contains__, "/opt/rocm") == ("/opt/rocm/lib",)
    assert sysdeps_dirs("/opt/rocm", present.__contains__) == (None, None)


def test_therock_tree_appends_sysdeps_lib_once():
    present = {"/tr/lib", "/tr/lib/rocm_sysdeps", "/tr/lib/rocm_sysdeps/lib"}
    assert rocm_lib_dirs(("/tr/lib",), present.__contains__, "/tr") == ("/tr/lib", "/tr/lib/rocm_sysdeps/lib")
    assert sysdeps_dirs("/tr/", present.__contains__) == ("/tr/lib/rocm_sysdeps", "/tr/lib/rocm_sysdeps/lib")


def test_rocm_is_a_leaf_that_imports_no_installer_module():
    """config imports this file, so it must import nothing of any stage (or config would reach upward)."""
    tree = ast.parse(_ROCM_SRC.read_text(encoding="utf-8"))
    bad = [ast.unparse(n) for n in ast.walk(tree)
           if (isinstance(n, ast.ImportFrom) and (n.level or (n.module or "").startswith("install")))
           or (isinstance(n, ast.Import) and any(a.name.startswith("install") for a in n.names))]
    assert bad == []
