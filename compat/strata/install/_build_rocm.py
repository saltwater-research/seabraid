# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""ROCm layout for the BUILD stage and its services: which HIP clang driver, CMake prefix path and runtime lib dirs.

OWNER     : this file owns "given a planned rocm Build, which driver / prefix path / lib dirs". build.py (configure +
            build env) and services.py (runtime LD_LIBRARY_PATH) call it. The sysdeps probe itself lives in the leaf
            _rocm.py, because config.py needs it too and config must not import a BUILD-stage module.
CONTRACT  : rocm_layout(b, exists=os.path.exists) -> RocmLayout.  `exists` is the only filesystem access (injectable, so
            tests use fake roots).  Imports: plan's types and _rocm only.
FAILURE   : BuildError naming EVERY path tried when no clang++ driver is found.  Never falls back to hipcc.
CHECKER   : tests/install/test_build.py (TheRock-shaped, /opt/rocm-shaped and hipcc-only roots).
"""
from __future__ import annotations

import os
from typing import Callable

from ._rocm import rocm_lib_dirs, sysdeps_dirs
from .plan import Build

# Joined with "/" not os.path.join: these name files on the Linux host being configured, not on the control machine.
_CLANG_REL = ("lib/llvm/bin/clang++", "llvm/bin/clang++")      # TheRock first, then the classic /opt/rocm layout


class RocmError(Exception):
    """Raised for an unusable ROCm tree; build.py re-raises it as BuildError (this module must not import build)."""


class RocmLayout:
    def __init__(self, clang: str, prefix_path: tuple, lib_dirs: tuple):
        self.clang, self.prefix_path, self.lib_dirs = clang, prefix_path, lib_dirs


def rocm_layout(b: Build, exists: Callable[[str], bool] = os.path.exists) -> RocmLayout:
    """WHY clang++ and not hipcc: CMake's HIP language needs the clang driver; the hipcc wrapper breaks its compiler probe."""
    root = b.toolkit_root.rstrip("/")
    tried = []
    clang = None
    if os.path.basename(b.compiler or "") == "clang++":          # detection already found a driver: keep it if it is real
        tried.append(b.compiler)
        clang = b.compiler if exists(b.compiler) else None
    for rel in _CLANG_REL if clang is None else ():
        cand = f"{root}/{rel}"
        tried.append(cand)
        if exists(cand):
            clang = cand
            break
    if clang is None:
        raise RocmError(f"{b.vendor} ROCm at {root}: no clang++ driver for CMake's HIP language (hipcc is not accepted); tried: "
                        + ", ".join(tried))
    prefix_dir, _ = sysdeps_dirs(root, exists)
    return RocmLayout(clang, (root, *([prefix_dir] if prefix_dir else [])), rocm_lib_dirs(b.lib_dirs, exists, root))
