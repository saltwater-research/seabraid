# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""ROCm sysdeps probe shared by CONFIG and BUILD: a leaf that imports no installer stage.

OWNER     : this file owns "which TheRock rocm_sysdeps dirs exist under a ROCm tree". config.py (runtime lib dirs) and
            _build_rocm.py (configure prefix path + build/runtime LD_LIBRARY_PATH) both call it; neither recomputes it.
CONTRACT  : sysdeps_dirs(root, exists) -> (prefix dir | None, lib dir | None);
            rocm_lib_dirs(detected, exists, root="") -> tuple.  Arguments are str / tuple / Callable only.
WHY A LEAF: stage order is detect <- plan <- config <- build/services <- setup. Both config and build need this probe, so
            it lives below both and imports nothing of theirs; a module that depends on nothing cannot close a cycle.
FAILURE   : none of its own: a missing dir is simply not returned (absence is a valid layout, /opt/rocm has no sysdeps).
CHECKER   : tests/install/test_rocm.py (behaviour, and that this file imports no installer module).
"""
from __future__ import annotations

from typing import Callable


def sysdeps_dirs(root: str, exists: Callable[[str], bool]) -> tuple:
    """(prefix dir, lib dir) of TheRock's bundled system libs, each only if it exists on disk.

    WHY by existence: the /opt/rocm layout has no rocm_sysdeps and must keep working; a TheRock tree needs it on both the
    CMake prefix path and LD_LIBRARY_PATH or the engine fails to load its deps at start.
    """
    base = root.rstrip("/") + "/lib/rocm_sysdeps"
    return base if exists(base) else None, (base + "/lib") if exists(base + "/lib") else None


def rocm_lib_dirs(detected: tuple, exists: Callable[[str], bool], root: str = "") -> tuple:
    """Runtime library dirs: what detection found, then each <lib>/rocm_sysdeps/lib that exists (root/lib is also probed)."""
    out = list(detected)
    for d in [*detected, *([root.rstrip("/") + "/lib"] if root else [])]:
        extra = d.rstrip("/") + "/rocm_sysdeps/lib"
        if extra not in out and exists(extra):
            out.append(extra)
    return tuple(out)
