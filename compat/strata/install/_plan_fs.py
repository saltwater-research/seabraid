# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Filesystem helpers for the planner (split out of plan.py for size).  UNTESTED-ON-HARDWARE.

OWNER     : mount matching and the "move the model to NVMe" advice text.
CONTRACT  : under(path, mount) -> bool;  move_advice(model_bytes, fs, headroom) -> str, where fs is plan's adapted
            filesystem tuples (mount, rotational, free_bytes, fstype).  stdlib only; imports nothing from plan.
FAILURE   : advice only; a missing figure yields "no non-rotational filesystem was detected", never a guess.
CHECKER   : tests/install/test_plan.py (rotational-disk and longest-prefix tests).
"""
from __future__ import annotations

GIB = 1 << 30


def under(path: str, mount: str) -> bool:
    p, m = path.replace("\\", "/").rstrip("/"), mount.rstrip("/")
    return m == "" or p == m or p.startswith(m + "/")


def move_advice(model_bytes: int, fs: list, headroom: float) -> str:
    need = int(model_bytes * (1 + headroom))
    real = [f for f in fs if f[1] is False and f[3] not in ("tmpfs", "devtmpfs", "ramfs", "squashfs") and f[2] is not None]
    fit = [f for f in real if f[2] >= need]
    if fit:
        best = max(fit, key=lambda f: f[2])
        return f"move the model to {best[0]} (non-rotational, {best[2] / GIB:.0f} GiB free)"
    top = max(real, key=lambda f: f[2], default=None)
    have = f"the largest non-rotational filesystem is {top[0]} with {top[2] / GIB:.0f} GiB free" if top else "no non-rotational filesystem was detected"
    return f"free space on an NVMe filesystem (model + {headroom:.0%} headroom = {need / GIB:.0f} GiB; {have})"
