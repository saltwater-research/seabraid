# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""PLAN stage: a card excluded for lack of free VRAM names who holds it (detect's Gpu.holders), per card, never guessed."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from install import plan as P

G = lambda x: int(x * (1 << 30))  # noqa: E731


@dataclass(frozen=True)
class Gpu:
    index: int
    vendor: str
    name: str
    arch: Optional[str]
    vram_total_bytes: Optional[int]
    vram_free_bytes: Optional[int]
    holders: tuple = ()


@dataclass(frozen=True)
class Toolkit:
    vendor: str
    kind: str
    root: str
    version: str
    compiler: str
    host_compiler: Optional[str] = None
    lib_dirs: tuple = ()


@dataclass(frozen=True)
class Host:
    gpus: tuple
    toolkits: tuple = (Toolkit("amd", "rocm", "/opt/r", "7", "/opt/r/clang++"),
                       Toolkit("nvidia", "cuda", "/opt/c", "12", "/opt/c/nvcc", host_compiler="/usr/bin/gcc"))
    filesystems: tuple = ()
    blockers: tuple = ()


MODEL = P.ModelFacts(48, 512, G(50), G(3), 2560, 1510400)
H = lambda pid, gib, name: (pid, G(gib), name)  # noqa: E731


def _excluded(*gpus) -> dict:
    return dict(P.make_plan(Host(tuple(gpus)), MODEL, "/m.gguf").excluded)


def test_reason_names_holders_biggest_first_capped_at_three_with_more():
    held = (H(16488, 8.7, "python"), H(1203, 0.2, "Xorg"), H(77, 0.1, "obs"), H(5, 0.1, "x"), H(6, 0.1, "y"))
    why = _excluded(Gpu(0, "amd", "R9700", "gfx1201", G(32), G(1.2), held))[("amd", 0)]
    assert why.endswith("in use by python(16488) 8.7 GiB, Xorg(1203) 0.2 GiB, obs(77) 0.1 GiB, +2 more"), why
    assert "x(5)" not in why and "y(6)" not in why


def test_exactly_three_holders_has_no_more_suffix():
    held = (H(1, 2.0, "a"), H(2, 1.0, "b"), H(3, 0.5, "c"))
    why = _excluded(Gpu(0, "amd", "R9700", "gfx1201", G(32), G(1.2), held))[("amd", 0)]
    assert why.endswith("in use by a(1) 2.0 GiB, b(2) 1.0 GiB, c(3) 0.5 GiB") and "more" not in why


def test_no_holders_says_no_process_attributed():
    why = _excluded(Gpu(0, "amd", "R9700", "gfx1201", G(32), G(1.2)))[("amd", 0)]
    assert why.endswith("; no process attributed") and "in use by" not in why


def test_holders_on_one_vendors_card_zero_never_leak_to_the_others_card_zero():
    amd = Gpu(0, "amd", "R9700", "gfx1201", G(32), G(0.1), (H(111, 8.7, "amd_hog"),))
    nv = Gpu(0, "nvidia", "RTX 3060", "sm_86", G(12), G(0.1), (H(222, 3.1, "nv_hog"),))
    ex = _excluded(amd, nv)
    assert "amd_hog(111)" in ex[("amd", 0)] and "nv_hog" not in ex[("amd", 0)]
    assert "nv_hog(222)" in ex[("nvidia", 0)] and "amd_hog" not in ex[("nvidia", 0)]
