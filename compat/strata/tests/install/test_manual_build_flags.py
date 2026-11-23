# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""README's manual-build cmake flags must equal install/build.py configure_cmd's: one source of truth, checked.

WHY: the manual block once passed only -DGPU_TARGETS for HIP; the engine then defaulted to gfx1100 and failed to link.
The installer's flags were right and the README was not, so the README is now parsed and compared with them."""
from __future__ import annotations

import re
import shlex

import pytest

from install import build as B
from test_build import AMD, FS, NV, ROOT, plan

README = (ROOT / "README.md").read_text(encoding="utf-8")
WINDOW = 16


def readme_block(marker: str) -> str:
    m = re.search(rf"<!-- manual-cmake: {marker} -->\s*```bash\n(.*?)```", README, re.S)
    assert m, f"README has no manual-cmake block for {marker}"
    return m.group(1).replace("\\\n", " ")


def readme_cmake(marker: str) -> list[str]:
    """The `cmake -S` command of the fenced block that follows `<!-- manual-cmake: marker -->`, as an argv."""
    line = next(l for l in readme_block(marker).splitlines() if l.startswith("cmake -S"))
    return shlex.split(line)


def substitute(argv: list[str], values: dict[str, str]) -> list[str]:
    def sub(tok: str) -> str:
        for k, v in values.items():
            tok = re.sub(rf"\${k}\b|\${{{k}}}", lambda _m: v, tok)
        return tok
    return [sub(t) for t in argv]


def flags(argv) -> list[str]:
    """Everything but the source/build dir pair, which differ by design (the README names its own build dir)."""
    a, out, i = list(argv), [], 0
    while i < len(a):
        if a[i] in ("-S", "-B"):
            i += 2
            continue
        out.append(a[i])
        i += 1
    return out


def installer(b, vendor_flag: str):
    r = B.Runner(dry_run=True, echo=lambda s: None)
    B.build_all(plan(b, window=WINDOW), str(ROOT), "/work", r, jobs=2, exists=FS.__contains__)
    return next(c for c in r.commands if c.label.startswith("configure") and vendor_flag in str(c))


def test_hip_manual_flags_equal_configure_cmd():
    c = installer(AMD, "STRATA_ENABLE_HIP=ON")
    values = {"BL": str(ROOT), "SEABRAID": str(ROOT.parents[1]), "ARCH": "gfx1201", "ROCM_ROOT": AMD.toolkit_root,
              "HIP_CLANG": AMD.toolkit_root + "/llvm/bin/clang++", "WINDOW": str(WINDOW)}
    assert flags(substitute(readme_cmake("hip"), values)) == flags(c.argv), "README HIP flags drifted from configure_cmd"
    block = readme_block("hip")     # the exports the README carries must be the ones the installer sets
    assert "ROCM_PATH=$ROCM_ROOT" in block and c.env["ROCM_PATH"] == AMD.toolkit_root
    assert "HIP_PLATFORM=amd" in block and c.env["HIP_PLATFORM"] == "amd"


def test_cuda_manual_flags_equal_configure_cmd():
    c = installer(NV, "STRATA_ENABLE_CUDA=ON")
    values = {"BL": str(ROOT), "SEABRAID": str(ROOT.parents[1]), "ARCH": "86;89", "CUDA_ROOT": NV.toolkit_root, "NVCC": NV.compiler,
              "CUDA_HOST_CC": NV.host_compiler, "CUDA_HOST_CXX": B.cxx_for(NV.host_compiler), "WINDOW": str(WINDOW)}
    assert flags(substitute(readme_cmake("cuda"), values)) == flags(c.argv), "README CUDA flags drifted from configure_cmd"


def test_extractor_rejects_a_missing_block():
    with pytest.raises(AssertionError):
        readme_cmake("nonexistent")
