# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Toolkit probes for :mod:`detect`: CUDA, ROCm and SYCL.

The one decision worth defending here is that the CUDA host compiler is chosen
BY COMPILING, newest first, and a value is published only when a compile
returned 0.  The alternative -- a table of "nvcc X supports gcc <= Y" -- was
tried against the test system and was wrong: a recent ``gcc`` can be newer
than nvcc 12.4 accepts (anything past 13 is rejected), and a build configured from the table fails
at the first translation unit with ``#error -- unsupported GNU version!``.  The
probe costs about a second and removes the single most likely build failure.
"""

from __future__ import annotations

import glob
import logging
import os
import re
import shutil
import tempfile
from typing import Optional

from ._log import describe
from ._probe import _run

_log = logging.getLogger(__name__)

_TEST_CU = """#include <cstdio>
__global__ void seabraid_probe_kernel() {}
int main() { seabraid_probe_kernel<<<1, 1>>>(); printf("ok\\n"); return 0; }
"""


def _first_existing(paths) -> Optional[str]:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


# ---------------------------------------------------------------------- CUDA


def parse_nvcc_version(text: str) -> Optional[str]:
    """``nvcc --version`` -> "12.4".  Pure, for the fixture tests."""
    m = re.search(r"release\s+(\d+\.\d+)", text)
    return m.group(1) if m else None


def host_compiler_candidates() -> list[str]:
    """Every gcc on PATH, newest first.

    Newest first because a newer gcc is the better choice when nvcc accepts it;
    the loop stops at the first that compiles, so the result is "the newest
    supported", which is what a hand-written build line would have picked.
    """
    seen: dict[int, str] = {}
    plain = shutil.which("gcc")
    for d in os.environ.get("PATH", "").split(os.pathsep):
        if not d:
            continue
        for path in glob.glob(os.path.join(d, "gcc-[0-9]*")):
            m = re.search(r"gcc-(\d+)$", path)
            if m and os.access(path, os.X_OK):
                seen.setdefault(int(m.group(1)), path)
    out = [seen[v] for v in sorted(seen, reverse=True)]
    if plain and plain not in out:
        out.append(plain)
    return out


def verify_host_compiler(
    nvcc: str, candidates: list[str], archs: tuple = ()
) -> tuple[Optional[str], tuple[str, ...]]:
    """Compile a test file with each candidate; return the first that works.

    ``archs`` are the detected NVIDIA architectures, passed through so the probe
    also catches an nvcc too old for the card actually installed -- not only the
    host-compiler version gate.
    """
    tried: list[str] = []
    with tempfile.TemporaryDirectory(prefix="seabraid-cuda-probe-") as tmp:
        src = os.path.join(tmp, "probe.cu")
        with open(src, "w", encoding="utf-8") as fh:
            fh.write(_TEST_CU)
        for cc in candidates:
            tried.append(cc)
            cmd = [nvcc]
            for arch in archs:
                cmd += ["-arch=" + arch]
            cmd += ["-ccbin", cc, "-c", src, "-o", os.path.join(tmp, "probe.o")]
            # 90s: a first-time nvcc compile loads the whole toolchain, far slower than --version.
            if _run(cmd, timeout=90.0, cwd=tmp) is not None:
                return cc, tuple(tried)
    return None, tuple(tried)


def _install_root(tool: str) -> str:
    """The install root of a tool that lives in <root>/bin (the tool's own directory when it is not under bin)."""
    bindir = os.path.dirname(os.path.realpath(tool))
    return os.path.dirname(bindir) if os.path.basename(bindir) == "bin" else bindir


def _locate_nvcc(explicit: Optional[str]) -> Optional[str]:
    """nvcc: an explicit root, then the environment, then PATH, then the conventional install locations."""
    roots = [r for r in (explicit, *(os.environ.get(v) for v in ("CUDA_HOME", "CUDA_PATH", "CUDAToolkit_ROOT"))) if r]
    for root in roots:
        nvcc = _first_existing([os.path.join(root, "bin", "nvcc"), os.path.join(root, "nvcc")])
        if nvcc:
            return nvcc
    return shutil.which("nvcc") or _first_existing(sorted(glob.glob("/usr/local/cuda*/bin/nvcc"), reverse=True))


def find_cuda(explicit: Optional[str] = None, archs: tuple = ()):
    """The CUDA toolkit, or ``None``.  Searches an explicit root, then the
    environment, then PATH, then the conventional install locations."""
    from .detect import Toolkit  # noqa: PLC0415 -- cycle: detect owns the types

    nvcc = _locate_nvcc(explicit)
    if nvcc is None:
        return None
    root = _install_root(nvcc)
    out = _run([nvcc, "--version"], timeout=20.0)
    version = parse_nvcc_version(out) if out is not None else None
    chosen, tried = verify_host_compiler(nvcc, host_compiler_candidates(), archs)
    lib_dirs = tuple(
        d for d in (os.path.join(root, "lib64"), os.path.join(root, "lib")) if os.path.isdir(d)
    )
    return Toolkit(
        vendor="nvidia",
        kind="cuda",
        root=root,
        version=version,
        compiler=nvcc,
        host_compiler=chosen,
        host_compiler_tried=tried,
        lib_dirs=lib_dirs,
    )


# ---------------------------------------------------------------------- ROCm


def is_rocm_root(root: Optional[str]) -> bool:
    """A directory is a ROCm root only if it holds the compiler a build needs.

    This test exists because of a real failure, caught by running the probe on
    one host: ``ldconfig`` lists a distro ``libamdhip64.so`` under
    /usr/lib, from which the search derived the "root" /usr/lib -- a directory
    with no hipcc, no headers and no version.  Worse than useless: a plausible
    wrong answer SILENCED the blocker that would have told the user to point at
    the real install.  A root that cannot compile is not a root.
    """
    return bool(root) and os.path.exists(os.path.join(root, "bin", "hipcc"))


def rocm_root_from_ldconfig(text: str) -> Optional[str]:
    """Find a ROCm root from ``ldconfig -p`` output, or ``None``.

    Last resort, because ROCm may sit on a data mount, absent
    from PATH and from every conventional location, and the linker cache is the
    one place a non-standard install still registers itself.  The candidate is
    accepted only if :func:`is_rocm_root` passes, so a library without its
    toolchain yields nothing rather than a wrong answer.
    """
    for line in text.splitlines():
        m = re.search(r"=>\s*(\S*/libamdhip64\.so[^\s]*)", line)
        if not m:
            continue
        libdir = os.path.dirname(os.path.realpath(m.group(1)))
        for root in (os.path.dirname(libdir), os.path.dirname(os.path.dirname(libdir))):
            if is_rocm_root(root):
                return root
    return None


def rocm_root_from_tool(which=shutil.which) -> Optional[str]:
    """A ROCm root reached through one of its tools on PATH, or None.

    A TheRock tarball is commonly unpacked outside every conventional location and surfaced by symlinking
    its tools into ~/.local/bin -- which is a common layout.  Following such a link back to
    its real file and up out of bin/ finds the install that a path search cannot guess.  hipcc is tried
    first because it is the one a build actually needs; the query tools only point the way.
    """
    for tool in ("hipcc", "amd-smi", "rocminfo", "rocm-smi"):
        found = which(tool)
        if not found:
            continue
        bindir = os.path.dirname(os.path.realpath(found))
        root = os.path.dirname(bindir) if os.path.basename(bindir) == "bin" else bindir
        if is_rocm_root(root):
            return root
    return None


def _locate_hipcc(explicit: Optional[str]) -> Optional[str]:
    """hipcc: an explicit or environment root, then PATH, /opt, a TheRock tarball in a home directory, then the tool's own report."""
    for root in (explicit, os.environ.get("ROCM_PATH"), os.environ.get("ROCM_HOME")):
        if is_rocm_root(root):
            return os.path.join(root, "bin", "hipcc")
    found = (shutil.which("hipcc")
             or _first_existing(sorted(glob.glob("/opt/rocm*/bin/hipcc"), reverse=True))
             or _first_existing(sorted(glob.glob(os.path.expanduser("~/rocm*/bin/hipcc")), reverse=True)))
    if found:
        return found
    root_from_tool = rocm_root_from_tool()
    return os.path.join(root_from_tool, "bin", "hipcc") if root_from_tool else None


def _rocm_root_and_compiler(explicit: Optional[str]) -> Optional[tuple]:
    """(root, hipcc), or None.  With no hipcc anywhere the dynamic linker's cache may still name a root."""
    hipcc = _locate_hipcc(explicit)
    if hipcc is not None:
        return _install_root(hipcc), hipcc
    out = _run(["ldconfig", "-p"], timeout=20.0)
    root = rocm_root_from_ldconfig(out) if out is not None else None
    return None if root is None else (root, os.path.join(root, "bin", "hipcc"))


def _rocm_version(root: str) -> Optional[str]:
    """The ROCm version from its version file, else from the install path, else None."""
    vfile = _first_existing(
        [os.path.join(root, ".info", "version"), os.path.join(root, "share", "doc", "rocm", "VERSION")]
    )
    version = None
    if vfile:
        try:
            with open(vfile, "r", encoding="utf-8") as fh:
                version = fh.read().strip().split("-")[0] or None
        except OSError as exc:
            _log.warning("ROCm version file unreadable (%s); falling back to the version in the install path", describe(exc))
    m = re.search(r"(\d+\.\d+\.\d+)", root)
    return version or (m.group(1) if m else None)


def find_rocm(explicit: Optional[str] = None):
    """The ROCm installation, or ``None``.

    ``rocminfo`` and ``amd-smi`` are deliberately not consulted: they may be absent
    on a host that builds with ROCm.  What a build needs is ``hipcc`` and a lib directory.
    """
    from .detect import Toolkit  # noqa: PLC0415

    found = _rocm_root_and_compiler(explicit)
    if found is None:
        return None
    root, hipcc = found
    lib_dirs = tuple(
        d for d in (os.path.join(root, "lib"), os.path.join(root, "lib64")) if os.path.isdir(d)
    )
    return Toolkit(
        vendor="amd",
        kind="rocm",
        root=root,
        version=_rocm_version(root),
        compiler=hipcc,
        lib_dirs=lib_dirs,
    )


# ---------------------------------------------------------------------- SYCL


def find_sycl():
    from .detect import Toolkit  # noqa: PLC0415

    icpx = shutil.which("icpx")
    if icpx is None:
        return None
    bindir = os.path.dirname(os.path.realpath(icpx))
    root = os.path.dirname(bindir) if os.path.basename(bindir) == "bin" else bindir
    version = None
    out = _run([icpx, "--version"], timeout=20.0)
    if out is not None:
        m = re.search(r"(\d+\.\d+\.\d+)", out)
        version = m.group(1) if m else None
    lib_dirs = tuple(d for d in (os.path.join(root, "lib"),) if os.path.isdir(d))
    return Toolkit(
        vendor="intel", kind="sycl", root=root, version=version, compiler=icpx, lib_dirs=lib_dirs
    )


def _probe_cuda(gpus: tuple, cuda_path: Optional[str], out: list, blockers: list) -> None:
    from .detect import Blocker  # noqa: PLC0415

    nv_archs = tuple(sorted({g.arch for g in gpus if g.vendor == "nvidia" and g.arch}))
    tk = find_cuda(cuda_path, archs=nv_archs)
    if tk is None:
        blockers.append(
            Blocker(
                where="toolkit:cuda",
                what="an NVIDIA GPU is present but no CUDA toolkit was found",
                fix="install the CUDA toolkit, or pass --cuda-path <root containing bin/nvcc>",
            )
        )
        return
    out.append(tk)
    if tk.host_compiler is None:
        blockers.append(
            Blocker(
                where="toolkit:cuda",
                what="CUDA %s was found but none of its candidate host compilers (%s) "
                "compiled a test file"
                % (tk.version or "?", ", ".join(tk.host_compiler_tried) or "none tried"),
                fix="install a gcc this nvcc supports (nvcc --version names the ceiling) "
                "and make sure it is on PATH as gcc-<N>",
            )
        )


def _probe_rocm(gpus: tuple, rocm_path: Optional[str], out: list, blockers: list) -> None:
    from .detect import Blocker  # noqa: PLC0415

    tk = find_rocm(rocm_path)
    if tk is not None:
        out.append(tk)
        return
    archs = ", ".join(sorted({g.arch for g in gpus if g.vendor == "amd" and g.arch}))
    blockers.append(
        Blocker(
            where="toolkit:rocm",
            what="an AMD GPU (%s) is present but no ROCm root was found" % (archs or "unknown arch"),
            fix="pass --rocm-path <root containing bin/hipcc>, or set $ROCM_PATH; "
            "a TheRock wheel install is a valid root",
        )
    )


def _probe_sycl(out: list, blockers: list) -> None:
    from .detect import Blocker  # noqa: PLC0415

    tk = find_sycl()
    if tk is not None:
        out.append(tk)
        return
    blockers.append(
        Blocker(
            where="toolkit:sycl",
            what="an Intel GPU is present but no oneAPI/SYCL compiler was found",
            fix="source oneAPI setvars.sh, then rerun detection",
        )
    )


def probe_toolkits(
    gpus: tuple,
    blockers: list,
    rocm_path: Optional[str] = None,
    cuda_path: Optional[str] = None,
) -> tuple:
    """The toolkit each detected vendor needs; a missing one becomes a Blocker that names its fix."""
    out: list = []
    vendors = {g.vendor for g in gpus}
    if "nvidia" in vendors:
        _probe_cuda(gpus, cuda_path, out, blockers)
    if "amd" in vendors:
        _probe_rocm(gpus, rocm_path, out, blockers)
    if "intel" in vendors:
        _probe_sycl(out, blockers)
    return tuple(out)
