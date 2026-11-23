# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""seabraid installer, stage 4 (BUILD): fetch Strata, apply our patches, build one tree per vendor.  UNTESTED-ON-HARDWARE.

OWNER     : this file owns "how a Plan becomes binaries": the STRATA_BASE pin read, the patch application, and every
            CMake option and built artifact layout.  Engine/helper targets come from the plan; opt-in file moves add
            the sums tool to the main tree.  No arch or toolkit path
            literal lives here.  BuildResult is the type it produces.
CONTRACT  : build_all(plan, project_dir, work_dir, runner, ...) -> BuildResult;  Runner (the single exec point);
            strata_setup_cmd() builds a call of Strata's OWN setup.py (model download / pack / tokenizer are not
            reimplemented).  Imports: plan's types only (detect <- plan <- config <- build/services <- setup).
FAILURE   : fail fast with BuildError naming the step (the patch file for a git-am conflict, the vendor for a missing
            toolkit piece).  A failed `git am` is aborted so the tree is not left half-applied.  Dry-run runs nothing.
CHECKER   : tests/install/test_build.py (captured dry-run command lists; one assertion sabotaged and seen to fail).
Linux first; Windows is out of scope (target paths are POSIX).  Parallel jobs = cores (os.cpu_count()), stated in the printed plan.
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath as PP
from typing import Callable, Mapping, Optional, Sequence

from ._build_rocm import RocmError, rocm_layout
from ._capture import Captured, capture
from ._log import scrub
from .plan import Build, Plan

log = logging.getLogger(__name__)
STRATA_REPO = "https://github.com/Niko1221/Strata"      # default source; override for a mirror (air-gapped hosts)
# -DSTRATA_WINDOW_MAX_T comes from Plan.window_max_t (one owner, also sizes the shm channel) and goes to EVERY build.
_GIT_ID = ("-c", "user.name=seabraid-installer", "-c", "user.email=installer@localhost")   # `git am` needs a committer


class BuildError(Exception):
    pass


@dataclass(frozen=True)
class Cmd:
    argv: tuple
    cwd: Optional[str] = None
    env: Mapping = field(default_factory=dict)      # additions to the process environment
    label: str = ""

    def __str__(self) -> str:
        env = " ".join(f"{k}={v}" for k, v in self.env.items())
        return (f"(cd {self.cwd}) " if self.cwd else "") + (env + " " if env else "") + " ".join(self.argv)


class Runner:
    """The one place a command is executed.  dry_run=True records and prints but never runs (tests read .commands)."""

    # WHY dry_run has NO default: it used to default to True, so a caller that forgot it got a silent no-op (a real
    # install printed "[dry-run]" and built nothing).  Required means a forgotten argument is a TypeError at the call site.
    def __init__(self, dry_run: bool, echo: Callable[[str], None] = print):
        self.dry_run, self.echo, self.commands = dry_run, echo, []

    def run(self, cmd: Cmd) -> int:
        self.commands.append(cmd)
        self.echo(("[dry-run] " if self.dry_run else "[run] ") + str(cmd))
        if self.dry_run:
            return 0
        env = {**os.environ, **{k: str(v) for k, v in cmd.env.items()}}
        return subprocess.run(list(cmd.argv), cwd=cmd.cwd, env=env).returncode

    def run_capture(self, cmd: Cmd, until: str, timeout: float) -> Captured:
        """Like run, but returns the output and stops the process at the first line containing `until` (see _capture).
        Dry-run executes nothing and returns an empty, not-found capture: callers must treat that as 'no data'."""
        self.commands.append(cmd)
        self.echo(("[dry-run] " if self.dry_run else "[run] ") + str(cmd))
        if self.dry_run:
            return Captured(0, "", False, False)
        return capture(cmd.argv, cmd.cwd, cmd.env, until, timeout)


@dataclass(frozen=True)
class BuildResult:
    strata_dir: str
    server_exe: str                 # the main engine binary
    helper_exes: Mapping            # vendor -> seabraid-helper binary
    build_dirs: Mapping             # vendor -> build dir
    jobs: int


# WHY a full 40-hex SHA only: a tag or branch can be moved after review, and an abbreviated SHA can become ambiguous as
# the repository grows; only the full object name makes "the verified Strata" mean one tree for everyone who builds it.
_FULL_SHA = re.compile(r"[0-9a-fA-F]{40}")


def read_strata_base(project_dir: str) -> str:
    """The pinned Strata commit, read from patches/strata/STRATA_BASE (its single owner).  No pin, or a pin that is
    not a full 40-hex SHA (a tag, a branch, a short SHA), = refuse."""
    f = Path(project_dir) / "patches" / "strata" / "STRATA_BASE"
    words = f.read_text().split() if f.is_file() else []
    if not words:
        log.error("no Strata pin: patches/strata/STRATA_BASE is missing or empty")
        raise BuildError("no Strata pin: patches/strata/STRATA_BASE is missing or empty")
    if not _FULL_SHA.fullmatch(words[0]):
        log.error("Strata pin %r is not a full 40-hex commit SHA; refusing tags, branches and short SHAs", scrub(words[0], 60))
        raise BuildError("patches/strata/STRATA_BASE must hold a full 40-hex commit SHA, not a tag, branch or short SHA")
    return words[0]


def normalize_arch(kind: str, arch: str) -> str:
    """Detected arch -> the CMake spelling.  cuda: 'sm_86' / 'compute_86' -> '86' (suffix letters kept, '90a');
    rocm: 'gfx1201:xnack-' -> 'gfx1201'.  Anything else is refused: a wrong arch builds the wrong kernels silently."""
    if kind == "cuda":
        m = re.fullmatch(r"(?:sm_|compute_)?(\d+[a-z]?)", arch)
    else:
        m = re.fullmatch(r"(gfx[0-9a-f]+)(?::.*)?", arch)
    if not m:
        raise BuildError(f"unrecognised {kind} architecture {arch!r}: refusing to guess compile targets")
    return m.group(1)


def _layout(b: Build, exists: Callable[[str], bool]):
    try:
        return rocm_layout(b, exists)
    except RocmError as e:
        raise BuildError(str(e)) from None


def _env(b: Build, exists: Callable[[str], bool] = os.path.exists) -> dict:
    env, libs = {}, b.lib_dirs
    if b.kind == "rocm":
        env.update(ROCM_PATH=b.toolkit_root, HIP_PLATFORM="amd")
        libs = _layout(b, exists).lib_dirs          # same dirs the services get: one owner (_build_rocm)
    if libs:
        env["LD_LIBRARY_PATH"] = ":".join([*libs, os.environ.get("LD_LIBRARY_PATH", "")]).rstrip(":")
    return env


def cxx_for(host_compiler: str) -> str:
    """The g++ that pairs with the CUDA host gcc: gcc-13 -> g++-13, /usr/bin/gcc -> /usr/bin/g++.  Derived from the one
    detected host compiler, never detected again, so the three compilers cannot disagree."""
    d, base = host_compiler.rpartition("/")[0], host_compiler.rpartition("/")[2]      # POSIX paths, also when tested on Windows
    return (d + "/" if d else "") + re.sub(r"^gcc", "g++", base)


def configure_cmd(b: Build, strata_dir: str, build_dir: str, project_dir: str, ggml_dir: Optional[str], window_max_t: int,
                  exists: Callable[[str], bool] = os.path.exists) -> Cmd:
    if b.kind == "cuda" and not b.host_compiler:     # nvcc rejects a too-new default gcc: never fall back to it
        raise BuildError(f"no working CUDA host compiler for {b.vendor}; tried: {', '.join(b.host_compiler_tried) or 'nothing'}")
    if not b.toolkit_root or not b.compiler:
        raise BuildError(f"{b.vendor} {b.kind} toolkit root/compiler missing from detection: cannot configure")
    archs = ";".join(normalize_arch(b.kind, a) for a in b.archs)
    a = ["cmake", "-S", strata_dir, "-B", build_dir, "-DCMAKE_BUILD_TYPE=Release", "-DSTRATA_BUILD_TESTS=OFF",
         # the Strata patch adds <SEABRAID_STRATA_SOURCE_DIR>/strata-ext itself, so this is the project ROOT
         f"-DSEABRAID_STRATA_SOURCE_DIR={project_dir}",
         # The v7 ladder adapter consumes the full checkout, while the helper consumes compat/strata.
         f"-DSTRATA_SEABRAID_SOURCE_DIR={Path(project_dir).parents[1]}"]
    if b.kind == "cuda":
        host_cxx = cxx_for(b.host_compiler)
        a += ["-DSTRATA_ENABLE_CUDA=ON", "-DSTRATA_ENABLE_HIP=OFF", f"-DCMAKE_CUDA_ARCHITECTURES={archs}",
              f"-DCMAKE_CUDA_COMPILER={b.compiler}", f"-DCUDAToolkit_ROOT={b.toolkit_root}",
              # Keep one GCC version across C, C++ and CUDA to avoid ABI mismatches.
              # CUDA-only executables also need g++ as the host link driver for libstdc++.
              f"-DCMAKE_CUDA_HOST_COMPILER={host_cxx}",
              f"-DCMAKE_C_COMPILER={b.host_compiler}", f"-DCMAKE_CXX_COMPILER={host_cxx}",
              # WHY: since v0.2.14 STRATA_MMQ defaults to OFF (Strata's own prompt kernels alone); through v0.2.13.1 every CUDA build had
              # llama.cpp's MMQ prompt kernels.  ON keeps the MMQ prompt path that patch 0013 (D4) hooks via mmq_plan().
              "-DSTRATA_MMQ=ON"]
    else:
        lay = _layout(b, exists)
        # GPU_TARGETS next to CMAKE_HIP_ARCHITECTURES: the working TheRock cache carries both with the same value.
        a += ["-DSTRATA_ENABLE_HIP=ON", "-DSTRATA_ENABLE_CUDA=OFF", f"-DCMAKE_HIP_ARCHITECTURES={archs}", f"-DGPU_TARGETS={archs}",
              f"-DCMAKE_HIP_COMPILER={lay.clang}", f"-DCMAKE_HIP_COMPILER_ROCM_ROOT={b.toolkit_root}",
              "-DCMAKE_PREFIX_PATH=" + ";".join(lay.prefix_path),
              # WHY: Strata's own setup.py always builds HIP with these three (v0.2.12 setup.py:5246).  v0.2.12 calls
              # fused::want_dm_f16 / dm_f16_used from prefill.cpp without stubbing them for STRATA_PREFILL_MMQ=OFF (the
              # CMake default), so a HIP `strata` without PREFILL_MMQ fails to link.  Match the release config, not CMake's.
              # v0.2.14+: STRATA_PREFILL_MMQ builds the MMQ path only together with STRATA_MMQ=ON (default OFF).  Upstream's own release
              # build passes STRATA_PREFILL_MMQ=ON STRATA_MMQ=OFF (its own prompt kernels only); we pass ON/ON to keep llama.cpp's MMQ prompt
              # path that patch 0013 (D4) hooks, the equivalent of the v0.2.13.1 build the patches were validated against.
              "-DSTRATA_PREFILL_MMQ=ON", "-DSTRATA_MMQ=ON", "-DSTRATA_Q6K_EXPERTS=ON", "-DSTRATA_MMQ_KQUANTS=ON"]
    # WHY helpers inherit MAIN's window: patch 0001 Channel::connect() REFUSES a window_max_t mismatch ("seabraid channel geometry
    # mismatch"), so per-vendor defaults (CUDA 8 vs HIP 16) would never connect.  Do not "simplify" this back to per-build defaults.
    a.append(f"-DSTRATA_WINDOW_MAX_T={window_max_t}")
    if ggml_dir:
        a.append(f"-DSTRATA_GGML_DIR={ggml_dir}")
    return Cmd(tuple(a), env=_env(b, exists), label=f"configure {b.vendor}")


def build_cmd(b: Build, build_dir: str, jobs: int, exists: Callable[[str], bool] = os.path.exists,
              *, file_moves: bool = False) -> Cmd:
    argv = ["cmake", "--build", build_dir, "--parallel", str(jobs)]
    for t in build_targets(b, file_moves):
        argv += ["--target", t]
    return Cmd(tuple(argv), env=_env(b, exists), label=f"build {b.vendor}")


def build_targets(b: Build, file_moves: bool) -> tuple[str, ...]:
    """One main-engine tree supplies the per-pack tool; helpers do not need another copy."""
    return (*b.targets, "seabraid-sums") if file_moves and "strata" in b.targets else b.targets


def strata_setup_cmd(strata_dir: str, args: Sequence[str]) -> Cmd:
    """Strata's own setup.py (model download, pack, tokenizer).  `args` are its flags (--model, --gguf-dir,
    --data-dir, --prepare-pack ...); the setup CLI owner chooses them.  --no-start is always added: seabraid's
    services start the server, not Strata."""
    return Cmd(("python3", str(PP(strata_dir) / "setup.py"), "--yes", "--no-start", *args), label="strata setup")


def patch_files(project_dir: str) -> tuple[Path, ...]:
    """Match CI's ordered compatibility series followed by the required v7 ladder integration."""
    root = Path(project_dir)
    patches = tuple(sorted((root / "patches" / "strata").glob("*.patch")))
    if not patches:
        raise BuildError("no patches/strata/*.patch found")
    adapter = root.parents[1] / "adapters" / "strata" / "strata-v7-seabraid.patch"
    if not adapter.is_file():
        raise BuildError(f"required Strata v7 integration patch missing: {adapter}")
    return (*patches, adapter)


def fetch_and_patch(project_dir: str, strata_dir: str, runner: Runner, repo: str = STRATA_REPO) -> None:
    base = read_strata_base(project_dir)
    if not runner.dry_run and Path(strata_dir).exists() and any(Path(strata_dir).iterdir()):
        raise BuildError(f"{strata_dir} already exists and is not empty: choose a fresh work dir (patches apply once)")
    patches = patch_files(project_dir)
    steps = [Cmd(("git", "clone", repo, strata_dir), label="fetch Strata"),
             Cmd(("git", "-C", strata_dir, "checkout", "--detach", base), label=f"pin Strata {base[:12]}")]
    for s in steps:
        if runner.run(s):
            raise BuildError(f"{s.label} failed")
    for p in patches:
        if runner.run(Cmd(("git", "-C", strata_dir, *_GIT_ID, "am", str(p)), label=f"apply {p.name}")):
            runner.run(Cmd(("git", "-C", strata_dir, "am", "--abort"), label="abort git am"))
            raise BuildError(f"patch {p.name} does not apply to Strata {base[:12]}; tree restored, nothing built")


def built_exe_candidates(build_dir: str, name: str) -> tuple[str, ...]:
    """CMake's current output first, then layouts supported for previous builds (POSIX installer paths)."""
    subdirs = ("",) if name == "strata" else ("seabraid-strata-ext", "seabraid", "", "bin")
    return tuple(str(PP(Path(build_dir).as_posix(), subdir, name)) for subdir in subdirs)


def find_built_exe(build_dir: str, name: str, exists: Callable[[str], bool] = os.path.isfile) -> str:
    """Resolve prior or freshly built artifacts through the build owner's single layout definition."""
    return next((path for path in built_exe_candidates(build_dir, name) if exists(path)), "")


def find_built_exes(strata_dir: str, name: str) -> dict[str, str]:
    """Existing outputs by vendor, shared by prebuilt helpers and the manifest tool lookup."""
    found = {}
    for directory in sorted(Path(strata_dir).glob("build-*")):
        path = find_built_exe(str(directory), name)
        if path:
            found[directory.name.removeprefix("build-")] = os.path.normpath(path)
    return found


def _find_exe(runner: Runner, build_dir: str, name: str, exists: Callable[[str], bool]) -> str:
    """Missing build artifacts retain their expected path so verification names the missing output."""
    expected = built_exe_candidates(build_dir, name)[0]
    return expected if runner.dry_run else find_built_exe(build_dir, name, exists) or expected


def _build_vendor(b: Build, strata_dir: str, project_dir: str, ggml_dir: Optional[str], window_max_t: int, jobs: int,
                  runner: Runner, exists: Callable[[str], bool], *, file_moves: bool = False) -> tuple:
    """Configure and build one vendor's tree; returns (build_dir, strata exe or "", seabraid-helper exe or "")."""
    bd = str(PP(strata_dir) / f"build-{b.vendor}")
    for c in (configure_cmd(b, strata_dir, bd, project_dir, ggml_dir, window_max_t, exists),
              build_cmd(b, bd, jobs, exists, file_moves=file_moves)):
        if runner.run(c):
            raise BuildError(f"{c.label} failed (command: {c})")
    outputs = _vendor_outputs(b, bd, runner, exists, file_moves)
    return bd, outputs.get("strata", ""), outputs.get("seabraid-helper", "")


def _vendor_outputs(b: Build, build_dir: str, runner: Runner, exists: Callable[[str], bool],
                    file_moves: bool) -> dict[str, str]:
    """Verify every requested artifact, including optional tools, before the build can be consumed."""
    outputs = {name: _find_exe(runner, build_dir, name, exists) for name in build_targets(b, file_moves)}
    _verify_outputs(runner, "", outputs, exists)
    return outputs


def _verify_outputs(runner: Runner, server: str, helpers: Mapping, exists: Callable[[str], bool]) -> None:
    # WHY: exit code 0 is not proof a binary exists; a BuildResult naming paths nothing built made a real install write
    # a config pointing at nothing.  Dry-run built nothing, so there is nothing to check.
    if runner.dry_run:
        return
    gone = [p for p in ([server] if server else []) + list(helpers.values()) if not exists(p)]
    if gone:
        raise BuildError("build reported success but these outputs do not exist: " + ", ".join(gone))


def build_all(plan: Plan, project_dir: str, work_dir: str, runner: Runner, repo: str = STRATA_REPO,
              ggml_dir: Optional[str] = None, jobs: Optional[int] = None,
              exists: Callable[[str], bool] = os.path.exists, *, file_moves: bool = False) -> BuildResult:
    """Build each vendor under work_dir/strata; file_moves also requests and verifies the main tree's sums tool."""
    if not plan.ready:
        raise BuildError("plan is not ready; refusing to build:\n" + plan.explain())
    if plan.window_max_t <= 0:
        raise BuildError("plan.window_max_t is unset; main and helpers must be built with one explicit window")
    jobs = jobs or os.cpu_count() or 1
    runner.echo(f"parallel jobs = {jobs} (all cores)")
    strata_dir = str(PP(work_dir) / "strata")
    fetch_and_patch(project_dir, strata_dir, runner, repo)
    dirs, helpers, server = {}, {}, ""
    for b in plan.builds:
        dirs[b.vendor], exe, helper = _build_vendor(b, strata_dir, project_dir, ggml_dir, plan.window_max_t, jobs, runner, exists,
                                                   file_moves=file_moves)
        server = exe or server
        if helper:
            helpers[b.vendor] = helper
    _verify_outputs(runner, server, helpers, exists)
    return BuildResult(strata_dir, server, helpers, dirs, jobs)
