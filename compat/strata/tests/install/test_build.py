# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Tests for install/build.py: dry-run command lists on fixture plans.  Nothing is executed."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from install import build as B
from install.plan import Build, ModelFacts, Plan, Role, TierSizes

ROOT = Path(__file__).resolve().parents[2]
AMD = Build("amd", "rocm", ("strata",), ("gfx1201",), (0,), "/rocm-x", "7.13", "/rocm-x/llvm/bin/clang++", None, ("/rocm-x/lib",))
NV = Build("nvidia", "cuda", ("seabraid-helper",), ("sm_86", "sm_89"), (1, 2), "/cuda-y", "12.4", "/cuda-y/bin/nvcc", "/usr/bin/gcc-13", ())
NV_MAIN = Build("nvidia", "cuda", ("strata",), ("sm_86",), (0,), "/cuda-y", "12.4", "/cuda-y/bin/nvcc", "/usr/bin/gcc-13", ())


# Fake filesystem: only these paths "exist".  Every rocm root used below needs a clang++ driver listed here.
HOME_ROOT = "/home"   # a variable, so no literal developer path sits in this file
FS = {"/rocm-x/llvm/bin/clang++", "/elsewhere/rocm/clang++", HOME_ROOT + "/dev/rocm/clang++"}
THEROCK = {"/tr/lib/llvm/bin/clang++", "/tr/lib/rocm_sysdeps", "/tr/lib/rocm_sysdeps/lib"}     # TheRock layout (the test system)
OPT = {"/opt/rocm-y/llvm/bin/clang++"}                                                      # classic /opt/rocm layout


def plan(*builds, window=16) -> Plan:
    m = ModelFacts(3, 5, 100, 10, 8)
    role = Role("main", 0, builds[0].vendor, "c", builds[0].archs[0], 0, None, 1)
    return Plan(m, "all-vram", (role,), tuple(builds), TierSizes({(builds[0].vendor, 0): 100}, 0, 0, None), window_max_t=window)


def run(p: Plan, **kw):
    r = B.Runner(dry_run=True, echo=lambda s: None)
    fs = kw.pop("fs", FS)
    res = B.build_all(p, str(ROOT), "/work", r, jobs=kw.pop("jobs", 12), exists=fs.__contains__, **kw)
    return r, res


def cfg(r: B.Runner, vendor_flag: str) -> str:
    return next(str(c) for c in r.commands if c.label.startswith("configure") and vendor_flag in str(c))


def test_amd_main_plus_nvidia_helper_commands():
    r, res = run(plan(AMD, NV))
    builds = [str(c) for c in r.commands if c.label.startswith("build")]
    assert "--target strata" in builds[0] and "--target seabraid-helper" in builds[1]
    assert all("--parallel 12" in b for b in builds)
    assert res.build_dirs["amd"] != res.build_dirs["nvidia"]
    assert res.helper_exes == {"nvidia": "/work/strata/build-nvidia/seabraid-strata-ext/seabraid-helper"}


def test_amd_main_configure_uses_detected_rocm_layout():
    r, _ = run(plan(AMD, NV))
    amd = cfg(r, "STRATA_ENABLE_HIP=ON")
    assert f"-DSEABRAID_STRATA_SOURCE_DIR={ROOT}" in amd
    assert "-DCMAKE_HIP_ARCHITECTURES=gfx1201" in amd
    assert "-DCMAKE_HIP_COMPILER=/rocm-x/llvm/bin/clang++" in amd
    assert "ROCM_PATH=/rocm-x" in amd
    assert "-DCMAKE_PREFIX_PATH=/rocm-x" in amd.split()
    assert "-DGPU_TARGETS=gfx1201" in amd.split()


def test_mixed_builds_share_the_window_and_keep_mmq_enabled():
    r, _ = run(plan(AMD, NV))
    amd, nv = cfg(r, "STRATA_ENABLE_HIP=ON"), cfg(r, "STRATA_ENABLE_CUDA=ON")
    assert "-DSTRATA_WINDOW_MAX_T=16" in amd and "-DSTRATA_WINDOW_MAX_T=16" in nv    # AMD main + NVIDIA helper: both 16 (the test system)
    assert "-DCMAKE_CUDA_ARCHITECTURES=86;89" in nv
    assert "-DCUDAToolkit_ROOT=/cuda-y" in nv
    # Strata v0.2.12's HIP `strata` only links with PREFILL_MMQ on (missing fused:: stubs); setup.py's release flags
    assert all(f in amd.split() for f in ("-DSTRATA_PREFILL_MMQ=ON", "-DSTRATA_Q6K_EXPERTS=ON", "-DSTRATA_MMQ_KQUANTS=ON"))
    # v0.2.14+ defaults STRATA_MMQ to OFF; patch 0013 (D4) hooks the MMQ prompt path, so both the HIP and the CUDA build ask for it
    assert "-DSTRATA_MMQ=ON" in amd.split() and "-DSTRATA_MMQ=ON" in nv.split()


def test_cuda_configure_pins_c_and_cxx_to_the_cuda_host_compiler():
    r, _ = run(plan(AMD, NV))
    nv = cfg(r, "STRATA_ENABLE_CUDA=ON").split()
    assert "-DCMAKE_C_COMPILER=/usr/bin/gcc-13" in nv and "-DCMAKE_CXX_COMPILER=/usr/bin/g++-13" in nv
    assert "-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13" in nv
    assert B.cxx_for("gcc") == "g++" and B.cxx_for("/opt/x/gcc-15") == "/opt/x/g++-15"
    assert not any(a.startswith("-DCMAKE_C_COMPILER") for a in cfg(r, "STRATA_ENABLE_HIP=ON").split())   # HIP keeps its own


def test_every_configure_gets_exactly_one_and_the_same_explicit_window():
    amd_helper = Build("amd", "rocm", ("seabraid-helper",), ("gfx1201",), (1,), "/rocm-x", "7.13", "/rocm-x/llvm/bin/clang++", None, ("/rocm-x/lib",))
    r, _ = run(plan(NV_MAIN, amd_helper, window=8))
    cmds = [str(c) for c in r.commands if c.label.startswith("configure")]
    assert len(cmds) == 2 and all(c.split().count("-DSTRATA_WINDOW_MAX_T=8") == 1 for c in cmds)
    assert not any("-DSTRATA_WINDOW_MAX_T=auto" in c for c in cmds)


def test_nvidia_main_amd_helper_both_get_the_main_window_and_override_wins():
    amd_helper = B.Build("amd", "rocm", ("seabraid-helper",), ("gfx1201",), (0,), "/rocm-x", "7", "/rocm-x/llvm/bin/clang++", None, ())
    r, _ = run(plan(NV_MAIN, amd_helper, window=8))
    assert "-DSTRATA_WINDOW_MAX_T=8" in cfg(r, "STRATA_ENABLE_CUDA=ON") and "-DSTRATA_WINDOW_MAX_T=8" in cfg(r, "STRATA_ENABLE_HIP=ON")
    r, _ = run(plan(NV_MAIN, amd_helper, window=12))                      # an explicit override, already resolved into the plan
    assert all("-DSTRATA_WINDOW_MAX_T=12" in cfg(r, f) for f in ("STRATA_ENABLE_CUDA=ON", "STRATA_ENABLE_HIP=ON"))


def test_unset_window_is_refused():
    with pytest.raises(B.BuildError, match="window"):
        run(plan(NV_MAIN, window=0))


def test_fetch_pins_base_and_applies_every_patch_in_order():
    r, _ = run(plan(NV_MAIN))
    argvs = [c.argv for c in r.commands]
    base = (ROOT / "patches/strata/STRATA_BASE").read_text().split()[0]
    assert ("git", "-C", "/work/strata", "checkout", "--detach", base) in argvs
    am = [Path(a[-1]).name for a in argvs if "am" in a and a[-1].endswith(".patch")]
    assert am == [*sorted(p.name for p in (ROOT / "patches/strata").glob("*.patch")), "strata-v7-seabraid.patch"]


def test_nvidia_only_has_no_hip_and_one_tree():
    r, res = run(plan(NV_MAIN))
    c = cfg(r, "STRATA_ENABLE_CUDA=ON")
    assert "HIP" not in c.replace("STRATA_ENABLE_HIP=OFF", "")
    assert list(res.build_dirs) == ["nvidia"] and res.helper_exes == {}


_LEAK = r"gfx1201|sm_86|/opt/rocm|/usr/local/cuda|/home/"


def _commands_text(*builds) -> str:
    """Every command, with the paths the project legitimately owns (its own checkout and the work dir) masked.

    The checkout itself can live under /home (a developer's often does), so it is not a leak; anything else under /home is.
    """
    text = "\n".join(str(c) for c in run(plan(*builds))[0].commands)
    return text.replace(str(ROOT.parents[1]), "<checkout>").replace("/work", "<work>")


def test_no_hardcoded_arch_or_path_leaks():
    other = Build("amd", "rocm", ("strata",), ("gfx1100",), (0,), "/elsewhere/rocm", "7", "/elsewhere/rocm/clang++", None, ())
    text = _commands_text(other)
    assert "gfx1100" in text and "/elsewhere/rocm" in text
    assert not re.search(_LEAK, text)


def test_leak_guard_still_bites_on_a_foreign_home_path():
    """Masking the project root must not switch the guard off: a toolkit under someone's home is still a leak."""
    foreign = Build("amd", "rocm", ("strata",), ("gfx1100",), (0,), HOME_ROOT + "/dev/rocm", "7", HOME_ROOT + "/dev/rocm/clang++", None, ())
    assert re.search(_LEAK, _commands_text(foreign))


def test_cuda_without_host_compiler_is_refused_listing_tried():
    bad = Build("nvidia", "cuda", ("seabraid-helper",), ("sm_86",), (1,), "/c", "12", "/c/nvcc", None, (), ("gcc", "gcc-13"))
    with pytest.raises(B.BuildError, match="gcc, gcc-13"):
        run(plan(AMD, bad))


def test_unparseable_arch_is_refused():
    with pytest.raises(B.BuildError, match="unrecognised"):
        B.normalize_arch("cuda", "weird")
    assert B.normalize_arch("rocm", "gfx90a:xnack-") == "gfx90a" and B.normalize_arch("cuda", "sm_90a") == "90a"


def test_unready_plan_is_refused():
    p = plan(AMD)
    p = Plan(p.model, "infeasible", p.roles, p.builds, p.tiers, refusals=("x",))
    with pytest.raises(B.BuildError, match="not ready"):
        run(p)


def test_strata_base_read_from_patches_dir_and_missing_refused(tmp_path):
    with pytest.raises(B.BuildError):
        B.read_strata_base(str(tmp_path))
    (tmp_path / "patches/strata").mkdir(parents=True)
    (tmp_path / "patches/strata/STRATA_BASE").write_text(FULL_SHA + "\n")
    assert B.read_strata_base(str(tmp_path)) == FULL_SHA


FULL_SHA = "628f74955a56fcba092495b800dd02af75373a0f"


@pytest.mark.parametrize("pin", ["9f7d977", "v0.2.13.1", "main", FULL_SHA[:39], FULL_SHA + "0", "g" * 40, FULL_SHA[:-1] + "é"])
def test_strata_base_refuses_anything_but_a_full_sha_and_logs_once(tmp_path, caplog, pin):
    (tmp_path / "patches/strata").mkdir(parents=True)
    (tmp_path / "patches/strata/STRATA_BASE").write_text(pin + "\n", encoding="utf-8")
    with caplog.at_level("ERROR", logger=B.log.name), pytest.raises(B.BuildError, match="full 40-hex"):
        B.read_strata_base(str(tmp_path))
    assert len([r for r in caplog.records if r.levelname == "ERROR"]) == 1    # an error path logs exactly once


class _FailOnPatch(B.Runner):
    def run(self, cmd):
        super().run(cmd)
        return int(cmd.label == "apply 0002-ExpertCache-clamp-fills-to-the-slot-not-to-the-large.patch")


def test_patch_conflict_names_patch_and_aborts():
    r = _FailOnPatch(dry_run=True, echo=lambda s: None)
    with pytest.raises(B.BuildError, match="0002-ExpertCache"):
        B.build_all(plan(NV_MAIN), str(ROOT), "/work", r, jobs=2)
    assert r.commands[-1].argv[-2:] == ("am", "--abort") and not any(c.label.startswith("configure") for c in r.commands)


def test_strata_setup_is_called_not_reimplemented():
    c = B.strata_setup_cmd("/work/strata", ["--model", "IQ3_S"])
    assert c.argv[:2] == ("python3", "/work/strata/setup.py") and "--no-start" in c.argv and c.argv[-2:] == ("--model", "IQ3_S")


def _rocm(root: str, compiler: str, lib=None) -> Build:
    return Build("amd", "rocm", ("strata",), ("gfx1201:xnack-",), (0,), root, "7", compiler, None, lib if lib is not None else (root + "/lib",))


def test_therock_root_uses_clang_gpu_targets_and_rocm_sysdeps():
    """TheRock: HIP clang under lib/llvm, rocm_sysdeps on the CMake prefix path and (below) on LD_LIBRARY_PATH."""
    r, _ = run(plan(_rocm("/tr", "/tr/bin/hipcc")), fs=THEROCK)
    c = next(c for c in r.commands if c.label.startswith("configure"))
    assert "-DCMAKE_HIP_COMPILER=/tr/lib/llvm/bin/clang++" in c.argv and "-DGPU_TARGETS=gfx1201" in c.argv
    assert "-DCMAKE_HIP_ARCHITECTURES=gfx1201" in c.argv and "-DCMAKE_PREFIX_PATH=/tr;/tr/lib/rocm_sysdeps" in c.argv
    assert not any("hipcc" in a for a in c.argv)
    assert c.env["LD_LIBRARY_PATH"].startswith("/tr/lib:/tr/lib/rocm_sysdeps/lib")


def test_opt_rocm_root_has_clang_and_no_sysdeps_anywhere():
    r, _ = run(plan(_rocm("/opt/rocm-y", "/opt/rocm-y/bin/hipcc")), fs=OPT)
    c = next(c for c in r.commands if c.label.startswith("configure"))
    assert "-DCMAKE_HIP_COMPILER=/opt/rocm-y/llvm/bin/clang++" in c.argv and "-DCMAKE_PREFIX_PATH=/opt/rocm-y" in c.argv
    assert "rocm_sysdeps" not in str(c)


def test_detected_clang_that_exists_is_kept_and_one_that_does_not_is_re_resolved():
    kept = _rocm("/tr", "/tr/custom/clang++")
    r, _ = run(plan(kept), fs=THEROCK | {"/tr/custom/clang++"})
    assert "-DCMAKE_HIP_COMPILER=/tr/custom/clang++" in next(c for c in r.commands if c.label.startswith("configure")).argv
    r, _ = run(plan(kept), fs=THEROCK)
    assert "-DCMAKE_HIP_COMPILER=/tr/lib/llvm/bin/clang++" in next(c for c in r.commands if c.label.startswith("configure")).argv


def test_hipcc_only_root_is_refused_listing_every_path_tried():
    with pytest.raises(B.BuildError) as e:
        run(plan(_rocm("/hip", "/hip/bin/hipcc")), fs={"/hip/bin/hipcc"})
    assert "/hip/lib/llvm/bin/clang++" in str(e.value) and "/hip/llvm/bin/clang++" in str(e.value)


def test_runner_requires_dry_run():
    with pytest.raises(TypeError):
        B.Runner()


class _Silent(B.Runner):
    """Exit 0 for everything, builds nothing: the failure mode build_all must not believe."""
    def run(self, cmd):
        self.commands.append(cmd)
        return 0


def test_real_build_that_produces_no_files_raises_naming_the_path():
    r = _Silent(dry_run=False, echo=lambda s: None)
    with pytest.raises(B.BuildError, match=r"build-nvidia/seabraid/seabraid-helper|build-nvidia/strata"):
        B.build_all(plan(NV_MAIN), str(ROOT), "/work", r, jobs=2, exists=lambda p: False)


def test_real_build_with_outputs_present_returns():
    r = _Silent(dry_run=False, echo=lambda s: None)
    res = B.build_all(plan(NV_MAIN), str(ROOT), "/work", r, jobs=2, exists=lambda p: True)
    assert res.server_exe.endswith("strata") or res.helper_exes


def test_dry_run_skips_the_output_check():
    r, res = run(plan(NV_MAIN), fs=set())   # nothing exists, dry-run: no raise
    assert res.strata_dir


def test_the_project_root_the_installer_passes_derives_the_wire_dir_the_patched_cmake_needs():
    # WHY: patch 0001 derives the helper wire (channel.hpp) from SEABRAID_STRATA_SOURCE_DIR as "<it>/../../adapters/strata/wire"; the
    # installer's only knob is that variable, so it must resolve to a directory that really holds the header, or the
    # installer's Strata configure dies with the patch's FATAL_ERROR on a fresh checkout.
    patch = next((ROOT / "patches" / "strata").glob("0001-*.patch")).read_text(encoding="utf-8")
    rel = re.search(r'\$\{SEABRAID_STRATA_SOURCE_DIR\}/(\.\./\.\./[\w/.-]+)"', patch)
    assert rel, "patch 0001 no longer derives the wire dir from SEABRAID_STRATA_SOURCE_DIR; update this test and the installer together"
    r, _ = run(plan(AMD, NV))
    assert f"-DSEABRAID_STRATA_SOURCE_DIR={ROOT}" in cfg(r, "STRATA_ENABLE_HIP=ON")
    assert (ROOT / rel.group(1) / "strata" / "seabraid" / "channel.hpp").is_file()


def test_every_build_enables_the_v7_adapter_from_the_checkout_root():
    r, _ = run(plan(AMD, NV))
    flags = [c.argv for c in r.commands if c.label.startswith("configure")]
    assert all(f"-DSTRATA_SEABRAID_SOURCE_DIR={ROOT.parents[1]}" in argv for argv in flags)
    assert (ROOT.parents[1] / "adapters/strata/strata-v7-seabraid.patch").is_file()


def test_missing_v7_patch_is_refused_before_fetch(tmp_path):
    project = tmp_path / "compat/strata"
    patches = project / "patches/strata"
    patches.mkdir(parents=True)
    (patches / "STRATA_BASE").write_text(FULL_SHA)
    (patches / "0001.patch").write_text("placeholder")
    runner = B.Runner(dry_run=True, echo=lambda s: None)
    with pytest.raises(B.BuildError, match="required Strata v7 integration patch missing"):
        B.fetch_and_patch(str(project), "/work/strata", runner)
    assert runner.commands == []


def test_file_moves_requests_sums_only_in_the_main_tree():
    runner, _ = run(plan(AMD, NV), file_moves=True)
    builds = [c.argv for c in runner.commands if c.label.startswith("build")]
    assert builds[0][-2:] == ("--target", "seabraid-sums")
    assert "seabraid-sums" not in builds[1]
    normal, _ = run(plan(AMD, NV))
    assert not any("seabraid-sums" in c.argv for c in normal.commands)


def test_file_moves_build_refuses_success_without_the_sums_artifact():
    runner = _Silent(dry_run=False, echo=lambda s: None)
    with pytest.raises(B.BuildError, match="seabraid-strata-ext/seabraid-sums"):
        B.build_all(plan(NV_MAIN), str(ROOT), "/work", runner, jobs=2, file_moves=True,
                    exists=lambda path: path.endswith("/strata"))


def test_file_moves_build_accepts_the_actual_sums_artifact():
    runner = _Silent(dry_run=False, echo=lambda s: None)
    present = {"/work/strata/build-nvidia/strata", "/work/strata/build-nvidia/seabraid-strata-ext/seabraid-sums"}
    result = B.build_all(plan(NV_MAIN), str(ROOT), "/work", runner, jobs=2, file_moves=True,
                         exists=present.__contains__)
    assert result.server_exe == "/work/strata/build-nvidia/strata"


@pytest.mark.parametrize("subdir", ["seabraid-strata-ext", "seabraid", "", "bin"])
def test_prebuilt_helper_lookup_supports_build_outputs_and_legacy_layouts(tmp_path, subdir):
    from install._setup_support import find_prebuilt
    directory = tmp_path / "strata/build-nvidia" / subdir
    directory.mkdir(parents=True)
    helper = directory / "seabraid-helper"
    helper.write_text("binary")
    assert find_prebuilt(str(tmp_path)).helper_exes == {"nvidia": str(helper)}


def test_canonical_artifact_directory_wins_over_a_legacy_binary(tmp_path):
    for subdir in ("seabraid-strata-ext", "seabraid"):
        directory = tmp_path / subdir
        directory.mkdir()
        (directory / "seabraid-helper").write_text("binary")
    assert B.find_built_exe(str(tmp_path), "seabraid-helper") == (tmp_path / "seabraid-strata-ext/seabraid-helper").as_posix()


def test_build_artifact_directory_matches_the_patched_cmake():
    patch = next((ROOT / "patches/strata").glob("0001-*.patch")).read_text(encoding="utf-8")
    output = re.search(r'add_subdirectory\("\$\{SEABRAID_STRATA_SOURCE_DIR\}/strata-ext" "\$\{CMAKE_BINARY_DIR\}/([^\"]+)"', patch)
    assert output, "the helper output directory moved; update the installer lookup with CMake"
    for name in ("seabraid-helper", "seabraid-sums"):
        assert B.built_exe_candidates("/build", name)[0] == f"/build/{output.group(1)}/{name}"
