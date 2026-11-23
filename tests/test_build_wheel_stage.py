# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Checker for scripts/build_wheel.py's real setuptools build, run without cargo against a copy of the package: stray
files in seabraid/_lib/ neither ship nor get deleted, two builds of one commit are byte-identical whatever the path,
umask and mtimes, and the build survives Debian's setuptools patch. (Split from test_build_wheel.py at its size limit.)"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import zipfile

import pytest
from conftest import REPO

_spec = importlib.util.spec_from_file_location("build_wheel", os.path.join(REPO, "scripts", "build_wheel.py"))
build_wheel = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_wheel)


# A copy of what a build reads (the real package, pyproject and licence files), so setuptools builds a real wheel without
# cargo. What must NOT matter to the bytes: the path, the file times and the umask.
def _checkout_copy(destination, with_stray=False):
    shutil.copytree(os.path.join(REPO, "bindings", "python"), destination / "bindings" / "python",
                    ignore=shutil.ignore_patterns("__pycache__", "tests", "_lib", ".pytest_cache"))
    for name in ("pyproject.toml", "VERSION", "LICENSE", "NOTICE"):
        shutil.copy(os.path.join(REPO, name), destination / name)
    if with_stray:
        lib = destination / "bindings" / "python" / "seabraid" / "_lib"
        lib.mkdir()
        (lib / "stray.txt").write_text("junk")
        (lib / "user_note.txt").write_text("keepme")
    return destination


DEBIAN_SETUPTOOLS = """
import setuptools.command.install_lib as il
original = il.install_lib.finalize_options
def finalize_options(self):  # what Debian's setuptools patch does (R7 python3.11 run on Ubuntu, setuptools 78.1.1)
    self.install_layout = None
    self.set_undefined_options('install', ('install_layout', 'install_layout'))
    original(self)
il.install_lib.finalize_options = finalize_options
import setuptools.command.install_egg_info as ie
original_egg = ie.install_egg_info.finalize_options
def finalize_egg(self):
    self.prefix_option = None
    self.set_undefined_options('install', ('prefix_option', 'prefix_option'))
    original_egg(self)
ie.install_egg_info.finalize_options = finalize_egg
"""


def _build_in_subprocess(repo, out, umask=0o022, mtime=None, library=None, env=None, prelude=""):
    if mtime is not None:
        for path in repo.rglob("*"):
            os.utime(path, (mtime, mtime))
    script = os.path.join(REPO, "scripts", "build_wheel.py")
    code = (prelude + "\nimport importlib.util, os, pathlib;"
            f"hasattr(os, 'umask') and os.umask({umask});"
            f"s = importlib.util.spec_from_file_location('bw', {script!r});"
            "m = importlib.util.module_from_spec(s); s.loader.exec_module(m);"
            f"lib = {None if library is None else str(library)!r};"
            f"print(m.build_wheel(pathlib.Path({str(out)!r}), 'manylinux_2_28_x86_64', "
            f"pathlib.Path(lib) if lib else None, 'libseabraid.so', pathlib.Path({str(repo)!r})))")
    done = subprocess.run([sys.executable, "-c", code], cwd=repo.parent, capture_output=True, text=True,
                          env={**os.environ, **(env or {})})
    assert done.returncode == 0, done.stderr[-2000:]
    return done.stdout.strip().splitlines()[-1]


def test_stray_files_in_lib_dir_neither_ship_nor_get_deleted(tmp_path):
    pytest.importorskip("setuptools")
    repo = _checkout_copy(tmp_path / "repo", with_stray=True)
    fake = tmp_path / "libseabraid.so"
    fake.write_bytes(b"\x7fELF fake library")
    wheel = _build_in_subprocess(repo, tmp_path / "out", library=fake)
    members = zipfile.ZipFile(wheel).namelist()
    assert "seabraid/_lib/libseabraid.so" in members
    assert not [m for m in members if "stray" in m or "user_note" in m], members
    lib = repo / "bindings" / "python" / "seabraid" / "_lib"
    assert sorted(path.name for path in lib.iterdir()) == ["stray.txt", "user_note.txt"], "the build deleted or wrote user files"
    assert (lib / "user_note.txt").read_text() == "keepme"
    info = zipfile.ZipFile(wheel).getinfo("seabraid/_lib/libseabraid.so")
    assert os.name != "posix" or (info.external_attr >> 16) & 0o111, "the library lost its execute bit"


def test_a_pure_wheel_carries_no_library_even_when_the_lib_dir_has_one(tmp_path):
    pytest.importorskip("setuptools")
    repo = _checkout_copy(tmp_path / "repo", with_stray=True)
    (repo / "bindings" / "python" / "seabraid" / "_lib" / "libseabraid.so").write_bytes(b"stale binary")
    wheel = _build_in_subprocess(repo, tmp_path / "out")
    assert not [m for m in zipfile.ZipFile(wheel).namelist() if "_lib" in m]


# --- LOW 6: reproducible ----------------------------------------------------------------------------------------------

def test_two_builds_from_different_paths_umasks_and_mtimes_are_byte_identical(tmp_path):
    pytest.importorskip("setuptools")
    fake = tmp_path / "libseabraid.so"
    fake.write_bytes(b"\x7fELF fake library")
    first = _build_in_subprocess(_checkout_copy(tmp_path / "one"), tmp_path / "out1", umask=0o022, mtime=1_000_000_000, library=fake)
    second = _build_in_subprocess(_checkout_copy(tmp_path / "two" / "deeper"), tmp_path / "out2", umask=0o077,
                                  mtime=1_700_000_000, library=fake)
    assert os.path.basename(first) == os.path.basename(second)
    assert open(first, "rb").read() == open(second, "rb").read(), "the wheel depends on path, umask or mtime"
    times = {tuple(info.date_time) for info in zipfile.ZipFile(first).infolist()}
    assert len(times) == 1 and min(times)[0] >= 1980, times


def test_source_date_epoch_from_the_caller_wins_and_is_stamped_into_the_wheel(tmp_path):
    pytest.importorskip("setuptools")
    wheel = _build_in_subprocess(_checkout_copy(tmp_path / "repo"), tmp_path / "out", env={"SOURCE_DATE_EPOCH": "1700000000"})
    assert {tuple(i.date_time) for i in zipfile.ZipFile(wheel).infolist()} == {(2023, 11, 14, 22, 13, 20)}


def test_the_fallback_epoch_is_announced_on_stderr_with_its_reason(tmp_path, capsys):
    # R8 L7: the fallback used to be a bare `except: pass`; now it says why the commit time was not used.
    assert build_wheel.source_date_epoch({}, tmp_path) == build_wheel.FALLBACK_EPOCH
    err = capsys.readouterr().err
    assert "build_wheel: warning" in err and "no usable commit time" in err and "git failed" in err and "1980-01-01" in err
    assert build_wheel.source_date_epoch({"SOURCE_DATE_EPOCH": "42"}, tmp_path) == "42"
    assert capsys.readouterr().err == "", "a caller-supplied epoch is not a fallback and prints nothing"


def test_source_date_epoch_is_the_commit_time_or_a_fixed_date_without_git(tmp_path):
    assert build_wheel.source_date_epoch({"SOURCE_DATE_EPOCH": "42"}, tmp_path) == "42"
    assert build_wheel.source_date_epoch({}, tmp_path) == build_wheel.FALLBACK_EPOCH  # not a git checkout
    git = shutil.which("git")
    if git is None:
        pytest.skip("git is not installed")

    def run(*args):
        return subprocess.run([git, "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
                              check=True, capture_output=True, env={**os.environ, "GIT_COMMITTER_DATE": "@1234567890 +0000"})

    run("init", "-q")
    (tmp_path / "f").write_text("x")
    run("add", "f")
    run("commit", "-q", "-m", "m")
    assert build_wheel.source_date_epoch({}, tmp_path) == "1234567890"
    sub = tmp_path / "sub"
    sub.mkdir()
    assert build_wheel.source_date_epoch({}, sub) == build_wheel.FALLBACK_EPOCH, "a directory inside someone else's repo"


def test_the_wheel_builds_with_a_setuptools_that_expects_debians_install_layout(tmp_path):
    pytest.importorskip("setuptools")
    wheel = _build_in_subprocess(_checkout_copy(tmp_path / "repo"), tmp_path / "out", prelude=DEBIAN_SETUPTOOLS)
    assert wheel.endswith(".whl")
