# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Checker for scripts/build_wheel.py's pure tag logic: readelf text in, manylinux tag / rejected libraries out."""

from __future__ import annotations

import importlib.util
import json
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

VERSIONS = """
Version needs section '.gnu.version_r' contains 2 entries:
 0x0000: Version: 1  File: libgcc_s.so.1  Cnt: 1
 0x0010:   Name: GCC_3.0  Flags: none  Version: 6
 0x0020: Version: 1  File: libc.so.6  Cnt: 3
 0x0030:   Name: GLIBC_2.17  Flags: none  Version: 5
 0x0040:   Name: GLIBC_2.34  Flags: none  Version: 4
 0x0050:   Name: GLIBC_2.9  Flags: none  Version: 3
"""
DYNAMIC = """
 0x0000000000000001 (NEEDED)             Shared library: [libgcc_s.so.1]
 0x0000000000000001 (NEEDED)             Shared library: [libc.so.6]
 0x0000000000000001 (NEEDED)             Shared library: [ld-linux-x86-64.so.2]
 0x000000000000000e (SONAME)             Library soname: [libseabraid.so.0.5]
"""


def test_tag_uses_the_highest_glibc_minor_numerically():
    assert build_wheel.manylinux_tag(VERSIONS) == "manylinux_2_34_x86_64"  # 2.34 beats 2.9: not a string comparison


def test_no_glibc_symbols_refuses_to_guess():
    with pytest.raises(SystemExit):
        build_wheel.manylinux_tag("Version needs section '.gnu.version_r' contains 0 entries:")


def test_allowed_needed_passes_and_soname_is_not_a_dependency():
    assert build_wheel.disallowed_needed(DYNAMIC) == []


def test_unexpected_needed_library_is_reported():
    text = DYNAMIC + " 0x0000000000000001 (NEEDED)             Shared library: [libstdc++.so.6]\n"
    assert build_wheel.disallowed_needed(text) == ["libstdc++.so.6"]


# --- release build environment -------------------------------------------------------------------------------------

def _flags(env):
    return env["CARGO_ENCODED_RUSTFLAGS"].split("\x1f")


def test_remap_is_appended_after_the_callers_rustflags(tmp_path):
    env = build_wheel.release_env({"RUSTFLAGS": "-C target-cpu=x86-64-v2  -Dwarnings", "PATH": "p"}, tmp_path)
    assert _flags(env) == ["-C", "target-cpu=x86-64-v2", "-Dwarnings", f"--remap-path-prefix={tmp_path}=/seabraid"]
    assert "RUSTFLAGS" not in env and env["PATH"] == "p"


def test_the_callers_encoded_flags_are_kept_and_a_path_with_spaces_stays_one_flag(tmp_path):
    repo = tmp_path / "a b"
    env = build_wheel.release_env({"CARGO_ENCODED_RUSTFLAGS": "-C\x1fdebuginfo=2"}, repo)
    assert _flags(env) == ["-C", "debuginfo=2", f"--remap-path-prefix={repo}=/seabraid"]


def test_with_no_flags_set_only_the_remap_is_passed(tmp_path):
    assert _flags(build_wheel.release_env({}, tmp_path)) == [f"--remap-path-prefix={tmp_path}=/seabraid"]


TRIPLE = "x86_64-unknown-linux-gnu"
TARGET_VAR = "CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTFLAGS"


def test_the_target_variable_name_follows_cargos_rule():
    assert build_wheel.target_rustflags_variable(TRIPLE) == TARGET_VAR
    assert build_wheel.target_rustflags_variable("aarch64-apple-darwin") == "CARGO_TARGET_AARCH64_APPLE_DARWIN_RUSTFLAGS"


def test_build_rustflags_alone_is_no_longer_dropped(tmp_path):
    env = build_wheel.release_env({"CARGO_BUILD_RUSTFLAGS": "-C target-cpu=x86-64-v2"}, tmp_path)
    assert _flags(env) == ["-C", "target-cpu=x86-64-v2", f"--remap-path-prefix={tmp_path}=/seabraid"]
    assert "CARGO_BUILD_RUSTFLAGS" not in env


def test_the_target_specific_flags_are_kept_for_the_building_triple_only(tmp_path):
    environ = {TARGET_VAR: "-Ccodegen-units=1", "CARGO_TARGET_AARCH64_APPLE_DARWIN_RUSTFLAGS": "-Cother"}
    assert _flags(build_wheel.release_env(environ, tmp_path, TRIPLE)) == [
        "-Ccodegen-units=1", f"--remap-path-prefix={tmp_path}=/seabraid"]
    assert _flags(build_wheel.release_env(environ, tmp_path)) == [f"--remap-path-prefix={tmp_path}=/seabraid"]


def test_cargo_build_target_selects_the_target_variable_when_no_triple_is_passed(tmp_path):
    env = build_wheel.release_env({"CARGO_BUILD_TARGET": TRIPLE, TARGET_VAR: "-Ccodegen-units=1"}, tmp_path)
    assert _flags(env)[0] == "-Ccodegen-units=1"


def test_every_source_is_merged_in_cargos_order_with_the_remap_last(tmp_path):
    environ = {"CARGO_BUILD_RUSTFLAGS": "-Dbuild", TARGET_VAR: "-Dtarget", "RUSTFLAGS": "-Drustflags",
               "CARGO_ENCODED_RUSTFLAGS": "-Dencoded", "PATH": "p"}
    env = build_wheel.release_env(environ, tmp_path, TRIPLE)
    assert _flags(env) == ["-Dencoded", "-Drustflags", "-Dtarget", "-Dbuild", f"--remap-path-prefix={tmp_path}=/seabraid"]
    assert {"RUSTFLAGS", "CARGO_BUILD_RUSTFLAGS", TARGET_VAR}.isdisjoint(env) and env["PATH"] == "p"
    assert "CARGO_BUILD_RUSTFLAGS" in environ  # the caller's mapping is not mutated


# --- the path-leak scan --------------------------------------------------------------------------------------------

def _leaks(blob, tmp_path):
    roots = {"checkout": tmp_path / "wheel repo", "home directory": tmp_path / "home" / "al"}
    return build_wheel.leaked_paths(blob, roots)


def test_a_clean_binary_has_no_leaks(tmp_path):
    assert _leaks(b"\x7fELF /seabraid/src/lib.rs /rustc/abc/library/core", tmp_path) == []


def test_the_checkout_path_is_found_with_either_separator_or_escaped_backslashes(tmp_path):
    posix = (tmp_path / "wheel repo").as_posix()
    for text in (str(tmp_path / "wheel repo"), posix, posix.replace("/", "\\"), posix.replace("/", "\\\\")):
        found = _leaks(b"xx" + text.encode() + b"/src/lib.rs", tmp_path)
        assert any(hit.startswith("checkout: ") for hit in found), text


def test_the_home_directory_is_found_with_either_separator_after_it(tmp_path):
    home = (tmp_path / "home" / "al").as_posix()
    assert f"home directory: {home}" in _leaks(f"{home}/.cargo/registry".encode(), tmp_path)
    assert f"home directory: {home}" in _leaks(f"{home}\\.cargo".encode(), tmp_path)


def test_a_home_directory_name_that_is_a_prefix_of_another_is_not_a_leak(tmp_path):
    home = (tmp_path / "home" / "al").as_posix()
    assert _leaks(f"{home}ice/.cargo".encode(), tmp_path) == []  # a root "al" must not match a sibling "alice"


# --- macOS tag from the dylib --------------------------------------------------------------------------------------

BUILD_VERSION = """
Load command 9
      cmd LC_BUILD_VERSION
  cmdsize 32
 platform 1
    minos {minos}
      sdk 14.2
   ntools 1
"""
LEGACY = """
Load command 8
      cmd LC_VERSION_MIN_MACOSX
  cmdsize 16
  version 10.12
      sdk 10.15
"""


def test_macos_tag_comes_from_minos_and_a_modern_minor_is_zero():
    assert build_wheel.macos_platform_tag(BUILD_VERSION.format(minos="14.5"), "arm64\n") == "macosx_14_0_arm64"


def test_macos_tag_keeps_the_minor_below_eleven_and_reads_the_legacy_command():
    assert build_wheel.macos_platform_tag(LEGACY, "x86_64") == "macosx_10_12_x86_64"


def test_an_arm64_dylib_is_never_tagged_below_macos_11():
    assert build_wheel.macos_platform_tag(LEGACY, "arm64") == "macosx_11_0_arm64"


def test_a_universal_dylib_takes_the_highest_minimum_and_universal2():
    text = BUILD_VERSION.format(minos="11.0") + BUILD_VERSION.format(minos="12.3")
    assert build_wheel.macos_platform_tag(text, "x86_64 arm64") == "macosx_12_0_universal2"


def test_a_non_macos_platform_block_is_ignored_and_no_version_refuses_to_guess():
    ios = BUILD_VERSION.format(minos="15.0").replace("platform 1", "platform 2")
    with pytest.raises(SystemExit):
        build_wheel.macos_platform_tag(ios, "arm64")
    with pytest.raises(SystemExit):
        build_wheel.macos_platform_tag("Load command 0\n cmd LC_SEGMENT_64\n", "arm64")


def test_unknown_architectures_refuse():
    with pytest.raises(SystemExit):
        build_wheel.macos_platform_tag(LEGACY, "ppc")


# --- cwd independence ----------------------------------------------------------------------------------------------

def test_the_wheel_is_built_from_any_working_directory(tmp_path):
    pytest.importorskip("setuptools")
    work, out = tmp_path / "elsewhere", tmp_path / "out"
    work.mkdir()
    script = os.path.join(REPO, "scripts", "build_wheel.py")
    code = ("import importlib.util, pathlib;"
            f"s = importlib.util.spec_from_file_location('bw', {script!r});"
            "m = importlib.util.module_from_spec(s); s.loader.exec_module(m);"
            f"print(m.build_wheel(pathlib.Path({str(out)!r}), 'manylinux_2_28_x86_64'))")
    done = subprocess.run([sys.executable, "-c", code], cwd=work, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    name = os.path.basename(done.stdout.strip())
    assert name.startswith("seabraid-") and name.endswith("-py3-none-manylinux_2_28_x86_64.whl"), name


# --- where cargo leaves the library ----------------------------------------------------------------------------------

def _cargo_artifact(path):
    return json.dumps({"reason": "compiler-artifact", "target": {"name": "seabraid", "crate_types": ["cdylib"]},
                       "filenames": [str(path)]})


# --- the wheel really carries the library ----------------------------------------------------------------------------

def _wheel(path, members):
    """A zip with `members` = {name: unix mode}; the mode is stored the way wheel stores it (external_attr high word)."""
    with zipfile.ZipFile(path, "w") as archive:
        for name, mode in members.items():
            info = zipfile.ZipInfo(name)
            info.external_attr = (0o100000 | mode) << 16
            archive.writestr(info, b"\x7fELF")
    return path


def test_a_wheel_without_the_library_is_refused(tmp_path):
    wheel = _wheel(tmp_path / "a.whl", {"seabraid/__init__.py": 0o644})
    with pytest.raises(SystemExit, match="seabraid/_lib/libseabraid.so"):
        build_wheel.verify_wheel_library(wheel, "libseabraid.so", needs_exec_mode=True)


def test_a_library_without_execute_bits_is_refused_where_modes_matter(tmp_path):
    wheel = _wheel(tmp_path / "a.whl", {"seabraid/_lib/libseabraid.so": 0o644})
    with pytest.raises(SystemExit, match="not executable"):
        build_wheel.verify_wheel_library(wheel, "libseabraid.so", needs_exec_mode=True)
    build_wheel.verify_wheel_library(wheel, "libseabraid.so", needs_exec_mode=False)  # a Windows DLL has no mode


def test_an_executable_library_member_passes(tmp_path):
    wheel = _wheel(tmp_path / "a.whl", {"seabraid/_lib/libseabraid.so": 0o755})
    build_wheel.verify_wheel_library(wheel, "libseabraid.so", needs_exec_mode=True)


# --- concurrent builds in one checkout -------------------------------------------------------------------------------

def test_a_second_build_in_the_same_checkout_is_refused_and_the_lock_is_released_after(tmp_path):
    lock = tmp_path / "build.lock"
    with build_wheel.checkout_lock(lock):
        with pytest.raises(SystemExit, match="another"):
            with build_wheel.checkout_lock(lock):
                pass
    with build_wheel.checkout_lock(lock):  # released by the first holder's exit
        pass


def test_the_lock_file_is_named_per_checkout(tmp_path):
    assert build_wheel.lock_path(build_wheel.REPO) != build_wheel.lock_path(tmp_path / "other")


# --- main() end to end, cargo and setuptools faked -------------------------------------------------------------------
# The helpers above are unit-tested, but removing a helper's call from main() kept every one of those tests green (R3
# finding 20). These run main() itself against a scratch checkout with no cargo, no setuptools and no network.

TRIPLE_BUILT = "x86_64-unknown-linux-gnu"
LIBRARY_BYTES = b"clean library bytes"


def _lock_is_held(repo):
    try:
        with build_wheel.checkout_lock(build_wheel.lock_path(repo)):
            return False
    except SystemExit:
        return True


class Events(list):
    """The ordered record of what main() ran; `.commands` keeps every (command, kwargs) the fake _run received."""

    def __init__(self):
        super().__init__()
        self.commands = []
        self.wheel_mode = 0o755  # the mode the faked setuptools gives the library inside the wheel


@pytest.fixture
def faked_main(tmp_path, monkeypatch):
    """(repo, out, events): events is the ordered record of what main() ran, each with whether the lock was held."""
    repo, out = tmp_path / "checkout", tmp_path / "out"
    name = build_wheel.LIB_NAMES.get(sys.platform, "libseabraid.so")
    stale = repo / "target" / TRIPLE_BUILT / "release" / name
    for library, content in ((stale, b"a stale artifact: must not be read"), (_built_library(repo), LIBRARY_BYTES)):
        library.parent.mkdir(parents=True)
        library.write_bytes(content)
    lib_dir = repo / "bindings" / "python" / "seabraid" / "_lib"
    lib_dir.parent.mkdir(parents=True)  # the package directory exists in a checkout; _lib/ is created by main()
    events = Events()

    def fake_run(command, **kwargs):
        events.commands.append((command, kwargs))
        events.append(("cargo" if command[0] == "cargo" else "unexpected " + command[0], _lock_is_held(repo)))
        return _cargo_artifact(_built_library(repo))

    def fake_platform_tag(library):
        events.append(("tag", _lock_is_held(repo)))
        return "manylinux_2_17_x86_64"

    def fake_build_wheel(out_dir, tag, library, library_name):
        copied = library.read_bytes()  # raises if main() did not hand over the library cargo built
        events.append(("setuptools", _lock_is_held(repo), copied))
        assert library_name == name
        events.built_in = out_dir  # where main() asked for the wheel: absolute, whatever the cwd is by now
        return _wheel(out_dir / "seabraid-0-py3-none-test.whl", {f"seabraid/_lib/{name}": events.wheel_mode})

    real_verify = build_wheel.verify_wheel_library

    def spy_verify(wheel, library_name, needs_exec_mode):
        events.append(("verify", wheel.name, library_name))
        real_verify(wheel, library_name, needs_exec_mode)

    monkeypatch.setattr(build_wheel, "REPO", repo)
    monkeypatch.setattr(build_wheel, "LIB_DIR", lib_dir)
    monkeypatch.setattr(build_wheel, "_run", fake_run)
    monkeypatch.setattr(build_wheel, "platform_tag_of", fake_platform_tag)
    monkeypatch.setattr(build_wheel, "build_wheel", fake_build_wheel)
    monkeypatch.setattr(build_wheel, "verify_wheel_library", spy_verify)
    monkeypatch.setattr(build_wheel, "host_triple", lambda: pytest.fail("the triple comes from CARGO_BUILD_TARGET here"))
    monkeypatch.setenv("CARGO_BUILD_TARGET", TRIPLE_BUILT)
    monkeypatch.setenv("CARGO_TARGET_DIR", str(_built_library(repo).parents[2]))
    monkeypatch.setattr(sys, "argv", ["build_wheel.py", "--out", str(out)])
    return repo, out, events


def test_main_builds_under_the_lock_from_cargos_reported_path_and_verifies_the_wheel(faked_main, capsys):
    repo, out, events = faked_main
    assert build_wheel.main() == 0
    assert [event[0] for event in events] == ["cargo", "tag", "setuptools", "verify"]
    assert events[0] == ("cargo", True), "cargo must run while the checkout lock is held, or a second builder builds too"
    assert events[2][1:] == (True, LIBRARY_BYTES), "setuptools ran without the lock, or on the wrong library"
    assert events[3][1] == "seabraid-0-py3-none-test.whl"  # verify_wheel_library read back the wheel main() built
    assert capsys.readouterr().out.strip() == str(out / "seabraid-0-py3-none-test.whl")
    assert not build_wheel.LIB_DIR.exists(), "the source tree's _lib/ must not be created by a build"
    assert not _lock_is_held(repo), "the lock must be released afterwards"


def test_main_refuses_a_second_builder_before_it_builds_anything(faked_main):
    repo, out, events = faked_main
    with build_wheel.checkout_lock(build_wheel.lock_path(repo)):
        with pytest.raises(SystemExit, match="another"):
            build_wheel.main()
    assert events == [], f"the refused builder still ran: {events}"
    assert not out.exists() and not build_wheel.LIB_DIR.exists()


# Each assertion below pins one call site in main() that every test above survived losing (R4 finding 8): the cargo flags,
# the environment handed to cargo, the leaked-path check and the built-library check.

def test_main_passes_cargo_the_offline_locked_flags_the_checkout_and_the_remapped_environment(faked_main, monkeypatch):
    repo, _, events = faked_main
    for name in ("CARGO_ENCODED_RUSTFLAGS", "CARGO_BUILD_RUSTFLAGS", build_wheel.target_rustflags_variable(TRIPLE_BUILT)):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RUSTFLAGS", "-C target-cpu=x86-64-v2")  # the user's own flags: they must survive next to the remap
    assert build_wheel.main() == 0
    (command, kwargs), = events.commands
    assert command == ["cargo", "build", "--release", "--offline", "--locked", "--message-format=json-render-diagnostics"]
    assert kwargs["cwd"] == repo, "cargo must run in the checkout, not in the caller's directory"
    # CONTENT, not identity: comparing with a second call of release_env() passed for any environment at all (it only
    # pinned that the call site exists). The remap prefix and the caller's own flags must be in what cargo is given.
    flags = kwargs["env"]["CARGO_ENCODED_RUSTFLAGS"].split("\x1f")
    assert flags == ["-C", "target-cpu=x86-64-v2", f"--remap-path-prefix={repo}={build_wheel.REMAP_TO}"], flags
    assert "RUSTFLAGS" not in kwargs["env"], "the plain variable must be folded into the encoded one, not passed twice"
    assert kwargs["env"]["PATH"] == os.environ["PATH"], "the rest of the environment must reach cargo unchanged"


def _built_library(repo):
    name = build_wheel.LIB_NAMES.get(sys.platform, "libseabraid.so")
    return repo.parent / "custom cargo target" / TRIPLE_BUILT / "release" / name


@pytest.mark.parametrize("leak", ["checkout", "home directory"])
def test_main_fails_when_the_built_library_embeds_where_it_was_built(faked_main, leak, monkeypatch, tmp_path):
    repo, out, events = faked_main
    home = tmp_path / "somebody"
    monkeypatch.setattr(build_wheel.Path, "home", classmethod(lambda cls: home))
    where = repo if leak == "checkout" else home
    _built_library(repo).write_bytes(b"panic at " + str(where).encode() + b"/src/lib.rs")
    with pytest.raises(SystemExit, match="embeds where it was built") as caught:
        build_wheel.main()
    assert leak in str(caught.value)
    assert [event[0] for event in events] == ["cargo"], f"a leaking library went on to {events}"
    assert not list(out.glob("*.whl")) and not build_wheel.LIB_DIR.exists(), "nothing may be packaged from a leaking build"
    assert not _lock_is_held(repo), "the lock must be released after a failure"


def test_main_fails_when_cargo_produced_no_library(faked_main):
    repo, out, events = faked_main
    _built_library(repo).unlink()
    with pytest.raises(SystemExit, match="cargo did not produce"):
        build_wheel.main()
    assert [event[0] for event in events] == ["cargo"], f"a missing library went on to {events}"
    assert not list(out.glob("*.whl")) and not build_wheel.LIB_DIR.exists()
    assert not _lock_is_held(repo)


# Two more call sites of main() that no test above noticed losing (R5): the execute-bit check and the early resolve of --out.

@pytest.mark.skipif(sys.platform == "win32", reason="Windows files carry no execute bit; main() does not ask for one there")
def test_main_refuses_a_wheel_whose_library_is_not_executable(faked_main):
    _, _, events = faked_main
    events.wheel_mode = 0o644  # what a restrictive umask or a mis-copied source gives the library
    with pytest.raises(SystemExit, match="is not executable"):
        build_wheel.main()  # a main() that passed needs_exec_mode=False would return 0 here


def test_a_relative_out_is_resolved_against_the_callers_directory_before_anything_changes_it(faked_main, monkeypatch, tmp_path):
    repo, _, events = faked_main
    caller = tmp_path / "caller"
    caller.mkdir()
    monkeypatch.chdir(caller)
    monkeypatch.setattr(sys, "argv", ["build_wheel.py", "--out", "relative-out"])
    real_run = build_wheel._run

    def run_then_move(command, **kwargs):
        messages = real_run(command, **kwargs)
        os.chdir(repo)  # what the real packaging step does (setuptools runs from the checkout) before the wheel is written
        return messages

    monkeypatch.setattr(build_wheel, "_run", run_then_move)
    assert build_wheel.main() == 0
    assert events.built_in == caller / "relative-out", f"the wheel went to {events.built_in}, not under the caller's directory"
    assert (caller / "relative-out" / "seabraid-0-py3-none-test.whl").is_file()
    assert not (repo / "relative-out").exists(), "the relative --out followed the working-directory change into the checkout"


# --- LOW 4: the home directory is scanned for only when it names a person ----------------------------------------------

@pytest.mark.parametrize("home", ["/", "/usr", "/root"])
def test_a_short_home_directory_is_not_scanned_for(home, tmp_path):
    roots = build_wheel.leak_roots(tmp_path / "repo", build_wheel.Path(home))
    assert list(roots) == ["checkout"], roots
    assert build_wheel.leaked_paths(b"see a plain URL scheme// and /usr/lib", roots) == []


def test_a_two_component_home_directory_is_scanned_for(tmp_path):
    home = build_wheel.Path("/srv/alice")
    roots = build_wheel.leak_roots(tmp_path, home)
    assert roots == {"checkout": tmp_path, "home directory": home}
    assert build_wheel.leaked_paths(b"/srv/alice/.cargo/x", roots)


@pytest.mark.skipif(os.name == "nt", reason="Path.home() reads HOME only on POSIX")
def test_main_does_not_refuse_a_clean_library_when_home_is_the_filesystem_root(faked_main, monkeypatch):
    repo, out, events = faked_main
    monkeypatch.setenv("HOME", "/")
    _built_library(repo).write_bytes(b"a plain URL scheme// // /usr/lib clean")
    assert build_wheel.main() == 0


# --- LOW 5: --out is checked before cargo ------------------------------------------------------------------------------

def test_an_out_directory_that_cannot_be_made_is_a_one_line_exit_before_cargo(faked_main, tmp_path, monkeypatch):
    repo, _, events = faked_main
    blocker = tmp_path / "a file"
    blocker.write_text("not a directory")
    monkeypatch.setattr(sys, "argv", ["build_wheel.py", "--out", str(blocker / "wheels")])
    with pytest.raises(SystemExit, match=r"--out .* cannot be created or written"):
        build_wheel.main()
    assert events == [], f"cargo ran before --out was checked: {events}"
    assert not _lock_is_held(repo)


# --- LOW 3: files a user keeps in seabraid/_lib/ ----------------------------------------------------------------------

def test_main_refuses_stray_files_in_lib_dir_before_cargo_and_leaves_them_alone(faked_main):
    repo, out, events = faked_main
    build_wheel.LIB_DIR.mkdir()
    (build_wheel.LIB_DIR / "user_note.txt").write_text("keep me")
    with pytest.raises(SystemExit, match="user_note.txt"):
        build_wheel.main()
    assert events == [], f"the build ran past a stray file: {events}"
    assert (build_wheel.LIB_DIR / "user_note.txt").read_text() == "keep me"


def test_a_lib_dir_holding_only_the_library_name_is_accepted(tmp_path):
    (tmp_path / "libseabraid.so").write_bytes(b"x")
    build_wheel.check_lib_dir(tmp_path, "libseabraid.so")
    build_wheel.check_lib_dir(tmp_path / "absent", "libseabraid.so")
