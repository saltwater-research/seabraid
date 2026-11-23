# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Build-configuration decisions the CMake/Cargo side must keep: static checks plus one configure-only CMake run."""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import sys

import pytest
from conftest import REPO, code_only

# WHY SEABRAID_CMAKE: native-tests/installed/use_cmake.sh honours the same variable, so one setting tests the CMake floor.
CMAKE = os.environ.get("SEABRAID_CMAKE") or shutil.which("cmake")


def read(*parts: str) -> str:
    with open(os.path.join(REPO, *parts), encoding="utf-8") as handle:
        return handle.read()


def test_cargo_release_profile_decisions():
    toml = read("Cargo.toml")
    assert 'crate-type = ["cdylib"]' in toml, "an rlib nothing links is built and discarded"
    assert 'strip = "symbols"' in toml
    assert 'panic = "unwind"' in toml, "the FFI guard needs catch_unwind to work"
    assert "overflow-checks = true" in toml


# Every CMakeLists.txt in the tree. test_every_cmake_file_is_guarded fails when a new one is not listed here.
GUARDED_CMAKE = [
    "CMakeLists.txt",
    "adapters/ggml/CMakeLists.txt",
    "adapters/ggml/sdk_check/CMakeLists.txt",
    "adapters/strata/CMakeLists.txt",
    "compat/strata/strata-ext/CMakeLists.txt",
    "compat/strata/tests/CMakeLists.txt",
    "examples/consumer/CMakeLists.txt",
    "native-tests/installed/CMakeLists.txt",
    "scripts/ggml_fixtures/CMakeLists.txt",
    "tools/sim/CMakeLists.txt",
]
# The ones that register output-matching tests, and so must route them through check_output.cmake.
MATCHES_OUTPUT = ["CMakeLists.txt", "tools/sim/CMakeLists.txt", "examples/consumer/CMakeLists.txt"]


def test_every_cmake_file_is_guarded():
    import _common

    found = sorted(p for p in _common.tracked(REPO) if p.endswith("/CMakeLists.txt") or p == "CMakeLists.txt")
    assert found == sorted(GUARDED_CMAKE), "a CMakeLists.txt is missing from (or stale in) GUARDED_CMAKE"


@pytest.mark.parametrize("rel", GUARDED_CMAKE)
def test_cmake_files_check_exit_code_not_just_output(rel):
    # CTest ignores the exit code of a test that sets PASS_REGULAR_EXPRESSION, in set_tests_properties or as an
    # add_test property; every test that matches output goes through check_output.cmake instead.
    cmake = code_only(read(*rel.split("/")))
    assert "PASS_REGULAR_EXPRESSION" not in cmake, f"{rel}: CTest ignores the exit code with it"
    if rel in MATCHES_OUTPUT:
        assert "check_output.cmake" in cmake, rel


@pytest.mark.parametrize(
    "text, kept, dropped",
    [
        ("# whole line\nkeep()", "keep()", "whole"),
        ("keep() # trailing", "keep()", "trailing"),
        ("a #[[ inline block ]] b", "a", "block"),
        ("before\n#[=[ multi\nline ]] still ]=]\nafter", "after", "still"),
        ('set(x "^#define Y") # why', "^#define Y", "why"),
        ('echo "a # b" # real', "a # b", "real"),
        ("echo $# a#b", "$# a#b", "zzz"),
        ("run: cargo test   # yaml comment", "cargo test", "yaml"),
    ],
)
def test_code_only_strips_comments_but_not_look_alikes(text, kept, dropped):
    out = code_only(text)
    assert kept in out and dropped not in out, out


@pytest.mark.parametrize(
    "text, expected",
    [
        # R5: a single-quoted '#' is text (sh, TOML literal strings), not the start of a comment.
        ("grep '#' file # real", "grep '#' file"),
        ("a = 'x # y'  # why", "a = 'x # y'"),
        # R5: a #[[ inside a string is not a block comment; the old regex ate everything up to the next ]].
        ('set(a "#[[")\nkeep()\nset(b "]]")', 'set(a "#[[")\nkeep()\nset(b "]]")'),
        ("echo a#[[b]] c", "echo a#[[b]] c"),
        # R5: an escaped backslash does not escape the quote after it, so the string ends and the comment is a comment.
        ('set(p "C:\\\\")  # real', 'set(p "C:\\\\")'),
        # ... while an escaped quote does keep the string open.
        ('set(p "a\\" # still text")  # real', 'set(p "a\\" # still text")'),
        # R5: a '#' right after ')' starts a comment.
        ("set(a b)#comment\nnext()", "set(a b)\nnext()"),
        # A double quote inside single quotes (and the reverse) is text.
        ("""echo 'a "b' # c""", """echo 'a "b'"""),
        ("""echo "it's # not" # c""", """echo "it's # not\""""),
        # A real block comment still goes, including its closing line's trailing code, and a '#' after it still counts.
        ("x #[==[ a ]] b\nc ]==] y # z", "x  y"),
        ("a #[[ one ]] b # two", "a  b"),
        # An unterminated block opener is a line comment, never a swallow of the rest of the file.
        ("a #[[ never closed\nb", "a\nb"),
        # '#' after an escaped hash or inside $# stays.
        ("echo \\# $# a#b", "echo \\# $# a#b"),
        # R6b #13: ';' and '{' are comment predecessors (sh: `a;# c`, `{# c`) but '${#x}' (the length of x) is not a comment.
        ("echo a;# c\nnext", "echo a;\nnext"),
        ("{# c\nnext", "{\nnext"),
        ("n=${#x} # c", "n=${#x}"),
        # R6b #13: a double-quoted string spans lines (CMake multi-line argument, sh), so a comment after its closing quote goes
        # and a '#' at the start of a line inside it stays.
        ('set(a "l1\nl2") # c\nnext()', 'set(a "l1\nl2")\nnext()'),
        ('set(a "x\n# text\ny")', 'set(a "x\n# text\ny")'),
        ('echo "a\nb" # c', 'echo "a\nb"'),
        # ... but a LONE double quote (no partner in sight) must not swallow the rest of the file.
        ('echo "lone\nreal # c\nnext', 'echo "lone\nreal\nnext'),
        # R6b #13: sh $'...' escapes a quote with a backslash; plain '...' does not.
        ("printf $'a\\'b # in' # c", "printf $'a\\'b # in'"),
        ("printf 'a\\' # c", "printf 'a\\'"),
        # A CMake bracket argument is raw: its lone '"' must not open a string that outlives it (`MATCHES [=[["$\;]]=]`), and
        # a comment after it still goes.
        ('if(x MATCHES [=[["$\\;]]=] OR y) # c\n"later" # d', 'if(x MATCHES [=[["$\\;]]=] OR y)\n"later"'),
        # An unclosed bracket opener is not a bracket argument.
        ("a [=[ b # c", "a [=[ b"),
        # A '$' before a quote that is itself escaped is not the $'...' form.
        ("echo \\$'x # c", "echo \\$'x # c"),
    ],
)
def test_code_only_exact_output_for_the_review_counter_examples(text, expected):
    assert code_only(text) == expected


def test_installed_sdk_check_covers_move_version_epoch_exports_and_manifest():
    script = code_only(read("native-tests", "installed", "check_sdk.sh"))
    for needle in ('mv "$prefix" "$moved"', "nm -D --defined-only", "install_manifest.txt", "SEABRAID_REQUIRE_VERSION", "-x \"$lib\""):
        assert needle in script, needle
    assert "SEABRAID_REQUIRE_VERSION" in code_only(read("native-tests", "installed", "CMakeLists.txt"))


def sdk_toolchain_gap(linux_only: bool) -> str | None:
    """Why the SDK-building tests cannot run here, or None when they can."""
    if CMAKE is None or shutil.which("cargo") is None:
        return "needs cmake and cargo (the SDK is built and installed from this tree)"
    if linux_only and not sys.platform.startswith("linux"):
        return "needs Linux (the installed-SDK checks use nm, ELF and POSIX sh)"
    return None


def require_sdk_toolchain(linux_only: bool = True) -> None:
    """Skip when the toolchain is missing, unless CI says it must be there.

    WHY two switches: a skip reads as a pass, so SEABRAID_REQUIRE_SDK_TESTS=1 (set on exactly one CI leg, the one that
    has cmake and cargo and builds the SDK) turns a missing toolchain into a failure. SEABRAID_SKIP_SDK_TESTS=1 (the
    other legs) skips with the reason, so the SDK is built once per CI run and not once per Python version.
    """
    required = os.environ.get("SEABRAID_REQUIRE_SDK_TESTS") == "1"
    # WHY == "1": SEABRAID_SKIP_SDK_TESTS=0 (or empty) must not skip; it is the symmetric twin of the REQUIRE switch.
    skipped = os.environ.get("SEABRAID_SKIP_SDK_TESTS") == "1"
    if required and skipped:
        pytest.fail("SEABRAID_REQUIRE_SDK_TESTS=1 and SEABRAID_SKIP_SDK_TESTS=1 are both set; a run cannot both need and skip them")
    if skipped:
        pytest.skip("SEABRAID_SKIP_SDK_TESTS is set: the SDK tests run once per CI run, on the leg that sets SEABRAID_REQUIRE_SDK_TESTS=1")
    gap = sdk_toolchain_gap(linux_only)
    if gap and required:
        pytest.fail(f"SEABRAID_REQUIRE_SDK_TESTS=1 but this machine {gap}")
    if gap:
        pytest.skip(gap)


@pytest.mark.parametrize("value, skips", [("1", True), ("0", False), ("", False), ("true", False), ("yes", False), (" 1", False)])
def test_the_skip_switch_skips_only_on_exactly_1(monkeypatch, value, skips):
    # SEABRAID_SKIP_SDK_TESTS=0 (or empty, or any word) must not skip: it is the symmetric twin of the REQUIRE switch.
    monkeypatch.setattr(sys.modules[__name__], "sdk_toolchain_gap", lambda linux_only: None)  # a toolchain is "present"
    monkeypatch.delenv("SEABRAID_REQUIRE_SDK_TESTS", raising=False)
    monkeypatch.setenv("SEABRAID_SKIP_SDK_TESTS", value)
    if skips:
        with pytest.raises(pytest.skip.Exception, match="SEABRAID_SKIP_SDK_TESTS"):
            require_sdk_toolchain()
    else:
        require_sdk_toolchain()  # returns: the SDK tests run


def test_require_wins_over_a_missing_toolchain_and_conflicts_with_skip(monkeypatch):
    monkeypatch.setenv("SEABRAID_REQUIRE_SDK_TESTS", "1")
    monkeypatch.delenv("SEABRAID_SKIP_SDK_TESTS", raising=False)
    monkeypatch.setattr(sys.modules[__name__], "sdk_toolchain_gap", lambda linux_only: "needs the moon")
    with pytest.raises(pytest.fail.Exception, match="needs the moon"):
        require_sdk_toolchain()  # required but missing: a failure, never a skip
    monkeypatch.setenv("SEABRAID_SKIP_SDK_TESTS", "1")
    with pytest.raises(pytest.fail.Exception, match="both set"):
        require_sdk_toolchain()


def build_step(cmd: list[str]) -> None:
    """Run one build command; on failure show its whole output (cargo and cmake say why on stderr)."""
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if result.returncode != 0:
        pytest.fail(f"{' '.join(cmd)} exited {result.returncode}\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}")


def run_script(rel: str, *args: str, timeout: int = 900, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["sh", os.path.join(REPO, rel), *args], capture_output=True, text=True, timeout=timeout, env=env)


def test_check_sdk_passes_on_a_freshly_built_and_installed_sdk(tmp_path):
    # The needle test above only proves the script mentions its checks; this runs them. SEABRAID_SDK_PREFIX and
    # SEABRAID_SDK_BUILD point at an SDK a caller already built (a developer's shortcut; no CI job sets them, the CI
    # `sdk` job runs check_sdk.sh itself); otherwise one is built here.
    require_sdk_toolchain()
    prefix, build = os.environ.get("SEABRAID_SDK_PREFIX"), os.environ.get("SEABRAID_SDK_BUILD")
    if not prefix:
        build = str(tmp_path / "build")
        prefix = str(tmp_path / "sdk")
        for cmd in ([CMAKE, "-S", REPO, "-B", build], [CMAKE, "--build", build, "--parallel"], [CMAKE, "--install", build, "--prefix", prefix]):
            build_step(cmd)
    result = run_script("native-tests/installed/check_sdk.sh", prefix, *([build] if build else []))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "find_package(" in result.stdout and "passed" in result.stdout
    assert os.path.isdir(prefix), "check_sdk.sh must move the prefix back"


def test_install_layouts_put_the_sdk_where_libdir_says():
    require_sdk_toolchain()
    # CI containers can run as root; the script's absolute LIBDIRs are scratch paths, so the override is safe there. WHY only
    # when geteuid() == 0: a non-root run must keep the script's own root guard unexercised-but-armed, never silenced.
    env = dict(os.environ)
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        env["SEABRAID_LAYOUT_CHECK_ALLOW_ROOT"] = "1"
    else:
        env.pop("SEABRAID_LAYOUT_CHECK_ALLOW_ROOT", None)
    result = run_script("native-tests/installed/check_layouts.sh", env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "all 9 layouts passed" in result.stdout
    assert "install prefix != configure prefix: find_package, build and run passed" in result.stdout
    # WHY each line is pinned: the script prints an `ok:` line only after the assertions above it ran, so a block that
    # was deleted (or skipped by an early `exit 0`) stops printing its line and this test fails.
    for line in (
        "a ';' prefix is refused",  # default layout: the refusal is not limited to the absolute LIBDIR
        "absolute LIBDIR reinstalled",  # same prefix (also after touch -r), another, relative, quote/space/dollar/braces, ';'
        "absolute LIBDIR installed under umask 077: the package config is mode 644",
        "absolute LIBDIR with --prefix / under DESTDIR: the config names /include",
    ):
        assert line in result.stdout, f"check_layouts.sh no longer reports: {line}"


def stub_sdk(tmp_path):
    """A hand-made SDK prefix (header, versioned library, symlink) and a bin dir holding a stub nm: no cargo build needed."""
    parent = tmp_path / "parent"
    prefix = parent / "sdk"
    (prefix / "lib").mkdir(parents=True)
    (prefix / "include" / "seabraid").mkdir(parents=True)
    lib = prefix / "lib" / "libseabraid.so.0.1"
    lib.write_text("x")
    lib.chmod(0o755)
    (prefix / "lib" / "libseabraid.so").symlink_to(lib.name)
    (prefix / "include" / "seabraid" / "seabraid.h").write_text("int seabraid_stub(void);\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "nm").write_text("#!/bin/sh\necho '0000000000000000 T seabraid_stub'\n")
    (bin_dir / "nm").chmod(0o755)
    return parent, prefix, bin_dir


def stub_env(bin_dir, scratch_root):
    """The environment for a script run against stubs. WHY SEABRAID_CMAKE is dropped: it would put the real cmake before the stub."""
    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}", "TMPDIR": str(scratch_root)}
    env.pop("SEABRAID_CMAKE", None)
    return env


@pytest.mark.skipif(
    not sys.platform.startswith("linux") or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="needs Linux and a non-root user (chmod does not stop root)",
)
@pytest.mark.parametrize("checks_pass", [False, True], ids=["a-check-fails", "every-check-passes"])
def test_check_sdk_keeps_the_only_copy_when_it_cannot_move_the_sdk_back(tmp_path, checks_pass):
    # WHY stubs: the failure needs a prefix whose parent turns read-only AFTER check_sdk.sh moved the SDK away, which
    # a stub cmake does on its first call; a stub nm and a hand-made prefix keep the test to a second, with no cargo build.
    # WHY checks_pass: when a check already failed the script exits 1 whatever the restore does; only with every check
    # green (stub cmake succeeds after the refusal, stub ctest passes) does a restore that returns quietly exit 0 with
    # the SDK stranded, which is the case the exit code must catch.
    parent, prefix, bin_dir = stub_sdk(tmp_path)
    # Every cmake call fails; the first one also makes the parent read-only and prints the refusal text the script wants.
    if checks_pass:
        # The first call is the refusal check (must fail with the version text); every later call succeeds.
        marker = tmp_path / "first-call-done"
        (bin_dir / "cmake").write_text(
            f"#!/bin/sh\nif [ ! -e '{marker}' ]; then\n  : > '{marker}'\n  chmod a-w '{parent}'\n"
            "  echo 'The version found is not compatible with the version requested.'\n  exit 1\nfi\nexit 0\n"
        )
        (bin_dir / "ctest").write_text("#!/bin/sh\nexit 0\n")
    else:
        (bin_dir / "cmake").write_text(f"#!/bin/sh\nchmod a-w '{parent}'\necho 'The version found is not compatible with the version requested.'\nexit 1\n")
    for stub in bin_dir.iterdir():
        stub.chmod(0o755)
    scratch_root = tmp_path / "tmp"
    scratch_root.mkdir()
    env = stub_env(bin_dir, scratch_root)
    try:
        result = run_script("native-tests/installed/check_sdk.sh", str(prefix), timeout=60, env=env)
        assert result.returncode == 1, (result.returncode, result.stdout, result.stderr)
        if checks_pass:
            assert "consumer run all passed" in result.stdout, "the checks were meant to pass before the move back failed: " + result.stdout + result.stderr
        assert "COULD NOT MOVE THE SDK BACK" in result.stderr, result.stderr
        kept = [p for p in scratch_root.iterdir() if (p / "relocated-prefix" / "lib" / "libseabraid.so.0.1").exists()]
        assert len(kept) == 1, f"the scratch directory holding the only copy was deleted: {list(scratch_root.iterdir())}"
        assert str(kept[0] / "relocated-prefix") in result.stderr, "the message must name where the SDK is"
    finally:
        parent.chmod(parent.stat().st_mode | stat.S_IWUSR)


def test_a_libdir_the_install_rules_cannot_carry_is_refused_at_configure(tmp_path):
    # The LIBDIR is pasted into install(CODE) text that CMake parses again at install time; see CMakeLists.txt.
    require_sdk_toolchain(linux_only=False)
    # ';' splits a list; a leading '~' is IS_ABSOLUTE to CMake yet installs a directory named '~lib' under the cwd.
    for var, bad in [("LIBDIR", b) for b in ('lib"x', "lib$x", "lib\\x", "lib\nx", "lib;x", "~lib", "~")] + [("INCLUDEDIR", "inc;x"), ("INCLUDEDIR", "~inc")]:
        result = subprocess.run([CMAKE, "-S", REPO, "-B", str(tmp_path / "b"), f"-DCMAKE_INSTALL_{var}={bad}"], capture_output=True, text=True, timeout=120)
        assert result.returncode != 0 and f"CMAKE_INSTALL_{var}" in result.stderr and "cannot carry it" in result.stderr, (var, bad, result.stderr)
        shutil.rmtree(tmp_path / "b", ignore_errors=True)
    # Not refused: a '~' that is not leading, and a plain absolute path.
    for ok in ("lib~x", "/opt/seabraid/lib"):
        result = subprocess.run([CMAKE, "-S", REPO, "-B", str(tmp_path / "b"), f"-DCMAKE_INSTALL_LIBDIR={ok}"], capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, (ok, result.stderr)
        shutil.rmtree(tmp_path / "b", ignore_errors=True)


def test_editing_the_header_epoch_reconfigures_the_package_config(tmp_path):
    require_sdk_toolchain(linux_only=False)
    # The header owns SEABRAID_ABI_EPOCH and the package config republishes it; a stale config would tell consumers the wrong epoch.
    tree = tmp_path / "src"
    shutil.copytree(REPO, tree, ignore=shutil.ignore_patterns(".git", "target", "build", "build-*","__pycache__", ".pytest_cache"))
    build = tmp_path / "b"
    build_step([CMAKE, "-S", str(tree), "-B", str(build)])

    def published() -> str:
        text = (build / "SeabraidConfig.cmake").read_text(encoding="utf-8")
        return next(line for line in text.splitlines() if line.startswith("set(Seabraid_ABI_EPOCH"))

    header = tree / "include" / "seabraid" / "seabraid.h"
    text = header.read_text(encoding="utf-8")
    before = re.search(r"#define SEABRAID_ABI_EPOCH ([0-9]+)u", text).group(1)
    assert published() == f"set(Seabraid_ABI_EPOCH {before})"
    header.write_text(text.replace(f"#define SEABRAID_ABI_EPOCH {before}u", "#define SEABRAID_ABI_EPOCH 77u"), encoding="utf-8")
    # Building the library target runs the build system's regenerate check first (`--target help` does not).
    build_step([CMAKE, "--build", str(build), "--target", "seabraid_runtime_build"])
    assert published() == "set(Seabraid_ABI_EPOCH 77)"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="needs POSIX sh")
def test_check_layouts_refuses_to_run_as_root_without_the_override(tmp_path):
    # A stub `id` says uid 0; the script must stop before configuring anything (a DESTDIR regression as root writes the host).
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "id").write_text("#!/bin/sh\necho 0\n")
    (bin_dir / "id").chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k != "SEABRAID_LAYOUT_CHECK_ALLOW_ROOT"}
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    result = run_script("native-tests/installed/check_layouts.sh", timeout=60, env=env)
    assert result.returncode != 0 and "refusing to run as root" in result.stderr, result.stderr


REFUSAL_WORDINGS = {
    "cmake-4.x": ("The version found is not compatible with the version requested.", True),
    # The 3.22.6 text as measured, line wrapping included.
    "cmake-3.22": (
        'CMake Error at CMakeLists.txt:3 (find_package):\n  Could not find a configuration file for package "Seabraid" that is\n  compatible with requested version "0.6".\n\n'
        "  The following configuration files were considered but not accepted:\n\n    /x/lib/cmake/Seabraid/SeabraidConfig.cmake, version: 0.5.0\n",
        True,
    ),
    # The 3.22 sentence without the listing of this SDK's version (no config was considered), a listing of another
    # version, and an unrelated failure (a missing compiler) are not refusals.
    "cmake-3.22-sentence-without-the-listing": ('Could not find a configuration file for package "Seabraid" that is\n  compatible with requested version "0.6".', False),
    "listing-of-another-version": ('Could not find a configuration file for package "Seabraid" that is\n  compatible with requested version "0.6".\n    /x/SeabraidConfig.cmake, version: 7.7.7\n', False),
    "unrelated-failure": ("CMake Error: No CMAKE_C_COMPILER could be found.", False),
}


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="needs Linux and POSIX sh")
@pytest.mark.parametrize("name", list(REFUSAL_WORDINGS))
def test_check_sdk_accepts_the_version_refusal_by_meaning_on_both_cmake_wordings_and_nothing_else(tmp_path, name):
    # WHY stubs: the wording is CMake's, so each CMake series is stood in for by its measured text. A stub that prints the
    # text and fails on every call tells "accepted as the refusal" (the script goes on and stops at the NEXT cmake call:
    # "configure against the moved prefix failed") from "rejected" ("not because the SDK's version file refused it").
    text, accepted = REFUSAL_WORDINGS[name]
    parent, prefix, bin_dir = stub_sdk(tmp_path)
    (bin_dir / "cmake").write_text(f"#!/bin/sh\ncat <<'EOT'\n{text}\nEOT\nexit 1\n")
    (bin_dir / "cmake").chmod(0o755)
    scratch_root = tmp_path / "tmp"
    scratch_root.mkdir()
    result = run_script("native-tests/installed/check_sdk.sh", str(prefix), timeout=60, env=stub_env(bin_dir, scratch_root))
    assert result.returncode == 1, (result.stdout, result.stderr)
    assert ("is refused by the" in result.stdout) == accepted, (result.stdout, result.stderr)
    assert ("configure against the moved prefix failed" in result.stderr) == accepted, result.stderr
    assert ("not because the SDK's version file refused it" in result.stderr) == (not accepted), result.stderr
    assert (prefix / "lib" / "libseabraid.so.0.1").exists(), "the prefix was not moved back"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="needs Linux and POSIX sh")
def test_seabraid_cmake_is_honoured_or_refused(tmp_path):
    # use_cmake.sh is sourced by both scripts; running it alone shows which cmake they would use, with no toolchain.
    script = os.path.join(REPO, "native-tests", "installed", "use_cmake.sh")
    fake = tmp_path / "bin" / "cmake"
    fake.parent.mkdir()
    fake.write_text("#!/bin/sh\necho 'cmake version 9.9.9'\n")
    fake.chmod(0o755)
    other = tmp_path / "bin" / "cmake-3.22"
    other.write_text("#!/bin/sh\n")
    other.chmod(0o755)

    def source(value):
        env = {**os.environ, "SEABRAID_CMAKE": value} if value is not None else {k: v for k, v in os.environ.items() if k != "SEABRAID_CMAKE"}
        return subprocess.run(["sh", "-c", '. "$1"; scratch=$(mktemp -d); use_seabraid_cmake "$scratch"; command -v cmake; echo "shim=$(ls "$scratch")"; rm -rf "$scratch"', "_", script], capture_output=True, text=True, timeout=60, env=env)

    ok = source(str(fake))
    assert ok.returncode == 0 and f"using {fake}: cmake version 9.9.9" in ok.stdout and "shim=cmake-shim" in ok.stdout, (ok.stdout, ok.stderr)
    assert "/cmake-shim/cmake" in ok.stdout, "the shim must be first on PATH, not the binary's own directory (which may be /usr/bin)"
    assert source("").returncode == 0 and "using " not in source("").stdout, "an empty SEABRAID_CMAKE must leave PATH alone"
    assert source(None).returncode == 0 and "using " not in source(None).stdout
    for bad in (str(tmp_path / "missing" / "cmake"), str(tmp_path / "bin"), str(other)):
        refused = source(bad)
        assert refused.returncode == 1 and "is not an executable file named cmake" in refused.stderr, (bad, refused.stdout, refused.stderr)
