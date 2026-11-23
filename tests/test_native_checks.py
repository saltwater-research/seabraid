# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""native-tests/check_output.cmake must fail on a bad exit code even when the output is right (CTest's own
PASS_REGULAR_EXPRESSION does not).  Needs cmake and a POSIX shell; skipped otherwise."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
import warnings

import pytest
from conftest import REPO

# WHY SEABRAID_CMAKE: the same variable native-tests/installed/use_cmake.sh honours, so one setting tests the CMake floor.
CMAKE = os.environ.get("SEABRAID_CMAKE") or shutil.which("cmake")
SCRIPT = os.path.join(REPO, "native-tests", "check_output.cmake")
needs = pytest.mark.skipif(CMAKE is None or sys.platform == "win32", reason="needs cmake and a POSIX shell")


def run_check(*defs: str, cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([CMAKE, *defs, "-P", SCRIPT, "--", *cmd], capture_output=True, text=True, timeout=60)


def sh(script: str) -> list[str]:
    return ["sh", "-c", script]


@needs
def test_good_exit_and_matching_output_passes():
    assert run_check("-DEXPECT_REGEX=(^|\n)ok\n", cmd=sh("echo ok")).returncode == 0


@needs
def test_matching_output_with_a_failing_exit_code_fails():
    # The case CTest's PASS_REGULAR_EXPRESSION lets through.
    r = run_check("-DEXPECT_REGEX=ok", cmd=sh("echo ok; exit 3"))
    assert r.returncode != 0 and "exit code was '3', expected '0'" in r.stderr


@needs
def test_expected_nonzero_exit_code_passes_and_wrong_one_fails():
    assert run_check("-DEXPECT_REGEX=usage", "-DEXPECT_RC=2", cmd=sh("echo usage; exit 2")).returncode == 0
    assert run_check("-DEXPECT_REGEX=usage", "-DEXPECT_RC=2", cmd=sh("echo usage; exit 1")).returncode != 0


@needs
def test_good_exit_with_missing_output_fails():
    r = run_check("-DEXPECT_REGEX=output = ", cmd=sh("echo nothing"))
    assert r.returncode != 0 and "does not match" in r.stderr


@needs
def test_stderr_is_checked_too():
    assert run_check("-DEXPECT_REGEX=warned", cmd=sh("echo warned >&2")).returncode == 0


@needs
def test_forbidden_output_fails():
    r = run_check("-DEXPECT_REGEX=ok", "-DFORBID_REGEX=ok:false", cmd=sh("echo ok:false"))
    assert r.returncode != 0 and "forbidden" in r.stderr


@needs
def test_timeout_fails_and_names_the_timeout():
    # WHY the command prints the expected text first: without TIMEOUT it runs 5 s, exits 0 with matching output, and
    # the check PASSES, so this test fails exactly when the timeout handling is missing. (A bare `sleep 5` could not
    # tell: it also failed on the regex alone.)
    r = run_check("-DEXPECT_REGEX=x", "-DTIMEOUT=1", cmd=sh("echo x; sleep 5"))
    assert r.returncode != 0 and "timed out after 1 seconds" in r.stderr


@needs
def test_a_timeout_fails_even_when_the_expected_rc_names_it():
    r = run_check("-DEXPECT_REGEX=x", "-DTIMEOUT=1", "-DEXPECT_RC=Process terminated due to timeout", cmd=sh("echo x; sleep 5"))
    assert r.returncode != 0 and "timed out" in r.stderr


@needs
def test_empty_arguments_reach_the_command():
    # execute_process(COMMAND ${list}) dropped empty arguments: $# was 0.
    assert run_check("-DEXPECT_REGEX=n=2", cmd=["sh", "-c", 'echo "n=$#"', "_", "", ""]).returncode == 0


@needs
def test_semicolons_brackets_and_leading_newlines_arrive_unchanged():
    assert run_check(r"-DEXPECT_REGEX=\[a;b ]=] c\]", cmd=["sh", "-c", 'echo "[$1]"', "_", "a;b ]=] c"]).returncode == 0
    assert run_check("-DEXPECT_REGEX=len=3", cmd=["sh", "-c", 'echo "len=${#1}"', "_", "\nab"]).returncode == 0


@needs
def test_a_command_named_false_is_run_not_mistaken_for_no_command():
    assert run_check("-DEXPECT_REGEX=^$", "-DEXPECT_RC=1", cmd=["false"]).returncode == 0


@needs
def test_an_execute_process_keyword_argument_is_refused_loudly():
    r = run_check("-DEXPECT_REGEX=x", cmd=["echo", "OUTPUT_VARIABLE"])
    assert r.returncode != 0 and "'OUTPUT_VARIABLE' is an execute_process keyword" in r.stderr


@needs
def test_missing_command_or_regex_fails():
    assert subprocess.run([CMAKE, "-DEXPECT_REGEX=x", "-P", SCRIPT], capture_output=True, text=True).returncode != 0
    assert subprocess.run([CMAKE, "-P", SCRIPT, "--", "true"], capture_output=True, text=True).returncode != 0


def test_the_consumer_copy_of_check_output_matches_the_repository_one():
    # WHY a copy: examples/consumer is meant to be copied out of the tree on its own; the copy must not drift.
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    assert (root / "examples/consumer/check_output.cmake").read_bytes() == (root / "native-tests/check_output.cmake").read_bytes()


@needs
@pytest.mark.parametrize("value", ["abc", "-1", "1.5", "5\nfile(WRITE {marker} injected)"])
def test_a_timeout_that_is_not_a_whole_number_is_refused_and_never_executed(tmp_path, value):
    # The raw value used to be pasted into the text cmake_language(EVAL) runs: a newline plus file(WRITE ...) ran, and
    # "abc" was a parse error that never mentioned TIMEOUT.
    marker = tmp_path / "injected"
    r = run_check("-DEXPECT_REGEX=x", "-DTIMEOUT=" + value.replace("{marker}", marker.as_posix()), cmd=sh("echo x"))
    assert r.returncode != 0 and "-DTIMEOUT must be a whole number of seconds" in r.stderr, r.stderr
    assert not marker.exists(), "the TIMEOUT text was executed"


@needs
def test_a_timeout_returns_promptly_and_leaves_a_grandchild_while_a_normal_exit_waits_for_it():
    # Pins the wording in the script's header: the delay from a pipe-holding grandchild is on the normal-exit path.
    start = time.monotonic()
    r = run_check("-DEXPECT_REGEX=x", "-DTIMEOUT=1", cmd=sh("echo x; sleep 4 & sleep 30"))
    assert r.returncode != 0 and "timed out" in r.stderr
    assert time.monotonic() - start < 3.5, "the timeout path waited for the grandchild"
    start = time.monotonic()
    assert run_check("-DEXPECT_REGEX=x", cmd=sh("echo x; sleep 3 &")).returncode == 0
    assert time.monotonic() - start >= 2.5, "the normal-exit path did not wait for the pipe holder"


# The newest CMake whose execute_process synopsis was compared with check_output.cmake's keyword list (4.2.3 on the build
# box). WHY a ceiling: a CMake NEWER than this may document another keyword, which is a reason to review the list, not a
# reason to break the release gate of a runner whose CMake merely updated. Older CMakes document a subset of the list.
KEYWORDS_CHECKED_UP_TO = (4, 2)


def cmake_version() -> tuple[int, int]:
    out = subprocess.run([CMAKE, "--version"], capture_output=True, text=True, timeout=60).stdout
    major, minor = re.search(r"version (\d+)\.(\d+)", out).groups()
    return int(major), int(minor)


def keyword_verdict(missing: list[str], version: tuple[int, int]) -> str:
    """pass when nothing is missing; fail when something is missing on a CMake the list was checked against; else xfail."""
    if not missing:
        return "pass"
    return "xfail" if version > KEYWORDS_CHECKED_UP_TO else "fail"


@needs
def test_a_timeout_of_zero_is_not_no_limit_it_times_out_at_once():
    # check_output.cmake documents this in its header. WHY no version guard: the script decides 0 itself instead of passing
    # it to execute_process (whose meaning differs by version, pinned below), so the contract holds on every CMake.
    start = time.monotonic()
    r = run_check("-DEXPECT_REGEX=x", "-DTIMEOUT=0", cmd=sh("echo x; sleep 3"))
    assert r.returncode != 0 and "timed out after 0 seconds" in r.stderr, r.stderr
    assert time.monotonic() - start < 2.5, "the command was started and waited for"


# What execute_process(TIMEOUT 0) itself does, per CMake series, MEASURED (echo x; sleep 2 under TIMEOUT 0):
#   3.22.6: no limit (exit 0, output "x", ran the full 2 s)      4.2.3: "Process terminated due to timeout" at once.
# Nothing between 3.22 and 4.2, or beyond 4.2, was measured, so those versions xfail loudly instead of guessing. This
# test explains why check_output.cmake special-cases 0; if a series is measured, add it here.
MEASURED_TIMEOUT_ZERO = {(3, 22): "no limit", (4, 2): "times out at once"}


@needs
def test_what_execute_process_does_on_a_timeout_of_zero_is_recorded_per_measured_version(tmp_path):
    probe = tmp_path / "probe.cmake"
    probe.write_text('execute_process(COMMAND sh -c "echo x; sleep 2" TIMEOUT 0 RESULT_VARIABLE rc OUTPUT_VARIABLE out)\nmessage("RC=[${rc}] OUT=[${out}]")\n')
    start = time.monotonic()
    r = subprocess.run([CMAKE, "-P", str(probe)], capture_output=True, text=True, timeout=60)
    took = time.monotonic() - start
    version = cmake_version()
    observed = "times out at once" if "terminated due to timeout" in r.stderr and took < 1.5 else "no limit" if "RC=[0]" in r.stderr and took >= 1.5 else f"unknown ({r.stderr!r}, {took:.1f}s)"
    expected = MEASURED_TIMEOUT_ZERO.get(version)
    if expected is None:
        message = f"CMake {version[0]}.{version[1]} was never measured for execute_process(TIMEOUT 0); it behaves as: {observed}"
        warnings.warn(message)
        pytest.xfail(message)
    assert observed == expected, f"CMake {version[0]}.{version[1]}: execute_process(TIMEOUT 0) is now: {observed}; update MEASURED_TIMEOUT_ZERO and check_output.cmake's header"


@needs
def test_the_keyword_list_covers_every_keyword_this_cmake_documents_for_execute_process():
    # execute_process reads an argument equal to its own keyword as that keyword, so the script refuses those words;
    # a CMake release that adds a keyword must be noticed. Up to the pinned version a missing keyword FAILS; beyond it
    # the test warns and xfails (see KEYWORDS_CHECKED_UP_TO) so a runner image update does not turn the gate red.
    doc = subprocess.run([CMAKE, "--help-command", "execute_process"], capture_output=True, text=True, timeout=60).stdout
    synopsis = doc[doc.index("execute_process(") :]
    synopsis = synopsis[: synopsis.index("\n\n")]
    documented = set(re.findall(r"\b[A-Z][A-Z_]+\b", re.sub(r"<[^>]*>", "", synopsis)))
    text = open(SCRIPT, encoding="utf-8").read()
    listed = set(re.search(r"set\(_keywords ([^)]*)\)", text).group(1).split())
    assert documented, "the synopsis parser found nothing"
    missing = sorted(documented - listed)
    version = cmake_version()
    verdict = keyword_verdict(missing, version)
    if verdict == "xfail":
        message = f"CMake {version[0]}.{version[1]} documents execute_process keywords that check_output.cmake lacks: {missing}; review the list and raise KEYWORDS_CHECKED_UP_TO"
        warnings.warn(message)
        pytest.xfail(message)
    assert verdict == "pass", f"keywords missing from check_output.cmake: {missing}"


@pytest.mark.parametrize(
    "version, missing, verdict",
    [((4, 2), ["C"], "fail"), ((3, 22), ["C"], "fail"), ((4, 3), ["C"], "xfail"), ((5, 0), ["C"], "xfail"), ((4, 3), [], "pass"), ((4, 2), [], "pass")],
)
def test_the_keyword_pin_fails_inside_the_checked_range_and_xfails_beyond_it(version, missing, verdict):
    # The decision rule alone, so the pin is tested without needing a CMake newer than it.
    assert keyword_verdict(missing, version) == verdict
