# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Policy tests for .github/workflows/ci.yml, dependabot.yml and deny.toml.  Actions cannot be run locally, so the
policy (least privilege, pinned actions, required jobs and steps) is asserted on the file itself."""

from __future__ import annotations

import glob
import importlib.util
import os
import re
import tomllib

import pytest
from conftest import REPO, code_only

WORKFLOW = os.path.join(REPO, ".github", "workflows", "ci.yml")
TEXT = open(WORKFLOW, encoding="utf-8").read()
try:
    import yaml as _yaml
except ImportError:  # the parse-level tests skip; text-level tests still guard the policy
    _yaml = None


def parsed() -> dict:
    if _yaml is None:
        pytest.skip("PyYAML not installed")
    return _yaml.safe_load(TEXT)


def test_workflow_text_policy():
    assert "pull_request_target" not in TEXT
    assert "secrets." not in TEXT and "${{ secrets" not in TEXT
    assert not re.search(r"\b(curl|wget)\b", TEXT), "no downloads piped into a shell"
    assert not re.search(r"-latest\b", TEXT), "runner images must be pinned (ubuntu-24.04, windows-2022, macos-15)"
    assert re.search(r"(?m)^permissions:\n  contents: read\n", TEXT)


def test_every_action_is_pinned_by_full_sha_with_a_tag_comment():
    uses = re.findall(r"(?m)^\s*-?\s*uses:\s*(\S+)(.*)$", TEXT)
    assert uses, "no actions found; the pin test would be vacuous"
    for ref, tail in uses:
        assert re.fullmatch(r"[\w.-]+/[\w./-]+@[0-9a-f]{40}", ref), f"not pinned by 40-char SHA: {ref}"
        assert re.fullmatch(r"\s+#\s*v\d+(\.\d+)*\s*", tail), f"missing version tag comment on {ref}: {tail!r}"


def test_linux_installs_the_optional_test_dependencies_pinned():
    # Without PyYAML and numpy the parse-level policy tests and the numpy provider tests skip, and a skip in CI reads as a pass.
    assert re.search(r"pip install [^\n]*\bPyYAML==\d+\.\d+\.\d+", TEXT) and re.search(r"pip install [^\n]*\bnumpy==\d+\.\d+\.\d+", TEXT)


def test_no_retired_macos_runner():
    assert not re.search(r"macos-1[0-4]", TEXT), "GitHub retired the macOS 14 images on 2026-11-02"


def test_python_matrix_includes_the_pyproject_floor():
    with open(os.path.join(REPO, "pyproject.toml"), "rb") as fh:
        floor = tomllib.load(fh)["project"]["requires-python"].lstrip(">=").strip()
    versions = parsed()["jobs"]["linux"]["strategy"]["matrix"]["python-version"]
    assert floor in versions and len(versions) >= 2, versions
    py_steps = [s for s in parsed()["jobs"]["linux"]["steps"] if "setup-python" in s.get("uses", "")]
    assert py_steps and all(s["with"]["python-version"] == "${{ matrix.python-version }}" for s in py_steps)


def test_msrv_job_checks_the_declared_rust_version():
    with open(os.path.join(REPO, "Cargo.toml"), "rb") as fh:
        msrv = tomllib.load(fh)["package"]["rust-version"]
    run = "\n".join(s.get("run", "") for s in parsed()["jobs"]["msrv"]["steps"])
    assert f"rustup toolchain install {msrv}" in run and f"cargo +{msrv} check --all-targets" in run


def test_triggers_and_token_permissions():
    doc = parsed()
    triggers = doc.get("on", doc.get(True))
    assert set(triggers) == {"push", "pull_request", "workflow_dispatch"}
    assert doc["permissions"] == {"contents": "read"}
    for name, job in doc["jobs"].items():
        assert "permissions" not in job, f"job {name} must not widen the token"


def test_required_jobs_and_runners():
    jobs = parsed()["jobs"]
    assert {n: j["runs-on"] for n, j in jobs.items()} == {"linux": "ubuntu-24.04", "journey": "ubuntu-24.04", "windows": "windows-2022", "macos": "macos-15", "wheels": "ubuntu-24.04", "wheels-other": "${{ matrix.os }}", "msrv": "ubuntu-24.04", "sanitizers": "ubuntu-24.04", "sdk": "ubuntu-24.04", "strata-adapter": "ubuntu-24.04", "cmake-floor": "ubuntu-24.04", "miri": "ubuntu-24.04", "deny": "ubuntu-24.04"}


def test_cmake_floor_job_builds_and_checks_the_sdk_with_exactly_the_promised_minimum():
    # The README's "CMake 3.22 or newer" is a claim; this job is its checker (two checks once held only on 4.x).
    job = parsed()["jobs"]["cmake-floor"]
    text = "\n".join(s.get("run", "") for s in job["steps"])
    with open(os.path.join(REPO, "CMakeLists.txt"), encoding="utf-8") as fh:
        minimum = re.search(r"cmake_minimum_required\(VERSION (\d+\.\d+)", fh.read()).group(1)
    assert f"cmake=={minimum}." in text and f"grep -F {minimum}." in text
    for needle in ("SEABRAID_CMAKE=", "ctest\" --test-dir build322", "check_sdk.sh", "check_layouts.sh",
                   "tests/test_native_checks.py tests/test_build_config.py"):
        assert needle in text, needle


def test_linux_first_windows_and_macos_run_only_by_hand():
    # Owner, 2026-11-19: pushes run Linux only; the non-Linux jobs bill 2x/10x minutes and wait for a manual start.
    jobs = parsed()["jobs"]
    manual = {n for n, j in jobs.items() if j.get("if") == "github.event_name == 'workflow_dispatch'"}
    assert manual == {"windows", "macos", "wheels-other"}
    pushed = [j for n, j in jobs.items() if n not in manual]
    assert pushed and all("windows" not in str(j["runs-on"]) and "macos" not in str(j["runs-on"]) for j in pushed)


def test_the_sdk_tests_are_required_on_exactly_one_linux_leg_and_skipped_on_the_others():
    # WHY the matrix check: the `if:` expressions below name 3.11; if the matrix drops it, no leg requires the SDK tests,
    # no leg builds the SDK, and every assertion here would still pass.
    assert "3.11" in parsed()["jobs"]["linux"]["strategy"]["matrix"]["python-version"], "no matrix leg would build the SDK"
    steps = [s for s in parsed()["jobs"]["linux"]["steps"] if "pytest tests -q" in s.get("run", "") and "working-directory" not in s]
    required = [s for s in steps if s.get("env", {}).get("SEABRAID_REQUIRE_SDK_TESTS") == "1"]
    skipped = [s for s in steps if s.get("env", {}).get("SEABRAID_SKIP_SDK_TESTS") == "1"]
    assert len(required) == 1 and required[0]["if"] == "matrix.python-version == '3.11'", "exactly one leg requires the SDK tests"
    assert len(skipped) == 1 and skipped[0]["if"] == "matrix.python-version != '3.11'", "every other leg skips them"
    assert len(steps) == 2, "a third pytest step would build the SDK again"


def test_linux_job_runs_every_required_step():
    run = "\n".join(s.get("run", "") for s in parsed()["jobs"]["linux"]["steps"])
    for needle in (
        "cargo fmt --check",
        "cargo clippy",
        "-D warnings",
        "cargo test",
        "-DSEABRAID_BUILD_NATIVE_TESTS=ON",
        "ctest",
        "python -m pytest tests -q",
        "python -B -m unittest discover -s bindings/python",
    ):
        assert needle in run, needle
    assert any(s.get("working-directory") == "compat/strata" and "pytest tests" in s["run"] for s in parsed()["jobs"]["linux"]["steps"])


def _checkers_ci_runs() -> set[str]:
    """Checkers CI runs: each named in a `python checks/check_X.py` line, plus everything checks/run_all.py runs."""
    runs = _runs("linux")
    named = set(re.findall(r"python checks/(check_\w+\.py)", runs))
    if "python checks/run_all.py" in runs:
        spec = importlib.util.spec_from_file_location("run_all", os.path.join(REPO, "checks", "run_all.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        named |= set(mod.checkers())
    return named


def test_the_checkers_ci_runs_are_exactly_the_glob_except_the_post_rename_gate():
    # check_names.py needs --forbid NAME: it runs only after a rename.
    glob_set = {os.path.basename(p) for p in glob.glob(os.path.join(REPO, "checks", "check_*.py"))} - {"check_names.py"}
    assert _checkers_ci_runs() == glob_set


def test_ci_runs_the_checkers_through_run_all_with_the_strict_runtime_flag():
    assert "python checks/run_all.py" in _runs("linux")
    text = open(os.path.join(REPO, "checks", "run_all.py"), encoding="utf-8").read()
    assert '"check_error_paths.py": ["--strict-runtime"]' in text
    assert "lizard" in _runs("linux"), "the complexity checker inside run_all needs lizard installed first"


def test_journey_job_replays_the_first_time_user_journey_online_in_a_fresh_archive():
    job = parsed()["jobs"]["journey"]
    assert "if" not in job and "permissions" not in job
    with open(os.path.join(REPO, "pyproject.toml"), "rb") as fh:
        floor = tomllib.load(fh)["project"]["requires-python"].lstrip(">=").strip()
    assert floor in job["strategy"]["matrix"]["python-version"] and len(job["strategy"]["matrix"]["python-version"]) >= 3
    run = _runs("journey")
    assert run.index("git archive HEAD") < run.index("scripts/journey/run.py")
    replay = next(s["run"] for s in job["steps"] if "scripts/journey/run.py" in s.get("run", ""))
    assert "--offline" not in replay and "--tree" in replay and "--out" in replay  # CI has network; the record stays out of the tree
    assert any(s.get("with", {}).get("python-version") == "${{ matrix.python-version }}" for s in job["steps"])


def test_windows_and_macos_run_cargo_and_ctest():
    jobs = parsed()["jobs"]
    for name in ("windows", "macos"):
        run = "\n".join(s.get("run", "") for s in jobs[name]["steps"])
        assert "cargo test" in run and "ctest" in run and "-DSEABRAID_BUILD_NATIVE_TESTS=ON" in run


def test_sanitizer_job_instruments_both_languages():
    env = parsed()["jobs"]["sanitizers"]["env"]
    for var in ("CFLAGS", "CXXFLAGS"):
        assert "-fsanitize=address,undefined" in env[var] and "-fno-omit-frame-pointer" in env[var]


def test_deny_job_uses_the_policy_file():
    step = next(s for s in parsed()["jobs"]["deny"]["steps"] if "cargo-deny-action" in s.get("uses", ""))
    assert "advisories" in step["with"]["command"] and "licenses" in step["with"]["command"]


def test_deny_toml_policy():
    with open(os.path.join(REPO, "deny.toml"), "rb") as fh:
        cfg = tomllib.load(fh)
    assert sorted(cfg["licenses"]["allow"]) == ["Apache-2.0", "MIT", "Unicode-3.0"]
    assert cfg["sources"]["unknown-registry"] == "deny" and cfg["sources"]["unknown-git"] == "deny"
    assert cfg["sources"]["allow-registry"] == ["https://github.com/rust-lang/crates.io-index"] and cfg["sources"]["allow-git"] == []
    assert {"reqwest", "hyper", "ureq"} <= {b["name"] for b in cfg["bans"]["deny"]}
    assert cfg["advisories"]["yanked"] == "deny"


def test_dependabot_covers_actions_cargo_and_pip():
    if _yaml is None:
        pytest.skip("PyYAML not installed")
    cfg = _yaml.safe_load(open(os.path.join(REPO, ".github", "dependabot.yml"), encoding="utf-8"))
    assert {u["package-ecosystem"] for u in cfg["updates"]} == {"github-actions", "cargo", "pip"}
    assert all(u["schedule"]["interval"] == "weekly" for u in cfg["updates"])


def _runs(job: str) -> str:
    return "\n".join(s.get("run", "") for s in parsed()["jobs"][job]["steps"])


def test_no_job_or_step_is_allowed_to_fail():
    assert "continue-on-error" not in TEXT


def test_sdk_job_installs_the_sdk_and_builds_the_downstream_consumer():
    run = _runs("sdk")
    assert "cmake --install" in run and "native-tests/installed" in run
    assert run.index("cmake --install") < run.index("native-tests/installed")
    # The installed check is one script, run against the prefix and the build tree that installed it (for the manifest).
    assert 'sh native-tests/installed/check_sdk.sh "$RUNNER_TEMP/seabraid-sdk" build-sdk' in run
    script = code_only(open(os.path.join(REPO, "native-tests", "installed", "check_sdk.sh"), encoding="utf-8").read())
    assert "CMAKE_PREFIX_PATH" in script and "ctest" in script and "find_package" in script
    assert script.index('mv "$prefix" "$moved"') < script.index("-DCMAKE_PREFIX_PATH")  # relocate first, then consume


def test_sdk_job_runs_before_the_example_consumer_that_needs_the_restored_prefix():
    # check_sdk.sh moves the prefix and moves it back; the later steps must find it in place.
    steps = parsed()["jobs"]["sdk"]["steps"]
    runs = [s.get("run", "") for s in steps]
    check = next(i for i, r in enumerate(runs) if "check_sdk.sh" in r)
    later = [i for i, r in enumerate(runs) if "examples/consumer" in r or "ggml/sdk_check" in r]
    assert later, "no step builds the consumer or the ggml check, so the ordering below would hold vacuously"
    assert all(i > check for i in later)


def test_sdk_job_builds_and_runs_the_minimal_consumer_example_against_the_installed_sdk():
    job = parsed()["jobs"]["sdk"]
    assert "if" not in job
    steps = [s for s in job["steps"] if "examples/consumer" in s.get("run", "")]
    assert len(steps) == 1 and "if" not in steps[0]
    run = steps[0]["run"]
    assert "CMAKE_PREFIX_PATH" in run and "output = [2.5, 5]" in run
    assert _runs("sdk").index("cmake --install") < _runs("sdk").index("examples/consumer")
    text = code_only(open(os.path.join(REPO, "examples", "consumer", "CMakeLists.txt"), encoding="utf-8").read())
    assert "find_package(Seabraid CONFIG REQUIRED)" in text and "Seabraid::runtime" in text and "../minimal_provider.c" in text
    root_min = re.search(r"cmake_minimum_required\(VERSION ([0-9.]+)\)", open(os.path.join(REPO, "CMakeLists.txt"), encoding="utf-8").read())
    assert "cmake_minimum_required(VERSION %s)" % root_min.group(1) in text


def test_sdk_job_links_the_ggml_adapter_from_a_pinned_ggml_source():
    steps = parsed()["jobs"]["sdk"]["steps"]
    assert "sh adapters/ggml/sdk_check/run.sh" in _runs("sdk")
    fetch = [s for s in steps if s.get("with", {}).get("repository") == "ggml-org/llama.cpp"]
    assert len(fetch) == 1, "exactly one checkout of the engine source"
    pin = fetch[0]["with"]["ref"]
    assert re.fullmatch(r"[0-9a-f]{40}", pin), "the engine source is pinned by full commit SHA, not a tag or branch"
    assert pin in open(os.path.join(REPO, "adapters", "ggml", "MAP.md"), encoding="utf-8").read(), "the pin is the commit the adapter documents"
    assert steps.index(fetch[0]) < [i for i, s in enumerate(steps) if "run.sh" in s.get("run", "")][0]


def test_the_cargo_build_target_check_runs_in_linux_ctest_and_is_skipped_by_the_sanitizer_job():
    assert "seabraid_cargo_build_target" in open(os.path.join(REPO, "CMakeLists.txt"), encoding="utf-8").read()
    assert "seabraid_cargo_build_target" in _runs("sanitizers")
    assert "seabraid_cargo_build_target" not in _runs("linux"), "linux runs every ctest test, including this one"


def test_linux_builds_the_load_simulator():
    assert "-DSEABRAID_BUILD_TOOLS=ON" in _runs("linux")


def _unsafe_test_modules() -> list[str]:
    """Rust test module paths under src/ffi and src/diag, derived from the tree (the modules that call into unsafe code).

    A file is a test module when its module path has a ``tests`` segment; a file with an inline ``mod tests {`` adds ``<path>::tests``.
    """
    found = set()
    for top in ("ffi", "diag"):
        for path in glob.glob(os.path.join(REPO, "src", top, "**", "*.rs"), recursive=True) + [os.path.join(REPO, "src", top + ".rs")]:
            module = os.path.relpath(path, os.path.join(REPO, "src"))[:-3].replace(os.sep, "::")
            if "tests" in module.split("::"):
                found.add(module)
            elif re.search(r"^\s*mod tests\s*\{", open(path, encoding="utf-8").read(), re.M):
                found.add(module + "::tests")
    return sorted(found)


def _miri_filters(run: str) -> list[str]:
    return run.split(" -- ", 1)[1].split()


def _unmatched(filters: list[str], modules: list[str]) -> list[str]:
    """libtest filters are plain substrings of the full test name, which starts with the module path."""
    return [m for m in modules if not any(f in m + "::a_test" for f in filters)]


def test_the_unsafe_test_module_derivation_sees_the_placement_tests():
    modules = _unsafe_test_modules()
    assert "ffi::placement::tests::handle" in modules and "ffi::tests" in modules and "diag::tests" in modules


def test_the_unmatched_check_catches_the_filter_that_dropped_the_placement_tests():
    stale = ["ffi::tests", "ffi::sized::tests", "diag::tests"]  # the filter before the placement move
    assert any(m.startswith("ffi::placement::tests") for m in _unmatched(stale, _unsafe_test_modules()))


def _miri_steps(job: dict) -> list[tuple[str, list[str]]]:
    """(effective MIRIFLAGS, libtest filters) of every step that runs Miri: a step's env overrides the job's."""
    return [
        (step.get("env", {}).get("MIRIFLAGS", job["env"]["MIRIFLAGS"]), _miri_filters(step["run"]))
        for step in job["steps"]
        if "cargo +nightly miri test" in step.get("run", "")
    ]


def _aliasing_model_problems(job: dict) -> list[str]:
    """Why `job` does not run the same Miri tests once under Stacked Borrows (the default) and once under Tree Borrows."""
    steps = _miri_steps(job)
    tree = [filters for flags, filters in steps if "-Zmiri-tree-borrows" in flags.split()]
    stacked = [filters for flags, filters in steps if "-Zmiri-tree-borrows" not in flags.split()]
    problems = []
    if len(tree) != 1 or len(stacked) != 1:
        problems.append(f"want one Stacked Borrows and one Tree Borrows step, got {steps}")
    elif tree[0] != stacked[0]:
        problems.append(f"the two models run different tests: {stacked[0]} vs {tree[0]}")
    if any("if" in step for step in job["steps"]):
        problems.append("a step is conditional, so one model could be skipped")
    return problems


def test_miri_job_is_required_and_scoped_to_the_ffi_tests():
    job = parsed()["jobs"]["miri"]
    assert "continue-on-error" not in job
    steps = _miri_steps(job)
    assert steps, "no step runs cargo +nightly miri test"
    for flags, filters in steps:
        assert _unmatched(filters, _unsafe_test_modules()) == []  # every unsafe-touching test module is under Miri
        assert "-Zmiri-ignore-leaks" in flags  # the deliberately leaked close_timeout worker


def test_miri_runs_under_both_aliasing_models_with_the_same_filters():
    assert _aliasing_model_problems(parsed()["jobs"]["miri"]) == []


def test_the_aliasing_model_check_catches_a_dropped_or_narrowed_tree_borrows_step():
    job = parsed()["jobs"]["miri"]
    tree = next(s for s in job["steps"] if "-Zmiri-tree-borrows" in s.get("env", {}).get("MIRIFLAGS", ""))
    dropped = {**job, "steps": [s for s in job["steps"] if s is not tree]}
    narrowed = {**job, "steps": [s if s is not tree else {**s, "run": s["run"].replace(" diag::tests", "")}
                                 for s in job["steps"]]}
    assert _aliasing_model_problems(dropped) and _aliasing_model_problems(narrowed)


def test_pinned_pip_installs_take_wheels_only():
    # A pinned wheel may be missing on the newest CPython; compiling from source must not hide that.
    assert re.search(r"pip install [^\n]*--only-binary=:all: [^\n]*PyYAML==", TEXT)


def test_release_checklist_requires_linux_qualification_before_tagging():
    text = open(os.path.join(REPO, "docs", "RELEASING.md"), encoding="utf-8").read()
    assert "Before tagging, qualify the candidate on Linux x86-64" in text


def test_the_rust_suite_runs_once_per_platform_and_ctest_skips_it():
    jobs = parsed()["jobs"]
    for name in ("linux", "windows", "macos"):
        steps = jobs[name]["steps"]
        assert sum("cargo test" in s.get("run", "") for s in steps) == 1, name
        ctests = [s["run"] for s in steps if "ctest" in s.get("run", "")]
        assert ctests and all("-E '^seabraid_rust_core$'" in r for r in ctests), name
    assert "-E '^(seabraid_rust_core|" in _runs("sanitizers")


def test_the_linux_rust_and_ctest_steps_run_on_one_python_leg_only():
    linux = parsed()["jobs"]["linux"]["steps"]
    once = [s for s in linux if "cargo test" in s.get("run", "") or "ctest" in s.get("run", "")]
    assert once and all(s.get("if") == "matrix.python-version == '3.11'" for s in once)


def test_sanitizer_leak_detection_is_off_only_for_the_provider_smoke_test():
    steps = parsed()["jobs"]["sanitizers"]["steps"]
    off = [s for s in steps if "detect_leaks=0" in s.get("env", {}).get("ASAN_OPTIONS", "")]
    assert len(off) == 1 and "-R '^seabraid_provider_smoke$'" in off[0]["run"]
    assert "detect_leaks=1" in parsed()["jobs"]["sanitizers"]["env"]["ASAN_OPTIONS"]
    on = [s for s in steps if "ctest" in s.get("run", "") and s is not off[0]]
    assert on and all("seabraid_provider_smoke" in s["run"] for s in on)  # excluded from the leak-checking run


def test_the_linux_native_build_is_optimised():
    # 2026-11-19: an optimiser-only -Werror diagnostic broke the Release build while untyped builds passed.
    run = "\n".join(s.get("run", "") for s in parsed()["jobs"]["linux"]["steps"])
    assert "-DCMAKE_BUILD_TYPE=Release -DSEABRAID_BUILD_NATIVE_TESTS=ON -DSEABRAID_BUILD_TOOLS=ON" in run
