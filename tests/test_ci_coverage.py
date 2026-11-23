# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""CI must run what the repository ships: the link-level egress gate, the ggml fixtures, the Strata adapter compile.

OWNER: this file owns "which shipped native code CI builds or runs" (R8 M2, M3); tests/test_ci_workflow.py owns the policy
(pinning, permissions, job list).  ci.yml cannot run locally, so each claim is asserted on the parsed workflow and on the
files it points at.  FAILURE: a missing step, an unpinned source, or a pin that drifted from the repo's own pin fails.
"""

from __future__ import annotations

import os
import re

import pytest
from conftest import REPO, code_only

yaml = pytest.importorskip("yaml", reason="PyYAML not installed (CI installs it pinned)")

with open(os.path.join(REPO, ".github", "workflows", "ci.yml"), encoding="utf-8") as _fh:
    JOBS = yaml.safe_load(_fh)["jobs"]
STRATA_HELPER_LOSS_RUNNER = "compat/strata/tests/run_engine_helper_loss.sh"


def _text(rel: str) -> str:
    with open(os.path.join(REPO, rel), encoding="utf-8") as fh:
        return fh.read()


def _step_index(steps: list, needle: str) -> int:
    hits = [i for i, s in enumerate(steps) if needle in s.get("run", "")]
    assert hits, f"no step runs {needle!r}"
    return hits[0]


def test_the_release_library_is_built_before_the_root_tests_that_need_it():
    steps = JOBS["linux"]["steps"]
    root = [i for i, s in enumerate(steps) if "pytest tests -q" in s.get("run", "") and "working-directory" not in s]
    assert len(root) == 2
    assert _step_index(steps, "cargo build --release --locked") < min(root), "the link-level egress test would find no library"


def test_every_root_pytest_leg_points_the_egress_test_at_the_built_library_and_requires_it():
    steps = JOBS["linux"]["steps"]
    root = [s for s in steps if "pytest tests -q" in s.get("run", "") and "working-directory" not in s]
    for step in root:
        env = step["env"]
        assert env["SEABRAID_LIBRARY"].endswith("/target/release/libseabraid.so"), env
        assert env["SEABRAID_REQUIRE_EGRESS_LINK_TEST"] == "1", "a missing library must FAIL in CI, not skip"


def test_the_ggml_fixtures_are_built_and_run_against_the_pinned_ggml_source_in_the_sdk_job():
    steps = JOBS["sdk"]["steps"]
    fetch = [i for i, s in enumerate(steps) if s.get("with", {}).get("repository") == "ggml-org/llama.cpp"]
    fixtures = _step_index(steps, "scripts/ggml_fixtures/run.sh")
    assert len(fetch) == 1 and fetch[0] < fixtures
    assert "$GITHUB_WORKSPACE/llama.cpp/ggml" in steps[fixtures]["run"]


def test_the_ggml_fixture_wrapper_turns_the_fixtures_on_and_demands_that_ctest_finds_them():
    cmake = _text("scripts/ggml_fixtures/CMakeLists.txt")
    assert "SEABRAID_GGML_TESTS ON" in cmake and "enable_testing()" in cmake and "adapters/ggml" in cmake
    assert "--no-tests=error" in _text("scripts/ggml_fixtures/run.sh")
    assert "option(SEABRAID_GGML_TESTS" in _text("adapters/ggml/CMakeLists.txt"), "the switch the wrapper sets no longer exists"


def _strata_pin() -> str:
    return _text("compat/strata/patches/strata/STRATA_BASE").strip()


def test_the_strata_job_checks_out_the_pinned_sources_and_build_inputs():
    steps = JOBS["strata-adapter"]["steps"]
    fetch = [s for s in steps if s.get("with", {}).get("repository") == "Niko1221/Strata"]
    assert len(fetch) == 1
    assert fetch[0]["with"]["ref"] == _strata_pin(), "CI's Strata pin must be the one compat/strata/install/build.py reads"
    assert re.fullmatch(r"[0-9a-f]{40}", fetch[0]["with"]["ref"])
    sparse = fetch[0]["with"].get("sparse-checkout", "").split()
    assert {"include", "src", "tests", "cmake", "bench/micro", "third_party"} <= set(sparse), "the patched engine's build inputs"


def test_the_strata_job_builds_and_tests_the_adapter_after_applying_both_patch_series():
    steps = JOBS["strata-adapter"]["steps"]
    run = "\n".join(s.get("run", "") for s in steps)
    apply_step = steps[_step_index(steps, "compat/strata/patches/strata/*.patch")]
    assert apply_step.get("working-directory") == "strata", "Bash resolves the patch glob before Git starts"
    assert apply_step["run"].startswith("git -c user.name=ci -c user.email=ci@invalid am ")
    assert "apply --include" not in run, "an include-filtered apply skips the hunks the series needs (Q3 moved the wire out of it)"
    compat = run.index("compat/strata/patches/strata/*.patch")
    v7 = run.index("adapters/strata/strata-v7-seabraid.patch")
    assert compat < v7 < run.index("-DSEABRAID_BUILD_STRATA_ADAPTER=ON")
    assert '-DSTRATA_SOURCE="$GITHUB_WORKSPACE/strata"' in run and "cmake --build build-strata" in run
    assert "ctest --test-dir build-strata -R '^strata_' --no-tests=error" in run, "the adapter fixtures run, and none missing fails"


def test_the_strata_job_builds_compat_with_pinned_xxhash_and_checks_the_helper_units():
    steps = JOBS["strata-adapter"]["steps"]
    units = next(s for s in steps if s.get("name", "").startswith("Compat host tests"))["run"]
    assert "compat/strata/tests" in units and "ctest --test-dir build-compat --no-tests=error" in units
    # The host fixtures use Strata's vendored xxHash header even though they need no engine headers.
    assert '-DSTRATA_SOURCE="$GITHUB_WORKSPACE/strata"' in units
    assert "-fsyntax-only" in units and "-Iadapters/strata/wire" in units
    for tu in ("helper", "helper_serve", "card_bench", "card_bench_run"):
        assert tu in units and os.path.isfile(os.path.join(REPO, "compat", "strata", "strata-ext", "src", f"{tu}.cpp"))


def test_the_strata_job_requires_helper_loss_tests_from_the_same_patched_checkout():
    job = JOBS["strata-adapter"]
    steps = job["steps"]
    index = _step_index(steps, STRATA_HELPER_LOSS_RUNNER)
    step = steps[index]
    assert _step_index(steps, "adapters/strata/strata-v7-seabraid.patch") < index
    switches = {"if", "continue-on-error"}
    assert switches.isdisjoint(job) and switches.isdisjoint(step)
    assert f"sh /source/{STRATA_HELPER_LOSS_RUNNER} /source/strata" in step["run"]
    assert "--gpus" not in step["run"]


def test_the_helper_loss_runner_builds_and_requires_both_cpu_tests():
    run = code_only(_text(STRATA_HELPER_LOSS_RUNNER))
    assert "-DSTRATA_BUILD_TESTS=ON" in run and "-DSTRATA_ENABLE_CUDA=ON" in run
    assert "set -- seabraid_helper_loss_test lost_rows_test" in run
    assert re.search(r'^cmake --build "\$build" .*--target "\$@"$', run, re.M)
    assert 'for test_name in "$@"; do' in run
    assert 'ctest --test-dir "$build" -R "^${test_name}$" --no-tests=error' in run
    assert "export CUDA_VISIBLE_DEVICES=-1" in run


def test_the_strata_job_patch_order_is_the_order_the_adapter_readme_documents():
    readme = _text("adapters/strata/README.md")
    assert readme.index("git am compat/strata/patches/strata/*.patch") < readme.index("then `git am` this patch")


def test_the_built_library_has_one_env_name_everywhere():
    # R8 L10: the egress test read a short alias while every other consumer read SEABRAID_LIBRARY.
    import _common

    alias = re.compile(r"\bSEABRAID_" + r"LIB\b")
    offenders = [rel for rel in _common.tracked(REPO) if (text := _common.read_text(REPO, rel)) and alias.search(text)]
    assert offenders == [], f"the retired alias is still mentioned in {offenders}"


def test_optional_portability_checks_match_the_manual_workflow_jobs():
    manual = {name for name, job in JOBS.items() if job.get("if") == "github.event_name == 'workflow_dispatch'"}
    section = _text("docs/RELEASING.md").split("## Before you run it")[1].split("\n## ")[0]
    optional = next(line for line in section.splitlines() if line.startswith("- Optional portability checks"))
    assert set(re.findall(r"`([\w-]+)`", optional)) == manual
