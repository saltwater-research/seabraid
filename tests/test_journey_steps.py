# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""scripts/journey/steps.toml is the README and CONTRIBUTING journey as data.  These tests are its checkers: every
documented command must still appear in its document (the data cannot drift from the docs), the schema must hold, and
the runner must behave on a toy step list."""

from __future__ import annotations

import importlib.util
import os
import re
import sys

import pytest
from conftest import REPO

spec = importlib.util.spec_from_file_location("journey_run", os.path.join(REPO, "scripts", "journey", "run.py"))
run = importlib.util.module_from_spec(spec)
sys.modules["journey_run"] = run
spec.loader.exec_module(run)

STATE, STEPS = run.load()


def squash(text: str) -> str:
    """A doc's `\\` line continuations and indentation are one space: how a shell reads the command."""
    return re.sub(r"\s+", " ", re.sub(r"\\\n", " ", text))


DOCS = {name: squash(open(os.path.join(REPO, name), encoding="utf-8").read()) for name in ("README.md", "CONTRIBUTING.md")}
needs_bash = pytest.mark.skipif(os.name == "nt", reason="the runner drives bash with a POSIX PATH")


def test_schema_is_valid():
    assert run.validate(STATE, STEPS) == []
    assert len(STEPS) >= 20, "the journey is not the journey if it shrinks to a handful of steps"


def test_every_documented_command_appears_in_its_document():
    for step in STEPS:
        if "harness" in step:
            continue
        doc = DOCS[step["doc"]]
        for text in (step["command"], step.get("env_prefix", ""), *step.get("substitute", {})):
            assert squash(text) in doc, f"{step['id']}: {text!r} is not in {step['doc']}"


def test_readme_wheel_example_builds_before_installing_in_a_clean_venv():
    readme = open(os.path.join(REPO, "README.md"), encoding="utf-8").read()
    blocks = re.findall(r"(?ms)^```sh\n(.*?)^```$", readme)
    install = ".venv-wheel/bin/pip install --no-index dist/seabraid-*.whl"
    examples = [block for block in blocks if install in block]
    assert len(examples) == 1
    commands = examples[0].splitlines()
    expected = ['pip install "setuptools>=77"', "python scripts/build_wheel.py", "python3 -m venv .venv-wheel", install]
    positions = [commands.index(command) for command in expected]
    assert positions == sorted(positions)


def test_state_lines_are_documented_shell_state():
    corpus = " ".join(DOCS.values())
    for name, line in STATE.items():
        assert squash(line) in corpus, f"state {name!r}: {line!r} is in neither document"


def test_harness_steps_are_few_and_say_why():
    harness = [s for s in STEPS if "harness" in s]
    assert len(harness) <= 2 and all(len(s["harness"]) > 40 for s in harness)


def test_both_docs_are_walked_and_the_offline_recipe_exists():
    assert {s["doc"] for s in STEPS} == {"README.md", "CONTRIBUTING.md"}
    assert {s.get("when") for s in STEPS} >= {"online", "offline"}
    assert any(s.get("status") == "warn-expected" for s in STEPS), "the documented lizard stop is the expected WARN"


def test_validate_rejects_each_kind_of_bad_step():
    bad = [{"id": "a", "doc": "README.md", "command": "x", "status": "maybe"}, {"id": "a", "doc": "README.md", "command": "x", "output": "("},
           {"id": "b", "doc": "README.md", "command": "x", "state": ["nope"], "bogus": 1}, {"id": "c", "doc": "README.md", "command": "x", "status": "warn-expected"},
           {"id": "d", "doc": "README.md", "command": "x", "substitute": {"<y>": "z"}}, {"id": "e", "command": "x"}]
    errs = "\n".join(run.validate({}, bad))
    for needle in ("bad status", "bad regex", "duplicate id", "unknown state nope", "unknown key bogus", "needs warn_output", "placeholder '<y>'", "e: missing doc"):
        assert needle in errs, needle


TOY = """
[state]
mark = "export MARK=set"

[[step]]
id = "ok"
doc = "README.md"
command = "echo hello $MARK"
state = ["mark"]
output = 'hello set'

[[step]]
id = "net"
doc = "README.md"
command = "echo fetched"
status = "skip-needs-network"

[[step]]
id = "warn"
doc = "README.md"
command = "echo documented stop; exit 3"
status = "warn-expected"
warn_output = 'documented stop'
"""


def _toy(tmp_path, extra=""):
    steps = tmp_path / "toy.toml"
    steps.write_text(TOY + extra, encoding="utf-8")
    tree = tmp_path / "tree"
    tree.mkdir()
    return steps, tree, tmp_path / "out"


@needs_bash
def test_runner_records_pass_skip_warn_offline_and_exits_zero(tmp_path):
    steps, tree, out = _toy(tmp_path)
    assert run.main(["--steps", str(steps), "--tree", str(tree), "--out", str(out), "--offline"]) == 0
    rows = [line.split("\t") for line in (out / "index.tsv").read_text().splitlines()]
    assert [(r[1], r[3], r[4]) for r in rows] == [("ok", "0", "PASS"), ("net", "-", "SKIP"), ("warn", "3", "WARN")]
    step = (out / "steps" / "01-ok.txt").read_text()
    assert "SHELL-STATE: export MARK=set" in step and "hello set" in step and "STATUS: PASS" in step
    assert "SKIPPED: needs network" in (out / "steps" / "02-net.txt").read_text()
    assert "1 SKIP" in (out / "REPORT.md").read_text()


@needs_bash
def test_runner_online_runs_the_network_step_and_fails_on_an_unexpected_exit_or_a_missing_pattern(tmp_path):
    extra = '\n[[step]]\nid = "boom"\ndoc = "README.md"\ncommand = "echo before; false; echo after"\n\n[[step]]\nid = "drift"\ndoc = "README.md"\ncommand = "echo other"\noutput = "expected text"\n\n[[step]]\nid = "fix"\ndoc = "README.md"\ncommand = "echo ok"\nharness = "why this is not in the docs"\n'
    steps, tree, out = _toy(tmp_path, extra)
    assert run.main(["--steps", str(steps), "--tree", str(tree), "--out", str(out)]) == 1
    rows = {r.split("\t")[1]: r.split("\t")[4] for r in (out / "index.tsv").read_text().splitlines()}
    assert rows == {"ok": "PASS", "net": "PASS", "warn": "WARN", "boom": "FAIL", "drift": "FAIL", "fix": "PASS"}
    assert "`fix` is not in the docs: why this is not in the docs" in (out / "REPORT.md").read_text()
    assert "after" not in (out / "steps" / "04-boom.txt").read_text().split("----- OUTPUT -----")[1], "bash -e stops at the first failure"


def test_runner_refuses_a_used_tree_and_a_record_inside_the_tree(tmp_path, capsys):
    steps, tree, out = _toy(tmp_path)
    (tree / ".venv").mkdir()
    assert run.main(["--steps", str(steps), "--tree", str(tree), "--out", str(out)]) == 2
    (tree / ".venv").rmdir()
    assert run.main(["--steps", str(steps), "--tree", str(tree), "--out", str(tree / "out")]) == 2
    assert not out.exists() and "fresh tree" in capsys.readouterr().err


def test_clean_env_carries_no_inherited_variables(monkeypatch, tmp_path):
    monkeypatch.setenv("SOME_SECRET", "x")
    monkeypatch.setenv("VIRTUAL_ENV", "/not/this")
    env = run.clean_env(tmp_path)
    assert "SOME_SECRET" not in env and "VIRTUAL_ENV" not in env
    assert env["CUDA_VISIBLE_DEVICES"] == "" and env["PATH"].startswith(os.path.dirname(sys.executable) + ":")
