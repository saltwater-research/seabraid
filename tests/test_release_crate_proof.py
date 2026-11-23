# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Synthetic Cargo and Git observations for the portable crates.io proof. No network calls."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "release"))
import rel_core as core
import rel_crate_proof as proof


@pytest.fixture
def recorder(tmp_path, monkeypatch):
    repo = tmp_path / "export"
    repo.mkdir()
    (repo / "Cargo.toml").write_text('[package]\nname="sample"\nversion="1"\n')
    tested = tmp_path / "sample-1.crate"
    tested.write_bytes(b"sealed and tested")
    ctx = core.Ctx(repo, tmp_path, "sample", "owner", env={"CARGO_REGISTRIES_CRATES_IO_INDEX": "loopback",
                                                          "CARGO_REGISTRY_TOKEN": "unused"})
    calls = []
    answers = dict(status="", tag_type="tag", commit="a" * 40, tag_object="b" * 40, rc=0,
                   output="Updating crates.io index\nwarning: aborting upload due to dry run",
                   crate=tested.read_bytes(), staged=True)

    def command(args, cwd, extra_env):
        if args[0] == "git":
            return git_answer(args, answers)
        calls.append(args)
        assert not cwd.is_relative_to(repo)
        assert_isolated_dry_run(args, extra_env)
        if answers["staged"]:
            target = Path(extra_env["CARGO_TARGET_DIR"]) / "package" / "tmp-crate"
            target.mkdir(parents=True)
            (target / tested.name).write_bytes(answers["crate"])
        return answers["rc"], answers["output"]

    monkeypatch.setattr(ctx, "run", command)
    return ctx, tested, tmp_path / "proof.json", answers, calls


def assert_isolated_dry_run(args, extra_env):
    assert extra_env["CARGO_REGISTRIES_CRATES_IO_INDEX"] is None
    assert extra_env["CARGO_REGISTRY_TOKEN"] is None
    assert not Path(extra_env["CARGO_HOME"]).exists()
    assert "--dry-run" in args and "--offline" not in args
    assert args[args.index("--registry") + 1] == "crates-io"


def git_answer(args, answers):
    if args[1] == "status":
        return 0, answers["status"]
    if args[1] == "cat-file":
        return 0, answers["tag_type"]
    return 0, answers["tag_object" if args[-1] == "refs/tags/v1" else "commit"]


@pytest.mark.parametrize("skip_build", [True, False])
def test_records_exact_generated_archive(recorder, skip_build):
    ctx, tested, out, answers, calls = recorder
    proof.record(ctx, "v1", out, tested if skip_build else None)
    assert len(calls) == 1 and ("--no-verify" in calls[0]) == skip_build
    record = json.loads(out.read_text())
    assert record == proof.evidence("v1", answers["commit"], answers["tag_object"], tested.name,
                                    core.release_file_sha256(tested), "matched-tested-crate" if skip_build else "cargo-verified")


@pytest.mark.parametrize("change", [
    {"rc": 101}, {"output": "warning: aborting upload due to dry run"},
    {"output": "Updating crates.io index"}, {"staged": False},
    {"crate": b"different"}, {"tag_type": "commit"}, {"status": "?? unexpected"},
])
def test_refusals_emit_no_proof(recorder, change):
    ctx, tested, out, answers, _ = recorder
    answers.update(change)
    with pytest.raises(core.Refuse):
        proof.record(ctx, "v1", out, tested)
    assert not out.exists()


def test_ancestor_config_and_offline_mode_do_not_contact_registry(recorder):
    ctx, tested, out, _, calls = recorder
    ctx.env["CARGO_NET_OFFLINE"] = "true"
    with pytest.raises(core.Refuse, match="requires online"):
        proof.record(ctx, "v1", out, tested)
    del ctx.env["CARGO_NET_OFFLINE"]
    config = ctx.work / ".cargo" / "config.toml"
    config.parent.mkdir()
    config.write_text("")
    with pytest.raises(core.Refuse, match="redirect"):
        proof.record(ctx, "v1", out, tested)
    assert not calls and not out.exists()
