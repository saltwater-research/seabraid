# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Release receipt regressions. Every external command is handled by a local recorder."""

import datetime
import io
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "release"))
import publish
import rel_build
import rel_all
import rel_core as core
import rel_upload
import rel_registry
import rel_crate
import rel_crate_proof


@pytest.fixture
def built(tmp_path, monkeypatch):
    ctx = core.Ctx(tmp_path, tmp_path / "work", "sample", "owner", publish=True, upload_sdist=True, env={})
    ctx.dist_dir.mkdir(parents=True)
    ctx.export_dir.mkdir()
    (ctx.export_dir / "pyproject.toml").write_text('[project]\ndescription="test"\n')
    ctx.state = dict(cargo_rehearsed=True, head="a" * 40, export_commit="b" * 40, tag="v1", tag_object="c" * 40, version="1",
                     done=["preflight", "export", "build"], wheels=["sample.whl"],
                     sdists=["sample.tar.gz"], crate="sample-1.crate", source_archive="sample-source.tar.gz")
    for name in ["sample.whl", "sample.tar.gz", "sample-source.tar.gz", "sample-1.crate"]:
        (ctx.dist_dir / name).write_text(name)
    (ctx.work / "RELEASE_NOTES.md").write_text("committed notes")
    (ctx.repo / "CITATION.cff").write_text(f"date-released: {datetime.date.today().isoformat()}\n")
    ctx.verified_file = ctx.work / "VERIFIED"
    ctx.verified_file.write_text(ctx.state["head"])
    ctx.state["checked_sha256"] = {name: core.release_file_sha256(ctx.dist_dir / name)
                                   for name in ["sample.whl", "sample-1.crate"]}
    ctx.state["stage_checks"] = {s: core.stage_check_record(ctx, s, set()) for s in ("preflight", "export")}
    rel_build.seal_artifacts(ctx)
    receipt = ctx.state["artifact_receipt"]
    ctx.state["crates_io_proof"] = rel_crate_proof.evidence(
        receipt["tag"], receipt["export_commit"], receipt["tag_object"], receipt["crate"],
        receipt["sha256"][receipt["crate"]], "matched-tested-crate")
    ctx.stdin = io.StringIO("reviewed " + ctx.state["head"] + "\n")
    rel_upload.sign_off(ctx)
    ctx.state = core.load_state(ctx.work)
    transfers = []

    def record(argv, cwd, extra_env=None):
        if argv[:2] == ["git", "rev-parse"]:
            if argv[-1] == "refs/tags/" + ctx.state["tag"]:
                return 0, "c" * 40
            return 0, ctx.state["head" if cwd == ctx.repo else "export_commit"]
        if argv[:2] != ["git", "status"]:
            transfers.append(argv)
        return 0, ""

    monkeypatch.setattr(ctx, "run", record)
    monkeypatch.setattr(ctx, "tool", lambda name: name)
    monkeypatch.setattr(rel_upload, "ready", lambda *_: True)
    return ctx, transfers


@pytest.mark.parametrize("target,changed", [
    ("release", "RELEASE_NOTES.md"), ("release", "SHA256SUMS"),
    ("release", "sample-source.tar.gz"), ("release", "sample-1.crate"), ("pypi", "sample.whl"),
    ("testpypi", "sample.tar.gz")])
def test_bytes_changed_at_prompt_never_reach_transfer(built, monkeypatch, target, changed):
    ctx, transfers = built
    ctx.test_pypi = target == "testpypi"
    ctx.upload_sdist = True
    ctx.skip_steps = {"repo", "release", "pypi", "crates"} - {target}

    def confirm(*_):
        base = ctx.work if changed == "RELEASE_NOTES.md" else ctx.dist_dir
        (base / changed).write_bytes(b"changed while confirming")
        return True

    monkeypatch.setattr(core, "confirm", confirm)
    with pytest.raises(core.Refuse, match="release artifacts"):
        rel_upload.run(ctx)
    assert transfers == []


def test_full_source_has_its_own_hash_and_only_goes_to_release(built):
    ctx, _ = built
    name = ctx.state["source_archive"]
    assert name in ctx.state["artifact_receipt"]["sha256"]
    assert ctx.dist_dir / name in rel_upload.artifacts(ctx)
    ctx.upload_sdist = True
    assert str(ctx.dist_dir / name) not in rel_upload.uploads(ctx)
    rel_upload.prove_inputs(ctx)


def test_replaced_annotated_tag_cannot_substitute_for_the_sealed_object(built, monkeypatch):
    ctx, transfers = built
    old_run = ctx.run

    def changed(argv, cwd, extra_env=None):
        if argv[-1] == "refs/tags/v1":
            return 0, "d" * 40
        return old_run(argv, cwd, extra_env)

    monkeypatch.setattr(ctx, "run", changed)
    with pytest.raises(core.Refuse, match="refs/tags/v1 no longer matches"):
        rel_upload.prove_inputs(ctx)
    assert not transfers


def test_missing_or_invalidated_receipt_blocks_existing_build(built):
    ctx, _ = built
    publish.invalidate_build(ctx, "build")
    ctx.state = core.load_state(ctx.work)
    assert "artifact_receipt" not in ctx.state
    with pytest.raises(core.Refuse, match="stages not completed.*build"):
        rel_upload.prove_inputs(ctx)


def test_rewritten_file_and_matching_manifest_do_not_replace_build_receipt(built):
    ctx, _ = built
    (ctx.dist_dir / "sample.whl").write_text("substituted")
    (ctx.dist_dir / "SHA256SUMS").write_bytes(core.checksum_manifest(core.artifact_receipt(ctx)))
    with pytest.raises(core.Refuse, match="changed since the build"):
        rel_upload.prove_inputs(ctx)

def test_first_publish_uses_only_prepared_artifacts(tmp_path, monkeypatch):
    asked = []
    monkeypatch.setattr(rel_all, "guards", lambda *_: True)

    def run_stages(ctx, stages):
        asked.extend(stages)
        return True

    args = ["--name", "sample", "--work", str(tmp_path / "release")]
    assert rel_all.main([*args, "--publish"], run_stages) == 0
    assert asked == ["publish"]


@pytest.mark.parametrize(("key", "other"), [
    ("owner", "other"), ("name", "other"), ("upload_sdist", False),
    ("skip_gates", {"manual"}),
])
def test_changed_publish_scope_refuses_the_existing_receipt(built, key, other):
    ctx, transfers = built
    setattr(ctx, key, other)
    with pytest.raises(core.Refuse, match="changed since the build"):
        rel_upload.prove_inputs(ctx)
    assert not transfers


def test_python_build_archives_the_gated_export_revision(built, monkeypatch):
    ctx, _ = built
    asked = []

    def archive(argv, *_):
        asked.append(argv)
        return 1, "fixture stopped before extraction"

    monkeypatch.setattr(ctx, "run", archive)
    with pytest.raises(core.Refuse, match="git archive"):
        rel_build.make_buildtree(ctx, ctx.export_dir)
    assert asked[0][-1] == ctx.state["export_commit"]


def test_offline_success_alone_cannot_publish(built):
    ctx, transfers = built
    del ctx.state["crates_io_proof"]
    assert ctx.state["artifact_receipt"]["cargo_rehearsed"] is True
    with pytest.raises(core.Refuse, match="official crates.io dry-run proof"):
        rel_upload.prove_inputs(ctx)
    assert not transfers


@pytest.mark.parametrize("skipped_stage", ["preflight", "export"])
def test_resumed_build_keeps_earlier_skipped_checks(built, monkeypatch, skipped_stage):
    ctx, transfers = built
    original = ctx.state.copy()
    ctx.publish = False

    def preflight(ctx):
        ctx.state.update(head=original["head"], version=original["version"])
        return True

    def export(ctx):
        ctx.state.update(tag=original["tag"], export_commit=original["export_commit"])
        return True

    def build(ctx):
        for key in ("tag_object", "crate", "source_archive", "wheels", "sdists", "checked_sha256", "crates_io_proof"):
            ctx.state[key] = original[key]
        rel_build.seal_artifacts(ctx)
        return True

    monkeypatch.setattr(publish, "STAGE_FUNCS", dict(preflight=preflight, export=export, build=build, publish=rel_upload.run))
    for stage in ("preflight", "export", "build"):
        ctx.skip_gates = {"pytest"} if stage == skipped_stage else set()
        assert publish.run_stages(ctx, [stage])
    assert publish.run_stages(ctx, ["publish"])  # a preview of the incomplete qualification remains available
    assert ctx.state["artifact_receipt"]["stage_checks"][skipped_stage]["skipped"] == ["pytest"]
    ctx.publish = True
    with pytest.raises(core.Refuse, match=skipped_stage + " checks"):
        publish.check_flat_publish(ctx)
    assert not transfers
    ctx.publish = False
    assert publish.run_stages(ctx, ["preflight", "export", "build"])
    ctx.publish = True
    ctx.stdin = io.StringIO("reviewed " + ctx.state["head"] + "\n")
    rel_upload.prove_inputs(ctx)
    assert not transfers


def test_incomplete_preparation_refuses_before_network_checks(built, monkeypatch):
    ctx, transfers = built
    del ctx.state["artifact_receipt"]["stage_checks"]["export"]
    core.save_state(ctx)
    monkeypatch.setattr(rel_all, "common_guards", lambda *_: True)
    with pytest.raises(core.Refuse, match="export checks"):
        rel_all.guards(ctx, rel_all.parse([]))
    assert not transfers


def test_interrupted_export_must_finish_again_before_build(built, monkeypatch):
    ctx, transfers = built
    ctx.publish = False

    def interrupted(_):
        raise core.Refuse("interrupted while checking the export")

    monkeypatch.setitem(publish.STAGE_FUNCS, "export", interrupted)
    with pytest.raises(core.Refuse, match="interrupted"):
        publish.run_stages(ctx, ["export"])
    saved = core.load_state(ctx.work)
    assert "export" not in saved["done"] and "export" not in saved["stage_checks"]
    assert "artifact_receipt" not in saved
    monkeypatch.setitem(publish.STAGE_FUNCS, "export", lambda _: True)
    assert publish.run_stages(ctx, ["export"])
    assert ctx.state["stage_checks"]["export"] == core.stage_check_record(ctx, "export", set())
    assert "artifact_receipt" not in ctx.state and not transfers


def test_sealed_check_from_another_export_is_refused(built):
    ctx, _ = built
    ctx.state["artifact_receipt"]["stage_checks"]["export"]["export_commit"] = "d" * 40
    with pytest.raises(core.Refuse, match="export checks"):
        core.require_preparation_checks(ctx)


def test_legacy_receipt_previews_but_cannot_publish(built):
    ctx, transfers = built
    del ctx.state["artifact_receipt"]["stage_checks"]
    del ctx.state["stage_checks"]
    ctx.publish = False
    rel_upload.prove_inputs(ctx)
    ctx.publish = True
    with pytest.raises(core.Refuse, match="preflight checks"):
        rel_upload.prove_inputs(ctx)
    assert not transfers


@pytest.mark.parametrize("field,other", [
    ("registry", "loopback"), ("tag", "v2"), ("tag_object", "d" * 40),
    ("export_commit", "e" * 40), ("crate", "sample-2.crate"), ("sha256", "0" * 64),
    ("result", "offline success"), ("verification", "skipped"),
])
def test_external_proof_must_match_sealed_inputs(built, field, other):
    ctx, transfers = built
    proof = ctx.state.pop("crates_io_proof")
    ctx.crates_io_proof = ctx.work / "proof.json"
    ctx.crates_io_proof.write_text(json.dumps({**proof, field: other}))
    with pytest.raises(core.Refuse, match="official crates.io dry-run proof"):
        rel_upload.prove_inputs(ctx)
    assert not transfers and "crates_io_proof" not in ctx.state


def test_proof_attaches_to_existing_offline_build_without_reseal(built):
    ctx, transfers = built
    proof = ctx.state.pop("crates_io_proof")
    ctx.state["cargo_rehearsed"] = False
    rel_build.seal_artifacts(ctx)
    core.save_state(ctx)
    original = ctx.state["artifact_receipt"].copy()
    ctx.crates_io_proof = ctx.work / "proof.json"
    ctx.crates_io_proof.write_text(json.dumps(proof))
    ctx.publish = False
    assert publish.run_stages(ctx, ["publish"])
    ctx.crates_io_proof = None
    ctx.state = core.load_state(ctx.work)
    assert ctx.state["crates_io_proof"] == proof and ctx.state["artifact_receipt"] == original
    ctx.publish = True
    ctx.stdin = io.StringIO("reviewed " + ctx.state["head"] + "\n")
    rel_upload.prove_inputs(ctx)
    assert not transfers


def test_external_proof_requires_independent_crate_test(built):
    ctx, _ = built
    receipt = ctx.state["artifact_receipt"]
    del receipt["checked_sha256"][receipt["crate"]]
    with pytest.raises(core.Refuse, match="sealed, tested crate"):
        rel_crate_proof.check(ctx, receipt, required=True)


@pytest.mark.parametrize("repacked,reason", [(b"wrong", "differs"), (b"sample-1.crate", None)])
def test_repacking_never_sends_an_unbound_crate(built, monkeypatch, repacked, reason):
    ctx, _ = built
    calls = []

    def package(argv, cwd, extra_env):
        calls.append(argv)
        path = Path(extra_env["CARGO_TARGET_DIR"]) / "package"
        path.mkdir()
        (path / ctx.state["crate"]).write_bytes(repacked)
        return 0, ""

    monkeypatch.setattr(ctx, "run", package)
    if reason:
        with pytest.raises(core.Refuse, match=reason):
            rel_crate.check_repack(ctx)
    else:
        rel_crate.check_repack(ctx)
    assert calls == [["cargo", "package", "--locked", "--offline", "--no-verify"]]


def test_changed_coordinator_sha_or_unsigned_receipt_refuses(built):
    ctx, transfers = built
    ctx.verified_file.write_text("e" * 40)
    with pytest.raises(core.Refuse, match="not the verified commit"):
        rel_upload.prove_inputs(ctx)
    ctx.verified_file.write_text(ctx.state["head"])
    ctx.state["attestation"]["receipt_sha256"] = "f" * 64
    with pytest.raises(core.Refuse, match="human review not confirmed"):
        rel_upload.prove_inputs(ctx)
    assert transfers == []


def test_unchecked_or_changed_wheel_cannot_be_sealed(built):
    ctx, _ = built
    (ctx.dist_dir / "sample.whl").write_bytes(b"replacement after install")
    with pytest.raises(core.Refuse, match="wheel install or crate test"):
        rel_build.seal_artifacts(ctx)


def test_cargo_success_without_provider_witness_cannot_finish(built, monkeypatch):
    ctx, _ = built
    ctx.env.update(CARGO_REGISTRY_TOKEN="synthetic-release-key", CARGO_REGISTRY_CREDENTIAL_PROVIDER="untrusted")
    provider = ctx.export_dir / "scripts" / "release" / "rel_crate_credential.py"
    provider.parent.mkdir(parents=True)
    provider.touch()
    monkeypatch.setattr(rel_crate, "check_repack", lambda _: None)

    def fake_cargo(argv, cwd, env):
        assert env["CARGO_REGISTRY_TOKEN"] is None and env["CARGO_REGISTRY_CREDENTIAL_PROVIDER"] is None
        assert env["RELEASE_CRATE_TOKEN"] not in repr(argv)
        assert Path(env["CARGO_HOME"]).parent == cwd
        assert not list(Path(env["CARGO_HOME"]).glob("credentials*"))
        return 0, "fixture claims success without asking for a token"

    monkeypatch.setattr(ctx, "run", fake_cargo)
    with pytest.raises(core.Refuse, match="did not confirm authorization"):
        rel_crate.upload(ctx)
    assert "crates" not in ctx.state.get("published", [])


def test_crate_probe_isolated_from_local_manifest_and_index(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "Cargo.toml").write_text("[package]\nname='sample'")
    ctx = core.Ctx(repo, tmp_path / "work", "sample", "owner", env={})
    registry = rel_registry.Registry(ctx)
    calls = []

    def probe(argv, cwd, extra_env):
        calls.append(argv)
        assert cwd != repo and repo not in cwd.parents
        assert not Path(extra_env["CARGO_HOME"]).exists()
        return 0, "sample\nversion: 1.0.0 (from .)\n"

    ctx.run = probe
    assert registry.name_taken("crates", "sample") is None
    assert calls == [["cargo", "info", "sample", "--registry", "crates-io"]]
    ctx.env["CARGO_NET_OFFLINE"] = "true"
    assert registry.name_taken("crates", "sample") is None and len(calls) == 1


@pytest.mark.parametrize("rc,out,expected", [
    (0, "sample\nversion: 1.0.0\n", True),
    (101, "error: could not find `sample@1.0.0` in registry `crates-io`", False),
    (101, "config.json not found", None),
    (0, "version: 1.0.1", None),
])
def test_registry_observation_requires_a_definite_answer(rc, out, expected):
    assert rel_registry.crate_answer("sample@1.0.0", rc, out) is expected
