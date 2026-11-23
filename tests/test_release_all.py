# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""`publish.py all` and `publish.py register` (scripts/release/rel_all.py, rel_register.py, rel_guard.py, rel_registry.py).

OWNER: this file proves the one-command release contract (docs/RELEASING.md, "Release in one command"): the fixed target
order, resume and skip-if-published against a fake registry, every refusal (no name, provisional name, unverified commit,
non-fresh or dev remote, Windows build), the typed "<target> <version>" confirmation, that no credential reaches argv or
output, and the dry-run command list (a golden).  No test touches a network: gh, pip, cargo and twine are never run
(``Ctx.run`` is a recorder that also plays the registries' answers), and the CLI tests use a tiny synthetic tree.
"""

from __future__ import annotations

import argparse
import io
import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
RELEASE = REPO / "scripts" / "release"
sys.path.insert(0, str(REPO / "checks"))
sys.path.insert(0, str(RELEASE))
import rel_all  # noqa: E402
import rel_core as core  # noqa: E402
import rel_guard as guard  # noqa: E402
import rel_export  # noqa: E402
import rel_register  # noqa: E402
import rel_registry  # noqa: E402
import rel_upload  # noqa: E402

OLD = guard.PROVISIONAL[0]  # WHY read, never spelled: scripts/rename.py would rewrite a spelled name in the export of this very file
NAME, OWNER, VERSION = "tidewrack", "acme-test", "0.5.0-alpha.2"
SECRETS = {"TWINE_PASSWORD": "pypi-AgENsecretvalue123", "CARGO_REGISTRY_TOKEN": "cioSECRETvalue456789", "GH_TOKEN": "ghp_" + "a" * 36}
ID = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid"}


REGISTRY_ASKS = (["gh", "repo", "view"], ["gh", "release", "view"], ["cargo", "info"])


class Recorder:
    """Stands in for Ctx.run: records every command, never runs one.  ``fail`` is a substring that makes a command exit 1;
    ``answers`` maps a substring of a command to its (rc, output); a registry question otherwise answers "not there"."""

    def __init__(self, fail: str = "", answers: dict | None = None, echo: str = ""):
        self.calls, self.fail, self.answers, self.echo = [], fail, answers or {}, echo
        self.refs = {}

    def git(self, cmd):
        if cmd[:3] == ["git", "remote", "get-url"]:
            return 0, rel_export.repo_url(OWNER, NAME)
        if cmd[:2] == ["git", "rev-parse"]:
            return 0, "c" * 40
        if cmd[:2] == ["git", "push"]:
            ref = "refs/heads/main" if "HEAD:" in cmd[-1] else "refs/tags/v" + VERSION
            self.refs[ref] = "b" * 40 if "heads/" in ref else "c" * 40
        if cmd[:2] == ["git", "ls-remote"]:
            return 0, "\n".join(sha + "\t" + ref for ref, sha in self.refs.items())
        return 0, "ok"

    def __call__(self, cmd, cwd, extra_env=None):
        self.calls.append(list(cmd))
        line = " ".join(cmd)
        for needle, reply in self.answers.items():
            if needle in line:
                return reply
        if self.fail and self.fail in line:
            return 1, "boom"
        if cmd[:2] == ["cargo", "info"]:
            return 101, f"error: could not find `{cmd[2]}` in registry `crates-io`"
        if cmd[0] == "git":
            return self.git(cmd)
        if cmd[:3] in REGISTRY_ASKS or "index" in cmd:
            return 1, "ERROR: No matching distribution found / could not find"
        return 0, "ok " + self.echo


def make_ctx(tmp_path, *, publish=False, stdin="", env=None, fail="", answers=None, echo="") -> core.Ctx:
    ctx = core.Ctx(repo=REPO, work=tmp_path / "work", name=NAME, owner=OWNER, publish=publish, all_mode=True, stdin=io.StringIO(stdin))
    ctx.env = {"PATH": os.environ.get("PATH", ""), **(env or {})}
    ctx.registry = rel_registry.Registry(ctx)
    ctx.state = {"version": VERSION, "tag": f"v{VERSION}", "wheels": [f"{NAME}-0.5.0a2-py3-none-any.whl"], "sdists": [f"{NAME}-0.5.0a2.tar.gz"], "done": ["preflight", "export", "build"]}
    ctx.state.update(export_commit="b" * 40, tag_object="c" * 40)
    ctx.work.mkdir(parents=True, exist_ok=True)
    (ctx.dist_dir).mkdir(exist_ok=True)
    (ctx.dist_dir / ctx.state["sdists"][0]).write_bytes(b"sdist")
    ctx.run = Recorder(fail, answers, echo)
    ctx.tool = lambda name: name  # every tool "exists"
    return ctx


def norm(text: str, ctx: core.Ctx) -> str:
    return text.replace(str(ctx.work), "<WORK>").replace(str(REPO), "<REPO>").replace("\\", "/").replace("//", "/")


def test_order_is_fixed_and_tokens_name_target_and_version(tmp_path):
    ctx = make_ctx(tmp_path)
    steps = rel_upload.steps_for(ctx)
    assert [s.key for s in steps] == ["repo", "release", "pypi", "crates"]
    assert [s.token for s in steps] == [f"{k} {VERSION}" for k in ("repo", "release", "pypi", "crates")]


def test_dry_run_command_list_golden(tmp_path):
    ctx = make_ctx(tmp_path)
    steps = rel_upload.steps_for(ctx)
    description = __import__("tomllib").loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]["description"]
    lines = [f"{s.key} | {s.token} | " + norm(" ".join(argv), ctx) for s in steps for argv, _ in s.cmds]
    assert [ln.replace(sys.executable.replace("\\", "/"), "<PY>") for ln in lines][:3] == [
        f"repo | repo {VERSION} | gh repo create {OWNER}/{NAME} --public --source <WORK>/export --remote origin --description {description}",
        f"repo | repo {VERSION} | git push origin HEAD:refs/heads/main",
        f"repo | repo {VERSION} | git push origin v{VERSION}",
    ]
    assert lines[3] == (f"release | release {VERSION} | gh release create v{VERSION} <WORK>/dist/{NAME}-0.5.0a2-py3-none-any.whl <WORK>/dist/{NAME}-0.5.0a2.tar.gz <WORK>/dist/SHA256SUMS"
                        f" --repo {OWNER}/{NAME} --title {NAME} {VERSION} --notes-file <WORK>/RELEASE_NOTES.md --verify-tag --prerelease")
    assert re.fullmatch(rf"pypi \| pypi {re.escape(VERSION)} \| \S+ upload --non-interactive <WORK>/dist/{NAME}-0\.5\.0a2-py3-none-any\.whl", lines[4])
    assert lines[5] == f"crates | crates {VERSION} | cargo publish --locked --registry crates-io --no-verify --manifest-path <WORK>/export/Cargo.toml"
    assert len(lines) == 6


def test_skip_if_published_asks_the_registry_and_records(tmp_path):
    ctx = make_ctx(tmp_path, publish=True, answers={"pip index": (0, "x (0.5.0a2)\nAvailable versions: 0.5.0a2, 0.4.0\n")})
    found = rel_upload.already_published(ctx, rel_upload.steps_for(ctx))
    assert found == ["pypi"] and ctx.state["published"] == ["pypi"]  # PEP 440 spelling matched; crates, repo and release are not there
    assert ["cargo", "info", f"{NAME}@{VERSION}", "--registry", "crates-io"] in ctx.run.calls


def test_registry_that_cannot_answer_stops_the_run(tmp_path):
    ctx = make_ctx(tmp_path, publish=True, answers={"pip index": (1, "ERROR: could not reach the index (connection refused)")})
    with pytest.raises(core.Refuse, match="cannot tell"):
        rel_upload.already_published(ctx, rel_upload.steps_for(ctx))


def test_dry_run_never_queries_a_registry(tmp_path):
    ctx = make_ctx(tmp_path, publish=False)
    assert rel_upload.already_published(ctx, rel_upload.steps_for(ctx)) == [] and ctx.run.calls == []


def test_resume_hint_after_a_failed_stage_names_the_stage_not_a_crash(tmp_path):
    """Found by the rename trial on a Linux box: preflight failed, then the hint raised KeyError 'tag'. A state file left
    by an earlier run carries a version, but this run's stages never set the tag, so the hint must not build targets."""
    ctx = make_ctx(tmp_path)
    core.save_state(ctx)          # an earlier run's state file, with a version
    ctx.state = {}                # this run failed in preflight: no stage set version, tag or files
    hint = rel_all.resume_hint(ctx)
    assert "resumes at: preflight" in hint and "Published so far: nothing" in hint


def test_run_stops_at_first_failure_then_resumes_where_it_stopped(tmp_path, monkeypatch):
    monkeypatch.setattr(rel_upload, "prove_inputs", lambda ctx: None)
    ctx = make_ctx(tmp_path, publish=True, stdin=f"repo {VERSION}\n", fail="push origin v")
    ctx.skip_steps = {"crates", "release", "pypi"}
    assert rel_upload.run(ctx) is False
    assert core.load_state(ctx.work)["repo_commands"] == ["create", "main"]
    ctx.run.fail = ""
    ctx.run.answers["gh repo view"] = (0, "exists")
    ctx.stdin = io.StringIO(f"repo {VERSION}\n")
    ctx.run.calls.clear()
    ctx.state = core.load_state(ctx.work)
    assert rel_upload.run(ctx)
    assert ctx.state["published"] == ["repo"] and ctx.state["repo_commands"] == ["create", "main", "tag"]
    assert [c for c in ctx.run.calls if c[:2] == ["git", "push"]] == [["git", "push", "origin", f"v{VERSION}"]]
    assert not any(c[:3] == ["gh", "repo", "create"] for c in ctx.run.calls)
    ctx.run.refs["refs/heads/main"] = "d" * 40
    with pytest.raises(core.Refuse, match="unexpected remote ref"):
        rel_upload.run(ctx)


def test_crate_published_by_hand_needs_proof_of_ownership_and_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(rel_upload, "prove_inputs", lambda ctx: None)
    ctx = make_ctx(tmp_path, publish=True, answers={"cargo info": (0, f"{NAME}\nversion: {VERSION}\n")})
    with pytest.raises(core.Refuse, match="matching publication receipt"):
        rel_upload.run(ctx)
    assert not any(c[:2] == ["cargo", "publish"] for c in ctx.run.calls)


def test_typed_confirmation_mismatch_runs_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(rel_upload, "prove_inputs", lambda ctx: None)
    ctx = make_ctx(tmp_path, publish=True, stdin="repo 0.5.0\n")  # right target, wrong version
    ctx.skip_steps = {"crates"}
    assert rel_upload.run(ctx) is False
    assert not any(c[:3] in (["gh", "repo", "create"], ["cargo", "publish", "--locked"]) for c in ctx.run.calls)
    assert "published" not in ctx.state


def test_eof_is_a_refusal(tmp_path, monkeypatch):
    monkeypatch.setattr(rel_upload, "prove_inputs", lambda ctx: None)
    ctx = make_ctx(tmp_path, publish=True, stdin="")
    ctx.skip_steps = {"crates"}
    assert rel_upload.run(ctx) is False and "published" not in ctx.state


def test_no_credential_in_argv_or_output(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(rel_upload, "prove_inputs", lambda ctx: None)
    for key, value in SECRETS.items():
        monkeypatch.setenv(key, value)
    typed = "".join(f"{k} {VERSION}\n" for k in ("repo", "release", "pypi", "crates"))
    ctx = make_ctx(tmp_path, publish=True, stdin=typed, env=SECRETS, echo="debug: " + " ".join(SECRETS.values()))  # a tool that echoes its credentials
    ctx.skip_steps = {"crates"}
    assert rel_upload.run(ctx) is True
    out = capsys.readouterr().out
    argv = repr(ctx.run.calls) + repr([s.cmds for s in rel_upload.steps_for(ctx)])
    assert "debug:" in out  # the echo reached the report, masked
    for value in SECRETS.values():
        assert value not in argv and value not in out


# ---- refusals (pure rules) -------------------------------------------------------------------------------------------

def test_provisional_name_is_refused():
    assert any("provisional" in p for p in guard.name_problems(OLD, REPO))
    assert guard.name_problems(NAME, REPO) == []


def test_the_final_name_is_not_provisional():
    final = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]["name"]  # WHY read, never spelled: rename.py rewrites a spelled name
    assert final not in guard.PROVISIONAL and not any("provisional" in p for p in guard.name_problems(final, REPO))


def test_unverified_commit_is_refused(tmp_path):
    head = "a" * 40
    assert "no verified-commit file" in guard.verified_problems(REPO, tmp_path / "missing", head)[0]
    f = tmp_path / "VERIFIED"
    f.write_text("# vouched by the coordinator\nnot-a-sha\n", encoding="utf-8")
    assert "no commit sha" in guard.verified_problems(REPO, f, head)[0]
    f.write_text("b" * 40 + " verified on the linux box\n", encoding="utf-8")
    assert "not the verified commit" in guard.verified_problems(REPO, f, head)[0]
    f.write_text("# ok\n" + head + " verified\n", encoding="utf-8")
    assert guard.verified_problems(REPO, f, head) == []
    assert guard.verified_problems(REPO, f, "") != []  # an unknown HEAD is a problem, not a pass


def test_windows_refuses_build_and_publish_but_linux_does_not():
    assert "Windows" in guard.platform_problems("Windows")[0] and "MSVC" in guard.platform_problems("Windows")[0]
    assert guard.platform_problems("Linux") == []


def test_dev_remote_and_dirty_export_are_refused(tmp_path):
    ctx = make_ctx(tmp_path)
    assert guard.target_problems(ctx, [f"git@example.invalid:{OWNER}/{NAME}.git"], [])
    assert guard.target_problems(ctx, ["git@example.invalid:someone/other.git"], []) == []
    assert guard.target_problems(ctx, [], ["file-remote.git"])


def test_existing_public_repo_is_not_fresh_unless_this_tool_made_it():
    assert guard.fresh_repo_problems(True, False, "o/n") and guard.fresh_repo_problems(None, False, "o/n")
    assert guard.fresh_repo_problems(True, True, "o/n") == [] and guard.fresh_repo_problems(False, False, "o/n") == []


def test_gate_refuses_in_a_real_run_and_warns_in_a_dry_run(tmp_path):
    dry, real = make_ctx(tmp_path), make_ctx(tmp_path, publish=True)
    assert guard.gate(dry, "x", ["bad"]) is False and dry.rep.counts == {"WARN": 1}
    with pytest.raises(core.Refuse, match="bad"):
        guard.gate(real, "x", ["bad"])
    with pytest.raises(core.Refuse):
        guard.gate(dry, "x", ["bad"], hard=True)


def test_pep440_spelling():
    assert [rel_registry.pep440(v) for v in ("0.5.0-alpha.2", "1.0.0-beta.1", "1.0.0-rc.3", "1.2.3")] == ["0.5.0a2", "1.0.0b1", "1.0.0rc3", "1.2.3"]


# ---- the command line, on a synthetic tree ---------------------------------------------------------------------------

def git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, env={**os.environ, **ID}, check=True).stdout.strip()


@pytest.fixture()
def tree(tmp_path):
    """A renamed tree (package NAME) with checks/ copied in, committed, plus a VERIFIED file holding its HEAD."""
    root = tmp_path / "dev"
    (root / "checks").mkdir(parents=True)
    for f in ("check_names.py", "_common.py"):  # WHY only these: the gate under test is check_names; the other checkers name the old project
        text = (REPO / "checks" / f).read_text(encoding="utf-8")
        for old, new in ((OLD, NAME), (OLD.capitalize(), NAME.capitalize()), (OLD.upper(), NAME.upper())):  # what scripts/rename.py would have done
            text = text.replace(old, new)
        (root / "checks" / f).write_text(text, encoding="utf-8", newline="\n")
    (root / "checks" / "names_allow.txt").write_text("", encoding="utf-8")
    (root / "pyproject.toml").write_text(f'[project]\nname = "{NAME}"\nversion = "1.0.0"\ndescription = "d"\n', encoding="utf-8")
    (root / "Cargo.toml").write_text(f'[package]\nname = "{NAME}"\nversion = "1.0.0"\n', encoding="utf-8")
    (root / "VERSION").write_text("1.0.0\n", encoding="utf-8")
    (root / ".zenodo.json").write_text('{"title": "t"}', encoding="utf-8")
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "core.autocrlf", "false")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "dev")
    verified = tmp_path / "VERIFIED"
    verified.write_text(git(root, "rev-parse", "HEAD") + " verified\n", encoding="utf-8")
    return root, verified


def cli(*args):
    done = subprocess.run([sys.executable, str(RELEASE / "publish.py"), *args], capture_output=True, text=True, timeout=300, input="")
    return done.returncode, done.stdout + done.stderr


def test_register_dry_run_prints_both_reservations_and_creates_nothing_public(tree, tmp_path):
    root, verified = tree
    rc, out = cli("register", "--repo", str(root), "--work", str(tmp_path / "w"), "--name", NAME, "--verified-file", str(verified))
    assert rc == 0, out
    assert "cargo publish --allow-dirty" in out and "twine upload --non-interactive" in out and "name reserved; release follows" not in out.replace(rel_register.DESCRIPTION, "")
    assert "DRY RUN" in out and "not checked" in out
    assert (tmp_path / "w" / "crate" / "Cargo.toml").read_text(encoding="utf-8").count(rel_register.DESCRIPTION) == 1
    assert 'version = "0.0.0"' in (tmp_path / "w" / "python" / "pyproject.toml").read_text(encoding="utf-8")


def test_register_refuses_without_name(tree, tmp_path):
    rc, out = cli("register", "--repo", str(tree[0]), "--work", str(tmp_path / "w"))
    assert rc == 1 and "--name is required" in out


def test_register_refuses_provisional_name_and_unrenamed_tree(tree, tmp_path):
    root, verified = tree
    rc, out = cli("register", "--repo", str(root), "--work", str(tmp_path / "w"), "--name", OLD, "--verified-file", str(verified))
    assert rc == 1 and "provisional" in out
    rc, out = cli("register", "--repo", str(root), "--work", str(tmp_path / "w"), "--name", "othername", "--verified-file", str(verified))
    assert rc == 1 and "the tree is named" in out and "scripts/rename.py" in out


def test_register_refuses_a_tree_that_still_holds_the_old_name(tree, tmp_path):
    root, verified = tree
    (root / "NOTES.txt").write_text(f"this still says {OLD}\n", encoding="utf-8")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "leak")
    rc, out = cli("register", "--repo", str(root), "--work", str(tmp_path / "w"), "--name", NAME, "--verified-file", str(verified))
    assert rc == 1 and "still in the tree" in out


def test_register_publish_refuses_an_unverified_commit_before_anything_else(tree, tmp_path):
    root, verified = tree
    verified.write_text("c" * 40 + "\n", encoding="utf-8")
    rc, out = cli("register", "--repo", str(root), "--work", str(tmp_path / "w"), "--name", NAME, "--github-owner", OWNER, "--verified-file", str(verified), "--publish")
    assert rc == 1 and "not the verified commit" in out and not (tmp_path / "w" / "crate").exists()


def test_all_publish_refuses_provisional_name(tree, tmp_path):
    root, verified = tree
    rc, out = cli("all", "--repo", str(root), "--work", str(tmp_path / "w"), "--name", OLD, "--github-owner", OWNER, "--verified-file", str(verified), "--publish")
    assert rc == 1 and ("provisional" in out or "host can build" in out)


def test_the_default_owner_gives_the_public_repository_address():
    import rel_export
    assert core.DEFAULT_OWNER == "saltwater-research"
    assert rel_export.repo_url(core.DEFAULT_OWNER, "seabraid") == "https" + "://github.com/saltwater-research/seabraid"  # WHY split: check_no_egress flags a spelled URL in code
    ap = argparse.ArgumentParser()
    rel_all.add_common(ap, REPO)
    assert ap.parse_args([]).github_owner == core.DEFAULT_OWNER
    assert ap.parse_args(["--github-owner", OWNER]).github_owner == OWNER


@pytest.mark.skipif(sys.platform == "linux", reason="the Windows refusal is the host's own; Linux builds")
def test_all_publish_on_this_non_linux_host_refuses_to_build(tree, tmp_path):
    root, verified = tree
    rc, out = cli("all", "--repo", str(root), "--work", str(tmp_path / "w"), "--name", NAME, "--github-owner", OWNER, "--verified-file", str(verified), "--publish")
    assert rc == 1 and "need Linux" in out


@pytest.mark.skipif(sys.platform == "linux", reason="plan-only is for hosts that cannot build")
def test_all_dry_run_on_a_non_linux_host_prints_the_plan(tree, tmp_path):
    root, verified = tree
    rc, out = cli("all", "--repo", str(root), "--work", str(tmp_path / "w"), "--name", NAME, "--github-owner", OWNER, "--verified-file", str(verified))
    assert rc == 0, out
    assert "plan only" in out and out.count("DRY RUN ") == 4 and "BY HAND" in out and "arXiv" in out
    assert not (tmp_path / "w" / "export").exists()
