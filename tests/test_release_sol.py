# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the six defects an outside reviewer reported against the release tool (claims A-F).

OWNER: this file pins, one test group per claim, what the one-command release (scripts/release/) must do at the seams the
older tests did not reach: the filled public tree against the egress rule (A), a dry run followed by --publish and a resume
after a partial publish (B), the `origin` remote the repo step itself adds (C), gh's answer for a repository that does not
exist (D), a package name someone else holds (E), and the offline-versus-real rule for twine and the registries (F).
No test touches a network or runs gh, pip, cargo or twine: ``Ctx.run`` is the recorder of test_release_all.py, which also
plays the registries' answers.  Every URL is built from parts because checks/check_no_egress.py reads a spelled one as network use.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest
from conftest import REPO
from test_release_all import NAME, OWNER, VERSION, Recorder, make_ctx

import check_no_egress as egress  # noqa: E402  (conftest put checks/ on sys.path)

sys.path.insert(0, str(Path(REPO) / "scripts" / "release"))
import publish  # noqa: E402
import rel_all  # noqa: E402
import rel_build  # noqa: E402
import rel_core as core  # noqa: E402
import rel_export  # noqa: E402
import rel_guard as guard  # noqa: E402
import rel_preflight  # noqa: E402
import rel_registry  # noqa: E402
import rel_upload  # noqa: E402

TREE_NAME = rel_preflight.tree_name(Path(REPO))  # the package name the real pyproject.toml carries
GH_MISSING = f"GraphQL: Could not resolve to a Repository with the name '{OWNER}/{NAME}'. (repository)\n"  # gh 2.x, exit 1 (probed on gh 2.103.3)


def url(name: str, owner: str = OWNER, host: str = "github.com") -> str:
    return rel_export.repo_url(owner, name, host)


class CwdRecorder(Recorder):
    """Recorder that also keeps the directory each command ran in, and can answer git questions per directory."""

    def __init__(self, *args, git_answers=None, **kw):
        super().__init__(*args, **kw)
        self.cwds, self.git_answers = [], git_answers or {}

    def __call__(self, cmd, cwd, extra_env=None):
        self.cwds.append(Path(cwd))
        if cmd[0] == "git":
            for (sub, where), reply in self.git_answers.items():
                if cmd[1:1 + len(sub)] == list(sub) and Path(cwd).name == where:
                    return reply
        return super().__call__(cmd, cwd, extra_env)


# ---- A: the filled public tree against the egress rule ----------------------------------------------------------------

def write_placeholder_manifests(root: Path) -> None:
    """Give the fixture private metadata even when the test runs in a filled public export."""
    token = rel_preflight.PLACEHOLDER
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "{TREE_NAME}"\n[project.urls]\nHomepage = "{token}"\nSource = "{token}"\n'
        f'Issues = "{token}/issues"\nChangelog = "{token}/blob/main/CHANGELOG.md"\n', encoding="utf-8",
    )
    (root / "Cargo.toml").write_text(
        f'[package]\nname = "{TREE_NAME}"\nrepository = "{token}"\n', encoding="utf-8",
    )


def filled_tree(tmp_path: Path, owner: str = OWNER) -> Path:
    """Fill the fixture manifests with the export stage's own code."""
    root = tmp_path / "filled"
    root.mkdir()
    write_placeholder_manifests(root)
    ctx = make_ctx(tmp_path)
    ctx.name, ctx.owner = TREE_NAME, owner
    assert rel_export.fill_placeholder(ctx, root) == list(rel_export.FILL_FILES)
    return root


def hits(root: Path) -> list[str]:
    return [b for b in egress.scan(str(root))[0] if not b.startswith("unused ")]  # a two-file tree cannot use the real tree's rows


def test_a_filled_metadata_urls_pass_the_egress_check(tmp_path):
    root = filled_tree(tmp_path)
    assert "github.com" in (root / "Cargo.toml").read_text(encoding="utf-8")  # the fill really put the public address in
    assert hits(root) == []


@pytest.mark.parametrize("rel, text", [
    ("src/lib.rs", 'pub const U: &str = "{own}";\n'),                          # runtime scope
    ("scripts/tool.py", 'U = "{own}"\n'),                                       # same address, wrong place
    ("pyproject.toml", 'description = "{own}"\n'),                              # right file, not a URL field
    ("pyproject.toml", 'Homepage = "{evil}"\n'),                                # URL field, another host
    ("pyproject.toml", 'Homepage = "{other}"\n'),                               # URL field, another package
    ("pyproject.toml", 'Homepage = "{own}?next={evil}"\n'),                     # URL field with a second address smuggled in
    ("Cargo.toml", 'dep = {{ git = "{own}" }}\n'),                              # a git dependency is network use
    ("docs/page.py", 'Homepage = "{own}"\n'),                                   # the field name alone proves nothing
])
def test_a_sabotage_a_url_anywhere_else_still_fails(tmp_path, rel, text):
    root = filled_tree(tmp_path)
    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "a", encoding="utf-8") as fh:
        fh.write(text.format(own=url(TREE_NAME), evil=url(TREE_NAME, host="evil.example"), other=url("otherpkg")))
    found = hits(root)
    assert found and any(rel in h for h in found), found


def fill_ctx(tmp_path, fail=""):
    ctx = make_ctx(tmp_path, fail=fail)
    ctx.run = CwdRecorder(fail)
    ctx.name = TREE_NAME
    dest = tmp_path / "work" / "export"
    dest.mkdir(parents=True)
    write_placeholder_manifests(dest)
    return ctx, dest


def test_a_the_export_stage_runs_every_checker_on_the_filled_tree(tmp_path):
    ctx, dest = fill_ctx(tmp_path)
    assert rel_export.fill_and_check(ctx, dest) is True
    ran = [(c, d) for c, d in zip(ctx.run.calls, ctx.run.cwds) if c[-1].endswith("run_all.py")]
    assert ran and ran[-1][1] == dest, "checks/run_all.py (egress included) must run inside the FILLED export"
    assert "github.com" in (dest / "Cargo.toml").read_text(encoding="utf-8")  # it ran after the fill


def test_a_pytest_receives_the_filled_tree_and_can_write_without_changing_the_export(tmp_path):
    ctx, dest = fill_ctx(tmp_path)
    recorder = ctx.run

    def run(cmd, cwd, extra_env=None):
        if cmd[1:3] == ["-m", "pytest"]:
            assert cwd != dest
            assert rel_export.hash_tree(cwd) == rel_export.hash_tree(dest)
            assert url(TREE_NAME) in (cwd / "Cargo.toml").read_text(encoding="utf-8")
            (cwd / ".pytest_cache").mkdir()
        return recorder(cmd, cwd, extra_env)

    ctx.run = run
    assert rel_export.fill_and_check(ctx, dest) is True
    assert [sys.executable, "-m", "pytest", "-q", "tests", "bindings/python/tests"] in recorder.calls
    assert rel_export.junk_problems(dest) == []


@pytest.mark.parametrize("failed_gate", ["run_all.py", "-m pytest"])
def test_a_a_failing_gate_on_the_filled_tree_fails_the_stage(tmp_path, failed_gate):
    ctx, dest = fill_ctx(tmp_path, fail=failed_gate)
    assert rel_export.fill_and_check(ctx, dest) is False


# ---- B: dry run then --publish, and resume after a partial publish ----------------------------------------------------

def stage_funcs(monkeypatch, calls):
    """Stand-ins for the four stages that keep what the real ones keep in ``ctx.state`` (the export goes through the real
    ``rel_export.prepare``, whose ownership test is the code under proof)."""
    def extract_head(ctx, dest):
        dest.mkdir(parents=True)
        (dest / "f.txt").write_text("x", encoding="utf-8")
    monkeypatch.setattr(rel_export, "extract_head", extract_head)

    def preflight(ctx):
        calls.append("preflight")
        ctx.state.update(version=VERSION, head=ctx.head)
        return True

    def export(ctx):
        calls.append("export")
        ok = rel_export.prepare(ctx, ctx.export_dir, ctx.name)
        ctx.state.update(export_commit=f"commit-{len(calls)}", tag=f"v{VERSION}")
        return ok

    def build(ctx):
        calls.append("build")
        ctx.state.update(wheels=["w.whl"], sdists=["s.tar.gz"])
        return True

    monkeypatch.setattr(publish, "STAGE_FUNCS", {"preflight": preflight, "export": export, "build": build, "publish": lambda ctx: calls.append("publish") or True})


def staged_ctx(tmp_path, head="h1", publish_mode=False):
    ctx = make_ctx(tmp_path, publish=publish_mode)
    ctx.head = head
    return ctx


STAGES = ["preflight", "export", "build", "publish"]


def test_b_a_dry_run_then_publish_reuses_the_work_directory(tmp_path, monkeypatch):
    calls = []
    stage_funcs(monkeypatch, calls)
    assert publish.run_stages(staged_ctx(tmp_path), STAGES) is True
    assert publish.run_stages(staged_ctx(tmp_path, publish_mode=True), STAGES) is True  # was: "exists and was not made by this script"
    assert calls == STAGES * 2


def test_b_a_resume_after_a_partial_publish_keeps_the_export_and_the_completed_targets(tmp_path, monkeypatch):
    calls = []
    stage_funcs(monkeypatch, calls)
    first = staged_ctx(tmp_path, publish_mode=True)
    assert publish.run_stages(first, STAGES) is True
    first.state.update(published=["repo"], origin="o")  # the repo step finished, the release step did not
    core.save_state(first)
    export_commit = first.state["export_commit"]
    calls.clear()
    again = staged_ctx(tmp_path, publish_mode=True)
    assert publish.run_stages(again, STAGES) is True
    assert calls == ["preflight", "publish"], "export and build must not be redone: the tag is already public"
    assert again.state["published"] == ["repo"] and again.state["export_commit"] == export_commit and again.state["wheels"] == ["w.whl"]
    assert "export" in again.state["done"] and "build" in again.state["done"]


def test_b_a_moved_dev_head_after_a_partial_publish_is_refused(tmp_path, monkeypatch):
    stage_funcs(monkeypatch, [])
    first = staged_ctx(tmp_path, publish_mode=True)
    publish.run_stages(first, STAGES)
    first.state.update(published=["repo"])
    core.save_state(first)
    with pytest.raises(core.Refuse, match="already published"):
        publish.run_stages(staged_ctx(tmp_path, head="h2", publish_mode=True), STAGES)


def test_b_a_rehearsal_alone_does_not_pin_the_next_run(tmp_path, monkeypatch):
    calls = []
    stage_funcs(monkeypatch, calls)
    first = staged_ctx(tmp_path, publish_mode=True)
    publish.run_stages(first, STAGES)
    first.state.update(published=["testpypi"])  # a TestPyPI rehearsal is not a public release
    core.save_state(first)
    calls.clear()
    assert publish.run_stages(staged_ctx(tmp_path, head="h2", publish_mode=True), STAGES) is True
    assert calls == STAGES


# ---- C, D, E: the guards a real run passes before anything is made public -------------------------------------------

HEAD = "a" * 40


def guard_ctx(tmp_path, monkeypatch, *, export_remotes="", published=(), origin="", answers=None, publish_mode=True, date_problems=()):
    """A context whose git and registries are fakes; ``guards`` is the real function under test."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    verified = tmp_path / "VERIFIED"
    verified.write_text(HEAD + " ok\n", encoding="utf-8")
    monkeypatch.setattr(rel_all.platform, "system", lambda: "Linux")
    monkeypatch.setattr(rel_preflight, "citation_problems", lambda *_: list(date_problems))
    base = {
        "versions pip --pre": (0, "pip (1)\nAvailable versions: 1\n"),
        "cargo info serde --registry crates-io": (0, "serde #fixture\nversion: 1.0.0\n"),
        "gh auth status": (0, "ok"),
    }
    ctx = make_ctx(tmp_path, publish=publish_mode, answers={**base, **(answers or {})})
    ctx.run = CwdRecorder(answers={**base, **(answers or {})}, git_answers={
        (("rev-parse", "HEAD"), Path(REPO).name): (0, HEAD + "\n"),
        (("remote", "-v"), "export"): (0, export_remotes),
        (("remote", "-v"), Path(REPO).name): (0, "")})
    (ctx.export_dir / ".git").mkdir(parents=True, exist_ok=True)
    ctx.state["head"] = HEAD
    records = {s: core.stage_check_record(ctx, s, set()) for s in core.STAGE_GATES}
    ctx.state["stage_checks"] = {s: records[s] for s in ("preflight", "export")}
    ctx.state["artifact_receipt"] = {"stage_checks": records}
    if published:
        ctx.state.update(published=list(published), **({"origin": origin} if origin else {}))
    core.save_state(ctx)
    return ctx, argparse.Namespace(verified_file=verified)


def remote_lines(address: str) -> str:
    return f"origin\t{address} (fetch)\norigin\t{address} (push)\n"


def test_c_the_origin_the_repo_step_added_is_accepted_on_resume(tmp_path, monkeypatch):
    own = url(NAME)
    ctx, a = guard_ctx(tmp_path, monkeypatch, export_remotes=remote_lines(own), published=["repo"], origin=own,
                       answers={"gh repo view": (0, "ok")})
    rel_all.guards(ctx, a)  # was: Refuse "the export already has a remote"


def test_c_a_preexisting_remote_is_still_refused(tmp_path, monkeypatch):
    ctx, a = guard_ctx(tmp_path, monkeypatch, export_remotes=remote_lines(url(NAME)))  # nothing recorded: not this tool's
    with pytest.raises(core.Refuse, match="already has a remote"):
        rel_all.guards(ctx, a)


@pytest.mark.parametrize("current", [url("elsewhere"), url(NAME, owner="someone-else")])
def test_c_a_remote_that_is_not_the_recorded_origin_is_refused(tmp_path, monkeypatch, current):
    ctx, a = guard_ctx(tmp_path, monkeypatch, export_remotes=remote_lines(current), published=["repo"], origin=url(NAME))
    with pytest.raises(core.Refuse, match="already has a remote"):
        rel_all.guards(ctx, a)


def test_c_the_repo_step_records_the_origin_it_added(tmp_path, monkeypatch):
    monkeypatch.setattr(rel_upload, "prove_inputs", lambda ctx: None)
    ctx = make_ctx(tmp_path, publish=True, stdin=f"repo {VERSION}\n", answers={"remote get-url": (0, url(NAME) + "\n")})
    ctx.skip_steps = {"release", "pypi", "crates"}
    assert rel_upload.run(ctx) is True
    assert ctx.state["origin"] == url(NAME)


def test_d_gh_reports_a_missing_repository_as_absent_not_unknown(tmp_path):
    ctx = make_ctx(tmp_path, publish=True, answers={"gh repo view": (1, GH_MISSING)})
    assert rel_registry.Registry(ctx).name_taken("repo", NAME, OWNER) is False
    assert guard.fresh_repo_problems(False, False, f"{OWNER}/{NAME}") == []


def test_d_a_fresh_repository_does_not_stop_the_real_run(tmp_path, monkeypatch):
    ctx, a = guard_ctx(tmp_path, monkeypatch, answers={"gh repo view": (1, GH_MISSING)})
    rel_all.guards(ctx, a)  # was: Refuse "cannot tell whether ... exists"


def test_d_an_unreadable_gh_answer_is_still_unknown(tmp_path):
    ctx = make_ctx(tmp_path, publish=True, answers={"gh repo view": (1, "error connecting to api.github.com")})
    assert rel_registry.Registry(ctx).name_taken("repo", NAME, OWNER) is None


@pytest.mark.parametrize("needle, reply, label", [
    (f"versions {NAME} --pre", (0, f"{NAME} (0.1.0)\nAvailable versions: 0.1.0\n"), "PyPI"),
    (f"cargo info {NAME} --registry crates-io", (0, f"{NAME} #something\nversion: 0.0.1\n"), "crates.io"),
])
def test_e_a_package_name_held_by_someone_else_is_refused_before_the_repo_exists(tmp_path, monkeypatch, needle, reply, label):
    ctx, a = guard_ctx(tmp_path, monkeypatch, answers={"gh repo view": (1, GH_MISSING), needle: reply})
    with pytest.raises(core.Refuse, match=label):
        rel_all.guards(ctx, a)
    assert not any(c[:3] == ["gh", "repo", "create"] for c in ctx.run.calls)


def test_e_a_name_this_tool_published_is_ours(tmp_path, monkeypatch):
    answers = {"gh repo view": (0, "ok"), f"versions {NAME} --pre": (0, "Available versions: 0.1.0, 0.5.0a2\n"),
               f"cargo info {NAME} --registry crates-io": (0, f"{NAME} #fixture\nversion: {VERSION}\n")}
    ctx, a = guard_ctx(tmp_path, monkeypatch, published=["repo", "pypi", "crates"], origin=url(NAME), export_remotes=remote_lines(url(NAME)), answers=answers)
    rel_all.guards(ctx, a)


def test_e_a_registry_that_cannot_answer_about_the_name_refuses(tmp_path, monkeypatch):
    ctx, a = guard_ctx(tmp_path, monkeypatch, answers={"gh repo view": (1, GH_MISSING), f"versions {NAME} --pre": (1, "ERROR: connection refused")})
    with pytest.raises(core.Refuse, match="cannot tell"):
        rel_all.guards(ctx, a)


def test_e_a_skipped_step_needs_no_name_check(tmp_path, monkeypatch):
    ctx, a = guard_ctx(tmp_path, monkeypatch, answers={"gh repo view": (1, GH_MISSING), f"versions {NAME} --pre": (0, "Available versions: 0.1.0\n")})
    ctx.skip_steps = {"pypi"}
    rel_all.guards(ctx, a)


# ---- F: what a dry run may skip and a real run may not ----------------------------------------------------------------

def test_f_a_dry_run_asks_no_registry_and_a_real_run_asks_both_names(tmp_path, monkeypatch):
    dry, a = guard_ctx(tmp_path / "d", monkeypatch, publish_mode=False)
    rel_all.guards(dry, a)
    assert not [c for c in dry.run.calls if c[:2] in (["cargo", "info"], ["gh", "repo"]) or "index" in c]
    real, a = guard_ctx(tmp_path / "r", monkeypatch, answers={"gh repo view": (1, GH_MISSING)})
    rel_all.guards(real, a)
    asked = [" ".join(c) for c in real.run.calls]
    assert any(f"versions {NAME}" in c for c in asked) and f"cargo info {NAME} --registry crates-io" in asked


def test_f_a_missing_twine_on_a_host_that_builds_fails_the_build_stage(tmp_path):
    ctx = make_ctx(tmp_path)
    ctx.twine_command = lambda: None
    assert rel_build.twine_check(ctx) is False and ctx.rep.counts == {"FAIL": 1}


NO_INDEX = "error: failed to update registry\nCould not resolve host: index.crates.io\n"


def test_f_cargo_publish_dry_run_without_the_index_skips_in_a_dry_run_and_fails_in_a_real_run(tmp_path):
    dry = make_ctx(tmp_path / "d", answers={"publish --dry-run": (101, NO_INDEX)})
    assert rel_build.publish_dry_run(dry, dry.export_dir) is True and dry.rep.counts == {"SKIP": 1}
    real = make_ctx(tmp_path / "r", publish=True, answers={"publish --dry-run": (101, NO_INDEX)})
    assert rel_build.publish_dry_run(real, real.export_dir) is False and real.rep.counts == {"FAIL": 1}


def test_f_preflight_treats_missing_twine_and_gh_as_a_warning_only_in_a_dry_run(tmp_path):
    for publish_mode, status in ((False, "WARN"), (True, "FAIL")):
        ctx = make_ctx(tmp_path / str(publish_mode), publish=publish_mode)
        ctx.tool = lambda name: None if name in ("twine", "gh") else name
        rel_preflight.check_tools(ctx)
        assert ctx.rep.counts.get(status) == 2 and ("FAIL" in ctx.rep.counts) == publish_mode


def test_public_contact_is_the_one_allowed_address_at_its_domain():
    """Owner decision 2026-11-21: the public identity is dev@saltyresearch.com; any other address there is still flagged."""
    import rel_secrets
    assert rel_secrets.line_kinds('authors = ["Saltwater Research LLC <dev@saltyresearch.com>"]') == []
    other = "someone" + "@" + rel_secrets.PUBLIC_CONTACT.split("@")[1]  # WHY assembled: the tree scan must not see a literal address
    assert "email-address" in rel_secrets.line_kinds(f"contact: {other}")

def test_stale_sealed_release_date_refuses_before_network_checks(tmp_path, monkeypatch):
    ctx, args = guard_ctx(tmp_path, monkeypatch, date_problems=["release date is out of date"])
    with pytest.raises(core.Refuse, match="release date is out of date"):
        rel_all.guards(ctx, args)
    assert not any(cmd[0] != "git" for cmd in ctx.run.calls)
