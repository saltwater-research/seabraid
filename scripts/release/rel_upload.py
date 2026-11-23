# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Stage 4, publish: create the public repository, the GitHub release, and the PyPI and crates.io uploads.

OWNER: this module owns the ordered list of outward steps and what each one runs.  Making anything public is the
OWNER's action: without ``--publish`` this stage only PRINTS the commands (a dry run) and changes nothing; with
``--publish`` it prints each step's exact commands and requires the owner to type that step's token before running it.
CONTRACT: ``run(ctx) -> bool``.  Inputs: the export and dist directory from the earlier stages, proven by ``ctx.state``
(same HEAD, every earlier stage passed).  Steps: repo, release, pypi, crates (`all` mode only changes the typed token); with ``--test-pypi`` the only step is a
TestPyPI upload (a rehearsal: run it first, check the install, then run without the flag).  Completed steps are recorded in
the state file, so a rerun after a failure resumes at the first step not done.
CREDENTIALS: gh and twine use their own login mechanisms. Cargo uses the receipt-bound provider in rel_crate;
its token stays out of logs and command arguments. Twine runs with
--non-interactive so it can never prompt for (and echo into a log) a password.
FAILURE: the first failing step stops the stage; its output (redacted) is shown; earlier completed steps stay recorded.
"""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
import rel_crate
import rel_crate_proof
import shlex
import tomllib
from pathlib import Path

import rel_core as core
import rel_guard as guard
import rel_preflight
import rel_repo

STAGE = "publish"
HERE = Path(__file__).resolve().parent


def artifacts(ctx: core.Ctx) -> list[Path]:
    d = ctx.dist_dir
    # Plan-only runs use illustrative filenames. Validation belongs to the real build receipt.
    names = ctx.state.get("wheels", []) + ctx.state.get("sdists", []) + core.source_archives(ctx)
    if "crate" in ctx.state:
        names.append(ctx.state["crate"])
    return [d / n for n in names] + [d / "SHA256SUMS"]


def uploads(ctx: core.Ctx) -> list[str]:
    """Files twine uploads: the wheels, plus the sdist only with --upload-sdist (see rel_build's docstring for why)."""
    names = ctx.state.get("wheels", []) + (ctx.state.get("sdists", []) if ctx.upload_sdist else [])
    return [str(ctx.dist_dir / n) for n in names]


def build_steps(ctx: core.Ctx) -> list[core.Step]:
    export, slug, tag, version = ctx.export_dir, f"{ctx.owner}/{ctx.name}", ctx.state["tag"], ctx.state["version"]
    twine = ctx.twine_command()
    need = () if twine else ("twine",)
    twine = twine or ["twine"]
    src = export if (export / "pyproject.toml").exists() else ctx.repo  # WHY: an `all` dry run on a host that cannot build has no export
    with open(src / "pyproject.toml", "rb") as fh:
        description = tomllib.load(fh)["project"]["description"]
    pre = ["--prerelease"] if "-" in version else []
    if ctx.test_pypi:
        return [core.Step("testpypi", f"testpypi:{ctx.name}", "upload the wheels to TestPyPI (a rehearsal; the version is burned there too)",
                     "TestPyPI keeps this filename forever; the real PyPI is untouched.",
                     [([*twine, "upload", "--non-interactive", "--repository", "testpypi", *uploads(ctx)], ctx.dist_dir)], need)]
    return [
        core.Step("repo", slug, f"create the PUBLIC GitHub repository {slug} and push the single commit and tag {tag}",
             "The repository becomes visible to everyone the moment it is created. Deleting it later does not unpublish what was cloned.",
             [(["gh", "repo", "create", slug, "--public", "--source", str(export), "--remote", "origin", "--description", description], export),
              (["git", "push", "origin", "HEAD:refs/heads/main"], export),
              (["git", "push", "origin", tag], export)], ("gh", "git")),
        core.Step("release", tag, f"create the GitHub release {tag} with the wheels, sdist, SHA256SUMS and the CHANGELOG notes",
             "Zenodo archives only releases created AFTER you enable the repository in its GitHub settings: do that first if you want a DOI for this tag.",
             [(["gh", "release", "create", tag, *map(str, artifacts(ctx)), "--repo", slug, "--title", f"{ctx.name} {version}", "--notes-file", str(ctx.work / "RELEASE_NOTES.md"), "--verify-tag", *pre], export)], ("gh",)),
        core.Step("pypi", f"pypi:{ctx.name}", f"upload the wheels of {ctx.name} {version} to PyPI",
             "IRREVERSIBLE: a PyPI version and filename can be yanked but never reused or removed from history.",
             [([*twine, "upload", "--non-interactive", *uploads(ctx)], ctx.dist_dir)], need),
        core.Step("crates", f"crates:{ctx.name}", f"publish the crate {ctx.name} {version} to crates.io",
             "A checksum-bound credential authorizes exactly the sealed archive. A crates.io version is permanent.",
             [(rel_crate.arguments(ctx), ctx.work)], ("cargo",)),
    ]


def steps_for(ctx: core.Ctx) -> list[core.Step]:
    """The ordered steps of this run.  WHY this order and no other: the cheapest to undo first (repo, release), then the two
    permanent uploads; a failure stops at that step and a rerun resumes there.  In `all` mode the typed token is
    "<target> <version>" (the target and the version both read back by the owner)."""
    steps = build_steps(ctx)
    if not ctx.all_mode or ctx.test_pypi:
        return steps
    version = ctx.state["version"]
    return [dataclasses.replace(s, token=f"{s.key} {version}") for s in steps]


def found_on_registry(ctx: core.Ctx, step: core.Step) -> bool:
    have = ctx.registry.published(step.key, ctx.name, ctx.state["version"], ctx.owner, ctx.state["tag"])
    if have is None:
        raise core.Refuse(f"cannot tell whether {step.key} {ctx.state['version']} is already on its registry; check the network and rerun")
    if step.key == "repo" and ctx.state.get("repo_commands"):
        if not have:
            raise core.Refuse("the recorded release repository is missing")
        return False  # The branch and tag checkpoints still need reconciliation.
    if have and step.key in ("repo", "crates"):
        raise core.Refuse(f"{step.key} already exists without a matching publication receipt; review its ownership and bytes")
    return have


def already_published(ctx: core.Ctx, steps: list[core.Step]) -> list[str]:
    """Ask the registry about each step (a real run only) and record the ones that already exist.  A registry that cannot
    answer stops the run: guessing "not published" would upload twice, guessing "published" would skip a target."""
    if not ctx.publish or ctx.registry is None:
        return []
    found = []
    for step in [s for s in steps if s.key not in ctx.state.get("published", [])]:
        if found_on_registry(ctx, step):
            found.append(step.key)
            ctx.rep.add(STAGE, "SKIP", f"{step.key}: already published on its registry; skipped")
            ctx.state.setdefault("published", []).append(step.key)
    core.save_state(ctx)
    return found


def credential_hint(ctx: core.Ctx, step: core.Step) -> tuple[str, str]:
    """(status, message): whether a login SEEMS to exist.  Presence only; no value is read into a message."""
    home = Path(ctx.env.get("HOME") or ctx.env.get("USERPROFILE") or Path.home())
    if step.key in ("repo", "release"):
        rc, _ = ctx.run(["gh", "auth", "status"], ctx.repo)
        return ("PASS", "gh is logged in") if rc == 0 else ("FAIL", "gh is not logged in: run `gh auth login` (see docs/RELEASING.md)")
    if step.key in ("pypi", "testpypi"):
        return ("PASS", "a twine credential source exists (TWINE_PASSWORD or ~/.pypirc; value not shown)") if twine_login_seen(ctx, home) else ("WARN", "no TWINE_PASSWORD or ~/.pypirc; twine will fail unless a keyring or trusted publishing supplies one (docs/RELEASING.md)")
    return ("PASS", "the checksum-bound Cargo provider has a token") if ctx.env.get("CARGO_REGISTRY_TOKEN") else ("FAIL", "CARGO_REGISTRY_TOKEN is required for checksum-bound publication")


def twine_login_seen(ctx: core.Ctx, home: Path) -> bool:
    return bool(ctx.env.get("TWINE_PASSWORD")) or (home / ".pypirc").exists()



def show_step(ctx: core.Ctx, step: core.Step, prefix: str) -> str:
    lines = [f"{prefix} {step.title}", f"  effect: {step.effect}"]
    lines += [f"  $ (in {cwd}) {shlex.join(argv)}" for argv, cwd in step.cmds]
    return "\n".join(lines)


def sign_off(ctx: core.Ctx) -> None:
    """A verified SHA identifies the revision; the human review also covers this exact receipt."""
    sha = ctx.state["head"]
    problems = guard.verified_problems(ctx.repo, ctx.verified_file or guard.default_verified_file(), sha)
    if problems:
        raise core.Refuse("; ".join(problems))
    if ctx.skip_gates:
        raise core.Refuse("a real release requires every preparation gate")
    encoded = json.dumps(ctx.state["artifact_receipt"], sort_keys=True).encode()
    witness = {"head": sha, "receipt_sha256": hashlib.sha256(encoded).hexdigest(), "items": list(rel_preflight.MANUAL_ITEMS)}
    if ctx.state.get("attestation") == witness:
        return
    if not core.confirm(ctx, "reviewed " + sha, "Confirm for this verified revision:\n" + "\n".join(witness["items"])):
        raise core.Refuse("human review not confirmed; no release action is authorized")
    ctx.state["attestation"] = witness
    core.save_state(ctx)


def prove_inputs(ctx: core.Ctx) -> None:
    done = set(ctx.state.get("done", []))
    missing = [s for s in ("preflight", "export", "build") if s not in done]
    if missing:
        raise core.Refuse(f"stages not completed in this work directory: {', '.join(missing)}; run them first (the publish stage never builds)")
    prove_revision(ctx, ctx.repo, "HEAD", ctx.state.get("head"))
    prove_revision(ctx, ctx.export_dir, "HEAD", ctx.state.get("export_commit"))
    tag_ref = f"refs/tags/{ctx.state['tag']}"
    prove_revision(ctx, ctx.export_dir, tag_ref, ctx.state.get("tag_object"))
    prove_revision(ctx, ctx.export_dir, tag_ref + "^{commit}", ctx.state.get("export_commit"))
    # Cargo reads the working tree, including source files that Git would ordinarily ignore.
    rc, dirty = core.git(ctx, ctx.export_dir, "status", "--porcelain=v1", "--untracked-files=all", "--ignored=matching")
    if rc or dirty.strip():
        raise core.Refuse("the export has unverified edits or files; rerun export and build")
    core.prove_artifacts(ctx)
    require_crates = ctx.publish and not ctx.test_pypi and "crates" not in ctx.skip_steps
    rel_crate_proof.check(ctx, ctx.state["artifact_receipt"], required=require_crates)
    if ctx.publish:
        prove_publish_policy(ctx)


def prove_publish_policy(ctx: core.Ctx) -> None:
    core.require_preparation_checks(ctx)
    problems = rel_preflight.citation_problems(ctx.repo, datetime.date.today())
    if problems:
        raise core.Refuse("; ".join(problems))
    sign_off(ctx)


def prove_revision(ctx: core.Ctx, cwd: Path, ref: str, expected: str | None) -> None:
    rc, found = core.git(ctx, cwd, "rev-parse", "--verify", ref)
    if rc or not expected or found.strip() != expected:
        raise core.Refuse(f"{cwd.name} {ref} no longer matches the gated commit; rerun preflight, export and build")


def print_checklist(ctx: core.Ctx, header: str) -> None:
    text = (HERE / "post_publish.md").read_text(encoding="utf-8")
    for key, value in {"{owner}": ctx.owner, "{name}": ctx.name, "{version}": ctx.state["version"], "{tag}": ctx.state["tag"]}.items():
        text = text.replace(key, value)
    print(f"\n{header}\n{text}")


def ready(ctx: core.Ctx, step: core.Step) -> bool:
    """Tools present and a login seems to exist (presence only).  A FAIL here stops the step before anything is asked."""
    missing = [t for t in step.needs if not ctx.tool(t)]
    if missing:
        ctx.rep.add(STAGE, "FAIL", f"{step.key}: needs {', '.join(missing)} on PATH (docs/RELEASING.md says how to install it)")
        return False
    status, message = credential_hint(ctx, step)
    ctx.rep.add(STAGE, status, f"{step.key}: {message}")
    return status != "FAIL"





def execute(ctx: core.Ctx, step: core.Step) -> bool:
    """Resume at each verified repository command and recheck sealed inputs after prompting."""
    shown = []
    for index, (argv, cwd) in enumerate(step.cmds):
        prove_inputs(ctx)
        command = ("create", "main", "tag")[index] if step.key == "repo" else None
        if command and command in rel_repo.reconcile(ctx):
            rel_repo.record(ctx, command)
            continue
        rc, out = rel_crate.upload(ctx) if step.key == "crates" else ctx.run(argv, cwd)
        if rc != 0:
            ctx.rep.add(STAGE, "FAIL", f"{step.key}: `{' '.join(argv[:3])}` exited {rc}", "\n".join(out.splitlines()[-15:]))
            return False
        if command:
            rel_repo.completed(ctx, command)
        shown.append(out)
    ctx.rep.add(STAGE, "PASS", f"{step.key}: done", "\n".join(shown))
    return True


def run_step(ctx: core.Ctx, step: core.Step, index: int, total: int) -> bool:
    # execute revalidates the receipt and the crate checksum after any confirmation.
    if not ready(ctx, step):
        return False
    if not core.confirm(ctx, step.token, show_step(ctx, step, f"STEP {index}/{total}:")):
        ctx.rep.add(STAGE, "FAIL", f"{step.key}: not confirmed; nothing was done for this step and the run stops here")
        return False
    if not execute(ctx, step):
        return False
    ctx.state.setdefault("published", []).append(step.key)
    core.save_state(ctx)
    return True


def dry_run(ctx: core.Ctx, steps: list[core.Step], checklist: bool = True) -> bool:
    for i, step in enumerate(steps, start=1):
        print(show_step(ctx, step, f"DRY RUN {i}/{len(steps)}: would"), flush=True)
        missing = [t for t in step.needs if not ctx.tool(t)]
        ctx.rep.add(STAGE, "WARN" if missing else "PASS", f"{step.key}: " + (f"needs {', '.join(missing)} (not found)" if missing else "tools found") + "; needs network and a login at --publish time")
    ctx.rep.add(STAGE, "NOTE", "DRY RUN: nothing was created, uploaded or pushed; rerun with --publish to do these steps, each after a typed confirmation")
    if checklist:
        print_checklist(ctx, "After a real publish you would do this (printed again then):")
    return True


def check_recorded_repository(ctx: core.Ctx) -> None:
    if ctx.publish and ("repo" in ctx.state.get("published", []) or ctx.state.get("repo_commands")):
        verified = rel_repo.check_refs(ctx)
        if "repo" in ctx.state.get("published", []) and verified != {"create", "main", "tag"}:
            raise core.Refuse("the recorded repository lacks the verified branch or tag")


def run(ctx: core.Ctx) -> bool:
    prove_inputs(ctx)
    check_recorded_repository(ctx)
    steps = steps_for(ctx)
    already_published(ctx, [s for s in steps if s.key not in ctx.skip_steps])
    steps = [s for s in steps if s.key not in ctx.skip_steps and s.key not in ctx.state.get("published", [])]
    if not steps:
        ctx.rep.add(STAGE, "SKIP", "no step left to run (all done or skipped)")
        return True
    if not ctx.publish:
        return dry_run(ctx, steps)
    # Each step gets its own confirmation and checksum-bound execution.
    for i, step in enumerate(steps, start=1):
        if not run_step(ctx, step, i, len(steps)):
            return False
    print_checklist(ctx, "Published. Now do this, in this order:")
    return True

