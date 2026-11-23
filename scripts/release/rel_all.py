# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""`publish.py all`: every package manager and archive in one owner-run command, from one verified export.

OWNER: this module owns the one-command flow (guards, then the four stages, then the ordered targets); the guards' rules
are rel_guard's, the steps are rel_upload's, the published-state queries are rel_registry's.
CONTRACT: ``main(argv, run_stages)``; ``run_stages(ctx, stages) -> bool`` is publish.py's stage runner, passed in so this
module never imports the entry point.  Targets, in this fixed order: GitHub repo, GitHub release, PyPI, crates.io; then the by-hand
targets (Zenodo, Software Heritage, Wayback Machine, arXiv, GitHub settings), which the run names so nothing the docs promise is silent.
DRY RUN IS THE DEFAULT and prints the exact commands.  A real run needs ``--publish`` and, per target, the typed text
"<target> <version>"; credentials come only from the environment or the tools' own logins.
FAILURE: the first failing target stops the run; finished targets stay recorded in ``<work>/state.json`` and a rerun
resumes at the first unfinished one (a real run also asks each registry, so a target published by hand is skipped).
"""

from __future__ import annotations

import argparse
import datetime
import platform
import sys
import tempfile
from pathlib import Path

import rel_core as core
import rel_guard as guard
import rel_preflight
import rel_registry
import rel_upload

ORDER_NOTE = "order: repo -> release -> pypi -> crates (cheapest to undo first, permanent uploads last), then the by-hand targets"
MANUAL_TARGETS = (
    "Zenodo DOI: enable the repository before the release or deposit the full tagged source archive from SHA256SUMS with .zenodo.json; use one route (post_publish.md item 3)",
    "Software Heritage and Wayback Machine archive requests (post_publish.md items 4 and 5; pages need no account)",
    "arXiv: no registry or API; submit by hand after the DOI exists (docs/RELEASING.md, arXiv section; post_publish.md item 8)",
    "GitHub settings (private vulnerability reporting, branch protection): post_publish.md items 1 and 2",
)


def add_common(ap: argparse.ArgumentParser, here: Path) -> None:
    ap.add_argument("--name", help="the FINAL product name (default for `all`: the tree's own name; `register` requires it)")
    ap.add_argument("--github-owner", default=core.DEFAULT_OWNER, help=f"GitHub user or organisation for the public repository (default: {core.DEFAULT_OWNER}, owner decision 2026-11-21)")
    ap.add_argument("--repo", type=Path, default=here, help="the dev repository (default: the one holding this script)")
    ap.add_argument("--work", type=Path, help="work directory outside the repo (default: a temp dir)")
    ap.add_argument("--verified-file", type=Path, default=None, help=f"file holding the verified commit sha (default: {guard.default_verified_file()})")
    ap.add_argument("--publish", action="store_true", help="really do it, each target after a typed confirmation")


def parse(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="publish.py all", description=__doc__.splitlines()[0])
    add_common(ap, Path(__file__).resolve().parents[2])
    ap.add_argument("--dev-branch", default="main")
    ap.add_argument("--crates-io-proof", type=Path, help="official crates.io dry-run proof for the sealed crate")
    ap.add_argument("--skip-step", action="append", default=[], choices=("repo", "release", "pypi", "crates"))
    ap.add_argument("--skip-gate", action="append", default=[], choices=core.GATES, help="skip one gate (dry runs only)")
    ap.add_argument("--upload-sdist", action="store_true")
    return ap.parse_args(argv)


def make_ctx(a: argparse.Namespace, suffix: str = "release") -> core.Ctx:
    repo = a.repo.resolve()
    name = a.name or rel_preflight.tree_name(repo)
    work = (a.work or Path(tempfile.gettempdir()) / f"{name}-{suffix}").resolve()
    if not core.outside(work, repo):
        raise core.Refuse(f"--work {work} is inside the repository; choose a directory outside it")
    if a.publish and getattr(a, "skip_gate", None):
        raise core.Refuse(f"--publish cannot skip a gate (asked to skip: {sorted(set(a.skip_gate))})")
    ctx = core.Ctx(repo=repo, work=work, name=name, owner=a.github_owner, dev_branch=getattr(a, "dev_branch", "main"), publish=a.publish,
                   upload_sdist=getattr(a, "upload_sdist", False), skip_gates=set(getattr(a, "skip_gate", [])), skip_steps=set(getattr(a, "skip_step", [])), all_mode=True)
    ctx.registry = rel_registry.Registry(ctx)
    ctx.verified_file = a.verified_file
    ctx.crates_io_proof = getattr(a, "crates_io_proof", None)
    return ctx


def head_of(ctx: core.Ctx) -> str:
    rc, out = core.git(ctx, ctx.repo, "rev-parse", "HEAD")
    return out.strip() if rc == 0 else ""


def remotes(ctx: core.Ctx, cwd: Path) -> list[str]:
    if not (cwd / ".git").exists():
        return []
    rc, out = core.git(ctx, cwd, "remote", "-v")
    return sorted({ln.split()[1] for ln in out.splitlines() if len(ln.split()) > 1}) if rc == 0 else []


def common_guards(ctx: core.Ctx, verified_file: Path | None, system: str) -> bool:
    """The rules both `all` and `register` obey.  Returns False when the host cannot build (plan-only dry run)."""
    problems = guard.platform_problems(system)
    can_build = not problems
    if ctx.publish:
        guard.gate(ctx, f"host can build and publish ({system})", problems)
    elif problems:
        ctx.rep.add("guard", "NOTE", f"plan only: {problems[0]}")
    guard.gate(ctx, f"name {ctx.name!r} is final", guard.name_problems(ctx.name, ctx.repo))
    guard.gate(ctx, "HEAD is the verified commit", guard.verified_problems(ctx.repo, verified_file or guard.default_verified_file(), head_of(ctx)))
    return can_build


def network_guards(ctx: core.Ctx) -> None:
    """Real runs only (a dry run stays offline-capable and asks no registry): the network answers, the public GitHub repository
    is fresh (this tool's own, or absent), and both package names are free or already ours.  WHY all of it before the first
    stage: the repo step is the first public action, and a name taken elsewhere must stop the run before it, not after."""
    down = ctx.registry.reachable()
    guard.gate(ctx, "network reachable", [f"no answer from {h}" for h in down])
    state = core.load_state(ctx.work)
    published = state.get("published", [])
    exists = ctx.registry.name_taken("repo", ctx.name, ctx.owner)
    guard.gate(ctx, "public repository is fresh", guard.fresh_repo_problems(exists, bool(state.get("repo_commands")) or "repo" in published, f"{ctx.owner}/{ctx.name}"))
    taken = {key: ctx.registry.name_taken(key, ctx.name) for key in ("pypi", "crates") if key not in ctx.skip_steps}
    guard.gate(ctx, "package names are free or already ours", guard.package_name_problems(taken, published, ctx.name))


def own_origin(ctx: core.Ctx) -> str:
    """The ``origin`` address the repo step recorded, or "" when that step has not finished (nothing is then expected)."""
    state = core.load_state(ctx.work)
    return state.get("origin", "") if state.get("repo_commands") or "repo" in state.get("published", []) else ""


def plan_only(ctx: core.Ctx) -> bool:
    """A host that cannot build still prints every outward command, from the version alone (no export, no wheel)."""
    version = core.read_version(ctx.repo)
    pep = rel_registry.pep440(version)
    ctx.state = {"version": version, "tag": f"v{version}", "wheels": [f"{ctx.name}-{pep}-<platform>.whl"], "sdists": [f"{ctx.name}-{pep}.tar.gz"], "done": []}
    for line in ("preflight (checks, pytest, secret scan)", "export (one commit, renamed, re-checked)", "build (sdist, wheel, twine check, cargo package)"):
        print(f"PLAN (not run on this host): stage {line}")
    return rel_upload.dry_run(ctx, rel_upload.steps_for(ctx))


def resume_hint(ctx: core.Ctx) -> str:
    state = core.load_state(ctx.work)
    done = state.get("published", [])
    # WHY ctx.state and not the file: the targets need this run's tag and files; a run that failed before export has none,
    # even when an earlier run's state file carries a version, and a rerun starts again at preflight anyway.
    staged = "tag" in (ctx.state or {})
    nxt = next((s.key for s in rel_upload.steps_for(ctx) if s.key not in done and s.key not in ctx.skip_steps), "none") if staged else "preflight"
    return f"resume: fix the cause and rerun the same command. Published so far: {', '.join(done) or 'nothing'}. It resumes at: {nxt} (work dir {ctx.work})"


def guards(ctx: core.Ctx, a: argparse.Namespace) -> bool:
    """Every refusal, in order.  Returns whether this host can build (False: plan only, dry run)."""
    can_build = common_guards(ctx, a.verified_file, platform.system())
    guard.gate(ctx, "public target is not the private dev remote", guard.target_problems(ctx, remotes(ctx, ctx.repo), remotes(ctx, ctx.export_dir), own_origin(ctx)))
    if ctx.publish:
        guard.gate(ctx, "release dates match today", rel_preflight.citation_problems(ctx.repo, datetime.date.today()))
        ctx.state = core.load_state(ctx.work)
        core.require_preparation_checks(ctx)
        network_guards(ctx)
    return can_build


def report_outcome(ctx: core.Ctx, ok: bool) -> None:
    for line in MANUAL_TARGETS if ok else ():
        print(f"BY HAND (no command the tool can run): {line}")
    if not ok:
        print(resume_hint(ctx))


def main(argv: list[str], run_stages) -> int:
    try:
        a = parse(argv)
        ctx = make_ctx(a)
        print(("PUBLISH MODE: each target asks for a typed confirmation" if ctx.publish else "DRY RUN: nothing is uploaded, pushed or created")
              + f"\nname {ctx.name}, owner {ctx.owner}, work dir {ctx.work}\n{ORDER_NOTE}", flush=True)
        can_build = guards(ctx, a)
        # A real invocation consumes the receipt from a separate preparation run.
        stages = ["publish"] if ctx.publish else ["preflight", "export", "build", "publish"]
        ok = run_stages(ctx, stages) if can_build else plan_only(ctx)
        report_outcome(ctx, ok)
    except core.Refuse as exc:
        print(f"[FAIL  ] refused: {core.redact(str(exc))}")
        return 1
    print("\nsummary: " + ", ".join(f"{k} {v}" for k, v in sorted(ctx.rep.counts.items())))
    return 0 if ok and not ctx.rep.failed() else 1


if __name__ == "__main__":
    sys.exit("run it as: python scripts/release/publish.py all")
