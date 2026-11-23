#!/usr/bin/env python3
# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Release the project: preflight, export a clean one-commit repo, build, and (only with --publish) publish.

One command:  python scripts/release/publish.py register --name FINAL [--publish]   (reserve the name on every registry)
              python scripts/release/publish.py all [--name FINAL --github-owner OWNER] [--publish]   (every target; see rel_all.py)
Usage:  python scripts/release/publish.py --name FINAL_NAME --github-owner OWNER [--work DIR]            (dry run)
        python scripts/release/publish.py --name FINAL_NAME --github-owner OWNER --publish --test-pypi   (rehearsal)
        python scripts/release/publish.py --name FINAL_NAME --github-owner OWNER --publish               (the real thing)
The human guide is docs/RELEASING.md; read it first.

OWNER: this file owns the command line and the stage order; each stage lives in its own module (rel_preflight, rel_export,
rel_build, rel_upload), shared plumbing in rel_core, credential shapes in rel_secrets.
DRY RUN IS THE DEFAULT: without --publish every stage up to the uploads runs for real (checks, export, builds) and the
publish stage only PRINTS what it would do.  With --publish each outward step prints its exact commands and waits for a
typed confirmation.  Nothing here ever stores or prints a credential.
RESUME: ``<work>/state.json`` records the stages that passed for one dev HEAD; ``--only`` / ``--skip`` pick stages, and a
rerun of the publish stage resumes after the last step that completed.  preflight starts a new run and clears the state.
EXIT: 0 every selected stage passed, 1 a gate failed or a step was refused, 2 bad usage.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rel_build  # noqa: E402
import rel_core as core  # noqa: E402
import rel_export  # noqa: E402
import rel_preflight  # noqa: E402
import rel_upload  # noqa: E402
import rel_registry  # noqa: E402

STAGE_FUNCS = {"preflight": rel_preflight.run, "export": rel_export.run, "build": rel_build.run, "publish": rel_upload.run}
STEP_KEYS = ("repo", "release", "pypi", "crates", "testpypi")
# WHY a gate that can be skipped for a dry run can never be skipped for --publish: a skipped gate is an unproven claim.
UNSKIPPABLE_WHEN_PUBLISHING = {"local-checks", "pytest", "manual"}


def parse(argv: list[str] | None) -> argparse.Namespace:
    here = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="the FINAL product name (the tree is renamed to it in the export)")
    ap.add_argument("--github-owner", default=core.DEFAULT_OWNER, help=f"GitHub user or organisation that will own the public repository (default: {core.DEFAULT_OWNER}, owner decision 2026-11-21)")
    ap.add_argument("--repo", type=Path, default=here, help="the dev repository (default: the one holding this script)")
    ap.add_argument("--work", type=Path, help="work directory outside the repo (default: a temp dir named after the name)")
    ap.add_argument("--dev-branch", default="main", help="the branch a release must be cut from (default: main)")
    ap.add_argument("--verified-file", type=Path, help="coordinator file containing the verified full commit sha")
    ap.add_argument("--crates-io-proof", type=Path, help="official crates.io dry-run proof for the sealed crate")
    ap.add_argument("--only", default="", help=f"comma list of stages to run (default all: {','.join(core.STAGES)})")
    ap.add_argument("--skip", default="", help="comma list of stages to skip")
    ap.add_argument("--skip-gate", action="append", default=[], choices=core.GATES, help="skip one gate (dry runs only)")
    ap.add_argument("--skip-step", action="append", default=[], choices=STEP_KEYS, help="skip one publish step")
    ap.add_argument("--publish", action="store_true", help="really publish, each step after a typed confirmation")
    ap.add_argument("--test-pypi", action="store_true", help="with --publish: only upload to TestPyPI (run this first)")
    ap.add_argument("--upload-sdist", action="store_true", help="also upload the sdist to PyPI (see rel_build.py for why not by default)")
    return ap.parse_args(argv)


def stage_list(text: str) -> list[str]:
    names = [s for s in text.split(",") if s]
    bad = [s for s in names if s not in core.STAGES]
    if bad:
        raise core.Refuse(f"unknown stage(s) {bad}; the stages are {', '.join(core.STAGES)}")
    return names


def select_stages(only: str, skip: str) -> list[str]:
    picked, skipped = stage_list(only) or list(core.STAGES), stage_list(skip)
    return [s for s in core.STAGES if s in picked and s not in skipped]


def build_ctx(a: argparse.Namespace) -> core.Ctx:
    repo = a.repo.resolve()
    work = (a.work or Path(tempfile.gettempdir()) / f"{a.name}-release").resolve()
    if not core.outside(work, repo):
        raise core.Refuse(f"--work {work} is inside the repository; choose a directory outside it")
    if a.publish and UNSKIPPABLE_WHEN_PUBLISHING & set(a.skip_gate):
        raise core.Refuse(f"--publish cannot skip a gate (asked to skip: {sorted(set(a.skip_gate))})")
    if a.test_pypi and not a.publish:
        raise core.Refuse("--test-pypi only means something with --publish (the dry run already lists the TestPyPI step)")
    ctx = core.Ctx(repo=repo, work=work, name=a.name, owner=a.github_owner, dev_branch=a.dev_branch, publish=a.publish, test_pypi=a.test_pypi,
                    upload_sdist=a.upload_sdist, skip_gates=set(a.skip_gate), skip_steps=set(a.skip_step), verified_file=a.verified_file,
                    crates_io_proof=a.crates_io_proof)
    if a.publish and not a.test_pypi:
        ctx.registry = rel_registry.Registry(ctx)
    return ctx


def reuse_after_publishing(ctx: core.Ctx, stage: str) -> bool:
    """True when ``stage`` must not run again.  WHY: once a target is public the export, tag and artifacts are what that target
    holds; the export and build stages would replace them with different bytes and a different commit (resume = publish only)."""
    if stage not in ("export", "build") or not core.went_public(ctx.state):
        return False
    ctx.rep.add(stage, "SKIP", "a public command has completed; the earlier export, tag and artifacts are reused")
    return True


def refuse_moved_head(ctx: core.Ctx, prior: dict) -> None:
    """After a preflight that follows a partial publish: the dev HEAD must still be the commit that run was made from."""
    if core.went_public(prior) and ctx.state.get("head") != prior.get("head"):
        raise core.Refuse(f"{ctx.work} already published from dev commit {str(prior.get('head'))[:12]}, but HEAD is now "
                          f"{str(ctx.state.get('head'))[:12]}; a public release is not rebuilt from another commit: finish it from that commit or start over with a new --work")


def invalidate_build(ctx: core.Ctx, stage: str) -> None:
    if stage not in core.STAGE_GATES:
        return
    # Persist invalidation before a command can fail or be interrupted.
    ctx.state.setdefault("stage_checks", {}).pop(stage, None)
    ctx.state["done"] = [s for s in ctx.state.get("done", []) if s != stage]
    if stage in ("export", "build"):
        ctx.state.pop("crates_io_proof", None)
        ctx.state.pop("artifact_receipt", None)
    core.save_state(ctx)


def finish_stage(ctx: core.Ctx, stage: str, ok: bool) -> None:
    if ok and stage in ("preflight", "export"):
        ctx.state["stage_checks"][stage] = core.stage_check_record(ctx, stage, ctx.skip_gates)
    done = [s for s in ctx.state.get("done", []) if s != stage]
    ctx.state["done"] = done + [stage] if ok else done
    core.save_state(ctx)


def run_stages(ctx: core.Ctx, stages: list[str]) -> bool:
    ctx.state = core.load_state(ctx.work)
    prior: dict = {}
    for stage in stages:
        if stage == "preflight":
            prior, ctx.state = ctx.state, core.next_run_state(ctx.state)  # a new run: only what next_run_state names is trusted
        elif "version" not in ctx.state:
            raise core.Refuse(f"no preflight in {ctx.work}; run the preflight stage first")
        print(f"\n== stage: {stage} " + "=" * (60 - len(stage)), flush=True)
        if reuse_after_publishing(ctx, stage):
            continue
        invalidate_build(ctx, stage)
        ok = STAGE_FUNCS[stage](ctx)
        if stage == "preflight":
            refuse_moved_head(ctx, prior)
        finish_stage(ctx, stage, ok)
        if not ok:
            ctx.rep.add(stage, "FAIL", f"stage {stage} failed; later stages not run")
            return False
        ctx.rep.add(stage, "PASS", f"stage {stage} complete")
    return True


def subcommand(argv: list[str]) -> int | None:
    """`all` and `register` (None when argv is the older flat command line).  WHY lazy and injected: the one-command flow
    must not import this entry point back, so it is handed run_stages."""
    if argv[:1] == ["crates-io-proof"]:
        import rel_crate_proof
        return rel_crate_proof.main(argv[1:])
    if argv[:1] == ["all"]:
        import rel_all
        return rel_all.main(argv[1:], run_stages)
    if argv[:1] == ["register"]:
        import rel_register
        return rel_register.main(argv[1:])
    return None


def check_flat_publish(ctx: core.Ctx) -> None:
    """The flat release CLI has the same remote ownership checks as the all command."""
    import rel_all

    ctx.state = core.load_state(ctx.work)
    rel_upload.prove_inputs(ctx)
    if not ctx.test_pypi:
        rel_all.network_guards(ctx)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    code = subcommand(argv)
    if code is not None:
        return code
    a = parse(argv)
    try:
        ctx = build_ctx(a)
        stages = select_stages(a.only, a.skip)
        if ctx.publish and "publish" in stages:
            # The older flat entry point must consume the same prepared artifacts.
            check_flat_publish(ctx)
            stages = ["publish"]
        mode = "PUBLISH MODE: each outward step asks for a typed confirmation" if ctx.publish else "DRY RUN: checks, export and builds run; nothing is uploaded, pushed or created"
        print(f"{mode}\nname {ctx.name}, owner {ctx.owner}, work dir {ctx.work}, stages {', '.join(stages)}", flush=True)
        ok = run_stages(ctx, stages)
    except core.Refuse as exc:
        print(f"[FAIL  ] refused: {core.redact(str(exc))}")
        return 1
    print("\nsummary: " + ", ".join(f"{k} {v}" for k, v in sorted(ctx.rep.counts.items())))
    return 0 if ok and not ctx.rep.failed() else 1


if __name__ == "__main__":
    sys.exit(main())
