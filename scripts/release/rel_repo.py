# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Per-command GitHub repository receipts. Refuse a changed branch, tag, or push destination."""

from __future__ import annotations

import re

import rel_core as core


def git_output(ctx: core.Ctx, *args: str) -> str:
    rc, out = core.git(ctx, ctx.export_dir, *args)
    if rc:
        raise core.Refuse(f"cannot verify repository state with git {args[0]}: {out}")
    return out.strip()


def origin(ctx: core.Ctx) -> str:
    """Check both Git destinations, including pushurl and insteadOf rewrites."""
    fetch = git_output(ctx, "remote", "get-url", "--all", "origin")
    push = git_output(ctx, "remote", "get-url", "--push", "--all", "origin")
    slug = re.escape(f"{ctx.owner}/{ctx.name}")
    if fetch != push or not re.fullmatch(rf"(?:https:/{{2}}github\.com/|git@github\.com:){slug}(?:\.git)?", push, re.I):
        raise core.Refuse("origin does not point exclusively to the release repository")
    return fetch


def record(ctx: core.Ctx, command: str) -> None:
    saved = ctx.state.setdefault("repo_commands", [])
    if command not in saved:
        saved.append(command)
        core.save_state(ctx)


def check_refs(ctx: core.Ctx) -> set[str]:
    """Return verified completed pushes. A removed or replaced ref requires owner review."""
    if origin(ctx) != ctx.state.get("origin"):
        raise core.Refuse("origin changed since repository creation")
    tag = "refs/tags/" + ctx.state["tag"]
    expected = {"refs/heads/main": ctx.state["export_commit"], tag: ctx.state["tag_object"]}
    raw = git_output(ctx, "ls-remote", "--refs", "origin")
    done = {"create"}
    for entry in raw.splitlines():
        commit, _, ref = entry.partition("\t")
        if ref not in expected or commit != expected[ref]:
            raise core.Refuse(f"unexpected remote ref or commit at {ref}; review the remote before resuming")
        done.add("main" if ref == "refs/heads/main" else "tag")
    if not set(ctx.state.get("repo_commands", [])).issubset(done):
        raise core.Refuse("a recorded repository ref is missing; review the remote before resuming")
    return done


def reconcile(ctx: core.Ctx) -> set[str]:
    if not ctx.state.get("repo_commands"):
        return set()
    return check_refs(ctx)


def completed(ctx: core.Ctx, command: str) -> None:
    if command == "create":
        ctx.state["origin"] = origin(ctx)
        record(ctx, command)
    if command not in check_refs(ctx):
        raise core.Refuse(f"repository command {command} did not establish the expected remote ref")
    record(ctx, command)
