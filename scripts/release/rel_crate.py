# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Bind Cargo's generated upload to the tested archive. Fail closed on unknown inputs or authority."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import rel_core as core


def arguments(ctx: core.Ctx) -> list[str]:
    # A clean working directory prevents Cargo from discovering project-specific registry configuration.
    return ["cargo", "publish", "--locked", "--registry", "crates-io", "--no-verify",
            "--manifest-path", str(ctx.export_dir / "Cargo.toml")]


def clear_cargo_env(ctx: core.Ctx) -> dict:
    return {key: None for key in ctx.env if key.upper().startswith(("CARGO_", "RELEASE_CRATE_"))}


def check_repack(ctx: core.Ctx) -> None:
    with tempfile.TemporaryDirectory(dir=ctx.work) as fresh:
        env = {**clear_cargo_env(ctx), **ctx.cargo_env(), "CARGO_TARGET_DIR": fresh}
        rc, out = ctx.run(["cargo", "package", "--locked", "--offline", "--no-verify"], ctx.export_dir, env)
        if rc:
            raise core.Refuse("could not reproduce the sealed crate: " + out)
        name = ctx.state["crate"]
        repacked = core.release_file_sha256(Path(fresh) / "package" / name)
    if repacked != ctx.state["artifact_receipt"]["sha256"][name]:
        raise core.Refuse("Cargo repack differs from the sealed crate; no crate transfer is permitted")


def configure(ctx: core.Ctx, fresh: Path, digest: str) -> dict:
    """Only Cargo's publish-operation credential callback can obtain the release token."""
    if core.inherited_cargo_config(fresh):
        raise core.Refuse("Cargo configuration above the temporary publish directory could override its registry")
    home = fresh / "cargo-home"
    home.mkdir()
    provider = ctx.export_dir / "scripts" / "release" / "rel_crate_credential.py"
    if not provider.is_file():
        raise core.Refuse("the verified export lacks the checksum-bound Cargo credential provider")
    invocation = json.dumps([sys.executable, "-I", "-B", str(provider)], ensure_ascii=False)
    (home / "config.toml").write_text(f"[registry]\ncredential-provider = {invocation}\nglobal-credential-providers = []\n", encoding="utf-8")
    token = ctx.env.get("CARGO_REGISTRY_TOKEN")
    if not token:
        raise core.Refuse("checksum-bound crate publication requires CARGO_REGISTRY_TOKEN in the owner's environment")
    env = clear_cargo_env(ctx)
    env.update(CARGO_HOME=str(home), CARGO_TARGET_DIR=str(fresh / "target"), RELEASE_CRATE_TOKEN=token,
               RELEASE_CRATE_NAME=ctx.name, RELEASE_CRATE_VERSION=ctx.state["version"], RELEASE_CRATE_SHA256=digest,
               RELEASE_CRATE_WITNESS=str(fresh / "authorized.sha256"))
    return env


def upload(ctx: core.Ctx) -> tuple[int, str]:
    check_repack(ctx)
    digest = ctx.state["artifact_receipt"]["sha256"][ctx.state["crate"]]
    with tempfile.TemporaryDirectory(dir=ctx.work) as directory:
        fresh = Path(directory)
        env = configure(ctx, fresh, digest)
        rc, out = ctx.run(arguments(ctx), fresh, env)
        if rc:
            return rc, out
        witness = Path(env["RELEASE_CRATE_WITNESS"])
        expected = {"name": ctx.name, "vers": ctx.state["version"], "cksum": digest}
        if not witness.is_file() or json.loads(witness.read_text(encoding="utf-8")) != expected:
            raise core.Refuse("Cargo did not confirm authorization of the exact sealed checksum; inspect the registry before retrying")
    return 0, out
