# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Record an official Cargo dry run and validate it against an independently tested crate.

This is an operator-transferred receipt from a trusted machine. Cargo owns the registry
protocol. Only this recorder contacts the index, with a fresh cache and a fixed command.
Import never contacts a registry and fails closed on a different tag or archive.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import tomllib

import rel_core as core
import rel_crate

INDEX_MESSAGE = "Updating crates.io index"
STOP_MESSAGE = "warning: aborting upload due to dry run"
VERIFIED = "cargo-verified"
MATCHED = "matched-tested-crate"


def evidence(tag: str, commit: str, tag_object: str, crate: str, digest: str, mode: str) -> dict:
    return {"schema": "crates-io-dry-run-v1", "registry": "crates-io", "tag": tag,
            "export_commit": commit, "tag_object": tag_object, "crate": crate, "sha256": digest,
            "verification": mode, "result": STOP_MESSAGE}


def git_value(ctx: core.Ctx, *argv: str) -> str:
    rc, out = core.git(ctx, ctx.repo, *argv, extra_env={"GIT_OPTIONAL_LOCKS": "0"})
    if rc:
        raise core.Refuse(f"cannot verify tagged checkout: {out}")
    return out.strip()


def revision(ctx: core.Ctx, tag: str) -> tuple[str, str]:
    ref = "refs/tags/" + tag
    if git_value(ctx, "cat-file", "-t", ref) != "tag":
        raise core.Refuse("an annotated export tag is required")
    commit = git_value(ctx, "rev-parse", "--verify", ref + "^{commit}")
    if git_value(ctx, "rev-parse", "--verify", "HEAD") != commit:
        raise core.Refuse("the checkout must be at the export tag")
    if git_value(ctx, "status", "--porcelain=v1", "--untracked-files=all", "--ignored=matching"):
        raise core.Refuse("the tagged checkout must be clean, including ignored files")
    return commit, git_value(ctx, "rev-parse", "--verify", ref)


def crate_name(ctx: core.Ctx, tag: str) -> str:
    with (ctx.repo / "Cargo.toml").open("rb") as manifest:
        package = tomllib.load(manifest)["package"]
    if tag != "v" + package["version"]:
        raise core.Refuse("the export tag must match the manifest version")
    return f"{package['name']}-{package['version']}.crate"


def registry_environment(ctx: core.Ctx, fresh: Path) -> dict:
    if ctx.env.get("CARGO_NET_OFFLINE", "").lower() == "true":
        raise core.Refuse("the official crates.io dry run requires online Cargo")
    if core.inherited_cargo_config(fresh):
        raise core.Refuse("a Cargo configuration above the proof directory can redirect the registry")
    return {**rel_crate.clear_cargo_env(ctx), "CARGO_HOME": str(fresh / "home"),
            "CARGO_TARGET_DIR": str(fresh / "target"), "CARGO_REGISTRIES_CRATES_IO_PROTOCOL": "sparse"}


def run_cargo(ctx: core.Ctx, fresh: Path, skip_build: bool) -> None:
    args = ["cargo", "publish", "--dry-run", "--locked", "--registry", "crates-io"]
    if skip_build:
        args.append("--no-verify")
    rc, out = ctx.run([*args, "--manifest-path", str(ctx.repo / "Cargo.toml")], fresh, registry_environment(ctx, fresh))
    messages = {line.strip() for line in out.splitlines()}
    if rc or not {INDEX_MESSAGE, STOP_MESSAGE}.issubset(messages):
        raise core.Refuse(f"Cargo did not complete an official crates.io dry run (exit {rc}):\n{out}")


def dry_run(ctx: core.Ctx, fresh: Path, filename: str, tested: Path | None) -> tuple[str, str]:
    if tested and tested.name != filename:
        raise core.Refuse("the supplied tested crate filename differs from the tagged manifest")
    expected = core.release_file_sha256(tested) if tested else None
    run_cargo(ctx, fresh, skip_build=tested is not None)
    digest = generated_sha256(fresh / "target" / "package", filename)
    if expected is not None and digest != expected:
        raise core.Refuse("Cargo dry-run archive differs from the supplied tested crate")
    return digest, MATCHED if tested else VERIFIED


def generated_sha256(package: Path, filename: str) -> str:
    # Cargo versions place the generated archive directly here or in the dry-run staging directories.
    candidates = [package / sub / filename for sub in (".", "tmp-crate", "tmp-registry")]
    hashes = {core.release_file_sha256(path) for path in candidates if path.exists()}
    if len(hashes) != 1:
        raise core.Refuse("Cargo dry-run archive missing or inconsistent in its fresh target")
    return hashes.pop()


def record(ctx: core.Ctx, tag: str, output: Path, tested: Path | None) -> None:
    if not core.outside(output.parent, ctx.repo):
        raise core.Refuse("choose a proof destination outside the tagged checkout")
    if output.exists():
        raise core.Refuse("choose a new proof filename; the destination already exists")
    before = revision(ctx, tag)
    name = crate_name(ctx, tag)
    with tempfile.TemporaryDirectory(dir=output.parent) as directory:
        digest, mode = dry_run(ctx, Path(directory), name, tested)
    if revision(ctx, tag) != before:
        raise core.Refuse("the export revision changed during the Cargo dry run")
    with output.open("x", encoding="utf-8") as dest:
        json.dump(evidence(tag, *before, name, digest, mode), dest, indent=2)
        dest.write("\n")
    ctx.rep.add("crates-io-proof", "PASS", f"{name} SHA-256 {digest} ({mode}); proof written to {output}")


def check(ctx: core.Ctx, receipt: dict, required: bool) -> None:
    """The Linux build receipt is authoritative for both packaged and unpacked-tested bytes."""
    proof = ctx.state.get("crates_io_proof")
    if ctx.crates_io_proof is not None:
        try:
            proof = json.loads(ctx.crates_io_proof.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise core.Refuse("cannot read the crates.io dry-run proof") from exc
    if proof is None and not required and ctx.crates_io_proof is None:
        return
    mode = proof.get("verification") if isinstance(proof, dict) else None
    name = receipt["crate"]
    digest = receipt["sha256"][name]
    expected = evidence(receipt["tag"], receipt["export_commit"], receipt["tag_object"], name, digest, mode)
    if mode not in (MATCHED, VERIFIED) or proof != expected or receipt["checked_sha256"].get(name) != digest:
        raise core.Refuse("the official crates.io dry-run proof is missing or differs from the sealed, tested crate; use --crates-io-proof")
    ctx.state["crates_io_proof"] = proof


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="publish.py crates-io-proof", description=__doc__.splitlines()[0])
    parser.add_argument("--export", type=Path, required=True, help="clean checkout of the annotated export tag")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output", type=Path, required=True, help="new JSON file outside the checkout")
    parser.add_argument("--tested-crate", type=Path, help="match the independently tested archive and skip rebuilding")
    args = parser.parse_args(argv)
    try:
        output = args.output.absolute()
        ctx = core.Ctx(args.export.resolve(), output.parent, "", "")
        record(ctx, args.tag, output, args.tested_crate.absolute() if args.tested_crate else None)
    except (core.Refuse, OSError, KeyError, ValueError) as exc:
        print(f"[FAIL  ] refused: {core.redact(str(exc))}")
        return 1
    return 0
