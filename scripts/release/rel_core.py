# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Shared plumbing of the release script: the run context, reporting, subprocesses, redaction, typed confirmation, state.

OWNER: this module owns how a stage talks to the owner (Report), how it runs a tool (run) and how an irreversible step is
confirmed (confirm); the stage modules own what is checked or done.  CONTRACT: no stage imports another stage; every
stage imports this module; this module imports only rel_secrets (one-way edges).
FAILURE: ``Refuse`` is a gate saying no, with the reason as its message; the orchestrator prints it and stops the stage.
A tool that is missing is exit status 127 with a message, never an exception, so "twine is not installed" is a line of
output the owner can act on.
CREDENTIALS: this module never reads, stores or prints a credential value.  Tools find their own (gh auth, ~/.pypirc,
TWINE_* variables, cargo login).  Everything a subprocess prints passes through ``redact`` before it reaches the terminal.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import time
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

import rel_secrets

STAGES = ("preflight", "export", "build", "publish")
GATES = ("local-checks", "pytest", "manual")  # names accepted by --skip-gate
STAGE_GATES = {"preflight": GATES, "export": ("local-checks", "pytest"), "build": ()}
# WHY a default: the owner decided on 2026-11-21 that the public repository lives under this organisation; --github-owner still overrides it.
DEFAULT_OWNER = "saltwater-research"
# WHY the LLC and not the dev identity: owner, 2026-11-21, "This project specifically will be tagged with dev@saltyresearch.com";
# the public squash names the copyright holder. The private dev repository keeps its own commit identity.
AUTHOR_NAME = "Saltwater Research LLC"
AUTHOR_EMAIL = rel_secrets.PUBLIC_CONTACT
SENSITIVE_ENV = re.compile(r"(?i)(TOKEN|PASSWORD|PASSWD|SECRET|API_?KEY|CREDENTIAL)")


class Refuse(Exception):
    """A gate said no.  The message is the whole explanation shown to the owner."""


def redact(text: str, environ: dict | None = None) -> str:
    """Remove token shapes and the VALUES of credential-looking environment variables from text about to be printed."""
    for key, value in (environ if environ is not None else os.environ).items():
        if SENSITIVE_ENV.search(key) and len(value) >= 6:
            text = text.replace(value, "[REDACTED]")
    for _, rx in rel_secrets.SHAPES:
        text = rx.sub("[REDACTED]", text)
    return text


class Report:
    """One line per check: ``[PASS  ] stage: message``.  Counts feed the final verdict and the exit status."""

    def __init__(self, out: TextIO | None = None):
        self.out = out or sys.stdout
        self.counts: dict[str, int] = {}

    def add(self, stage: str, status: str, message: str, detail: str = "") -> None:
        self.counts[status] = self.counts.get(status, 0) + 1
        print(f"[{status:<6}] {stage}: {redact(message)}", file=self.out, flush=True)
        for line in redact(detail).splitlines()[:40]:
            print(f"           {line}", file=self.out)

    def failed(self) -> bool:
        return self.counts.get("FAIL", 0) > 0


@dataclass
class Step:
    """One outward step: ``cmds`` run as subprocesses, in order, after the owner types ``token``."""

    key: str
    token: str
    title: str
    effect: str
    cmds: list  # list of (argv, cwd)
    needs: tuple  # tool names


@dataclass
class Ctx:
    repo: Path
    work: Path
    name: str
    owner: str
    dev_branch: str = "main"
    publish: bool = False
    test_pypi: bool = False
    upload_sdist: bool = False
    skip_gates: set = field(default_factory=set)
    skip_steps: set = field(default_factory=set)
    # WHY no bytecode: a tool run inside the export (a checker, pytest) must not leave __pycache__ in the tree that is committed.
    env: dict = field(default_factory=lambda: {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    rep: Report = field(default_factory=Report)
    # WHY these two live here: `publish.py all` and `register` add guards without a second Ctx type.  all_mode makes the typed
    # token "<target> <version>"; registry (rel_registry.Registry) lets the publish stage skip what is already on a registry.
    all_mode: bool = False
    registry: object | None = None
    verified_file: Path | None = None
    stdin: TextIO = field(default_factory=lambda: sys.stdin)
    state: dict = field(default_factory=dict)
    crates_io_proof: Path | None = None

    @property
    def export_dir(self) -> Path:
        return self.work / "export"

    @property
    def dist_dir(self) -> Path:
        return self.work / "dist"

    def tool(self, name: str) -> str | None:
        """Full path of an executable on the run's PATH.  WHY resolved here: on Windows a bare name misses .cmd shims."""
        return shutil.which(name, path=self.env.get("PATH"))

    def run(self, cmd: list[str], cwd: Path | str, extra_env: dict | None = None) -> tuple[int, str]:
        env = {**self.env, **(extra_env or {})}
        return run(cmd, cwd, {key: val for key, val in env.items() if val is not None})

    def cargo_env(self, offline: bool = True) -> dict:
        # WHY an external target dir: the export must stay byte-identical to its commit.  WHY offline by default: the crate
        # has no dependencies, so a cargo that wants the network during a dry run is a fault to see; only the real publish
        # step passes offline=False.
        env = {"CARGO_TARGET_DIR": str(self.work / "cargo-target")}
        return {**env, "CARGO_NET_OFFLINE": "true"} if offline else env

    def twine_command(self) -> list[str] | None:
        """The twine invocation, or None.  WHY the executable first: it is what a stub or a pipx install provides."""
        exe = self.tool("twine")
        if exe:
            return [exe]
        rc, _ = self.run([sys.executable, "-c", "import twine"], self.repo)
        return [sys.executable, "-m", "twine"] if rc == 0 else None


def run(cmd: list[str], cwd: Path | str, env: dict | None = None) -> tuple[int, str]:
    """(exit status, redacted merged output).  A missing tool is 127 with a message, never an exception."""
    resolved = shutil.which(cmd[0], path=(env or os.environ).get("PATH")) or cmd[0]
    try:
        done = subprocess.run([resolved, *cmd[1:]], cwd=cwd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    except OSError as exc:
        return 127, f"{cmd[0]}: cannot run ({exc.strerror})"
    return done.returncode, redact(done.stdout + done.stderr, env)


def _artifact_group(names: object, key: str, suffix: str) -> list[str]:
    if not isinstance(names, list) or not names:
        raise Refuse(f"release artifacts are missing the {key} list; rerun the build stage")
    # Artifact names also become command arguments. Accept package basenames on every OS.
    if not all(isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.+-]*", name)
               and name.endswith(suffix) for name in names):
        raise Refuse(f"release artifacts have an invalid {key} filename; rerun the build stage")
    if len(set(names)) != len(names):
        raise Refuse(f"release artifacts contain duplicate {key} filenames; rerun the build stage")
    return names


def release_file_sha256(path: Path) -> str:
    try:
        info = path.lstat()
        if (not stat.S_ISREG(info.st_mode) or getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
                or path.resolve() != path.absolute()):
            raise Refuse(f"release artifacts require a plain file: {path.name}; rerun the build stage")
        with path.open("rb") as source:
            return hashlib.file_digest(source, "sha256").hexdigest()
    except OSError as exc:
        raise Refuse(f"release artifacts cannot be read: {path.name}; rerun the build stage") from exc


def source_archives(ctx: Ctx) -> list[str]:
    return [] if "source_archive" not in ctx.state else _artifact_group([ctx.state["source_archive"]], "source_archive", ".tar.gz")


def release_asset_names(ctx: Ctx) -> list[str]:
    wheels = _artifact_group(ctx.state.get("wheels"), "wheels", ".whl")
    sdists = _artifact_group(ctx.state.get("sdists"), "sdists", ".tar.gz")
    source = source_archives(ctx)
    if set(sdists) & set(source):
        raise Refuse("the full source archive must have its own filename")
    crate = _artifact_group([ctx.state.get("crate")], "crate", ".crate")
    if not source:
        raise Refuse("release artifacts require the full tagged source archive")
    return wheels + sdists + source + crate


def stage_check_record(ctx: Ctx, stage: str, skipped: set) -> dict:
    """Preflight checks the source revision. Export and build additionally identify their output."""
    state = ctx.state
    record = {"head": state["head"], "version": state["version"], "name": ctx.name, "owner": ctx.owner,
              "skipped": sorted(skipped.intersection(STAGE_GATES[stage]))}
    if stage in ("export", "build"):
        record.update(export_commit=state["export_commit"], tag=state["tag"])
    if stage == "build":
        record["tag_object"] = state["tag_object"]
    return record


def require_preparation_checks(ctx: Ctx) -> None:
    """A sealed legacy receipt lacks the history needed to authorize a first or resumed publication."""
    receipt = ctx.state.get("artifact_receipt")
    sealed = receipt.get("stage_checks") if isinstance(receipt, dict) else None
    completed = ctx.state.get("stage_checks")
    if not isinstance(sealed, dict) or not isinstance(completed, dict):
        raise Refuse("preflight checks lack a complete release record; fully prepare the release before publishing")
    for stage in STAGE_GATES:
        record = sealed.get(stage)
        if stage not in ctx.state.get("done", []) or record != stage_check_record(ctx, stage, set()):
            raise Refuse(f"{stage} checks are missing, skipped, or belong to another release; fully prepare the release before publishing")
        if stage != "build" and completed.get(stage) != record:
            raise Refuse(f"{stage} checks have changed since the build; fully prepare the release before publishing")


def artifact_receipt(ctx: Ctx) -> dict:
    """Snapshot build outputs beside the stage state so a resumed publish can detect drift."""
    names = release_asset_names(ctx)
    try:
        entries = {path.name for path in ctx.dist_dir.iterdir()}
    except OSError as exc:
        raise Refuse("release artifacts directory cannot be read; rerun the build stage") from exc
    if entries - {"SHA256SUMS"} != set(names):
        raise Refuse("release artifacts directory has unexpected or missing files; rerun the build stage")
    return {
        "head": ctx.state["head"],
        "export_commit": ctx.state["export_commit"],
        "tag": ctx.state["tag"],
        "tag_object": ctx.state.get("tag_object"),
        "version": ctx.state["version"],
        "wheels": ctx.state["wheels"],
        "sdists": ctx.state["sdists"],
        "source_archive": ctx.state["source_archive"],
        "crate": ctx.state["crate"],
        "owner": ctx.owner,
        "name": ctx.name,
        "upload_sdist": ctx.upload_sdist,
        "skip_gates": sorted(ctx.skip_gates),
        # Absence preserves read-only previews of old receipts. Real publication requires all three records.
        **({"stage_checks": {**ctx.state["stage_checks"], "build": stage_check_record(ctx, "build", ctx.skip_gates)}}
           if "stage_checks" in ctx.state else {}),
        "cargo_rehearsed": ctx.state.get("cargo_rehearsed") is True,
        "sha256": {name: release_file_sha256(ctx.dist_dir / name) for name in names},
        "checked_sha256": ctx.state.get("checked_sha256", {}).copy(),
        "notes_sha256": release_file_sha256(ctx.work / "RELEASE_NOTES.md"),
    }


def checksum_manifest(receipt: dict) -> bytes:
    """Use one canonical checksum list for the saved receipt and release attachment."""
    hashes = receipt["sha256"]
    return "".join(f"{hashes[name]}  {name}\n" for name in sorted(hashes)).encode("utf-8")


def prove_artifacts(ctx: Ctx) -> None:
    saved = ctx.state.get("artifact_receipt")
    if not isinstance(saved, dict):
        raise Refuse("release artifacts have no build receipt; rerun the build stage")
    current = artifact_receipt(ctx)
    if current["skip_gates"] != saved.get("skip_gates"):
        raise Refuse("release gate policy changed since the build; rerun all preparation gates")
    if current != saved:
        raise Refuse("release artifacts or release notes changed since the build; rerun the build stage")
    expected = hashlib.sha256(checksum_manifest(current)).hexdigest()
    if release_file_sha256(ctx.dist_dir / "SHA256SUMS") != expected:
        raise Refuse("release artifacts have a changed checksum list; rerun the build stage")


def git(ctx: Ctx, cwd: Path | str, *args: str, extra_env: dict | None = None) -> tuple[int, str]:
    return ctx.run(["git", *args], cwd, extra_env)


def author_env() -> dict:
    """Identity and a UTC timestamp for the one export commit.  WHY UTC: a local offset would tell where the owner lives."""
    stamp = f"{int(time.time())} +0000"
    return {
        "GIT_AUTHOR_NAME": AUTHOR_NAME, "GIT_AUTHOR_EMAIL": AUTHOR_EMAIL, "GIT_AUTHOR_DATE": stamp,
        "GIT_COMMITTER_NAME": AUTHOR_NAME, "GIT_COMMITTER_EMAIL": AUTHOR_EMAIL, "GIT_COMMITTER_DATE": stamp,
    }


def confirm(ctx: Ctx, token: str, what: str) -> bool:
    """Print exactly what is about to happen and require the owner to type ``token``.

    WHY typed and not y/N: a stray Enter or a pasted "y" must not publish anything; the token names the thing being
    created, so the owner reads it before typing it.  EOF (no terminal, a closed pipe) is a refusal.
    """
    print(redact(what), flush=True)
    print(f"Type exactly   {token}   to do this, anything else (or end of input) skips it and stops the run: ", end="", flush=True)
    answer = ctx.stdin.readline()
    print()
    return answer.strip() == token


def state_path(work: Path) -> Path:
    return work / "state.json"


def load_state(work: Path) -> dict:
    try:
        return json.loads(state_path(work).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        raise Refuse(f"{state_path(work)} is unreadable ({exc}); delete the work directory to start over") from exc


def save_state(ctx: Ctx) -> None:
    ctx.work.mkdir(parents=True, exist_ok=True)
    # An interruption must leave the previous complete command receipt readable.
    with tempfile.NamedTemporaryFile("w", dir=ctx.work, delete=False, encoding="utf-8") as tmp:
        json.dump(ctx.state, tmp, indent=2, sort_keys=True)
        tmp.write("\n")
        tmp.flush()
        os.fsync(tmp.fileno())
    os.replace(tmp.name, state_path(ctx.work))


PUBLIC_STEPS = ("repo", "release", "pypi", "crates")  # the steps that make something permanent; testpypi is a rehearsal


def went_public(state: dict) -> bool:
    return bool(state.get("repo_commands")) or any(key in PUBLIC_STEPS for key in state.get("published", []))


def next_run_state(prior: dict) -> dict:
    """The state a new preflight starts from.  WHY two things survive it: ``export`` is how rel_export recognises its own work
    directory (without it a dry run followed by --publish refuses "not made by this script"); and a run that already made
    something public keeps everything (export commit, tag, artifacts, ``published``, ``origin``), because the tag on GitHub is
    that export's commit and a rebuilt export would have a different one.  Anything else starts clean."""
    if went_public(prior):
        return dict(prior)
    return {"done": [], **({"export": prior["export"]} if "export" in prior else {})}


def read_version(repo: Path | str) -> str:
    return (Path(repo) / "VERSION").read_text(encoding="utf-8").strip()


def inherited_cargo_config(cwd: Path) -> bool:
    """A fresh CARGO_HOME alone cannot suppress Cargo's ancestor discovery."""
    return any((p / ".cargo" / file).exists() for p in (cwd, *cwd.parents) for file in ("config", "config.toml"))


def outside(path: Path, repo: Path) -> bool:
    return not path.resolve().is_relative_to(repo.resolve())
