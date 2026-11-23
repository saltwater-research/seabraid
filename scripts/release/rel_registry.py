# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""What is already published: registry questions, asked through the registries' own tools (pip, cargo, gh).

OWNER: this module owns "is version V of NAME already on registry R" and "is NAME taken"; rel_upload (skip what is done) and
rel_register (do not reserve a name that is taken) ask, they never query a registry themselves.
CONTRACT: ``Registry(ctx).published(key, name, version, owner, tag) -> True | False | None`` and ``name_taken(key, name,
owner)``; None means "could not tell" and the caller stops rather than guess.  ``reachable() -> [what did not answer]``.
WHY no HTTP client here: checks/check_no_egress.py keeps network code out of the tree by design, and the registries' own
tools already speak their protocols and carry the owner's login; so every question is one subprocess (``Ctx.run``, which a
test replaces with a recorder; no test touches a network).  pip asks PyPI, cargo asks crates.io, gh asks GitHub.
CREDENTIALS: none are read here; the tools use their own.
FAILURE: an answer this module cannot read is None, never a guess.
"""

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

import rel_core as core

# WHY the gh line: `gh repo view` on a repository that does not exist exits 1 and says "GraphQL: Could not resolve to a Repository with
# the name 'o/n'. (repository)" (probed on gh 2.103.3); without it the fresh public repository read as "cannot tell" and a first run refused.
NOT_FOUND = ("No matching distribution", "could not find", "not found", "no matching package", "Could not resolve to a Repository")


def pep440(version: str) -> str:
    """The PEP 440 spelling of a semver pre-release: 0.5.0-alpha.2 -> 0.5.0a2.  WHY here: PyPI lists versions this way."""
    return re.sub(r"-(alpha|beta|rc)\.?(\d+)$", lambda m: {"alpha": "a", "beta": "b", "rc": "rc"}[m.group(1)] + m.group(2), version)


def crate_answer(spec: str, rc: int, out: str) -> bool | None:
    """Only the registry's explicit missing-package diagnostic establishes absence."""
    if rc == 0:
        version = re.search(r"^version: ([0-9A-Za-z.+-]+)(?: \(latest [^)]*\))?$", out, re.M)
        expected = spec.partition("@")[2]
        if version and (not expected or version[1] == expected):
            return True
        return None
    missing = f"error: could not find `{spec}` in registry `"
    if any(line.startswith(missing) and line.endswith(("crates-io`", "crates.io-index`")) for line in out.splitlines()):
        return False
    return None


class Registry:
    def __init__(self, ctx: core.Ctx):
        self.ctx = ctx

    def _ask(self, cmd: list[str]) -> tuple[bool | None, str]:
        """(True: the tool found it, False: the tool says it is not there, None: no usable answer), and the tool's output."""
        rc, out = self.ctx.run(cmd, self.ctx.repo)
        if rc == 0:
            return True, out
        return (False if any(marker in out for marker in NOT_FOUND) else None), out

    def _crate(self, spec: str) -> bool | None:
        """Cargo prefers a workspace member and can reuse a stale index. Isolate both locations."""
        if self.ctx.env.get("CARGO_NET_OFFLINE", "").lower() == "true":
            return None
        self.ctx.work.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=self.ctx.work) as fresh:
            cwd = Path(fresh)
            if core.inherited_cargo_config(cwd) or any((p / "Cargo.toml").exists() for p in cwd.parents):
                return None
            env = {key: None for key in self.ctx.env if key.upper().startswith("CARGO_")}
            env["CARGO_HOME"] = str(cwd / "cargo-home")
            rc, out = self.ctx.run(["cargo", "info", spec, "--registry", "crates-io"], cwd, env)
        return crate_answer(spec, rc, out)

    def _pypi_versions(self, name: str) -> tuple[bool | None, list[str]]:
        found, out = self._ask([sys.executable, "-m", "pip", "index", "versions", name, "--pre"])
        listed = re.search(r"Available versions:\s*(.+)", out)
        if "WARNING: Retrying" in out or (found and not listed):
            return None, []
        return found, [v.strip() for v in listed.group(1).split(",")] if listed else []

    def published(self, key: str, name: str, version: str, owner: str = "", tag: str = "") -> bool | None:
        if key == "pypi":
            found, versions = self._pypi_versions(name)
            return (pep440(version) in versions) if found else found
        if key == "crates":
            return self._crate(f"{name}@{version}")
        if key == "repo":
            return self._ask(["gh", "repo", "view", f"{owner}/{name}"])[0]
        if key == "release":
            return self._ask(["gh", "release", "view", tag, "--repo", f"{owner}/{name}"])[0]
        return False  # nothing else has a registry to ask; the state file is its only memory

    def name_taken(self, key: str, name: str, owner: str = "") -> bool | None:
        if key == "pypi":
            return self._pypi_versions(name)[0]
        if key == "crates":
            return self._crate(name)
        if key == "repo":
            return self._ask(["gh", "repo", "view", f"{owner}/{name}"])[0]
        return False

    def reachable(self) -> list[str]:
        """Registries that gave no usable answer to a question about a package that certainly exists; empty = usable."""
        probes = (("PyPI", self._pypi_versions("pip")[0]), ("crates.io", self._crate("serde")), ("GitHub", self._ask(["gh", "auth", "status"])[0]))
        return [label for label, answer in probes if answer is not True]
