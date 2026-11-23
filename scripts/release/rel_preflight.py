# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Stage 1, preflight: everything that must be true of the DEV tree before an export is worth making.

OWNER: this module owns the list of preflight gates; the checks themselves are owned elsewhere (checks/check_version.py,
checks/run_all.py, scripts/rename.py's name rule) and are CALLED here, never re-implemented.
CONTRACT: ``run(ctx) -> bool``; records the facts later stages need in ``ctx.state`` (version, head, tree name).  Reads
the dev tree and runs tools in it; writes nothing but the state file.
FAILURE: every gate prints PASS/FAIL/WARN/SKIP/MANUAL and the stage continues past a FAIL so the owner sees every problem
in one run; the stage as a whole fails if any gate did.  MANUAL items cannot be proven by a script: they are listed with
the evidence the script CAN show, and --publish refuses to go on while they are unconfirmed.
"""

from __future__ import annotations

import datetime
import re
import sys
import tomllib
from pathlib import Path

import rel_core as core
import rel_secrets

STAGE = "preflight"
NAME_RULE = re.compile(r"^[a-z][a-z0-9]{2,20}$")  # the same rule scripts/rename.py applies to NEW_NAME
PLACEHOLDER = "http://localhost/@REPO@"
MANUAL_ITEMS = (
    "The release candidate was verified on the Linux test machine (that verification run is recorded in docs/RESEARCH-LOG.md).",
    "A GPU run of this version was done and recorded in docs/RESEARCH-LOG.md.",
    "The README's claims match what docs/RESEARCH-LOG.md records; nothing is claimed that no run backs.",
    "The final product name is the one you want forever: PyPI and crates.io names and versions are permanent.",
)


def tree_name(root: Path) -> str:
    with open(root / "pyproject.toml", "rb") as fh:
        return tomllib.load(fh)["project"]["name"]


def name_problems(final: str, root: Path) -> list[str]:
    """Why ``final`` cannot be published from ``root``, as a list; empty when it can (rename or already named so)."""
    out = []
    if not NAME_RULE.match(final):
        out.append(f"the name {final!r} must match {NAME_RULE.pattern} (lower-case letters and digits, valid for PyPI and crates.io)")
    with open(root / "Cargo.toml", "rb") as fh:
        cargo_name = tomllib.load(fh)["package"]["name"]
    if cargo_name != tree_name(root):
        out.append(f"pyproject.toml names the package {tree_name(root)!r} but Cargo.toml names it {cargo_name!r}")
    return out


def leftover_name_problems(final: str, old: str, root: Path) -> list[str]:
    """After a rename: the old name must be gone from every file.  WHY here and not only in check_names: a retired
    name must never ship, so this gate is stated by name and refuses by name."""
    if final == old:
        return []
    done = core.run([sys.executable, str(root / "checks" / "check_names.py"), "--root", str(root), "--forbid", old], root)
    return [] if done[0] == 0 else [f"the old name {old!r} is still in the tree; refusing to publish under {final!r}", done[1]]


def metadata_problems(root: Path) -> list[str]:
    out = []
    with open(root / "pyproject.toml", "rb") as fh:
        project = tomllib.load(fh)["project"]
    with open(root / "Cargo.toml", "rb") as fh:
        package = tomllib.load(fh)["package"]
    for key in ("description", "readme", "license", "urls", "keywords", "classifiers", "authors"):
        if not project.get(key):
            out.append(f"pyproject.toml [project] has no {key}")
    out += [f"pyproject.toml project.urls.{k} is {v!r}, not the placeholder {PLACEHOLDER}" for k, v in project.get("urls", {}).items() if not v.startswith(PLACEHOLDER)]
    for key in ("description", "license", "repository", "readme", "keywords", "categories", "include"):
        if not package.get(key):
            out.append(f"Cargo.toml [package] has no {key}")
    if package.get("publish") is False:
        out.append("Cargo.toml [package] still says publish = false")
    return out


def check_git(ctx: core.Ctx) -> None:
    rc, out = core.git(ctx, ctx.repo, "status", "--porcelain")
    clean = rc == 0 and not out.strip()
    ctx.rep.add(STAGE, "PASS" if clean else "FAIL", "git tree is clean" if clean else "git tree is not clean (commit or remove these first)", "" if clean else out)
    rc, branch = core.git(ctx, ctx.repo, "rev-parse", "--abbrev-ref", "HEAD")
    ok = rc == 0 and branch.strip() == ctx.dev_branch
    ctx.rep.add(STAGE, "PASS" if ok else "FAIL", f"on the dev branch {ctx.dev_branch!r}" if ok else f"on branch {branch.strip()!r}, expected {ctx.dev_branch!r} (pass --dev-branch if that is intended)")
    rc, head = core.git(ctx, ctx.repo, "rev-parse", "HEAD")
    ctx.state["head"] = head.strip() if rc == 0 else ""


def check_script(ctx: core.Ctx, label: str, cmd: list[str], gate: str | None = None) -> None:
    if gate in ctx.skip_gates:
        ctx.rep.add(STAGE, "SKIP", f"{label} (skipped by --skip-gate {gate})")
        return
    rc, out = ctx.run(cmd, ctx.repo)
    ctx.rep.add(STAGE, "PASS" if rc == 0 else "FAIL", label, "" if rc == 0 else "\n".join(out.splitlines()[-25:]))


def check_name(ctx: core.Ctx) -> None:
    problems = name_problems(ctx.name, ctx.repo)
    old = tree_name(ctx.repo)
    ctx.state["old_name"] = old
    if problems:
        ctx.rep.add(STAGE, "FAIL", "final name", "\n".join(problems))
    elif old == ctx.name:
        ctx.rep.add(STAGE, "PASS", f"final name {ctx.name!r} is the tree's name; no rename needed")
    else:
        ctx.rep.add(STAGE, "PASS", f"final name {ctx.name!r}; the export will be renamed from {old!r} by scripts/rename.py and re-checked with the old name forbidden")


def changelog_problems(text: str, version: str) -> list[str]:
    """Why the changelog cannot be released as ``version``.  WHY a FAIL: release notes are cut from the version's own
    section only (rel_build.release_notes), so an entry left under ``## Unreleased`` above it would silently never ship."""
    heads = [(i, ln[3:].strip().lower()) for i, ln in enumerate(text.splitlines()) if ln.startswith("## ")]
    release = next((i for i, h in heads if h.startswith(version.lower())), None)
    stray = next((i for i, h in heads if h == "unreleased" and i < (release or 0)), None)
    if stray is None:
        return []
    return [f"CHANGELOG.md has a '## Unreleased' section (line {stray + 1}) above '## {version}': merge it into the release section, whose notes are the only ones published"]


def citation_problems(root: Path, today: datetime.date) -> list[str]:
    """CITATION.cff ``date-released`` must be ``today``.  WHY today and not the tag date: the tag does not exist yet when
    preflight runs, and the release is cut and tagged the day preflight passes (CHANGELOG: 'the date is set when the tag is
    made'), so today is the one date the tag can have; a stale date would put the previous release's date on the Zenodo record."""
    m = re.search(r"^date-released:\s*['\"]?(\d{4}-\d{2}-\d{2})", (root / "CITATION.cff").read_text(encoding="utf-8"), re.M)
    if m is None:
        return ["CITATION.cff has no date-released: YYYY-MM-DD"]
    if m.group(1) != today.isoformat():
        return [f"CITATION.cff date-released is {m.group(1)}, not today ({today.isoformat()}); set it to the tag date before releasing"]
    return []


def check_changelog(ctx: core.Ctx, version: str) -> None:
    text = (ctx.repo / "CHANGELOG.md").read_text(encoding="utf-8")
    problems = changelog_problems(text, version)
    heading = next((ln for ln in text.splitlines() if ln.startswith("## ") and ln[3:].startswith(version)), None)
    if problems:
        ctx.rep.add(STAGE, "FAIL", "CHANGELOG.md has entries outside the release section", "\n".join(problems))
    elif heading is None:
        ctx.rep.add(STAGE, "WARN", f"CHANGELOG.md has no '## {version}' section; the release notes would be 'Unreleased'")
    elif "unreleased" in heading.lower():
        ctx.rep.add(STAGE, "WARN", f"CHANGELOG.md heading still reads {heading!r}; set its date before tagging")
    else:
        ctx.rep.add(STAGE, "PASS", f"CHANGELOG.md has {heading!r}")


def check_citation(ctx: core.Ctx) -> None:
    problems = citation_problems(ctx.repo, datetime.date.today())
    ctx.rep.add(STAGE, "FAIL" if problems else "PASS", "CITATION.cff date-released is today" if not problems else "CITATION.cff date-released is stale", "\n".join(problems))


def check_tools(ctx: core.Ctx) -> None:
    soft = "FAIL" if ctx.publish else "WARN"  # RULE (offline vs real): a dry run WARNs about a missing upload tool, --publish refuses
    for tool, need in (("git", "FAIL"), ("cargo", "FAIL"), ("twine", soft), ("gh", soft)):
        found = ctx.tool(tool)
        hint = {"twine": "install: python -m pip install twine", "gh": "install the GitHub CLI, then gh auth login", "cargo": "install Rust from the rustup page"}.get(tool, "")
        ctx.rep.add(STAGE, "PASS" if found else need, f"{tool} is available" if found else f"{tool} not found ({hint}); needed for {'every stage' if tool in ('git', 'cargo') else 'twine check / --publish'}")
    rc, _ = ctx.run([sys.executable, "-c", "import setuptools, sys; sys.exit(0 if int(setuptools.__version__.split('.')[0]) >= 77 else 1)"], ctx.repo)
    ctx.rep.add(STAGE, "PASS" if rc == 0 else "FAIL", "setuptools >= 77 is importable (sdist and wheel build)" if rc == 0 else "setuptools >= 77 is missing: python -m pip install -U setuptools")


def check_manual(ctx: core.Ctx) -> None:
    rc, out = core.git(ctx, ctx.repo, "log", "-1", "--format=%cs", "--", "docs/RESEARCH-LOG.md")
    ctx.rep.add(STAGE, "NOTE", f"docs/RESEARCH-LOG.md last changed {out.strip() or 'never (untracked)'}; its newest headings:")
    heads = [ln for ln in (ctx.repo / "docs" / "RESEARCH-LOG.md").read_text(encoding="utf-8").splitlines() if ln.startswith("## ")][:4]
    ctx.rep.add(STAGE, "NOTE", "", "\n".join(heads))
    if "manual" in ctx.skip_gates:
        ctx.rep.add(STAGE, "SKIP", "manual confirmations (skipped by --skip-gate manual; refused with --publish)")
        return
    for item in MANUAL_ITEMS:
        ctx.rep.add(STAGE, "MANUAL", item)
    ctx.rep.add(STAGE, "NOTE", "the real publish command records the human sign-off against the verified revision and sealed artifacts")


def run(ctx: core.Ctx) -> bool:
    before = ctx.rep.counts.get("FAIL", 0)
    py = sys.executable
    check_git(ctx)
    version = core.read_version(ctx.repo)
    ctx.state.update(version=version, name=ctx.name, owner=ctx.owner)
    check_script(ctx, f"VERSION {version}, Cargo.toml, Cargo.lock, CMake and pyproject agree", [py, "checks/check_version.py"])
    check_name(ctx)
    problems = metadata_problems(ctx.repo)
    ctx.rep.add(STAGE, "FAIL" if problems else "PASS", "package metadata (pyproject.toml, Cargo.toml) is complete" if not problems else "package metadata is incomplete", "\n".join(problems))
    check_changelog(ctx, version)
    check_citation(ctx)
    check_tools(ctx)
    check_script(ctx, "local checkers (python checks/run_all.py)", [py, "checks/run_all.py"], "local-checks")
    check_script(ctx, "tests (pytest tests bindings/python/tests)", [py, "-m", "pytest", "-q", "tests", "bindings/python/tests"], "pytest")
    rc, listing = core.git(ctx, ctx.repo, "ls-files", "-z", "--cached", "--others", "--exclude-standard")
    hits = rel_secrets.scan_tree(str(ctx.repo), [p for p in listing.split("\0") if p]) if rc == 0 else ["git ls-files failed; cannot scan"]
    ctx.rep.add(STAGE, "FAIL" if hits else "PASS", "no token, key or personal address in the tree" if not hits else f"{len(hits)} secret/identifier hit(s) in the tree", "\n".join(hits[:30]))
    check_manual(ctx)
    return ctx.rep.counts.get("FAIL", 0) == before
