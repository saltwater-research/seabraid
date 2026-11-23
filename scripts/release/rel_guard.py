# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""The refusals that stand in front of `publish.py all` and `publish.py register`, one pure function per rule.

OWNER: this module owns WHEN a one-command release or a name reservation is refused; it owns the list of retired
temporary names.  What a stage checks about the tree stays in rel_preflight (called, never copied).  Each function returns a list of
problems (empty = fine) and has no side effects, so a test pins every refusal without running a release.
CONTRACT: ``gate(ctx, label, problems)`` is the single place a problem list becomes an outcome: a real run (--publish)
REFUSES, a dry run prints WARN (the owner sees what a real run would say, and the dry run still prints its plan).
VERIFIED FILE: the coordinator writes the commit it verified (full sha first on a line; ``#`` lines and text after the sha
are ignored) to ``--verified-file``, default ``~/.release/VERIFIED``.  WHY a file and not a flag: the owner must not be able
to type past the gate, and the file says who vouched for which commit.
FAILURE: refuse loudly with the fix in the message; never guess (an unreadable file or an unknown HEAD is a problem).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import rel_core as core
import rel_preflight

# WHY a constant here: Seabraid is the FINAL product name (owner decision, 2026-11-21); the names that must never ship are the
# retired ones (the temporary project name and the old add-on name), which a tree cannot say about itself.  Add a name to this tuple and every guard refuses it.  WHY spelled
# in two halves: scripts/rename.py rewrites a name wherever it occurs, which would turn this constant into the NEW name and make
# every guard refuse the final one; the halves keep it the retired name for good (check_names looks for the whole word, which
# is not here).
PROVISIONAL = ("lo" + "om", "blam" + "mer")  # the temporary project name, then the add-on name retired 2026-11-21 (see scripts/retire_name.py)
SHA = re.compile(r"^[0-9a-f]{40}$")
BUILD_REFUSAL = ("the build and publish stages need Linux (a native wheel and crate): on Windows there is no MSVC and Git's link.exe "
                 "shadows the linker. Dry runs still work here and print the plan; run the real thing on the Linux build host")


def default_verified_file() -> Path:
    return Path(os.environ.get("RELEASE_VERIFIED_FILE") or Path.home() / ".release" / "VERIFIED")


def name_problems(name: str, root: Path) -> list[str]:
    out = [f"{name!r} is a provisional name ({', '.join(PROVISIONAL)}, retired; the final name is Seabraid, decided 2026-11-21); nothing is published under it"] if name in PROVISIONAL else []
    return out + rel_preflight.name_problems(name, root)


def renamed_tree_problems(name: str, root: Path) -> list[str]:
    """For register: the tree itself must already carry ``name`` and no provisional name may remain in it."""
    tree = rel_preflight.tree_name(root)
    if tree != name:
        return [f"the tree is named {tree!r}, not {name!r}: run scripts/rename.py --from {tree} --to {name}, commit, then register"]
    out = []
    for old in PROVISIONAL:
        out += rel_preflight.leftover_name_problems(name, old, root)
    return out


def verified_problems(repo: Path, verified_file: Path, head: str) -> list[str]:
    try:
        text = verified_file.read_text(encoding="utf-8")
    except OSError:
        return [f"no verified-commit file at {verified_file}: the coordinator writes the sha it verified there (see docs/RELEASING.md)"]
    first = next((ln.split()[0].lower() for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")), "")
    if not SHA.match(first):
        return [f"{verified_file} holds no commit sha (expected the sha first on a line)"]
    if head != first:
        return [f"HEAD {head[:12] or '(unknown)'} is not the verified commit {first[:12]} recorded in {verified_file}; verify this commit or check it out"]
    return []


def platform_problems(system: str) -> list[str]:
    return [] if system == "Linux" else [f"this is {system}: {BUILD_REFUSAL}"]


def target_problems(ctx: core.Ctx, dev_remotes: list[str], export_remotes: list[str], own_origin: str = "") -> list[str]:
    """The public target must never be the private dev remote, and the export must start with no remote at all (fresh).

    ``own_origin`` is the address the repo step recorded in the state file after ``gh repo create`` added ``origin``.  WHY an
    exception and not a relaxed rule: once that step has run, the export legitimately holds exactly that remote, and a resume
    must not refuse its own work; any other remote, or this one when nothing recorded it, is still a pre-existing remote."""
    slug = f"{ctx.owner}/{ctx.name}".lower()

    def ours(url: str) -> bool:
        return bool(own_origin) and url == own_origin and slug in url.lower().replace(".git", "")

    out = [f"the dev repository's remote {url!r} is the public target {slug}; the public repository must be a fresh one" for url in dev_remotes if slug in url.lower().replace(".git", "")]
    return out + [f"the export already has a remote ({url!r}); a fresh public repository starts with none" for url in export_remotes if not ours(url)]


def package_name_problems(taken: dict, ours: list[str], name: str) -> list[str]:
    """Why the package ``name`` may not be used.  ``taken`` maps a registry key ("pypi", "crates") to ``name_taken``'s answer
    (True, False, None); ``ours`` is the steps this tool already published (the state file).

    WHY here and before the repo step: rel_upload asks whether THIS VERSION is published, which a name held by someone else
    at another version answers "no"; the owner would learn it only after the GitHub repository was public.  A name that exists
    and is not recorded as this tool's own is somebody else's; one the registry cannot answer about is refused, never guessed."""
    out = []
    for key, label in (("pypi", "PyPI"), ("crates", "crates.io")):
        if key not in taken or key in ours:
            continue
        if taken[key] is None:
            out.append(f"cannot tell whether the name {name!r} is free on {label}; check the network and rerun")
        elif taken[key]:
            out.append(f"the name {name!r} already exists on {label} and this tool did not publish it; choose another name (or, if an earlier run of this tool made it, rerun in that run's --work directory)")
    return out


def fresh_repo_problems(exists: bool | None, recorded: bool, slug: str) -> list[str]:
    if exists is None:
        return [f"cannot tell whether {slug} exists (gh gave no answer); log in with gh auth login and check the network"]
    if exists and not recorded:
        return [f"the GitHub repository {slug} already exists and this tool did not create it; the public repo must be fresh (no history). Delete it or choose another owner/name"]
    return []


def gate(ctx: core.Ctx, label: str, problems: list[str], hard: bool = False) -> bool:
    """Report one rule.  A real run refuses on any problem; a dry run warns and goes on, unless ``hard`` (register: a dry run
    of a reservation under a name it may not reserve is meaningless, so it refuses too).  Returns True when clear."""
    if not problems:
        ctx.rep.add("guard", "PASS", label)
        return True
    if ctx.publish or hard:
        raise core.Refuse(f"{label}: " + "; ".join(problems))
    ctx.rep.add("guard", "WARN", f"{label} (a real run would refuse)", "\n".join(problems))
    return False
