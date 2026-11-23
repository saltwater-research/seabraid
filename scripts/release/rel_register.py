# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""`publish.py register --name FINAL`: reserve the final name on every registry before the first real release.

OWNER: this module owns the placeholder packages and the order of reservations; the refusals are rel_guard's, the typed
confirmation and the steps' execution are rel_upload's, "is the name taken" is rel_registry's.
CONTRACT: ``main(argv)``.  REFUSES unless ``--name`` is given, is not provisional, equals the tree's own name (the tree has
been renamed: scripts/rename.py), and the provisional name is gone from every file (check_names --forbid).  Targets:
  crates.io  a minimal 0.0.0 crate, description "name reserved; release follows"; skipped when the name is taken.
  PyPI       a minimal 0.0.0 sdist (pure Python, no code); skipped when the name is taken.
  GitHub     availability of OWNER/NAME only.  Nothing is created: the repository comes with `all`, and it must be fresh.
A reservation is PUBLIC and permanent (a 0.0.0 version cannot be reused), so a real run needs ``--publish`` and the typed
text "<target> 0.0.0" per target.  DRY RUN prints the files it would write and the exact commands; it writes into the work
directory only.  CREDENTIALS: the tools' own logins (cargo login, TWINE_*); never in argv, never printed.
FAILURE: the first failing target stops the run; a rerun skips targets the registry now shows as taken.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import rel_all
import rel_core as core
import rel_guard as guard
import rel_registry
import rel_upload

VERSION = "0.0.0"
DESCRIPTION = "name reserved; release follows"
SDIST_SNIPPET = "import sys, setuptools.build_meta as b; print(b.build_sdist(sys.argv[1]))"


def write_crate(root: Path, name: str) -> Path:
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / "Cargo.toml").write_text(f'[package]\nname = "{name}"\nversion = "{VERSION}"\nedition = "2021"\ndescription = "{DESCRIPTION}"\nlicense = "Apache-2.0"\n', encoding="utf-8")
    (root / "src" / "lib.rs").write_text(f"//! {DESCRIPTION}\n", encoding="utf-8")
    return root


def write_python(root: Path, name: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text(f'[build-system]\nrequires = ["setuptools>=77"]\nbuild-backend = "setuptools.build_meta"\n\n[project]\nname = "{name}"\nversion = "{VERSION}"\n'
                                         f'description = "{DESCRIPTION}"\nlicense = "Apache-2.0"\n', encoding="utf-8")
    return root


def steps(ctx: core.Ctx) -> list[core.Step]:
    crate, py, dist = ctx.work / "crate", ctx.work / "python", ctx.work / "dist"
    twine = ctx.twine_command() or ["twine"]
    sdist = str(dist / f"{ctx.name}-{VERSION}.tar.gz")
    return [
        core.Step("crates", f"crates {VERSION}", f"reserve the crate name {ctx.name} on crates.io with an empty {VERSION} crate",
                  "IRREVERSIBLE: the name is yours from now on; 0.0.0 can be yanked, never reused.", [(["cargo", "publish", "--allow-dirty"], crate)], ("cargo",)),
        core.Step("pypi", f"pypi {VERSION}", f"reserve the PyPI name {ctx.name} with an empty {VERSION} sdist",
                  "IRREVERSIBLE: the name is yours from now on; 0.0.0 can be yanked, never reused.",
                  [([sys.executable, "-c", SDIST_SNIPPET, str(dist)], py), ([*twine, "upload", "--non-interactive", sdist], dist)], ("twine",)),
    ]


def prepare(ctx: core.Ctx) -> None:
    write_crate(ctx.work / "crate", ctx.name)
    write_python(ctx.work / "python", ctx.name)
    (ctx.work / "dist").mkdir(parents=True, exist_ok=True)


def parse(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="publish.py register", description=__doc__.splitlines()[0])
    rel_all.add_common(ap, Path(__file__).resolve().parents[2])
    return ap.parse_args(argv)


def asked(ctx: core.Ctx, key: str, label: str) -> bool | None:
    """Is the name taken on ``key``?  None = not asked (dry run: it never contacts a registry).  A real run that gets no
    answer refuses: guessing 'free' would try to upload over a stranger's name."""
    if not ctx.publish:
        ctx.rep.add("register", "NOTE", f"{label}: not checked (a dry run never contacts a registry); a real run asks first")
        return None
    taken = ctx.registry.name_taken(key, ctx.name, ctx.owner)
    if taken is None:
        raise core.Refuse(f"cannot tell whether the name is taken on {label}; check the network and rerun")
    return taken


def availability(ctx: core.Ctx) -> list[str]:
    """Registry targets still free to reserve, after reporting each one (GitHub is reported, never reserved)."""
    repo_taken = asked(ctx, "repo", f"GitHub {ctx.owner}/{ctx.name}")
    if repo_taken is not None:
        ctx.rep.add("register", "WARN" if repo_taken else "PASS", "GitHub repository " + ("EXISTS: the release needs a fresh one, look at it" if repo_taken else "name is free; `all` creates it"))
    free = []
    for key, label in (("crates", "crates.io"), ("pypi", "PyPI")):
        taken = asked(ctx, key, label)
        if taken:
            ctx.rep.add("register", "SKIP", f"{label}: the name is already taken; nothing to reserve there")
        else:
            ctx.rep.add("register", "PASS", f"{label}: the name is " + ("available" if taken is False else "to be checked at publish time"))
            free.append(key)
    return free


def guards(ctx: core.Ctx, a: argparse.Namespace) -> None:
    guard.gate(ctx, f"name {ctx.name!r} is final", guard.name_problems(ctx.name, ctx.repo), hard=True)
    guard.gate(ctx, "the tree has been renamed to it", guard.renamed_tree_problems(ctx.name, ctx.repo), hard=True)
    guard.gate(ctx, "HEAD is the verified commit", guard.verified_problems(ctx.repo, a.verified_file or guard.default_verified_file(), rel_all.head_of(ctx)))
    if ctx.publish:
        raise core.Refuse("name reservations have no sealed artifact receipt; live registration is held")


def reserve(ctx: core.Ctx, todo: list[core.Step]) -> bool:
    if not ctx.publish:
        return rel_upload.dry_run(ctx, todo, checklist=False)
    return all(rel_upload.run_step(ctx, s, i, len(todo)) for i, s in enumerate(todo, 1))  # all() stops at the first False


def main(argv: list[str]) -> int:
    try:
        a = parse(argv)
        if not a.name:
            raise core.Refuse("--name is required: register reserves the FINAL name, and only a name you chose")
        ctx = rel_all.make_ctx(a, "register")
        ctx.all_mode = False
        print(("PUBLISH MODE: each reservation asks for a typed confirmation" if ctx.publish else "DRY RUN: nothing is uploaded or created") + f"\nreserving {ctx.name!r}, work dir {ctx.work}", flush=True)
        guards(ctx, a)
        ctx.state = core.load_state(ctx.work)
        free = availability(ctx)
        prepare(ctx)
        ok = reserve(ctx, [s for s in steps(ctx) if s.key in free and s.key not in ctx.state.get("published", [])])
    except core.Refuse as exc:
        print(f"[FAIL  ] refused: {core.redact(str(exc))}")
        return 1
    print("\nsummary: " + ", ".join(f"{k} {v}" for k, v in sorted(ctx.rep.counts.items())))
    return 0 if ok and not ctx.rep.failed() else 1


if __name__ == "__main__":
    sys.exit("run it as: python scripts/release/publish.py register --name NAME")
