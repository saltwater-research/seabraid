# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Stage 2, export: turn HEAD of the dev repo into a FRESH repository holding one commit of the scrubbed tree.

OWNER: this module owns how the public tree is derived from the dev tree.  The dev repo keeps its full history; nothing
here touches it (it is read through ``git archive`` only).
CONTRACT: ``run(ctx) -> bool``.  Output is ``<work>/export``, a git repository with ONE commit by the noreply author
(no trailers, UTC timestamps) and the annotated tag ``v<VERSION>``.  Steps, in this order and each gating the next:
archive HEAD -> rename old name to final name (scripts/rename.py) -> re-run the gates on the archive -> fill the repository
placeholder -> prove only that placeholder changed -> run every checker (checks/run_all.py) and pytest on the FILLED tree ->
git init, commit, tag -> prove the commit holds every file.
WHY the checkers run before and after the fill: the gated tree carries the loopback placeholder and differs from the filled
one only in those lines. The second run proves the egress allowance in checks/_egress_meta.py. Pytest runs on the filled tree
because that is the metadata that readers and public CI get.
FAILURE: fail-fast; the first failing step stops the stage and the export directory is left in place for inspection.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import tarfile
import tempfile
import urllib.parse
from pathlib import Path

import rel_core as core
import rel_guard
import rel_preflight
import rel_secrets

STAGE = "export"
FILL_FILES = ("pyproject.toml", "Cargo.toml")


def repo_url(owner: str, name: str, host: str = "github.com") -> str:
    return urllib.parse.urlunsplit(("https", host, f"/{owner}/{name}", "", ""))


def hash_tree(root: Path) -> dict[str, str]:
    out = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file() and ".git" not in p.relative_to(root).parts):
        out[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def extract_head(ctx: core.Ctx, dest: Path) -> None:
    tar_path = ctx.work / "head.tar"
    rc, out = core.git(ctx, ctx.repo, "archive", "--format=tar", "-o", str(tar_path), "HEAD")
    if rc != 0:
        raise core.Refuse(f"git archive HEAD failed: {out}")
    dest.mkdir(parents=True)
    with tarfile.open(tar_path) as tar:
        tar.extractall(dest, filter="data")  # "data": refuse absolute paths, links out of the tree and device files
    tar_path.unlink()


def gate(ctx: core.Ctx, label: str, cmd: list[str], root: Path, skip: str | None = None) -> bool:
    if skip in ctx.skip_gates:
        ctx.rep.add(STAGE, "SKIP", f"{label} (skipped by --skip-gate {skip})")
        return True
    rc, out = ctx.run(cmd, root)
    ctx.rep.add(STAGE, "PASS" if rc == 0 else "FAIL", label, "" if rc == 0 else "\n".join(out.splitlines()[-25:]))
    return rc == 0


def run_gates(ctx: core.Ctx, root: Path, old: str, label: str) -> bool:
    py = sys.executable
    ok = gate(ctx, f"{label}: python checks/run_all.py", [py, "checks/run_all.py"], root, "local-checks")
    if old != ctx.name:
        ok &= gate(ctx, f"{label}: checks/check_names.py --forbid {old}", [py, "checks/check_names.py", "--forbid", old], root)
        problems = rel_preflight.leftover_name_problems(ctx.name, old, root)
        ctx.rep.add(STAGE, "FAIL" if problems else "PASS", f"{label}: the old name {old!r} is gone" if not problems else f"{label}: old name still present", "\n".join(problems))
        ok &= not problems
    for retired in rel_guard.PROVISIONAL:  # WHY always, not only after a rename: the retired temporary name must never ship, whatever the final name
        problems = rel_preflight.leftover_name_problems(ctx.name, retired, root)
        ctx.rep.add(STAGE, "FAIL" if problems else "PASS", f"{label}: the retired temporary name {retired!r} is gone" if not problems else f"{label}: retired temporary name still present", "\n".join(problems))
        ok &= not problems
    hits = rel_secrets.scan_tree(str(root))
    ctx.rep.add(STAGE, "FAIL" if hits else "PASS", f"{label}: no token, key, credential file or personal address" if not hits else f"{label}: {len(hits)} secret/identifier hit(s)", "\n".join(hits[:30]))
    return ok and not hits


# WHY exempt: these files NAME the placeholder (they are the tool that fills it), so they cannot be required to lack it.
PLACEHOLDER_EXEMPT = ("scripts/release/", "tests/test_release_publish.py", "docs/RELEASING.md")


def fill_file(path: Path, token: str, url: str) -> bool:
    """Rewrite one file's non-comment placeholder lines; True when anything changed."""
    lines = path.read_bytes().decode("utf-8").splitlines(keepends=True)
    out = [ln if ln.lstrip().startswith("#") else ln.replace(token, url) for ln in lines]
    if out == lines:
        return False
    path.write_bytes("".join(out).encode("utf-8"))
    return True


def leftover_placeholders(root: Path) -> list[str]:
    found = [p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and ".git" not in p.parts and b"@REPO@" in p.read_bytes()]
    return [p for p in found if not p.startswith(PLACEHOLDER_EXEMPT) and not (p in FILL_FILES and _only_comments(root / p))]


def fill_placeholder(ctx: core.Ctx, root: Path) -> list[str]:
    """Replace the placeholder on every non-comment line of FILL_FILES; a comment that explains it stays as written."""
    url = repo_url(ctx.owner, ctx.name)
    changed = [rel for rel in FILL_FILES if fill_file(root / rel, rel_preflight.PLACEHOLDER, url)]
    left = leftover_placeholders(root)
    if left:
        raise core.Refuse(f"the repository placeholder is still in: {left}")
    return changed


def _only_comments(path: Path) -> bool:
    return all(ln.lstrip().startswith("#") for ln in path.read_text(encoding="utf-8").splitlines() if "@REPO@" in ln)


def prove_only_placeholder(before: dict[str, str], after: dict[str, str]) -> list[str]:
    changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    return [f"{k} changed by the placeholder fill but is not one of {FILL_FILES}" for k in changed if k not in FILL_FILES]


def commit_tree(ctx: core.Ctx, root: Path, version: str) -> str:
    tag, steps = f"v{version}", [["init", "-q", "-b", "main"], ["add", "-A", "-f"]]
    for args in steps:
        rc, out = core.git(ctx, root, *args)
        if rc != 0:
            raise core.Refuse(f"git {args[0]} failed: {out}")
    env = core.author_env(ctx.repo)
    for args in (["commit", "-q", "-m", f"{ctx.name} {version}"], ["tag", "-a", tag, "-m", f"{ctx.name} {version}"]):
        rc, out = core.git(ctx, root, *args, extra_env=env)
        if rc != 0:
            raise core.Refuse(f"git {args[0]} failed: {out}")
    rc, sha = core.git(ctx, root, "rev-parse", "HEAD")
    return sha.strip()


def commit_identity(ctx: core.Ctx, root: Path) -> str:
    """The author and committer actually recorded in the export commit, read back from git, never assumed."""
    return core.git(ctx, root, "log", "-1", "--format=%an <%ae>")[1].strip()


def junk_problems(root: Path) -> list[str]:
    """Generated files that must never be committed.  WHY a name list: the cheap, certain cases; the allow-list in
    ``prove_commit_complete`` (disk equals commit) and the secret scan cover the rest."""
    junk = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "target", "build", "dist", ".venv", "venv"}
    return [f"generated path in the export: {p.relative_to(root).as_posix()}" for p in sorted(root.rglob("*")) if (junk & set(p.relative_to(root).parts)) or p.suffix == ".pyc"][:20]


def prove_commit_complete(ctx: core.Ctx, root: Path) -> list[str]:
    rc, listing = core.git(ctx, root, "ls-files", "-z")
    tracked = {p for p in listing.split("\0") if p}
    on_disk = set(hash_tree(root))
    rc2, count = core.git(ctx, root, "rev-list", "--count", "HEAD")
    problems = [f"on disk but not in the commit: {p}" for p in sorted(on_disk - tracked)] + [f"in the commit but not on disk: {p}" for p in sorted(tracked - on_disk)]
    if count.strip() != "1":
        problems.append(f"the export has {count.strip()} commits, expected exactly 1")
    return problems


def prepare(ctx: core.Ctx, dest: Path, old: str) -> bool:
    """Guard the work directory, archive HEAD into ``dest`` and rename it; False when the rename failed."""
    if not core.outside(dest, ctx.repo):
        raise core.Refuse(f"the work directory {ctx.work} is inside the repository; choose one outside it")
    if dest.exists():
        if ctx.state.get("export") != str(dest):
            raise core.Refuse(f"{dest} exists and was not made by this script; remove it or choose another --work")
        shutil.rmtree(dest)
    ctx.work.mkdir(parents=True, exist_ok=True)
    extract_head(ctx, dest)
    ctx.rep.add(STAGE, "PASS", f"git archive HEAD ({ctx.state.get('head', '')[:12]}) extracted to {dest}")
    ctx.state["export"] = str(dest)
    if old == ctx.name:
        return True
    rc, out = ctx.run([sys.executable, str(ctx.repo / "scripts" / "rename.py"), "--from", old, "--to", ctx.name, "--root", str(dest)], ctx.repo)
    ctx.rep.add(STAGE, "PASS" if rc == 0 else "FAIL", f"scripts/rename.py {old} -> {ctx.name}", "\n".join(out.splitlines()[-6:]))
    return rc == 0


def gate_on_copy(ctx: core.Ctx, dest: Path, old: str) -> bool:
    # WHY the gates run on a COPY: checkers can write caches and build output; the committed tree must be the
    # archive plus the placeholder fill and nothing else, so the archive itself is never a working directory.
    scratch = ctx.work / "gate-tree"
    if scratch.exists():
        shutil.rmtree(scratch)
    shutil.copytree(dest, scratch)
    ok = run_gates(ctx, scratch, old, "gates on the export")
    if ok:
        shutil.rmtree(scratch)
    return ok


def pytest_on_copy(ctx: core.Ctx, dest: Path) -> bool:
    # Pytest and native build tests can write caches and artifacts. Keep their working tree out of the commit.
    label = "after the fill: pytest tests bindings/python/tests"
    cmd = [sys.executable, "-m", "pytest", "-q", "tests", "bindings/python/tests"]
    if "pytest" in ctx.skip_gates:
        return gate(ctx, label, cmd, dest, "pytest")
    with tempfile.TemporaryDirectory(dir=ctx.work, prefix="filled-gate-") as scratch:
        root = Path(scratch) / "tree"
        shutil.copytree(dest, root)
        return gate(ctx, label, cmd, root, "pytest")


def fill_and_check(ctx: core.Ctx, dest: Path) -> bool:
    before = hash_tree(dest)
    ctx.rep.add(STAGE, "PASS", f"repository address placeholder filled in {', '.join(fill_placeholder(ctx, dest))}: {repo_url(ctx.owner, ctx.name)}")
    stray = prove_only_placeholder(before, hash_tree(dest))
    ctx.rep.add(STAGE, "FAIL" if stray else "PASS", "the gated tree and the exported tree differ only in the placeholder lines" if not stray else "the fill changed more than the placeholder", "\n".join(stray))
    ok = not stray
    # WHY every checker, egress included: the public CI runs the whole suite on this very tree, so the tree the owner publishes
    # must have passed it; the filled repository address is allowed by checks/_egress_meta.py and by nothing broader.
    ok &= gate(ctx, "after the fill: python checks/run_all.py", [sys.executable, "checks/run_all.py"], dest, "local-checks")
    hits = rel_secrets.scan_tree(str(dest)) + junk_problems(dest)
    ctx.rep.add(STAGE, "FAIL" if hits else "PASS", "after the fill: no secret, identifier or generated file" if not hits else "after the fill: secret, identifier or generated-file hits", "\n".join(hits[:30]))
    return ok and not hits and pytest_on_copy(ctx, dest)


def commit_and_prove(ctx: core.Ctx, dest: Path, version: str) -> bool:
    sha = commit_tree(ctx, dest, version)
    problems = prove_commit_complete(ctx, dest)
    ctx.rep.add(STAGE, "FAIL" if problems else "PASS", f"one commit {sha[:12]} by {commit_identity(ctx, dest)}, tag v{version}, holding every file" if not problems else "the export commit is wrong", "\n".join(problems[:20]))
    ctx.state.update(export_commit=sha, tag=f"v{version}")
    return not problems


def run(ctx: core.Ctx) -> bool:
    if not ctx.owner:
        raise core.Refuse("--github-owner is required to fill the repository address")
    version = ctx.state.get("version") or core.read_version(ctx.repo)
    old = ctx.state.get("old_name") or rel_preflight.tree_name(ctx.repo)
    dest = ctx.export_dir
    if not (prepare(ctx, dest, old) and gate_on_copy(ctx, dest, old)):
        return False
    return fill_and_check(ctx, dest) and commit_and_prove(ctx, dest, version)
