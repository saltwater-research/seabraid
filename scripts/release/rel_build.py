# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Stage 3, build: the sdist, the platform wheel, the crate listing and a crates.io dry run, with SHA256SUMS.

OWNER: this module owns what the release artifacts are and how they are checked before an upload.  The wheel is built by
scripts/build_wheel.py (its owner decides the platform tag and the library checks); this module only runs it.
CONTRACT: ``run(ctx) -> bool``; input is the committed export (``ctx.state['export']``); outputs are ``<work>/dist/*``
(sdist, wheel, SHA256SUMS) and ``<work>/RELEASE_NOTES.md``.  Builds happen in a throwaway ``git archive`` of the export
(``<work>/buildtree``) so the export stays byte-identical to its commit and its tag.
WHY the sdist is built but not uploaded by default: pyproject.toml explains that an sdist cannot build the Rust library, so
``pip install`` of it would give a reference-only package that looks like the engine; it goes into the GitHub release
(it is the source a reader can inspect) and ``--upload-sdist`` opts it into PyPI.
FAILURE: fail-fast per step with the tool's last lines; a missing OPTIONAL tool (twine) is a WARN with the install
command, a missing REQUIRED one (cargo, setuptools) is a FAIL.  Nothing here uploads or contacts a registry
(CARGO_NET_OFFLINE=true).
"""

from __future__ import annotations

import gzip
import shutil
import sys
import tempfile
import tarfile
from pathlib import Path

import rel_core as core
import rel_crate_proof

STAGE = "build"
# WHY an allow-list of crate paths: `cargo package --list` is the exact file list a crate would contain; anything outside
# these prefixes (tests, docs, scratch, CI files) is junk the Cargo.toml include list failed to keep out.
CRATE_ALLOWED = ("src/", "include/", "native-tests/", "build.rs", "Cargo.", "VERSION", "README.md", "LICENSE", "NOTICE", ".cargo_vcs_info.json")
SDIST_SNIPPET = "import sys, setuptools.build_meta as b; print(b.build_sdist(sys.argv[1]))"


def normalize_sdist(path: Path, epoch: str) -> None:
    """Rewrite setuptools' sdist metadata deterministically while preserving payloads and modes.

    WHY here: setuptools owns package contents, while this release stage owns reproducible artifact bytes; fixed tar/gzip
    headers let SOURCE_DATE_EPOCH control rebuilds even when setuptools leaves host time in its archive.
    """
    timestamp = int(epoch)
    temporary = path.with_suffix(path.suffix + ".normalized")
    try:
        with tarfile.open(path, "r:gz") as source, temporary.open("wb") as raw:
            compressed = gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0)
            _rewrite_sdist_tar(source, compressed, timestamp)
    except (OSError, tarfile.TarError, EOFError, ValueError) as exc:
        temporary.unlink(missing_ok=True)
        raise core.Refuse(f"could not normalize Python sdist {path.name}: {exc}") from exc
    temporary.replace(path)


def _rewrite_sdist_tar(source: tarfile.TarFile, compressed: gzip.GzipFile, timestamp: int) -> None:
    with compressed, tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as target:
        for member in source.getmembers():
            member.mtime = timestamp
            member.uid = member.gid = 0
            member.uname = member.gname = ""
            member.pax_headers = {key: value for key, value in member.pax_headers.items()
                                  if key not in ("atime", "ctime", "mtime", "uid", "gid", "uname", "gname")}
            payload = source.extractfile(member) if member.isfile() else None
            target.addfile(member, payload)


def seal_artifacts(ctx: core.Ctx) -> None:
    receipt = core.artifact_receipt(ctx)
    required = [*receipt["wheels"], receipt["crate"]]
    if receipt["checked_sha256"] != {name: receipt["sha256"][name] for name in required}:
        raise core.Refuse("release artifacts differ from the wheel install or crate test input; run the build again")
    rel_crate_proof.check(ctx, receipt, required=False)
    sums = core.checksum_manifest(receipt)
    try:
        (ctx.dist_dir / "SHA256SUMS").write_bytes(sums)
    except OSError as exc:
        raise core.Refuse("could not write the release artifact checksum list") from exc
    ctx.state["artifact_receipt"] = receipt
    ctx.rep.add(STAGE, "PASS", "SHA256SUMS written and release artifacts sealed", sums.decode("utf-8").rstrip())


def release_notes(changelog: str, version: str) -> tuple[str, str]:
    """(section title, body) for ``version``, else for Unreleased.  WHY a fallback: a first release often has no dated heading."""
    lines, picked, out = changelog.splitlines(), None, []
    for want in (version, "Unreleased"):
        for i, ln in enumerate(lines):
            if ln.startswith("## ") and ln[3:].startswith(want):
                picked = (ln[3:], i)
                break
        if picked:
            break
    if not picked:
        return "", ""
    for ln in lines[picked[1] + 1:]:
        if ln.startswith("## "):
            break
        out.append(ln)
    return picked[0], "\n".join(out).strip() + "\n"


def make_buildtree(ctx: core.Ctx, export: Path) -> Path:
    tree = ctx.work / "buildtree"
    if tree.exists():
        shutil.rmtree(tree)
    tar_path = ctx.work / "build.tar"
    # Archive the gated revision. Another checkout operation can move HEAD.
    rc, out = core.git(ctx, export, "archive", "--format=tar", "-o", str(tar_path), ctx.state["export_commit"])
    if rc != 0:
        raise core.Refuse(f"git archive of the export failed: {out}")
    tree.mkdir(parents=True)
    with tarfile.open(tar_path) as tar:
        tar.extractall(tree, filter="data")
    tar_path.unlink()
    return tree


def step(ctx: core.Ctx, label: str, cmd: list[str], cwd: Path, extra_env: dict | None = None, tail: int = 12) -> tuple[bool, str]:
    rc, out = ctx.run(cmd, cwd, extra_env)
    ctx.rep.add(STAGE, "PASS" if rc == 0 else "FAIL", label, "" if rc == 0 else "\n".join(out.splitlines()[-tail:]))
    return rc == 0, out


def build_python(ctx: core.Ctx, tree: Path, epoch: str) -> bool:
    """WHY SOURCE_DATE_EPOCH = the export commit time: the buildtree has no .git, and a rebuild of one tag must give the same bytes."""
    dist = ctx.dist_dir
    if dist.exists():
        shutil.rmtree(dist)
    dist.mkdir(parents=True)
    ok1, _ = step(ctx, "sdist (setuptools build_sdist)", [sys.executable, "-c", SDIST_SNIPPET, str(dist)], tree, {"SOURCE_DATE_EPOCH": epoch})
    if ok1:
        sdists = sorted(dist.glob("*.tar.gz"))
        if len(sdists) != 1:
            raise core.Refuse(f"expected one setuptools sdist, found {len(sdists)}")
        normalize_sdist(sdists[0], epoch)
    ok2, _ = step(ctx, "platform wheel (scripts/build_wheel.py)", [sys.executable, "scripts/build_wheel.py", "--out", str(dist)], tree, {"CARGO_NET_OFFLINE": "true", "SOURCE_DATE_EPOCH": epoch}, tail=20)
    return ok1 and ok2 and record_python_artifacts(ctx, dist)


def record_python_artifacts(ctx: core.Ctx, dist: Path) -> bool:
    wheels, sdists = sorted(dist.glob("*.whl")), sorted(dist.glob("*.tar.gz"))
    present = bool(wheels) and bool(sdists)
    ctx.rep.add(STAGE, "PASS" if present else "FAIL", f"artifacts: {', '.join(p.name for p in wheels + sdists) or 'none'}")
    ctx.state["wheels"] = [p.name for p in wheels]
    ctx.state["sdists"] = [p.name for p in sdists]
    return present


def twine_check(ctx: core.Ctx) -> bool:
    cmd = ctx.twine_command()
    if cmd is None:
        # RULE (offline vs real): a dry run WARNs or SKIPs only what needs the network or a login it may not have; twine is a
        # local tool, and this stage runs only on a host that can build, so a dry run without it proves nothing about the real run.
        ctx.rep.add(STAGE, "FAIL", "twine is not installed, so the distributions were not checked and --publish could not upload (install: python -m pip install twine)")
        return False
    files = [str(p) for p in sorted(ctx.dist_dir.iterdir()) if p.suffix in (".whl", ".gz")]
    return step(ctx, "twine check (metadata and README render)", [*cmd, "check", *files], ctx.dist_dir)[0]


def listed_files(listing: str) -> list[str]:
    """File paths in `cargo package --list` output (POSIX separators); cargo's own warning and error lines are not files."""
    return [ln.strip().replace("\\", "/") for ln in listing.splitlines() if ln.strip() and not ln.startswith(("warning", "error"))]


def crate_listing(ctx: core.Ctx, export: Path) -> bool:
    """cargo package --list, with every listed file checked against CRATE_ALLOWED."""
    ok, listing = step(ctx, "cargo package --list (what the crate would contain)", ["cargo", "package", "--allow-dirty", "--no-verify", "--list", "--offline"], export, ctx.cargo_env())
    files = listed_files(listing) if ok else []
    junk = [f for f in files if not f.startswith(CRATE_ALLOWED) and f != "docs/PLACEMENT-ABI.md"]
    good = ok and not junk and bool(files)
    ctx.rep.add(STAGE, "PASS" if good else "FAIL", f"crate holds {len(files)} files, none outside the allowed prefixes" if good else "crate content is wrong", "\n".join(junk[:20]))
    return good


def checked_wheel(ctx: core.Ctx, wheel: Path, tree: Path) -> bool:
    """An isolated interpreter tests the installed wheel using the same native weave smoke as CI."""
    digest = core.release_file_sha256(wheel)
    with tempfile.TemporaryDirectory(dir=ctx.work) as fresh:
        env = Path(fresh)
        py = str(env / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python"))
        commands = [
            ("fresh wheel environment", [sys.executable, "-I", "-m", "venv", str(env)]),
            ("install " + wheel.name, [py, "-I", "-m", "pip", "--isolated", "--disable-pip-version-check",
                                       "install", "--no-index", "--no-deps", "--no-cache-dir", str(wheel)]),
            ("load installed wheel and weave", [py, "-I", str(tree / "scripts" / "wheel_smoke.py")]),
        ]
        for label, argv in commands:
            if not step(ctx, label, argv, env)[0]:
                return False
    ctx.state["checked_sha256"][wheel.name] = digest
    return True


def checked_crate(ctx: core.Ctx, export: Path) -> bool:
    """Keep and test the exact archive; a fresh target prevents reuse of an earlier package."""
    name = f"{ctx.name}-{ctx.state['version']}"
    crate = ctx.dist_dir / (name + ".crate")
    with tempfile.TemporaryDirectory(dir=ctx.work) as fresh:
        env = {**ctx.cargo_env(), "CARGO_TARGET_DIR": fresh}
        if not step(ctx, "cargo package from gated export", ["cargo", "package", "--locked", "--offline"], export, env)[0]:
            return False
        try:
            shutil.copyfile(Path(fresh) / "package" / crate.name, crate)
            digest = core.release_file_sha256(crate)
            with tarfile.open(crate) as archive:
                archive.extractall(Path(fresh) / "unpacked", filter="data")
        except (OSError, tarfile.TarError) as exc:
            raise core.Refuse(f"cannot read the packaged crate {crate.name}: {exc}") from exc
        unpacked = Path(fresh) / "unpacked" / name
        if not step(ctx, "test unpacked " + crate.name, ["cargo", "test", "--locked", "--offline"], unpacked, env)[0]:
            return False
    ctx.state["crate"] = crate.name
    ctx.state["checked_sha256"][crate.name] = digest
    return True


def cargo_checks(ctx: core.Ctx, export: Path) -> bool:
    if not ctx.tool("cargo"):
        ctx.rep.add(STAGE, "FAIL", "cargo not found; the crate cannot be checked")
        return False
    return crate_listing(ctx, export) and checked_crate(ctx, export) and publish_dry_run(ctx, export)


def full_source(ctx: core.Ctx, export: Path) -> bool:
    stem = f"{ctx.name}-{ctx.state['version']}-source"
    archive = ctx.dist_dir / (stem + ".tar.gz")
    ok, _ = step(ctx, "full tagged source archive", ["git", "archive", "--format=tar.gz", "--prefix", stem + "/",
                  "-o", str(archive), ctx.state["export_commit"]], export)
    if ok:
        ctx.state["source_archive"] = archive.name
    return ok


# Lines cargo prints when it cannot reach (or has no copy of) the registry index.  WHY a list: `cargo publish --dry-run` asks the
# registry whether the version exists, even with --offline; without network that is a fact about the machine, not about the crate.
NO_REGISTRY = ("failed to update registry", "config.json not found", "Could not resolve host", "couldn't connect", "Network is unreachable", "failed to query replaced source")


def publish_dry_run(ctx: core.Ctx, export: Path) -> bool:
    rc, out = ctx.run(["cargo", "publish", "--dry-run", "--locked", "--offline"], export, ctx.cargo_env())
    ctx.state["cargo_rehearsed"] = rc == 0
    if rc == 0:
        ctx.rep.add(STAGE, "PASS", "offline Cargo dry run passed; official crates.io proof is separate")
    elif any(marker in out for marker in (*NO_REGISTRY, "cannot make API request in offline mode", "offline was specified", "--offline was specified")) and not ctx.publish:
        # RULE (offline vs real): a dry run stays offline-capable, so no index is a SKIP; --publish refuses, because the index
        # answer is the one thing this step proves that `cargo package` does not.
        ctx.rep.add(STAGE, "SKIP", "offline Cargo rehearsal could not use the index; record the official run with crates-io-proof")
        return True
    else:
        ctx.rep.add(STAGE, "FAIL", "cargo publish --dry-run" + (" (--publish needs the crates.io index: no network here)" if rc else ""), "\n".join(out.splitlines()[-20:]))
    return rc == 0


def write_notes(ctx: core.Ctx, tree: Path, version: str) -> None:
    title, body = release_notes((tree / "CHANGELOG.md").read_text(encoding="utf-8"), version)
    (ctx.work / "RELEASE_NOTES.md").write_text(f"# {ctx.name} {version}\n\n{body}" if body else f"# {ctx.name} {version}\n", encoding="utf-8")
    ctx.rep.add(STAGE, "PASS" if body else "WARN", f"release notes from CHANGELOG section {title!r} ({len(body.splitlines())} lines)" if body else "no CHANGELOG section found; release notes are only a title")


def run(ctx: core.Ctx) -> bool:
    export = Path(ctx.state.get("export", ""))
    if not (export / ".git").exists():
        raise core.Refuse("no export to build from: run the export stage first (same --work directory)")
    version = ctx.state["version"]
    rc, tag_object = core.git(ctx, export, "rev-parse", "--verify", "refs/tags/" + ctx.state["tag"])
    if rc:
        raise core.Refuse("could not identify the tagged export object")
    ctx.state["tag_object"] = tag_object.strip()
    tree = make_buildtree(ctx, export)
    rc, epoch = core.git(ctx, export, "log", "-1", "--format=%ct", ctx.state["export_commit"])
    if rc or not epoch.strip().isdigit():
        raise core.Refuse("could not read the gated export commit timestamp")
    ctx.state["checked_sha256"] = {}
    ok = build_python(ctx, tree, epoch.strip()) and twine_check(ctx)
    if ok:
        ok = all(checked_wheel(ctx, ctx.dist_dir / name, tree) for name in ctx.state["wheels"])
    ok &= cargo_checks(ctx, export)
    ok &= full_source(ctx, export)
    # Read notes from the same committed snapshot used to build the packages.
    write_notes(ctx, tree, version)
    if ok:
        seal_artifacts(ctx)
    return ok
