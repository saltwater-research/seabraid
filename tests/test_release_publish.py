# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""scripts/release/publish.py end to end, on a throwaway copy of the repo, with stubbed gh, twine and cargo.

OWNER: this file proves the release script's contract (docs/RELEASING.md): a dry run changes nothing outside its work
directory, the final-name gate refuses a leftover old name, the export is one commit by the public identity with no
trailers, --publish does nothing without the typed token, and no credential value reaches the output.
No test contacts a registry or a website: the stubs record the command lines they would have run, and the "remote" the gh
stub pushes to is a bare repository in a temporary directory.  The native wheel is replaced by a stub scripts/build_wheel.py
in the COPY only (a real build needs a C toolchain; a dry run on the Linux test machine builds the real one).
"""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tomllib
import urllib.parse

import pytest
from release_stubs import STUB, WHEEL_STUB

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RELEASE = os.path.join(REPO, "scripts", "release")
sys.path.insert(0, os.path.join(REPO, "checks"))
sys.path.insert(0, RELEASE)
import _common  # noqa: E402
import rel_build  # noqa: E402
import rel_export  # noqa: E402
import rel_preflight  # noqa: E402
import rel_secrets  # noqa: E402

# WHY read, never spelled: the export of this very file is renamed by scripts/rename.py, and a test that spells the old name would
# then spell the new one (the export's own pytest run, a release gate, caught exactly that).
with open(os.path.join(REPO, "pyproject.toml"), "rb") as _fh:
    OLD = tomllib.load(_fh)["project"]["name"]
FINAL = "tidewrack" if OLD != "tidewrack" else "reefmark"  # any name other than the tree's own
OWNER = "acme-test"
REPO_URL = urllib.parse.urlunsplit(("https", "github.com", f"/{OWNER}/{FINAL}", "", ""))
SKIP = ["--skip-gate", "local-checks", "--skip-gate", "pytest", "--skip-gate", "manual"]
ID = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid"}



def git(cwd, *args, **env):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, env={**os.environ, **ID, **env}, check=True).stdout


@pytest.fixture(scope="module")
def dev(tmp_path_factory):
    """A git repository holding this tree plus the wheel stub, on branch main with one clean commit."""
    root = tmp_path_factory.mktemp("dev")
    for rel in _common.tracked(REPO):
        if rel.startswith(("tests/", "bindings/python/tests/")):
            continue  # A release CLI fixture must not recurse into its own test suite.
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(os.path.join(REPO, rel), root / rel)
    (root / "scripts" / "build_wheel.py").write_text(WHEEL_STUB, encoding="utf-8")
    (root / "scripts" / "wheel_smoke.py").write_text(f"import {OLD}\n")
    for relative in ("tests", "bindings/python/tests"):
        (root / relative).mkdir(parents=True, exist_ok=True)
        name = "test_publication.py" if relative == "tests" else "test_binding_fixture.py"
        (root / relative / name).write_text("def test_fixture(): assert True\n")
    (root / "checks" / "run_all.py").write_text("print('synthetic release fixture checks')\n")
    with open(root / "pyproject.toml", "rb") as manifest:
        url = tomllib.load(manifest)["project"]["urls"]["Homepage"]
    for rel in rel_export.FILL_FILES:
        path = root / rel
        path.write_text(path.read_text(encoding="utf-8").replace(url, rel_preflight.PLACEHOLDER), encoding="utf-8")
    # WHY rewritten: preflight refuses a CITATION.cff date that is not today's, and the real tree keeps the previous release's
    # date until the day of the tag, so the copy plays the tag day.
    cff = (root / "CITATION.cff").read_text(encoding="utf-8")
    (root / "CITATION.cff").write_text(re.sub(r"(?m)^date-released: \S+", f"date-released: {datetime.date.today().isoformat()}", cff), encoding="utf-8", newline="\n")
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "core.autocrlf", "false")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "dev tree")
    return root


@pytest.fixture(scope="module")
def tools(tmp_path_factory):
    """A bin directory with gh, twine and cargo stubs that record their command lines in STUB_LOG."""
    bin_dir = tmp_path_factory.mktemp("bin")
    (bin_dir / "stub.py").write_text(STUB, encoding="utf-8")
    # Subprocess registry observations use only this fixture. Wheel installation uses its isolated venv pip.
    (bin_dir / "pip.py").write_text("import sys\nif 'pip' in sys.argv[1:]: print('Available versions: 1'); sys.exit(0)\nprint('ERROR: No matching distribution')\nsys.exit(1)\n")
    for tool in ("gh", "twine", "cargo", "git"):
        if os.name == "nt":
            (bin_dir / f"{tool}.cmd").write_text(f'@"{sys.executable}" "{bin_dir / "stub.py"}" {tool} %*\r\n', encoding="utf-8")
        else:
            path = bin_dir / tool
            path.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{bin_dir / "stub.py"}" {tool} "$@"\n', encoding="utf-8")
            path.chmod(0o755)
    return bin_dir


def publish(dev, tools, work, *args, stdin="", extra_env=None):
    log, remote = work.parent / (work.name + ".log"), work.parent / (work.name + "-remote.git")
    verified = tools / "fixture-verified"
    verified.write_text(git(dev, "rev-parse", "HEAD").strip())
    env = {**os.environ, "RELEASE_VERIFIED_FILE": str(verified), "PYTHONPATH": str(tools), "CARGO_REGISTRY_TOKEN": "fixture-upload-secret",
           "STUB_ORIGIN": REPO_URL, "STUB_GIT": shutil.which("git"), "PATH": str(tools) + os.pathsep + os.environ["PATH"], "STUB_LOG": str(log), "STUB_REMOTE": str(remote),
           "CARGO_NET_OFFLINE": "false",  # The cargo stub plays the registry. The enclosing suite may itself be offline.
           "HOME": str(work.parent), "USERPROFILE": str(work.parent), **(extra_env or {})}
    seen = len(log.read_text(encoding="utf-8").splitlines()) if log.exists() else 0  # the log is shared by every run in one work dir
    scope = [] if args[:1] == ("crates-io-proof",) else ["--repo", str(dev), "--work", str(work)]
    cmd = [sys.executable, os.path.join(RELEASE, "publish.py"), *scope, *args]
    done = subprocess.run(cmd, capture_output=True, text=True, input=stdin, env=env, timeout=600)
    calls = [json.loads(ln) for ln in log.read_text(encoding="utf-8").splitlines()[seen:]] if log.exists() else []
    return done, calls, remote


def cmdlines(calls):
    return [c["tool"] + " " + " ".join(c["argv"]) for c in calls]


def recorded_proof(dev, tools, work):
    path = work / "official-proof.json"
    if not path.exists():
        version = (work / "export" / "VERSION").read_text().strip()
        done, _, _ = publish(dev, tools, work, "crates-io-proof", "--export", str(work / "export"),
                             "--tag", "v" + version, "--output", str(path))
        assert done.returncode == 0, done.stdout + done.stderr
    return ["--crates-io-proof", str(path)]


@pytest.fixture(scope="module")
def dry(dev, tools, tmp_path_factory):
    work = tmp_path_factory.mktemp("w") / "work"
    done, calls, _ = publish(dev, tools, work, "--name", FINAL, "--github-owner", OWNER)
    return work, done, calls


def test_dry_run_passes_and_runs_no_upload(dev, dry):
    work, done, calls = dry
    assert done.returncode == 0, done.stdout + done.stderr
    assert "DRY RUN" in done.stdout and "nothing was created, uploaded or pushed" in done.stdout
    lines = cmdlines(calls)
    assert_dry_commands(lines)
    assert not any(map(is_outward, lines))
    assert git(dev, "status", "--porcelain").strip() == ""
    assert git(dev, "rev-list", "--count", "HEAD").strip() == "1"


def assert_dry_commands(lines):
    assert any(ln.startswith("twine check") for ln in lines)
    assert any(ln.startswith("cargo package") and "--list" in ln for ln in lines)
    assert any(ln.startswith("cargo publish") and "--dry-run" in ln for ln in lines)



def is_outward(line: str) -> bool:
    """A command line that would create, upload or push something (a dry-run cargo publish is not one)."""
    return line.startswith("gh ") or "upload" in line or (line.startswith("cargo publish") and "--dry-run" not in line)


def test_export_is_one_commit_by_the_public_identity_with_no_trailers(dry):
    work, _, _ = dry
    export = work / "export"
    assert git(export, "rev-list", "--count", "HEAD").strip() == "1"
    assert git(export, "log", "-1", "--format=%an|%ae|%cn|%ce").strip() == "Saltwater Research LLC|dev@saltyresearch.com|Saltwater Research LLC|dev@saltyresearch.com"
    message = git(export, "log", "-1", "--format=%B").strip()
    assert "\n" not in message and "Co-Authored" not in message
    assert git(export, "log", "-1", "--format=%(trailers)").strip() == ""
    version = (export / "VERSION").read_text(encoding="utf-8").strip()
    assert git(export, "tag", "--list").split() == [f"v{version}"]
    assert git(export, "status", "--porcelain").strip() == ""
    assert not list(export.rglob("*.pyc"))
    assert git(export, "log", "-1", "--format=%ad", "--date=iso").strip().endswith("+0000")


def test_export_carries_the_final_name_and_filled_urls(dry):
    work, _, _ = dry
    export = work / "export"
    assert old_name_leftovers(export) == []
    pyproject = (export / "pyproject.toml").read_text(encoding="utf-8")
    cargo = (export / "Cargo.toml").read_text(encoding="utf-8")
    assert f'Homepage = "{REPO_URL}"' in pyproject and f'name = "{FINAL}"' in pyproject
    assert f'repository = "{REPO_URL}"' in cargo
    assert not any("http://localhost/@" + "REPO@" in ln for ln in cargo.splitlines() if not ln.startswith("#"))


def old_name_leftovers(export) -> list[str]:
    """Paths whose name or bytes still hold the old name."""
    paths = [p for p in export.rglob("*") if ".git" not in p.parts]
    in_content = [p for p in paths if p.is_file() and OLD.encode() in p.read_bytes().lower()]
    return [p.relative_to(export).as_posix() for p in in_content + [p for p in paths if OLD in p.as_posix().lower()]]


def test_build_makes_sdist_wheel_and_checksums(dry):
    work, _, _ = dry
    dist = work / "dist"
    names = sorted(p.name for p in dist.iterdir())
    assert {".whl", ".crate", ".gz"} <= {Path(n).suffix for n in names}
    assert "SHA256SUMS" in names
    sums = (dist / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    assert sorted(ln.split("  ")[1] for ln in sums) == [n for n in names if n != "SHA256SUMS"]
    assert (work / "RELEASE_NOTES.md").read_text(encoding="utf-8").startswith(f"# {FINAL} ")


def test_name_gate_refuses_the_temporary_name_when_another_final_name_is_given(dev, tmp_path):
    # unit: an unrenamed tree must fail the gate for any other final name, and pass for its own name
    assert rel_preflight.leftover_name_problems(FINAL, OLD, dev)
    assert rel_preflight.leftover_name_problems(OLD, OLD, dev) == []
    assert rel_preflight.name_problems("Bad-Name", dev) and rel_preflight.name_problems("ab", dev)


def test_export_refuses_when_the_old_name_survives_the_rename(dev, tools, tmp_path):
    # a binary file holding the old name cannot be rewritten by scripts/rename.py; the gate must refuse by name
    sneaky = dev / "docs" / "blob.bin"
    sneaky.write_bytes(b"\0\0" + OLD.encode() + b"\0")
    git(dev, "add", "-f", "-A")
    git(dev, "commit", "-q", "-m", "plant")
    try:
        done, calls, _ = publish(dev, tools, tmp_path / "w", "--name", FINAL, "--github-owner", OWNER, *SKIP)
    finally:
        git(dev, "reset", "-q", "--hard", "HEAD~1")
    assert done.returncode == 1 and "old name" in done.stdout and "refusing to publish" in done.stdout
    assert not [ln for ln in cmdlines(calls) if ln.startswith(("gh", "twine upload", "cargo publish"))]


def test_final_name_equal_to_the_trees_name_needs_no_rename(dev, tools, tmp_path):
    done, _, _ = publish(dev, tools, tmp_path / "w", "--name", OLD, "--github-owner", OWNER, "--only", "preflight", *SKIP)
    assert done.returncode == 0, done.stdout
    assert "no rename needed" in done.stdout


def test_bad_name_and_bad_usage_are_refused(dev, tools, tmp_path):
    bad, _, _ = publish(dev, tools, tmp_path / "w1", "--name", "Bad-Name", "--github-owner", OWNER, "--only", "preflight", *SKIP)
    assert bad.returncode == 1 and "must match" in bad.stdout
    for extra, why in ((["--publish", "--skip-gate", "pytest"], "cannot skip a gate"), (["--test-pypi"], "only means something with --publish")):
        done, _, _ = publish(dev, tools, tmp_path / "w2", "--name", FINAL, "--github-owner", OWNER, *extra)
        assert done.returncode == 1 and why in done.stdout
    inside, _, _ = publish(dev, tools, dev / "work", "--name", FINAL, "--github-owner", OWNER)
    assert inside.returncode == 1 and "inside the repository" in inside.stdout
    assert subprocess.run([sys.executable, os.path.join(RELEASE, "publish.py")], capture_output=True).returncode == 2


@pytest.mark.parametrize("stdin", ["", "no\n", "yes\n", f"{OWNER}/{FINAL.upper()}\n"])
def test_publish_without_the_typed_token_does_nothing(dev, tools, dry, stdin):
    work, _, _ = dry
    done, calls, remote = publish(dev, tools, work, "--name", FINAL, "--github-owner", OWNER, "--only", "publish", "--publish",
                                  *recorded_proof(dev, tools, work), stdin=stdin)
    assert done.returncode == 1 and "not confirmed" in done.stdout
    lines = cmdlines(calls)
    assert not any(ln.startswith(("gh repo create", "gh release create", "twine upload", "cargo publish --locked")) for ln in lines)
    assert not remote.exists()
    assert json.loads((work / "state.json").read_text(encoding="utf-8")).get("published", []) == []


def test_publish_with_every_typed_token_runs_the_steps_in_order(dev, tools, dry):
    work, _, _ = dry
    version = (work / "export" / "VERSION").read_text(encoding="utf-8").strip()
    typed = f"reviewed {git(dev, 'rev-parse', 'HEAD').strip()}\n{OWNER}/{FINAL}\nv{version}\npypi:{FINAL}\ncrates:{FINAL}\n"
    done, calls, remote = publish(dev, tools, work, "--name", FINAL, "--github-owner", OWNER, "--only", "publish", "--publish",
                                  *recorded_proof(dev, tools, work), stdin=typed)
    assert done.returncode == 0, done.stdout + done.stderr
    lines = [ln for ln in cmdlines(calls) if not ln.startswith(("gh auth", "cargo package"))]
    order = [next(i for i, ln in enumerate(lines) if ln.startswith(p)) for p in ("gh repo create", "gh release create", "twine upload", "cargo publish --locked")]
    assert order == sorted(order)
    assert f"v{version}" in git(remote, "tag", "--list")
    assert "Published. Now do this" in done.stdout and "private vulnerability reporting".lower() in done.stdout.lower()
    check_create_and_upload(lines)


def check_create_and_upload(lines):
    create = next(ln for ln in lines if ln.startswith("gh repo create"))
    assert f"{OWNER}/{FINAL}" in create and "--public" in create and "--push" not in create
    upload = next(ln for ln in lines if ln.startswith("twine upload"))
    assert upload.count(".whl") == 1 and ".tar.gz" not in upload and "--non-interactive" in upload


def test_test_pypi_rehearsal_uploads_only_to_testpypi(dev, tools, tmp_path_factory):
    work = tmp_path_factory.mktemp("w2") / "work"
    done, _, _ = publish(dev, tools, work, "--name", FINAL, "--github-owner", OWNER, "--skip", "preflight,export,build", *SKIP)
    assert done.returncode == 1  # no state in a fresh work dir: refused, not guessed
    done, _, _ = publish(dev, tools, work, "--name", FINAL, "--github-owner", OWNER)
    assert done.returncode == 0
    done, calls, _ = publish(dev, tools, work, "--name", FINAL, "--github-owner", OWNER, "--only", "publish", "--publish", "--test-pypi", stdin=f"reviewed {git(dev, 'rev-parse', 'HEAD').strip()}\ntestpypi:{FINAL}\n")
    lines = cmdlines(calls)
    uploads = [ln for ln in lines if ln.startswith("twine upload")]
    assert done.returncode == 0 and len(uploads) == 1 and "--repository testpypi" in uploads[0]
    assert not [ln for ln in lines if ln.startswith(("gh repo", "gh release", "cargo publish --locked"))]


def test_no_credential_value_is_ever_printed(dev, tools, tmp_path_factory):
    secrets = {"TWINE_PASSWORD": "pypi-" + "A1b2C3d4" * 8, "CARGO_REGISTRY_TOKEN": "cio" + "Zy9Xw8Vu" * 5, "GH_TOKEN": "ghp_" + "Qq1Ww2Ee3R" * 4}
    work = tmp_path_factory.mktemp("w3") / "work"
    first, calls1, _ = publish(dev, tools, work, "--name", FINAL, "--github-owner", OWNER, extra_env=secrets)
    assert first.returncode == 0, first.stdout
    version = (work / "export" / "VERSION").read_text(encoding="utf-8").strip()
    typed = f"reviewed {git(dev, 'rev-parse', 'HEAD').strip()}\n{OWNER}/{FINAL}\nv{version}\npypi:{FINAL}\ncrates:{FINAL}\n"
    second, calls2, _ = publish(dev, tools, work, "--name", FINAL, "--github-owner", OWNER, "--only", "publish", "--publish",
                                *recorded_proof(dev, tools, work), stdin=typed, extra_env=secrets)
    assert first.returncode == 0 and second.returncode == 0, second.stdout
    output = first.stdout + first.stderr + second.stdout + second.stderr
    assert not any(value in output for value in secrets.values())
    assert "debug: TWINE_PASSWORD=[REDACTED]" in second.stdout  # the stub echoed its token; the script must have masked it
    assert not any(value in json.dumps(calls1 + calls2) for value in secrets.values())


def test_secret_scan_finds_each_family_without_quoting_it(tmp_path):
    planted = {
        "a.txt": "token = " + "pypi-" + "x" * 50,
        "b.txt": "k=" + "ghp_" + "y" * 36,
        "c.txt": "-----BEGIN " + "RSA PRIVATE KEY-----",
        "d.txt": "AKIA" + "ABCDEFGHIJKLMNOP",
        "e.txt": "api_key = '" + "z1" * 15 + "'",
        "f.txt": "write to someone" + "@" + "gmail.com",
        ".env": "A=1",
        "ok.txt": "contact t@example.invalid, unit x@0.service, hash " + "0" * 64,
    }
    for name, text in planted.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    hits = rel_secrets.scan_tree(str(tmp_path))
    assert sorted(h.split(":")[0] for h in hits) == [".env", "a.txt", "b.txt", "c.txt", "d.txt", "e.txt", "f.txt"]
    assert not any("x" * 20 in h or "z1" * 10 in h or "gmail" in h for h in hits)


def test_release_notes_pick_the_version_section_then_unreleased():
    log = "# C\n\n## Unreleased\n\nnew\n\n## 1.0.0 - 2026-01-01\n\nold one\n\n## 0.9\n\nolder\n"
    assert rel_build.release_notes(log, "1.0.0") == ("1.0.0 - 2026-01-01", "old one\n")
    assert rel_build.release_notes(log, "2.0.0") == ("Unreleased", "new\n")
    assert rel_build.release_notes("# C\n", "2.0.0") == ("", "")


def test_release_modules_stay_small_and_hold_no_url():
    for name in os.listdir(RELEASE):
        if name.endswith(".py"):
            text = open(os.path.join(RELEASE, name), encoding="utf-8").read()
            assert len(text.splitlines()) <= 400, name
            if name == "rel_crate_credential.py":
                text = text.replace("https" + "://index.crates.io/", "").replace("https" + "://github.com/rust-lang/crates.io-index", "")
            assert "://" not in text.replace("http://localhost/", ""), f"{name} spells an unexpected URL"
