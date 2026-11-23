# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Local command recorders and wheel builder for the release CLI integration tests."""

STUB = '''\
import io, json, os, subprocess, sys, tarfile, tomllib
from pathlib import Path
tool, argv = sys.argv[1], sys.argv[2:]
with open(os.environ["STUB_LOG"], "a", encoding="utf-8") as fh:
    fh.write(json.dumps({"tool": tool, "argv": argv}) + "\\n")
for key in ("TWINE_PASSWORD", "CARGO_REGISTRY_TOKEN", "GH_TOKEN"):  # a careless tool that echoes its credential when it uploads
    if os.environ.get(key) and argv[:1] in (["upload"], ["publish"], ["create"], ["repo"]) and "--dry-run" not in argv:
        print("debug: " + key + "=" + os.environ[key])
if tool == "git":
    if argv[-1].endswith("{commit}") and "^" not in argv[-1]:  # Windows .cmd removes this caret in the fixture shim.
        argv[-1] = argv[-1].replace("{commit}", "^{commit}")
    if argv[:2] in (["push", "origin"], ["ls-remote", "--refs"]):
        argv[argv.index("origin")] = os.environ["STUB_REMOTE"]
    sys.exit(subprocess.run([os.environ["STUB_GIT"], *argv]).returncode)
record_proof = argv[:1] == ["publish"] and "--dry-run" in argv and "--manifest-path" in argv
if tool == "cargo" and (record_proof or (argv[:1] == ["package"] and "--list" not in argv)):
    root = Path(argv[argv.index("--manifest-path") + 1]).parent if record_proof else Path.cwd()
    pkg = tomllib.loads((root / "Cargo.toml").read_text())["package"]
    name = pkg["name"] + "-" + pkg["version"]
    out = Path(os.environ["CARGO_TARGET_DIR"]) / "package"
    out.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out / (name + ".crate"), "w") as archive:
        info = tarfile.TarInfo(name + "/Cargo.toml")
        data = (root / "Cargo.toml").read_bytes()
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    if record_proof:
        print("Updating crates.io index\\nwarning: aborting upload due to dry run")
if tool == "cargo" and argv[:1] == ["info"]:
    if argv[1] == "serde":
        print("serde\\nversion: 1.0.0")
        sys.exit(0)
    print("error: could not find `" + argv[1] + "` in registry `crates-io`")
    sys.exit(101)
if tool == "cargo" and argv[:1] == ["publish"] and "--dry-run" not in argv:
    config = tomllib.loads((Path(os.environ["CARGO_HOME"]) / "config.toml").read_text())
    request = dict(v=1, kind="get", operation="publish", registry={"index-url":"sparse+" + "https" + "://index.crates.io/"},
                   name=os.environ["RELEASE_CRATE_NAME"], vers=os.environ["RELEASE_CRATE_VERSION"], cksum=os.environ["RELEASE_CRATE_SHA256"])
    result = subprocess.run([*config["registry"]["credential-provider"], "--cargo-plugin"], input=json.dumps(request)+"\\n", capture_output=True, text=True)
    if result.returncode or "Ok" not in result.stdout:
        sys.exit(3)
if tool == "gh" and argv[:2] in (["repo","view"], ["release","view"]):
    if argv[:2] == ["repo","view"] and Path(os.environ["STUB_REMOTE"]).exists():
        sys.exit(0)
    print("not found")
    sys.exit(1)
if tool == "cargo" and argv[:1] == ["package"]:
    print("Cargo.toml\\nbuild.rs\\nsrc/lib.rs\\ninclude/x/x.h\\nnative-tests/abi_layout.txt\\nREADME.md\\nLICENSE\\nNOTICE\\nVERSION")
if tool == "gh" and argv[:2] == ["repo", "create"]:
    src = argv[argv.index("--source") + 1]
    remote = os.environ["STUB_REMOTE"]
    subprocess.run(["git", "init", "-q", "--bare", remote], check=True)
    subprocess.run(["git", "-C", src, "remote", "add", "origin", os.environ["STUB_ORIGIN"]], check=True)
'''
WHEEL_STUB = '''\
# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Test double for scripts/build_wheel.py: writes an empty wheel named from pyproject.toml and VERSION."""
import sys, tomllib, zipfile
from pathlib import Path
root = Path(__file__).resolve().parent.parent
name = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]["name"]
version = (root / "VERSION").read_text(encoding="utf-8").strip().replace("-alpha.", "a")
out = Path(sys.argv[sys.argv.index("--out") + 1])
out.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(out / f"{name}-{version}-py3-none-any.whl", "w") as z:
    z.writestr(f"{name}/__init__.py", "")
    info = f"{name}-{version}.dist-info"
    z.writestr(f"{info}/METADATA", f"Metadata-Version: 2.1\\nName: {name}\\nVersion: {version}\\n")
    z.writestr(f"{info}/WHEEL", "Wheel-Version: 1.0\\nGenerator: fixture\\nRoot-Is-Purelib: true\\nTag: py3-none-any\\n")
    z.writestr(f"{info}/RECORD", "")
print(out)
'''
