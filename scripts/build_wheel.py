# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Build a platform wheel that carries the release cdylib at seabraid/_lib/<name>, offline, stdlib plus setuptools.

OWNER   : the packaging of the native library into a wheel, and the honest platform tag (the lookup is seabraid/_bundled.py).
CONTRACT: `python scripts/build_wheel.py [--out DIR]` -> prints the wheel path, from any working directory (the repository
          root is found from this file; a relative --out is relative to where the command ran). Platform checks use
          the ELF header and `readelf` for Linux, and `otool -l` + `lipo -archs` text for a macOS tag
          (the dylib's own minimum OS), merge every rustc-flag source and append --remap-path-prefix (release_env), list the
          build paths a binary embeds (leaked_paths, over leak_roots), and give the time every entry carries (source_date_epoch);
          check_out_dir() and check_lib_dir() are the up-front checks.
SIZE    : under 420 lines (raised from 410 by the loud epoch fallback; the repository ceiling is 500, and a split into a lock/verify module is the move past 450).
FAILURE : fails fast and loud (SystemExit with the reason): a missing tool, a failed build, any DT_NEEDED outside
          libc/libm/libgcc_s/libpthread/libdl/ld-linux, no GLIBC version found, an unsupported ELF, a library embedding
          the checkout path or a home directory of two or more components (written with slashes or backslashes), an unwritable --out
          (checked before cargo), a second run in the same checkout while one holds the lock (also before its cargo build).
          The source tree is never written to: the library goes straight into setuptools' private build directory, so a killed
          build leaves nothing behind, and anything a user keeps in seabraid/_lib/ is neither shipped nor deleted (the build
          refuses, before cargo, when _lib/ holds anything but the library's own name).
          Reproducible: every wheel entry's time is SOURCE_DATE_EPOCH (else the commit time, else a fixed 1980 date) and file
          modes are normalised to 0644 / 0755, so two builds of one commit from different paths are byte-identical. The built
          wheel is read back: it must hold seabraid/_lib/<name> with an execute bit (not on Windows, whose tag is the DLL's PE
          machine type). Cargo reports the exact library path, including target or target directory overrides.
CHECKER : tests/test_build_wheel*.py, tests/test_ci_workflow.py (wheel job), and the CI smoke install.
FINISH  : the wheel installs into a fresh venv with --no-index and seabraid.bundled_library() opens a native Runtime.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Mapping
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))  # the sibling module, whether run as a script or loaded by path
from wheel_elf_tag import check_elf, disallowed_needed, manylinux_tag  # noqa: E402
from wheel_pe_tag import windows_platform_tag  # noqa: E402
LIB_DIR = REPO / "bindings" / "python" / "seabraid" / "_lib"
# The file name cargo produces under target/release; the package looks for the same name (see _bundled.py).
LIB_NAMES = {"win32": "seabraid.dll", "darwin": "libseabraid.dylib"}
# The checkout path is rewritten to this in anything rustc embeds (debug info, panic locations, macros).
REMAP_TO = "/seabraid"


def target_rustflags_variable(triple: str) -> str:
    """Cargo's environment variable for one target triple: CARGO_TARGET_<TRIPLE>_RUSTFLAGS, upper case, `-` and `.` as `_`."""
    return "CARGO_TARGET_" + re.sub(r"[-.]", "_", triple).upper() + "_RUSTFLAGS"


def release_env(environ: Mapping[str, str], repo: Path = REPO, triple: str | None = None) -> dict[str, str]:
    """`environ` with every rustc-flag source merged and --remap-path-prefix APPENDED, never replacing the caller's flags.

    Sources, in cargo's precedence order: CARGO_ENCODED_RUSTFLAGS (split on the unit separator), RUSTFLAGS,
    CARGO_TARGET_<TRIPLE>_RUSTFLAGS (for `triple`, else CARGO_BUILD_TARGET, else none), CARGO_BUILD_RUSTFLAGS; ours last.

    WHY merge instead of "first one wins" like cargo: setting CARGO_ENCODED_RUSTFLAGS makes cargo ignore all the others,
    so a flag the release builder exported anywhere must be re-emitted here or it is dropped without a word. The cost is
    that a builder who set two of them gets both, where cargo alone would have used the first; a release wants the flags it
    was given. WHY CARGO_ENCODED_RUSTFLAGS at all: it survives a checkout path with spaces (RUSTFLAGS is split on
    whitespace). Config-file rustflags (`build.rustflags`, `[target.<cfg>]`) are not visible here and are shadowed by
    any of these variables, as cargo documents.
    """
    triple = triple or environ.get("CARGO_BUILD_TARGET") or None
    merged = [flag for flag in environ.get("CARGO_ENCODED_RUSTFLAGS", "").split("\x1f") if flag]
    sources = ["RUSTFLAGS", "CARGO_BUILD_RUSTFLAGS"]
    if triple:
        sources.insert(1, target_rustflags_variable(triple))  # cargo prefers the target's flags over build.rustflags
    for name in sources:
        merged += environ.get(name, "").split()
    merged.append(f"--remap-path-prefix={repo}={REMAP_TO}")
    env = {key: value for key, value in environ.items() if key not in sources}
    env["CARGO_ENCODED_RUSTFLAGS"] = "\x1f".join(merged)
    return env


def leaked_paths(binary: bytes, roots: Mapping[str, Path]) -> list[str]:
    """`label: text` for each build-location path a binary embeds, written with `/`, `\\` or an escaped `\\\\`.

    WHY the separator forms: a Windows path reaches a binary as C:\\x\\y (panic text), C:/x/y (rustc's own paths) or
    C:\\\\x\\\\y (a Debug-quoted string), and a scan for only str(path) misses two of them. A root that is a prefix of
    other directories (a home directory) must be followed by a separator, or "<home>/al" would match "<home>/alice".
    """
    hits = []
    for label, root in roots.items():
        posix = root.as_posix().rstrip("/")
        forms = {str(root), posix, posix.replace("/", "\\"), posix.replace("/", "\\\\")}
        tail = b"" if label == "checkout" else b"[/\\\\]"
        for form in sorted(forms):
            if form and re.search(re.escape(form.encode()) + tail, binary):
                hits.append(f"{label}: {form}")
    return hits


def leak_roots(repo: Path, home: Path) -> dict[str, Path]:
    """The roots leaked_paths() scans for: the checkout always, the home directory only when it names a person.

    WHY the two-component floor: a home of `/` (a container, HOME=/) or `/usr` is a prefix of every system path, so
    "scan for it" refuses a clean library on any "//" or "/usr/..." string (R7 e8). A directory like /srv/alice or
    D:\\work\\alice (two components below the anchor) is specific enough that finding it in a binary means a leak."""
    roots = {"checkout": repo}
    if len([part for part in home.parts if part != home.anchor]) >= 2:
        roots["home directory"] = home
    return roots


def check_out_dir(out: Path) -> None:
    """Create --out and write a probe file in it, or exit with a one-line reason: before cargo, not after ten minutes of it."""
    try:
        out.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=out):
            pass
    except OSError as error:
        raise SystemExit(f"--out {out} cannot be created or written to ({error.strerror or error}); nothing was built") from None


def check_lib_dir(lib_dir: Path, name: str) -> None:
    """Refuse when seabraid/_lib/ holds anything but the library's own file name; the contents are never touched.

    WHY refuse and not ignore: the build no longer reads _lib/ at all, so a stray file there cannot reach the wheel, but
    it is almost certainly somebody's leftover or hand-copied binary, and saying so beats silently building around it."""
    if not lib_dir.is_dir():
        return
    stray = sorted(path.name for path in lib_dir.iterdir() if path.name != name)
    if stray:
        raise SystemExit(f"{lib_dir} holds files this build neither ships nor deletes: {', '.join(stray)}; move them away and rerun")


# Used when neither SOURCE_DATE_EPOCH nor a git commit gives a time: 1980-01-01, the earliest instant a zip entry can carry.
FALLBACK_EPOCH = "315532800"


def source_date_epoch(environ: Mapping[str, str], repo: Path) -> str:
    """The timestamp for every wheel entry: the caller's SOURCE_DATE_EPOCH, else HEAD's commit time, else FALLBACK_EPOCH.

    WHY the commit time: it is the same for every checkout of one commit, wherever and whenever it is built; wall-clock
    or file mtimes made two builds of one commit differ. WHY the toplevel check: a source tarball unpacked inside some
    other git repository must not take that repository's commit time."""
    if environ.get("SOURCE_DATE_EPOCH"):
        return environ["SOURCE_DATE_EPOCH"]
    why = "no git commit time"
    try:
        git = ["git", "-C", str(repo)]
        top = subprocess.run(git + ["rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True).stdout.strip()
        if Path(top).resolve() == repo.resolve():
            stamp = subprocess.run(git + ["log", "-1", "--format=%ct"], capture_output=True, text=True, check=True).stdout.strip()
            if stamp.isdigit():
                return stamp
            why = f"git log printed {stamp!r}"
        else:
            why = f"{repo} sits inside the git repository at {top}"
    except (OSError, subprocess.CalledProcessError) as exc:
        why = f"git failed: {exc}"
    # WHY loud (R8 L7): this fallback used to be a bare `except: pass`; every entry then carried 1980 with no word of why.
    print(f"build_wheel: warning: no SOURCE_DATE_EPOCH and no usable commit time ({why}); using {FALLBACK_EPOCH} (1980-01-01)", file=sys.stderr)
    return FALLBACK_EPOCH


def built_library_path(messages: str, name: str) -> Path:
    """Use Cargo's artifact receipt so target configuration cannot send packaging to a stale default path."""
    libraries = set()
    for line in messages.splitlines():
        if not line.startswith("{"):  # Cargo permits plain-text output from external tools.
            continue
        message = json.loads(line)
        if message["reason"] != "compiler-artifact" or message["target"]["name"] != "seabraid":
            continue
        if "cdylib" in message["target"]["crate_types"]:
            paths = map(Path, message["filenames"])
            libraries.update(path for path in paths if path.name == name and path.is_absolute())
    if len(libraries) != 1:
        raise SystemExit(f"cargo did not produce exactly one {name}")
    return libraries.pop()


def verify_wheel_library(wheel: Path, name: str, needs_exec_mode: bool) -> None:
    """Fail unless the wheel holds seabraid/_lib/<name> (and, where modes exist, with an execute bit).

    WHY read the wheel back: a wheel is a valid zip whether or not the build step that adds the library ran, so one
    built from a mis-named or missing file would load no native library. The mode is normalised by the build (see
    build_wheel); a non-executable .so is refused by some loaders, so it is checked here too, not at install."""
    member = f"seabraid/_lib/{name}"
    with zipfile.ZipFile(wheel) as archive:
        try:
            info = archive.getinfo(member)
        except KeyError:
            raise SystemExit(f"{wheel.name} does not contain {member}; the native library did not reach the wheel") from None
    if needs_exec_mode and not (info.external_attr >> 16) & 0o111:
        raise SystemExit(f"{wheel.name}: {member} is not executable (mode {oct(info.external_attr >> 16)})")


def lock_path(repo: Path) -> Path:
    """The lock file for one checkout, in the temp directory so the tree gains no stray file."""
    return Path(tempfile.gettempdir()) / f"seabraid-build-wheel-{hashlib.sha1(str(repo).encode()).hexdigest()[:12]}.lock"


@contextlib.contextmanager
def checkout_lock(path: Path):
    """Hold an exclusive, non-blocking lock for the duration, or exit saying another build owns this checkout.

    WHY a lock: two builds in one checkout share target/, so the second would read a library the first is still
    writing, and cargo's own lock only serialises the compile, not our read of its output. An OS file lock needs no
    cleanup after a kill."""
    handle = open(path, "a+")
    try:
        try:
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise SystemExit(f"another scripts/build_wheel.py is running in this checkout (lock {path}); wait for it") from None
        yield
    finally:
        handle.close()  # closing releases the lock


def host_triple() -> str:
    """The host target triple from `rustc -vV` (what cargo builds for when no --target/CARGO_BUILD_TARGET is given)."""
    match = re.search(r"^host: (\S+)$", _run(["rustc", "-vV"]), re.M)
    if not match:
        raise SystemExit("rustc -vV printed no host triple; cannot find the target's rustflags variable")
    return match.group(1)


def macos_platform_tag(otool_load_commands: str, lipo_archs: str) -> str:
    """macosx_<major>_<minor>_<arch> from `otool -l` and `lipo -archs` text of the dylib.

    WHY from the dylib: the wheel must promise exactly the macOS the library was linked to run on; the interpreter's
    tag (what setuptools would pick) can be newer or a different architecture. From macOS 11 pip only recognises
    <major>_0, and arm64 exists only from 11.0, so both are normalised. A universal dylib takes the highest minimum.
    """
    found = []
    for block in otool_load_commands.split("Load command")[1:]:
        build = re.search(r"cmd LC_BUILD_VERSION.*?platform (\w+).*?minos (\d+)\.(\d+)", block, re.S)
        legacy = re.search(r"cmd LC_VERSION_MIN_MACOSX.*?version (\d+)\.(\d+)", block, re.S)
        if build and build.group(1) in ("1", "MACOS"):
            found.append((int(build.group(2)), int(build.group(3))))
        elif legacy:
            found.append((int(legacy.group(1)), int(legacy.group(2))))
    if not found:
        raise SystemExit("no macOS minimum version (LC_BUILD_VERSION / LC_VERSION_MIN_MACOSX) in otool -l output; "
                         "refusing to guess a macOS wheel tag")
    archs = sorted(set(lipo_archs.split()))
    if archs == ["arm64", "x86_64"]:
        arch = "universal2"
    elif archs in (["arm64"], ["x86_64"]):
        arch = archs[0]
    else:
        raise SystemExit(f"unexpected architectures from lipo: {lipo_archs!r}")
    major, minor = max(found)
    if arch != "x86_64":
        major, minor = max((major, minor), (11, 0))  # arm64 slices cannot target macOS below 11.0
    return f"macosx_{major}_{minor if major == 10 else 0}_{arch}"


def _run(command: list[str], **kwargs) -> str:
    try:
        done = subprocess.run(command, check=True, capture_output=True, text=True, **kwargs)
    except FileNotFoundError:
        raise SystemExit(f"required tool not found: {command[0]}") from None
    except subprocess.CalledProcessError as error:
        raise SystemExit(f"{' '.join(command)} failed ({error.returncode}):\n{error.stdout}\n{error.stderr}") from None
    return done.stdout


def linux_platform_tag(library: Path) -> str:
    with library.open("rb") as image:
        check_elf(image.read(64))
    env = {**os.environ, "LC_ALL": "C"}  # readelf's output is parsed using its English labels.
    bad = disallowed_needed(_run(["readelf", "-d", str(library)], env=env))
    if bad:
        raise SystemExit(f"{library.name} needs libraries a manylinux wheel may not rely on: {', '.join(bad)}")
    return manylinux_tag(_run(["readelf", "-V", str(library)], env=env))


def _bdist_wheel_class():
    try:  # setuptools >= 70.1 ships bdist_wheel itself; older ones rely on the wheel package
        from setuptools.command.bdist_wheel import bdist_wheel
    except ImportError:
        from wheel.bdist_wheel import bdist_wheel
    return bdist_wheel


def platform_tag_of(library: Path) -> str | None:
    """The honest platform tag for this host's library (Windows: its PE machine type, never the interpreter's arch)."""
    if sys.platform.startswith("linux"):
        return linux_platform_tag(library)
    if sys.platform == "darwin":
        return macos_platform_tag(_run(["otool", "-l", str(library)]), _run(["lipo", "-archs", str(library)]))
    return windows_platform_tag(library.read_bytes()) if sys.platform == "win32" else None


def build_wheel(out_dir: Path, platform_tag: str | None, library: Path | None = None, name: str = "libseabraid.so",
                repo: Path | None = None) -> Path:
    """Run setuptools in-process: non-pure (root-is-purelib false), tagged py3-none-<platform>, no network.

    `library`, when given, is copied to seabraid/_lib/<name> inside setuptools' private build directory and nowhere else.
    WHY not a package-data glob over the source tree's _lib/: that shipped whatever a user kept there, needed a copy
    into the tree that a kill left behind (and a `finally` that deleted the user's files), and made `pip install .`
    package a stale binary into a py3-none-any wheel. Every file's mode is normalised (0755 for the library, 0644 for the
    rest) and every entry's time is source_date_epoch(), so the bytes do not depend on umask, mtimes or checkout path."""
    import setuptools
    from setuptools.command.build_py import build_py
    from setuptools.command.install import install
    from setuptools.dist import Distribution

    repo = repo or REPO

    class BinaryDistribution(Distribution):
        def has_ext_modules(self) -> bool:  # makes the wheel platform-specific without any compiled extension
            return True

    class PlatformWheel(_bdist_wheel_class()):
        def get_tag(self):  # the library is loaded by ctypes, so no CPython ABI is involved: py3-none
            return "py3", "none", platform_tag or super().get_tag()[2]

    class PortableInstall(install):
        # WHY: Debian/Ubuntu's setuptools patches install_lib and install_egg_info to read install.install_layout, which
        # only Debian's patched stdlib distutils defines; setuptools' own distutils (always the one in use on Python 3.12+,
        # and on 3.11 without SETUPTOOLS_USE_DISTUTILS=stdlib) lacks it and bdist_wheel dies with AttributeError. None is
        # the value the Debian code treats as "not the deb layout", i.e. what every other setuptools behaves as.
        install_layout = None
        prefix_option = None  # install_egg_info reads this one the same way

    class LibraryBuildPy(build_py):
        def run(self):
            super().run()
            if library is not None:
                target = Path(self.build_lib) / "seabraid" / "_lib" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(library, target)
            for path in Path(self.build_lib).rglob("*"):
                if path.is_file():
                    path.chmod(0o755 if library is not None and path.name == name else 0o644)

    with tempfile.TemporaryDirectory() as scratch:
        # WHY in-process with pyproject.toml as the only metadata owner: no setup.py shim to drift from it. egg-info and
        # build/ go to a scratch directory so the checkout gains no stray files.
        argv = ["egg_info", "--egg-base", scratch, "build", "--build-base", scratch, "bdist_wheel",
                "--dist-dir", str(out_dir), "--bdist-dir", str(Path(scratch) / "bdist")]
        # WHY chdir: setuptools finds pyproject.toml and the package dirs from the working directory, and from anywhere
        # else it silently builds "unknown-0.0.0"; the old directory and environment are restored whatever happens.
        # WHY umask: setuptools writes the dist-info files (WHEEL, top_level.txt) with the caller's umask, and those modes
        # reach the zip, so a 077 builder got a different wheel; the library and package files are chmod-ed in LibraryBuildPy.
        previous, previous_epoch, previous_umask = os.getcwd(), os.environ.get("SOURCE_DATE_EPOCH"), os.umask(0o022)
        os.environ["SOURCE_DATE_EPOCH"] = source_date_epoch(os.environ, repo)
        os.chdir(repo)
        try:
            setuptools.setup(distclass=BinaryDistribution, script_args=argv,
                             cmdclass={"bdist_wheel": PlatformWheel, "build_py": LibraryBuildPy, "install": PortableInstall})
        finally:
            os.umask(previous_umask)
            os.chdir(previous)
            if previous_epoch is None:
                os.environ.pop("SOURCE_DATE_EPOCH", None)
            else:
                os.environ["SOURCE_DATE_EPOCH"] = previous_epoch
    wheels = sorted(out_dir.glob("seabraid-*.whl"), key=lambda path: path.stat().st_mtime)
    if not wheels:
        raise SystemExit("setuptools produced no wheel")
    return wheels[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=REPO / "dist", help="wheel output directory (default: dist/)")
    args = parser.parse_args()
    args.out = args.out.resolve()  # against the caller's directory, before anything changes it
    name = LIB_NAMES.get(sys.platform, "libseabraid.so")
    check_lib_dir(LIB_DIR, name)
    # WHY the lock covers the cargo build too: two builders in one checkout share target/, so the second one building
    # (and only then being refused) wasted a full build and could read a library the first was still writing.
    with checkout_lock(lock_path(REPO)):
        check_out_dir(args.out)  # before cargo: an unwritable --out must not cost a full build
        # --locked and --offline: the crate has no dependencies, so a build that wants the network is a bug to see, not hide.
        triple = os.environ.get("CARGO_BUILD_TARGET") or host_triple()
        messages = _run(["cargo", "build", "--release", "--offline", "--locked", "--message-format=json-render-diagnostics"],
                        cwd=REPO, env=release_env(os.environ, REPO, triple))
        built = built_library_path(messages, name)
        if not built.is_file():
            raise SystemExit(f"cargo did not produce {built}")
        leaks = leaked_paths(built.read_bytes(), leak_roots(REPO, Path.home()))
        if leaks:
            raise SystemExit(f"{built.name} embeds where it was built ({'; '.join(leaks)}); --remap-path-prefix did not cover it")
        wheel = build_wheel(args.out, platform_tag_of(built), built, name)
    verify_wheel_library(wheel, name, needs_exec_mode=sys.platform != "win32")
    print(wheel)
    return 0


if __name__ == "__main__":
    sys.exit(main())
