# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Shared plumbing for the release checkers: which files exist, which scope a path is in, how a run ends.

OWNER: this module is the only place that lists files and defines the scope split; every checker imports it,
so "what is the runtime" is one definition.  CONTRACT: pure functions over a repo root; no checker logic and no
import of any checker (one-way edge).  FAILURE: an unreadable file raises (a loud traceback, exit 1); only a binary file reads as ``None``.  Stdlib only.
"""

from __future__ import annotations

import os
import re
import subprocess

# WHY a tuple here: a post-merge tree may add scopes; checkers accept --runtime-scope to override without edits.
RUNTIME_SCOPES = ("src/", "include/", "adapters/", "bindings/python/seabraid/", "examples/")
COMPAT_SCOPE = "compat/strata/"
BINARY_EXT = (".png", ".jpg", ".gif", ".gguf", ".bin", ".pdf", ".zip", ".so", ".dll", ".exe", ".a", ".lib")
SOURCE_SUFFIXES = (".rs", ".c", ".h", ".cpp", ".hpp", ".cu", ".py", ".sh", ".cmake", ".cmake.in", "CMakeLists.txt")
_PRUNE = {".git", "target", "build", "dist", "__pycache__", ".pytest_cache", ".venv", "venv", "node_modules"}
# WHY an allow-list of top-level names for the no-.git walk: a tarball (`git archive`) ships exactly what `git ls-files`
# lists, but a walk of an unpacked archive also sees whatever the user put there afterwards: an SDK installed into the
# tree (`--prefix sdk`), a logs/ directory, a scratch checkout. Pruning by name cannot know them all, so the walk scans
# only the entries the repository ships at its top level. OWNER: this constant; tests/test_checkers.py compares it with
# `git ls-files` (a new top-level file or directory must be added here, and that test says so). Pruning below the top
# level (is_generated_dir) still applies inside these entries.
SHIPPED_TOP_LEVEL = frozenset({
    ".editorconfig", ".gitattributes", ".github", ".gitignore", ".zenodo.json", "CHANGELOG.md", "CITATION.cff", "CMakeLists.txt",
    "CODE_OF_CONDUCT.md", "CONTRIBUTING.md", "Cargo.lock", "Cargo.toml", "LICENSE", "LICENSE-MIT", "MAP.md", "NOTICE",
    "README.md", "SECURITY.md", "VERSION", "adapters", "bindings", "build.rs", "checks", "clippy.toml", "cmake", "compat",
    "deny.toml", "docs", "examples", "include", "native-tests", "pyproject.toml", "scripts", "src", "tests", "tools",
})


def is_generated_dir(parent: str, name: str) -> bool:
    """A directory no tarball ships: the fixed names above, egg-info, any CMake build tree, and any Python venv.

    WHY marker files, not names: the README's own wheel recipe makes ``.venv-wheel`` and users pick any venv name; a venv
    always holds ``pyvenv.cfg`` and a CMake build tree ``CMakeCache.txt``, so the walk prunes them whatever they are called.
    """
    if name in _PRUNE or name.endswith(".egg-info"):
        return True
    here = os.path.join(parent, name)
    return os.path.isfile(os.path.join(here, "pyvenv.cfg")) or os.path.isfile(os.path.join(here, "CMakeCache.txt"))


def default_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _git_files(root: str) -> list[str]:
    cmd = ["git", "-C", root, "ls-files", "-z", "--cached", "--others", "--exclude-standard"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout.decode("utf-8", "replace")
    return sorted(p for p in out.split("\0") if p and os.path.isfile(os.path.join(root, p)))


def _shipped(base: str, root: str, name: str) -> bool:
    """Below the top level every name is kept; AT the top level (the first os.walk yields the root as given) only shipped ones."""
    return base != root or name in SHIPPED_TOP_LEVEL


def _walked_files(root: str) -> list[str]:
    """The no-.git fallback: every file under the SHIPPED_TOP_LEVEL entries, minus generated directories."""
    found = []
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if not is_generated_dir(base, d) and _shipped(base, root, d)]
        names = [f for f in files if f != ".git" and _shipped(base, root, f)]
        found += [os.path.relpath(os.path.join(base, f), root).replace(os.sep, "/") for f in names]
    return sorted(found)


def tracked(root: str) -> list[str]:
    """Repo-relative POSIX paths: tracked plus untracked-not-ignored (so a pre-commit tree is checked too).

    WHY the walk fallback: the sabotage and rename tests copy the tree without .git; the checkers must still work.
    The fallback scans only SHIPPED_TOP_LEVEL entries, so an in-tree SDK prefix or a logs/ directory is never read.
    """
    # WHY exists, not isdir: in a linked worktree .git is a file.
    return _git_files(root) if os.path.exists(os.path.join(root, ".git")) else _walked_files(root)


def read_text(root: str, rel: str) -> str | None:
    """Text of a file, or None for binary (NUL in the first 8 KiB).  An unreadable file RAISES: never skip silently."""
    with open(os.path.join(root, rel), "rb") as fh:
        raw = fh.read()
    return None if b"\0" in raw[:8192] else raw.decode("utf-8", "replace")


def is_source(rel: str) -> bool:
    """A file of ours that carries code: the suffix list above; anything under fixtures/ is test data."""
    return rel.endswith(SOURCE_SUFFIXES) and "/fixtures/" not in f"/{rel}"


def in_scope(rel: str, scopes: tuple[str, ...] | list[str]) -> bool:
    return any(rel.startswith(s) for s in scopes)


def finish(name: str, hits: list[str], notes: list[str] | None = None) -> int:
    """Print every hit then one summary line; exit code 1 on any hit.  Loud by design: the summary names the checker."""
    for line in hits:
        print(line)
    for line in notes or []:
        print(line)
    print(f"{name}: {'FAILED, %d hit(s)' % len(hits) if hits else 'ok'}")
    return 1 if hits else 0


# English words that merely contain a project name; never a hit, never rewritten.  Keyed by the lower-case project name,
# valued by the whole lower-case words (empty until a rename name collides with a real word).  WHY a list and not a boundary rule: the old letter-edge rule let kName and
# NAMEX through the gate (review B8); the token now matches anywhere, so a legitimate English word is listed here by hand.
EXCLUDED_WORDS: dict = {}


def name_regex(name: str) -> re.Pattern:
    """The bare name token, any casing, with NO word boundaries: kName, NAMEX, name_x, libname and -lname all match.

    WHY this loose: a retired name that survives inside an identifier is still a retired name, and the earlier
    letter-edge rule is exactly what let kName and NAMEX through the gate.  The renamer maps the matched casing back
    (lower, UPPER, otherwise Capitalised) so one regex serves both the gate and the rewrite.
    """
    return re.compile(re.escape(name), re.IGNORECASE)


def is_excluded(name: str, text: str, start: int, end: int) -> bool:
    """True when the match sits inside a whole English word from EXCLUDED_WORDS (defence in depth over name_regex)."""
    lo, hi = start, end
    while lo > 0 and text[lo - 1].isalpha():
        lo -= 1
    while hi < len(text) and text[hi].isalpha():
        hi += 1
    return text[lo:hi].lower() in EXCLUDED_WORDS.get(name.lower(), ())
