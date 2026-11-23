# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""The two guides stay true: they exist, are linked, name only real files, show only real commands, and compile.

OWNER   : what "a working guide" means for docs/USER-GUIDE.md and docs/ENGINE-INTEGRATION.md. check_links.py owns link
          resolution and is not repeated here.
CONTRACT: (1) both files exist, are at most 400 lines, are listed first in README's Documentation section, and the adapter
          guide links ENGINE-INTEGRATION near its top; (2) every inline `code` token that starts inside a repo top-level
          directory (or names a top-level file) exists (a `*` is a glob and must match); (3) in sh fences, a python, sh,
          bash or cmake command line that names a repo path or script points at one that exists, resolved from the repo
          root or from the last `cd` of that fence; (4) every code block marked `<!-- compilable -->` is compiled with
          cc -std=c11 -Wall -Wextra -Werror -c against include/ (Linux with cc only; elsewhere skipped). Each guide must
          keep at least one marked block, so deleting the marker cannot silently turn the check off.
FAILURE : fail-fast, one message per problem naming the guide and line. Generated locations (build, target, dist) are
          not repo paths and are skipped by name.
CHECKER : this file; test_compile_rejects_a_broken_snippet shows the compile check bites.
"""

from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GUIDES = ["docs/USER-GUIDE.md", "docs/ENGINE-INTEGRATION.md"]
GENERATED = {"build", "target", "dist"}
TOP = {name for name in os.listdir(REPO) if name not in GENERATED and not name.startswith(".")}
FENCE = re.compile(r"^\s*```(\w*)\s*$")
SPAN = re.compile(r"`([^`\n]+)`")
PATHISH = re.compile(r"^[A-Za-z][\w.\-]*(/[\w.\-*]+)*/?$")


def lines_of(guide: str) -> list[str]:
    with open(os.path.join(REPO, guide), encoding="utf-8") as handle:
        return handle.read().splitlines()


def blocks(guide: str):
    """Yield (first_line_number, info_string, [lines], preceding_line) for every fenced block."""
    lines, open_at, info, body = lines_of(guide), None, "", []
    for number, line in enumerate(lines, 1):
        match = FENCE.match(line)
        if open_at is None and match:
            open_at, info, body = number, match.group(1), []
        elif open_at is not None and line.strip() == "```":
            yield open_at, info, body, lines[open_at - 2] if open_at > 1 else ""
            open_at = None
        elif open_at is not None:
            body.append(line)


def exists(path: str, *bases: str) -> bool:
    path = path.rstrip("/")
    for base in bases:
        full = os.path.join(base, path)
        if glob.glob(full) if "*" in path else os.path.exists(full):
            return True
    return False


@pytest.mark.parametrize("guide", GUIDES)
def test_guide_exists_and_is_short_enough(guide: str) -> None:
    assert os.path.isfile(os.path.join(REPO, guide)), guide
    assert len(lines_of(guide)) <= 400, f"{guide} is over the 400-line cap"


def test_readme_lists_both_guides_first_and_adapter_guide_points_at_integration() -> None:
    with open(os.path.join(REPO, "README.md"), encoding="utf-8") as handle:
        readme = handle.read()
    section = readme[readme.index("\n## Documentation") :]
    user, engine, abi = (section.find(f"docs/{n}") for n in ("USER-GUIDE.md", "ENGINE-INTEGRATION.md", "ABI.md"))
    assert 0 <= user < abi and 0 <= engine < abi, "README Documentation must list both guides before the ABI reference"
    assert "ENGINE-INTEGRATION.md" in "\n".join(lines_of("docs/ADAPTER-GUIDE.md")[:12]), "adapter guide must link it at the top"


@pytest.mark.parametrize("guide", GUIDES)
def test_every_repo_path_the_guide_names_exists(guide: str) -> None:
    fenced = {n for first, _, body, _ in blocks(guide) for n in range(first, first + len(body) + 2)}
    problems = []
    for number, line in enumerate(lines_of(guide), 1):
        spans = [] if number in fenced else [t.strip() for t in SPAN.findall(line)]
        problems += [f"{guide}:{number}: `{t}` does not exist" for t in spans if is_repo_path_span(t) and not exists(t, REPO)]
    assert not problems, "\n".join(problems)


def is_repo_path_span(token: str) -> bool:
    """A backticked span that names a path under a top-level entry (a bare word that equals one is not a path)."""
    return bool(PATHISH.match(token)) and token.split("/")[0] in TOP and ("/" in token or "." in token)


def logical_commands(body: list[str]) -> list[tuple[str, str]]:
    """(cwd-relative-to-repo, command segment) for each command of a fence; joins `\\` lines, follows `cd`."""
    joined, pending = [], ""
    for raw in body:
        pending += raw.rstrip("\\").strip() + " "
        if not raw.rstrip().endswith("\\"):
            joined.append(pending.strip())
            pending = ""
    cwd, out = "", []
    for line in joined:
        line = re.sub(r"\s+#.*$", "", line)
        for segment in re.split(r"\s*(?:&&|;)\s*", line):
            words = segment.split()
            if len(words) == 2 and words[0] == "cd":
                cwd = os.path.normpath(os.path.join(cwd, words[1])) if not words[1].startswith(("/", "$", "~")) else ""
            elif words:
                out.append((cwd, segment))
    return out


def names_a_repo_path(word: str) -> bool:
    """A command-line word that is meant as a path inside the repository (not a flag, an absolute path or a placeholder)."""
    if word.startswith(("-", "/", "$", ".", "~")) or "<" in word or "`" in word:
        return False
    return bool((word.split("/")[0] in TOP and "/" in word) or re.search(r"\.(py|sh|cmake)$", word))


def missing_paths(segment: str, cwd: str) -> list[str]:
    """The repo paths a shown command names that do not exist (only for the programs the guides run from the repo)."""
    words = [w for w in segment.split() if not re.match(r"^\w+=", w)]
    program = os.path.basename(words[0]) if words else ""
    if not (program.startswith("python") or program in ("sh", "bash", "cmake", "ctest", "gcc", "cc")):
        return []
    named = [w.strip("\"'") for w in words[1:]]
    return [w for w in named if names_a_repo_path(w) and not exists(w, REPO, os.path.join(REPO, cwd))]


@pytest.mark.parametrize("guide", GUIDES)
def test_shown_commands_name_real_scripts_and_paths(guide: str) -> None:
    problems = []
    for first, info, body, _ in blocks(guide):
        for cwd, segment in (logical_commands(body) if info in ("sh", "bash") else []):
            problems += [f"{guide}:{first}: `{segment}` names `{w}`, which does not exist" for w in missing_paths(segment, cwd)]
    assert not problems, "\n".join(problems)


def compilable(guide: str) -> list[tuple[int, str]]:
    return [(first, "\n".join(body) + "\n") for first, info, body, before in blocks(guide)
            if info == "c" and before.strip() == "<!-- compilable -->"]


@pytest.mark.parametrize("guide", GUIDES)
def test_each_guide_keeps_a_compilable_block(guide: str) -> None:
    assert compilable(guide), f"{guide} lost its <!-- compilable --> block"


def compile_c(source: str) -> subprocess.CompletedProcess:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "snippet.c")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(source)
        return subprocess.run(["cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-I", os.path.join(REPO, "include"),
                               "-c", path, "-o", os.path.join(tmp, "snippet.o")], capture_output=True, text=True)


needs_cc = pytest.mark.skipif(not sys.platform.startswith("linux") or shutil.which("cc") is None,
                              reason="the snippet compile runs on Linux with cc")


@needs_cc
@pytest.mark.parametrize("guide", GUIDES)
def test_compilable_blocks_compile(guide: str) -> None:
    for first, source in compilable(guide):
        done = compile_c(source)
        assert done.returncode == 0, f"{guide}:{first}: snippet does not compile\n{done.stderr}"


@needs_cc
def test_compile_rejects_a_broken_snippet() -> None:
    assert compile_c("#include <seabraid/seabraid.h>\nint f(void) { return no_such_symbol; }\n").returncode != 0
