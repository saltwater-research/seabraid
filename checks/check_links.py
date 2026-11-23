#!/usr/bin/env python3
# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Every relative link and #anchor in a tracked .md file points at something that exists.

OWNER: this checker owns "what a working doc link is"; nothing else in the repo resolves Markdown links.
CONTRACT: inline ``[t](target)``, images, reference definitions ``[x]: target`` and HTML ``href=``/``src=`` attributes
outside code fences (any indentation; a fence closes only on the same character, at least as long) and code spans.  A
target's balanced parentheses stay in the path and a ``?query`` is dropped before the file lookup.
http(s):, mailto: and any other scheme are skipped (no network, ever).  A path must name a tracked file or a directory
holding tracked files, resolved from the linking file's directory (a leading ``/`` means the repo root).  A ``#anchor`` into
a .md file must equal a GitHub heading slug of that file (lower-case, punctuation and emphasis ``_``/``*`` dropped, spaces
to ``-``, repeats get ``-1``, ``-2``) or an explicit ``id=``/``name=``; anchors into other files (``#L10``) are not checked.  Setext (underlined)
headings are not recognised.  SIZE: <=150 lines.  FAILURE: fail-fast and loud; exit 1 and ``path:line: broken link
(target): reason`` per link.  CHECKER: tests/test_checkers.py (slug and parsing rules) and the sabotage cases
links-missing-file and links-missing-anchor.
Run:  python3 checks/check_links.py [--root DIR]
"""

from __future__ import annotations

import argparse
import os
import posixpath
import re
import sys
from urllib.parse import unquote

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as common  # noqa: E402

FENCE = re.compile(r"^\s*(`{3,}|~{3,})(.*)$")
CODE_SPAN = re.compile(r"`+[^`]*`+")
INLINE = re.compile(r"\]\(\s*")
HTML_ATTR = re.compile(r"""\b(?:href|src)\s*=\s*["']([^"']+)["']""", re.IGNORECASE)
REFDEF = re.compile(r"^\s{0,3}\[[^\]]+\]:\s*(<[^>]+>|\S+)")
SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.*?)\s*#*\s*$")
HTML_ANCHOR = re.compile(r"""<a\s[^>]*?(?:id|name)\s*=\s*["']([^"']+)["']""", re.IGNORECASE)


def slug(heading: str) -> str:
    """GitHub's heading id: markup removed, lower-cased, everything but word characters, space and ``-`` dropped."""
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", heading)  # a link keeps only its text
    text = re.sub(r"<[^>]+>|[`*]", "", text)
    text = re.sub(r"(?<!\w)_+|_+(?!\w)", "", text)  # emphasis underscores go; snake_case keeps its own
    return re.sub(r"[^\w\- ]", "", text.lower()).replace(" ", "-")


def scan_lines(text: str):
    """Yield ``(line_number, line)`` for every line outside a fenced code block."""
    fence = None  # (character, length) of the open fence
    for number, line in enumerate(text.splitlines(), 1):
        m = FENCE.match(line)
        if fence is None and m:
            fence = (m.group(1)[0], len(m.group(1)))
        elif fence and m and m.group(1)[0] == fence[0] and len(m.group(1)) >= fence[1] and not m.group(2).strip():
            fence = None
        elif fence is None:
            yield number, line


def anchors(text: str) -> set[str]:
    found, seen = set(), {}
    for _, line in scan_lines(text):
        found.update(a.lower() for a in HTML_ANCHOR.findall(line))
        m = HEADING.match(line)
        if m:
            base = slug(m.group(1))
            count = seen.get(base, 0)
            seen[base] = count + 1
            found.add(base if count == 0 else f"{base}-{count}")
    return found


def inline_target(line: str, start: int) -> str:
    """The destination after ``](`` at ``start``: ``<...>`` as written, else up to the first space or unbalanced ``)``."""
    if line.startswith("<", start):
        return line[start + 1 : line.find(">", start)]
    depth = 0
    for i in range(start, len(line)):
        depth += {"(": 1, ")": -1}.get(line[i], 0)
        if depth < 0 or line[i].isspace():
            return line[start:i]
    return line[start:]


def targets(text: str):
    """Yield ``(line_number, target)`` for each link; code spans are blanked first so a quoted ``[x](y)`` is not a link."""
    for number, line in scan_lines(text):
        clean = CODE_SPAN.sub("", line)
        for m in INLINE.finditer(clean):
            yield number, inline_target(clean, m.end())
        for m in list(REFDEF.finditer(clean)) + list(HTML_ATTR.finditer(clean)):
            yield number, m.group(1).strip("<>")


def judge(rel: str, target: str, files: set[str], dirs: set[str], texts: dict[str, str]) -> str | None:
    """The reason a link is broken, or None.  ``texts`` caches the .md files read for their anchors."""
    if SCHEME.match(target):
        return None
    path, _, fragment = target.partition("#")
    path = path.partition("?")[0]  # a query string selects a view of the same file on GitHub
    if not path:
        dest = rel
    else:
        path = unquote(path)
        dest = posixpath.normpath(path[1:] if path.startswith("/") else posixpath.join(posixpath.dirname(rel), path))
        if dest == ".." or dest.startswith("../"):
            return "leaves the repository"
        if dest not in files and dest not in dirs:
            return "no such file or directory"
    if fragment and dest in files and dest.endswith(".md"):
        if unquote(fragment).lower() not in anchors(texts[dest]):
            return f"no heading or anchor '#{fragment}' in {dest}"
    return None


def scan(root: str) -> list[str]:
    files = set(common.tracked(root))
    dirs = {"."}
    for f in files:  # every ancestor of a tracked file is a directory (git tracks no empty ones)
        parts = f.split("/")[:-1]
        dirs.update("/".join(parts[:i]) for i in range(1, len(parts) + 1))
    texts = {f: (common.read_text(root, f) or "") for f in sorted(files) if f.endswith(".md")}
    hits = []
    for rel, text in texts.items():
        for number, target in targets(text):
            why = judge(rel, target, files, dirs, texts)
            if why:
                hits.append(f"{rel}:{number}: broken link ({target}): {why}")
    return hits


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--root", default=common.default_root())
    root = ap.parse_args(argv).root
    hits = scan(root)
    n = sum(1 for p in common.tracked(root) if p.endswith(".md"))
    return common.finish("check_links", hits, [f"note: {n} markdown file(s) checked"])


if __name__ == "__main__":
    sys.exit(main())
