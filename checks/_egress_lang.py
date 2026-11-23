# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Language handling for check_no_egress.py: pick a family per file, blank comments, extract patch additions.

OWNER: this module owns "which language is this file" and "what is a comment there"; patterns live in _egress_patterns.
CONTRACT: ``family_of(rel, text)``; ``strip_comments(family, text)`` keeps every newline (line numbers survive) and
every string literal (a URL inside a string is still a URL); ``patch_chunks(text)`` yields (target path, [(patch line,
added text)]) per hunk.  FAILURE: an unknown file type falls to "raw" (URLs only), never to "skip".
WHY a tokenizer regex: a string-aware strip is what stops a quoted URL being eaten as a ``//`` comment, and what
stops ``#include`` being eaten as a ``#`` comment (C-family files do not treat ``#`` as one at all).
"""

from __future__ import annotations

import os
import re

_EXT = {
    "c": (".c", ".h", ".cpp", ".hpp", ".cc", ".cxx", ".hh", ".cu", ".cuh", ".inl"),
    "rust": (".rs",),
    "py": (".py", ".pyi"),
    "sh": (".sh", ".bash"),
    "cmake": (".cmake", ".cmake.in"),
    "hash": (".toml", ".yml", ".yaml", ".txt", ".cfg", ".ini", ".lock", ".csv"),
    "doc": (".md", ".cff", ".rst"),
}
_DOC_NAMES = ("LICENSE", "NOTICE", "COPYING")
_SHEBANG = (("py", "python"), ("sh", "sh"), ("sh", "bash"), ("sh", "zsh"))
_C_TOKEN = re.compile(r"\"(?:\.|[^\"\\n])*\"|'(?:\[^'\n]+|[^'\\n])'|//[^\n]*|/\*.*?\*/", re.S)
_HASH_TOKEN = re.compile(r"\"\"\"[\s\S]*?\"\"\"|'''[\s\S]*?'''|\"(?:\.|[^\"\\n])*\"|'[^'\n]*'|(?:(?<=\s)|^)#[^\n]*", re.M)


def family_of(rel: str, text: str) -> str:
    base = os.path.basename(rel)
    if base == "CMakeLists.txt":
        return "cmake"
    if base.startswith(_DOC_NAMES):
        return "doc"
    for fam, exts in _EXT.items():
        if rel.endswith(exts):
            return fam
    if rel.endswith((".json", ".typed")):
        return "raw"
    first = text.split("\n", 1)[0]
    if first.startswith("#!"):
        return next((fam for fam, word in _SHEBANG if word in first), "sh")
    return "raw" if "." in base.lstrip(".") else "hash"


def _blank(m: re.Match) -> str:
    return re.sub(r"[^\n]", " ", m.group(0))


def strip_comments(family: str, text: str) -> str:
    """Text with comments overwritten by spaces; strings and newlines untouched."""
    if family in ("c", "rust"):
        tok, is_comment = _C_TOKEN, lambda s: s.startswith(("//", "/*"))
    elif family in ("py", "sh", "cmake", "hash"):
        tok, is_comment = _HASH_TOKEN, lambda s: s.startswith("#")
    else:
        return text
    return tok.sub(lambda m: _blank(m) if is_comment(m.group(0)) else m.group(0), text)


def patch_chunks(text: str):
    """Yield (target path, [(patch line number, added text)]) per hunk of a unified diff.

    Only '+' lines (not '+++') are code we would ship; context and removed lines are not scanned.
    """
    target, added = "", []
    for n, line in enumerate(text.split("\n"), start=1):
        if line.startswith("+++ "):
            target = re.sub(r"^b/", "", line[4:].split("\t")[0].strip())
        elif line.startswith("@@") or line.startswith("diff ") or line.startswith("From "):
            if added:
                yield target, added
            added = []
        elif line.startswith("+") and not line.startswith("+++"):
            added.append((n, line[1:]))
    if added:
        yield target, added
