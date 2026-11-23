# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Token, key and personal-address scan of a directory tree, plus the redaction shapes the release script prints with.

OWNER: this module is the only definition of "what a credential looks like" for the release tooling; rel_core imports
SHAPES to redact subprocess output, so the scan and the redaction cannot disagree.
CONTRACT: ``scan_tree(root) -> list[str]`` returns ``path:line: kind`` (and ``path: kind`` for a forbidden file name),
NEVER the matched text: a report that quoted the secret would itself be a leak.  Walks the files on disk, so it works on
an export before it is a git repository.  Imports nothing from the release package (one-way edge).
FAILURE: fail-fast and loud: any hit makes the caller refuse; a file that cannot be read is a hit, never skipped
(binary files, NUL in the first 8 KiB, are skipped: they hold no text a token could be read from).
CHECKER: tests/test_release_publish.py plants one token of each family and a forbidden file name.
"""

from __future__ import annotations

import fnmatch
import os
import re

# (kind, regex).  WHY shapes and not entropy: an entropy scan flags every hash in a lock file and gets muted; a shape
# list names the credentials the owner actually has accounts for (PyPI, crates.io, GitHub, Hugging Face, ...).
SHAPES: list[tuple[str, re.Pattern]] = [
    ("pypi-token", re.compile(r"pypi-[A-Za-z0-9_-]{40,}")),
    ("github-token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})")),
    ("crates-token", re.compile(r"\bcio[A-Za-z0-9]{30,}")),
    ("huggingface-token", re.compile(r"\bhf_[A-Za-z0-9]{30,}")),
    ("anthropic-or-openai-key", re.compile(r"\bsk-(?:ant-)?[A-Za-z0-9_-]{32,}")),
    ("aws-access-key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("slack-token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}")),
    ("private-key-block", re.compile(r"-----BEGIN (?:[A-Z]+ )*PRIVATE KEY-----")),
    # WHY a digit AND a letter, no "/" or ".": a path or a file name is not a secret ("tokenizer=/data/x", "secret = lib.so").
    ("credential-assignment", re.compile(r"(?i)(?:api[_-]?key|secret|token|passw(?:or)?d)(?![a-z])[\"']?\s*[:=]\s*[\"'](?=[A-Za-z0-9_+=-]*\d)(?=[A-Za-z0-9_+=-]*[A-Za-z])[A-Za-z0-9_+=-]{24,}[\"']")),
]
# File NAMES that mean credentials whatever they contain.
FORBIDDEN_NAMES = (".env", ".env.*", "*.pem", "*.p12", "*.pfx", ".pypirc", "credentials", "credentials.toml", "id_rsa", "id_ed25519", ".netrc")
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,})")
# WHY these domains and suffixes: the project's own commit address (the GitHub noreply form) and reserved test domains
# are not identifiers; ".service" and friends are systemd instance names (x@0.service), not mail domains.
EMAIL_OK = re.compile(r"(?:^|\.)(?:users\.noreply\.github\.com|example\.(?:com|org|net|invalid)|invalid|test|localhost|service|socket|target|timer)$", re.I)
# WHY one exact address and not its domain: the owner decided on 2026-11-21 that this project is published as
# dev@saltyresearch.com (public commit identity and package contact); any other address at that domain is still flagged.
PUBLIC_CONTACT = "dev@saltyresearch.com"
_SKIP_DIRS = {".git"}


def line_kinds(line: str) -> list[str]:
    kinds = [kind for kind, rx in SHAPES if rx.search(line)]
    kinds += ["email-address" for m in EMAIL.finditer(line) if not EMAIL_OK.search(m.group(1)) and m.group(0).lower() != PUBLIC_CONTACT][:1]
    return kinds


def scan_tree(root: str, files: list[str] | None = None) -> list[str]:
    """Hits for every file under ``root``, or only for ``files`` (repo-relative; the dev tree passes ``git ls-files`` so a
    venv or a build directory is never read)."""
    hits: list[str] = []
    for rel in files if files is not None else walk(root):
        if any(fnmatch.fnmatch(os.path.basename(rel), g) for g in FORBIDDEN_NAMES):
            hits.append(f"{rel}: forbidden file name (credentials)")
        else:
            hits += scan_file(os.path.join(root, rel), rel)
    return hits


def walk(root: str) -> list[str]:
    found = []
    for base, dirs, names in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS)
        found += [os.path.relpath(os.path.join(base, n), root).replace(os.sep, "/") for n in sorted(names)]
    return found


def scan_file(path: str, rel: str) -> list[str]:
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
    except OSError as exc:
        return [f"{rel}: unreadable ({exc.strerror}); cannot prove it holds no secret"]
    if b"\0" in raw[:8192]:
        return []
    out = []
    for n, line in enumerate(raw.decode("utf-8", "replace").splitlines(), start=1):
        out += [f"{rel}:{n}: {kind}" for kind in line_kinds(line)]
    return out
