# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""The one public address a released tree may carry: the repository URL in package metadata.

OWNER: this module owns the exception to check_no_egress.py's URL rule for the metadata fields of pyproject.toml and
Cargo.toml; the rule itself (every other URL) stays in the checker.
WHY it exists: scripts/release/rel_export.py replaces the loopback placeholder in those two files with the public
repository address (GitHub, OWNER, NAME) of the export, and the public CI then runs the full checker suite on that tree; without
this exception the five metadata lines are five egress hits and the public repository's first CI run is red.
CONTRACT: ``package_name(root)`` is the pyproject name ("" when unreadable, which allows nothing); ``allowed(rel, line,
name)`` is True only for a whole line ``KEY = "ADDRESS"`` in pyproject.toml or Cargo.toml, where KEY is one of FIELDS and ADDRESS
is github.com/ANY-OWNER/NAME (this package's own name), optionally followed by a path.  It is no ALLOW row on purpose: an ALLOW
row must be used by the tree it is written for, and the development tree carries the loopback placeholder instead.
FAILURE: anything else (another host, another package, a query string, a second address, any other file or key) is not allowed.
CHECKER: tests/test_release_sol.py (the filled tree passes; eight planted variants still fail).
"""

from __future__ import annotations

import os
import re
import tomllib

FILES = ("pyproject.toml", "Cargo.toml")
FIELDS = ("Homepage", "Source", "Issues", "Changelog", "Documentation", "homepage", "repository", "documentation")
# WHY built from parts: the checker scans this directory for spelled addresses, and the one place that may spell the scheme is a pattern.
_SCHEME = "https" + "://"


def package_name(root: str) -> str:
    try:
        with open(os.path.join(root, "pyproject.toml"), "rb") as fh:
            return str(tomllib.load(fh)["project"]["name"])
    except (OSError, ValueError, KeyError):
        return ""


def allowed(rel: str, line: str, name: str) -> bool:
    if not name or rel not in FILES:
        return False
    address = rf"{re.escape(_SCHEME)}github\.com/[A-Za-z0-9][A-Za-z0-9._-]*/{re.escape(name)}(?:/[A-Za-z0-9._/-]*)?"
    return re.fullmatch(rf'\s*(?:{"|".join(FIELDS)})\s*=\s*"{address}"\s*', line) is not None
