# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Resolve the package version from installed metadata, else from the repository VERSION file.

OWNER   : how Python learns its own version string (the VERSION file at the repository root is the source of truth).
CONTRACT: package_version() -> str; pep440(text) -> str. Every version string the package reports (this one, and the
          native library's through diagnostics.version()) is in the PEP 440 form the README shows ("0.5.0a2"); the
          repository VERSION file and the native seabraid_version() write the semver form ("0.5.0-alpha.2"), and
          pep440() is the one place that converts it. Imports: stdlib only.
SIZE    : under 70 lines.
FAILURE : with neither metadata nor a VERSION file (a loose copy of the sources) it returns "0+unknown" and logs a
          warning on logger "seabraid"; it never raises, because a version string must not break an import.
CHECKER : tests/test_diagnostics.py (metadata path, VERSION fallback, unknown path).
FINISH  : those tests pass and the built wheel reports the version in the VERSION file, in PEP 440 form.
"""

from __future__ import annotations

import logging
import re
from importlib import metadata
from pathlib import Path

_LOG = logging.getLogger("seabraid")
_DISTRIBUTION = "seabraid"
# bindings/python/seabraid/_version.py -> repository root, in a source checkout only.
_VERSION_FILE = Path(__file__).resolve().parent.parent.parent.parent / "VERSION"


_SEMVER_PRE = re.compile(r"^(\d+(?:\.\d+)*)-(alpha|beta|rc)\.(\d+)$")
_PRE_LETTER = {"alpha": "a", "beta": "b", "rc": "rc"}


def pep440(text: str) -> str:
    """"0.5.0-alpha.2" -> "0.5.0a2" (also beta -> b, rc -> rc); any other text is returned unchanged.

    WHY unchanged on a miss: a build-tagged or unknown form is the library's own word, and guessing at it would
    invent a version nobody released.
    """
    match = _SEMVER_PRE.match(text)
    return f"{match[1]}{_PRE_LETTER[match[2]]}{match[3]}" if match else text


def package_version() -> str:
    try:
        return metadata.version(_DISTRIBUTION)
    except metadata.PackageNotFoundError:
        _LOG.debug("seabraid is not installed; reading VERSION from the source checkout")
    try:
        return pep440(_VERSION_FILE.read_text(encoding="utf-8").strip())
    except OSError as error:
        _LOG.warning("Seabraid version unavailable: no package metadata and no VERSION file (%s)", type(error).__name__)
        return "0+unknown"
