# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Logging for the installer: one configuration point and the rule that logs carry no full paths.

OWNER     : the installer's log format, level and scrubbing.  Every module takes its own
            ``logging.getLogger(__name__)``; only ``seabraid_setup.main`` calls ``configure``.
CONTRACT  : ``configure(verbose)`` (idempotent), ``scrub(text, limit)``, ``describe(exc)``.  Leaf: imports nothing of ours.
FAILURE   : never raises; a log call must not take the installer down.
CHECKER   : tests/install/test_hardening.py (path scrubbing, exception descriptions, one handler after two configure calls).

PRIVACY   : logs may be pasted into bug reports, so they never carry a full path, a user name, a host name or an
            environment value.  User-facing messages (the plan, refusals that tell an operator what to fix) stay on
            the existing echo/print channel; this module is for DIAGNOSTICS.
"""

from __future__ import annotations

import logging
import re
import sys

ROOT_LOGGER = "install"
_PATHISH = re.compile(r"(?:[A-Za-z]:)?[\\/](?:[^\s'\"\\/:]+[\\/])*([^\s'\"\\/:]*)")


def scrub(text: str, limit: int = 160) -> str:
    """``text`` with every path cut to its basename, one line, at most ``limit`` characters.

    WHY a regex over whitespace tokens: tool stderr and OSError messages are free text, so we cannot know where a
    path starts; a path always contains a separator, and the basename is the part that helps a reader."""
    first = (text or "").strip().splitlines()[0] if (text or "").strip() else ""
    out = _PATHISH.sub(lambda m: m.group(1) or "<dir>", first)
    return out if len(out) <= limit else out[: limit - 3] + "..."


def describe(exc: BaseException) -> str:
    """A path-free description: ``FileNotFoundError: No such file or directory`` rather than the message that names the file."""
    if isinstance(exc, OSError) and exc.strerror:
        return "%s: %s" % (type(exc).__name__, exc.strerror)
    return "%s: %s" % (type(exc).__name__, scrub(str(exc)))


def configure(verbose: bool = False) -> None:
    """Send the installer's diagnostics to stderr: WARNING and up, or DEBUG with ``verbose``.  Safe to call twice."""
    log = logging.getLogger(ROOT_LOGGER)
    log.setLevel(logging.DEBUG if verbose else logging.WARNING)
    if not any(getattr(h, "_seabraid", False) for h in log.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("seabraid_setup: %(levelname)s %(name)s: %(message)s"))
        handler._seabraid = True  # type: ignore[attr-defined]  # marks our handler so a second configure() adds no duplicate
        log.addHandler(handler)
