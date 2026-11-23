# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""The three primitives every ``detect`` probe is built from: run a command, read
a file, parse an integer.  One owner, because a probe helper copied per module is
exactly how copies drifted -- ``_detect_host._read`` did not strip its result
while ``_detect_gpu._read`` did, and ``_detect_toolkit._run`` waited 90s and
returned the whole ``CompletedProcess`` where this one waits 20s and returns
stdout.  Both are now folded in: ``_read`` takes ``strip`` and ``_run`` takes
``cwd``, each call site passes what it needs EXPLICITLY, and no ``_detect_*``
module defines a probe helper of its own.

Failure contract: fail-open and LOGGED.  A primitive that cannot answer returns
``None``; it never raises, because a missing tool or an unreadable sysfs file is
a hardware FACT the caller reports, not an installer crash.  It does say why, in
the log: a missing command or file is DEBUG (the common, expected case on a host
without that vendor), a command that ran and failed, hung, or a file that exists
but cannot be read is WARNING.  Log text carries the command's basename and a
scrubbed first stderr line, never a full path.  Imports only ``_log``."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from typing import Optional

from ._log import describe, scrub

_log = logging.getLogger(__name__)


def _run(
    cmd: list[str], timeout: float = 20.0, cwd: Optional[str] = None
) -> Optional[str]:
    """Run a probe command.  ``None`` when it is missing, fails or hangs -- a
    probe that cannot answer reports nothing, it does not take the installer
    down with it.

    ``cwd`` exists because ``_detect_toolkit`` test-compiles a ``.cu`` file and
    must drop the ``probe.o`` it writes into a throwaway directory, not into
    wherever the installer was launched.  ``timeout`` stays a per-call argument:
    the same caller passes 90.0 for that compile, which can take far longer than
    a ``--version`` query."""
    name = os.path.basename(cmd[0])
    if shutil.which(cmd[0]) is None and not os.path.isabs(cmd[0]):
        _log.debug("probe %s: command not found", name)
        return None
    try:
        p = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False, cwd=cwd
        )
    except (OSError, subprocess.SubprocessError) as exc:
        _log.warning("probe %s did not run: %s", name, describe(exc))
        return None
    if p.returncode != 0:
        _log.warning("probe %s failed: exit %d: %s", name, p.returncode, scrub(p.stderr or "") or "(no stderr)")
        return None
    return p.stdout


def _read(path: str, strip: bool = True) -> Optional[str]:
    """Read a text file; ``None`` when it cannot be read.

    ``strip`` exists because ``_detect_host`` hands the text to line parsers
    (``parse_meminfo``, ``parse_mounts``, ``parse_os_release``) that anchor on the
    first character of a line, so it must pass ``strip=False`` to keep the file
    byte-for-byte; everyone else wants the trimmed value.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except FileNotFoundError:
        _log.debug("read %s: not present", os.path.basename(path))
        return None
    except OSError as exc:
        _log.warning("read %s failed: %s", os.path.basename(path), describe(exc))
        return None
    return text.strip() if strip else text


def _int(text: Optional[str]) -> Optional[int]:
    if text is None:
        return None
    try:
        return int(text.strip())
    except ValueError:
        _log.debug("not an integer: %r", scrub(text, 40))
        return None
