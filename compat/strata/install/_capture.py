# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""seabraid installer helper: run one command, read its output, stop it at a marker line.  UNTESTED-ON-HARDWARE.

OWNER     : the only place a child process is read-until-marker-then-stopped (build.Runner.run_capture delegates here).
CONTRACT  : Captured(rc, text, found, timed_out);  capture(cmd_argv, cwd, env, until, timeout) -> Captured.  stdlib only.
FAILURE   : never raises on the child's behaviour; a child that cannot start surfaces as rc=None, found=False and the
            OSError text in `text`.
CHECKER   : tests/install/test_capture.py runs real python children (marker then sleep; exit 3; silent sleep).
WHY       : seabraid-helper prints its bench line and then keeps SERVING, so waiting for exit would hang forever.
"""
from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

from ._log import describe

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Captured:
    rc: Optional[int]       # None = still alive when we stopped it
    text: str               # merged stdout+stderr
    found: bool             # a line containing `until` appeared
    timed_out: bool         # the deadline passed before it did (and the child was alive)


def capture(argv: Sequence[str], cwd: Optional[str], env: Mapping, until: str, timeout: float) -> Captured:
    full = {**os.environ, **{k: str(v) for k, v in env.items()}}
    try:
        p = subprocess.Popen(list(argv), cwd=cwd, env=full, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, errors="replace")
    except OSError as e:
        _log.warning("cannot start %s: %s", os.path.basename(argv[0]), describe(e))
        return Captured(None, f"cannot start {argv[0]}: {e}", False, False)
    lines, seen = [], threading.Event()

    def pump() -> None:
        for ln in p.stdout:
            lines.append(ln)
            if until in ln:
                seen.set()
    t = threading.Thread(target=pump, daemon=True)
    t.start()
    end = time.monotonic() + timeout
    while not seen.is_set() and p.poll() is None and time.monotonic() < end:
        seen.wait(0.05)
    alive = p.poll() is None
    if alive:
        p.terminate()                       # SIGTERM first: the helper unlinks its /dev/shm channel on it
        try:
            p.wait(10)
        except subprocess.TimeoutExpired:
            _log.warning("%s ignored SIGTERM for 10 s; killing it", os.path.basename(argv[0]))
            p.kill()
            p.wait()
    t.join(2)
    found = seen.is_set()
    return Captured(None if alive else p.returncode, "".join(lines), found, alive and not found)
