# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Who may reach the seabraid server: the loopback rule and the API-key file.

OWNER     : "is this bind address exposed, and if so is there a key" -- the one place that decides it.  services.py asks
            this module; it never re-implements the test.  Strata owns the authentication itself (its ``--api-key`` /
            ``STRATA_API_KEY``); this module only refuses to START an exposed server without one.
CONTRACT  : ``is_loopback_host(host)``, ``require_key_for_host(host, key_file)``, ``read_key_file(path) -> str``,
            ``key_wrapper_argv(argv, key_file) -> list``.  All raise ``ExposureError`` with the fix in the message.  Leaf.
FAILURE   : fail fast, before anything is written or started: a server bound beyond loopback with no key would serve the
            model (and every prompt it receives) to anyone who can reach the port.
CHECKER   : tests/install/test_exposure.py.
WHY A FILE: a key on the command line shows in ``ps`` and shell history, and a key written into a unit file is readable
            by whoever can read the unit.  The file stays where the user put it (mode 0600); the unit only names it, and
            the server wrapper reads it at start-up into the environment of that one process.
"""

from __future__ import annotations

import errno
import ipaddress
import logging
import os
import stat
from typing import Sequence

_log = logging.getLogger(__name__)

MAX_KEY_FILE_BYTES = 4096   # a key is a line; reading more means the wrong file
MIN_KEY_CHARS = 16          # short enough to type, long enough that guessing it over a LAN is not a plan


class ExposureError(Exception):
    pass


# WHY here: _exposure owns "is this bind exposed"; the server speaks plain HTTP (no TLS is added), so the key is readable on
# the wire to anyone on the path.  One sentence, used by the log line and the setup summary.
CLEARTEXT_NOTE = ("The API key is sent in clear over plain HTTP on this bind address; put a TLS reverse proxy "
                  "(nginx, caddy) in front of it, or keep --host on loopback and tunnel with SSH.")


def cleartext_warning(host: str) -> str | None:
    """CLEARTEXT_NOTE for a non-loopback ``host``, None for loopback."""
    return None if is_loopback_host(host) else CLEARTEXT_NOTE


def is_loopback_host(host: str) -> bool:
    """True only for ``localhost`` and loopback IP literals (127.0.0.0/8, ::1).  ``0.0.0.0``, ``::``, an empty string and
    every other name are NOT loopback: a name can resolve anywhere, and the wildcard addresses listen on every interface."""
    h = (host or "").strip().strip("[]").lower()
    if h == "localhost":
        return True
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


def _check_fd(fd: int, name: str, posix: bool) -> None:
    """Regular file, private, ours -- judged on the OPEN descriptor, so the file checked is the file read (no check-then-open gap)."""
    st = os.fstat(fd)
    if not stat.S_ISREG(st.st_mode):
        raise ExposureError("--api-key-file: %s is not a regular file" % name)
    if not posix:
        _log.warning("api key file permissions are not checked on this platform")
        return
    if st.st_mode & 0o077:
        raise ExposureError("--api-key-file: %s is readable by other users (mode %o). Fix: chmod 600 %s" % (name, st.st_mode & 0o777, name))
    if hasattr(os, "getuid") and st.st_uid != os.getuid():
        raise ExposureError("--api-key-file: %s is not owned by you; copy it to a file you own with mode 600" % name)


def _check_key(key: str, name: str) -> None:
    if len(key) < MIN_KEY_CHARS:
        raise ExposureError("--api-key-file: the key in %s is shorter than %d characters" % (name, MIN_KEY_CHARS))
    if any(c.isspace() or ord(c) < 0x21 or ord(c) == 0x7F for c in key):
        raise ExposureError("--api-key-file: %s must hold one key on one line, with no spaces or control characters" % name)


def _open_key(path: str, name: str) -> int:
    """Open without following a final symlink.  WHY O_NOFOLLOW: a key path that is a symlink could be swapped to any file;
    refusing it removes that, and fstat on the descriptor (not stat on the path) removes the check/open race."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        return os.open(path, flags)
    except OSError as exc:
        if exc.errno in (errno.ELOOP, getattr(errno, "EMLINK", -1)):
            _log.error("api key file is a symlink: %s", name)
            raise ExposureError("--api-key-file: %s is a symbolic link; pass the real file's path" % name) from None
        _log.error("api key file unreadable: %s (%s)", name, type(exc).__name__)
        raise ExposureError("--api-key-file: cannot read %s: %s" % (name, exc.strerror or type(exc).__name__)) from None


def read_key_file(path: str, posix: bool = os.name == "posix") -> str:
    """The key in ``path`` (one line).  Refuses a symlink, a file other users can read, one we do not own, an empty or short key,
    and a key that is not a single printable token -- each with the command or edit that fixes it.  The key is never logged.
    ``posix`` is a seam so the permission rules are testable on any host."""
    name = os.path.basename(path) or "<key file>"
    fd = _open_key(path, name)
    try:
        _check_fd(fd, name, posix)
        with os.fdopen(fd, "r", encoding="utf-8") as fh:
            fd = -1                                 # fdopen owns it now
            key = fh.read(MAX_KEY_FILE_BYTES + 1).strip()
    except (OSError, UnicodeDecodeError) as exc:
        _log.error("api key file unreadable: %s (%s)", name, type(exc).__name__)
        raise ExposureError("--api-key-file: cannot read %s: %s" % (name, getattr(exc, "strerror", None) or type(exc).__name__)) from None
    finally:
        if fd >= 0:
            os.close(fd)
    if len(key) > MAX_KEY_FILE_BYTES:
        raise ExposureError("--api-key-file: %s is larger than %d bytes" % (name, MAX_KEY_FILE_BYTES))
    _check_key(key, name)
    return key


def require_key_for_host(host: str, key_file: str | None) -> None:
    """Refuse a non-loopback ``host`` unless ``key_file`` names a valid key file.  Loopback needs nothing."""
    if is_loopback_host(host):
        if key_file:
            read_key_file(key_file)             # a key was asked for: validate it even though loopback does not need one
        return
    if not key_file:
        raise ExposureError(
            "--host %s is reachable from other machines, and the server would answer anyone who can reach it. "
            "Pass --api-key-file PATH (a file you own, mode 600, holding one key of %d+ characters) to serve beyond this PC, "
            "or leave --host at 127.0.0.1." % (host, MIN_KEY_CHARS))
    read_key_file(key_file)


# WHY a sh wrapper and not an EnvironmentFile: the key file stays the user's own file (no second copy of the secret anywhere),
# and nothing is interpolated into the script -- the key path and the real command arrive as positional arguments.
_WRAPPER = 'STRATA_API_KEY=$(cat -- "$1") || exit 1; export STRATA_API_KEY; shift; exec "$@"'


def key_wrapper_argv(argv: Sequence[str], key_file: str) -> list:
    """``argv`` run with STRATA_API_KEY read from ``key_file`` by /bin/sh at start-up (the key never appears in an argument)."""
    return ["/bin/sh", "-c", _WRAPPER, "seabraid-server", key_file, *argv]
