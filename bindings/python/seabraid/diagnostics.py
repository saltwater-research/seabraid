# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Version and logging bridge: native log records flow into Python's logging module.

OWNER   : version(), enable_logging(), disable_logging(); the one module-level reference that keeps the ctypes
          log callback alive for as long as the native library may call it.
CONTRACT: version() -> str (native version when a native library is loaded and exposes seabraid_version, else the package
          version); enable_logging(min_level="WARNING", library_path=None) -> bool; disable_logging() -> None.
          Records arrive on logger "seabraid" with the ABI status in record.seabraid_status. Imports: stdlib + siblings.
SIZE    : under 150 lines.
FAILURE : a library without seabraid_set_log_sink raises RuntimeError; a nonzero ABI status raises SeabraidError; no native
          library available is logged as a warning and returns False (reference mode has nothing to forward);
          an exception inside the callback is reported by ctypes through sys.unraisablehook, never swallowed.
CHECKER : tests/test_diagnostics.py (mocked libraries: install, forward, restore, two libraries, absent symbols).
FINISH  : those tests pass.
"""

from __future__ import annotations

import atexit
import ctypes as ct
import logging
from pathlib import Path

from . import _abi
from ._version import package_version, pep440
from .constants import LOG_DEBUG, LOG_ERROR, LOG_INFO, LOG_OFF, LOG_WARN
from .errors import SeabraidError

_LOG = logging.getLogger("seabraid")
_NATIVE_TO_PY = {LOG_DEBUG: logging.DEBUG, LOG_INFO: logging.INFO, LOG_WARN: logging.WARNING, LOG_ERROR: logging.ERROR}
_NAMES = {"DEBUG": LOG_DEBUG, "INFO": LOG_INFO, "WARN": LOG_WARN, "WARNING": LOG_WARN, "ERROR": LOG_ERROR,
          "OFF": LOG_OFF}
# Python level thresholds, ascending; the first one at or above the value picks the native level.
_PY_THRESHOLDS = ((logging.DEBUG, LOG_DEBUG), (logging.INFO, LOG_INFO), (logging.WARNING, LOG_WARN),
                  (logging.ERROR, LOG_ERROR))

# Every library a sink is installed on, with its callback. The callback object MUST stay referenced here: if it were
# collected the native side would call freed memory. WHY a list and not one slot: a sink on library B must not make
# library A forget it still forwards, or disable_logging and the exit hook would leave A calling into Python.
# Identity, not equality, picks the entry: a library handle is the thing the native side is attached to.
_installed: list[tuple[ct.CDLL, object]] = []
# Every callback ever installed, kept for the process lifetime. WHY: after a replacement a native thread may still be
# inside the previous sink (the native side does not wait for in-flight calls), so freeing it could crash that thread.
# The leak is one small ctypes object per enable_logging call.
_retired: list[object] = []
# WHY a flag: the hook must be registered once however many enable/disable cycles run; atexit has no "is registered".
_exit_hook_registered = False


def version() -> str:
    """The native library's version when one is loaded and reports it, else the package version; always PEP 440."""
    library = _abi.active_library()
    function = getattr(library, "seabraid_version", None) if library is not None else None
    raw = function() if function is not None else None
    return pep440(raw.decode("utf-8", "replace")) if raw else package_version()


def _native_level(level: int | str) -> int:
    if isinstance(level, str):
        try:
            return _NAMES[level.upper()]
        except KeyError:
            raise ValueError("min_level must be DEBUG, INFO, WARNING, ERROR, OFF or a logging level") from None
    return next((native for limit, native in _PY_THRESHOLDS if level <= limit), LOG_OFF)


def _forward(_context: int | None, level: int, status: int, message: bytes | None) -> None:
    text = message.decode("utf-8", "replace") if message else ""
    _LOG.log(_NATIVE_TO_PY.get(level, logging.ERROR), "%s", text, extra={"seabraid_status": status})


def _set_sink(library: ct.CDLL, callback: object, level: int) -> None:
    function = getattr(library, "seabraid_set_log_sink", None)
    if function is None:
        raise RuntimeError("This Seabraid library does not expose logging")
    # WHY a null LogCallback and not None: ctypes accepts None for a function-pointer argtype only on some platforms/versions,
    # and the real library refused it on Linux (found by the native test), so "restore the default sink" is a NULL pointer.
    status = function(_abi.LogCallback() if callback is None else callback, None, level)
    if status != 0:
        raise SeabraidError("seabraid_set_log_sink", status, _abi.last_error(library))


def enable_logging(min_level: int | str = "WARNING", library_path: str | Path | None = None) -> bool:
    """Forward native records at or above min_level to logging.getLogger("seabraid"); True when installed."""
    global _exit_hook_registered
    level = _native_level(min_level)
    library = _abi.resolve_library(library_path)
    if library is None:
        _LOG.warning("enable_logging: no native Seabraid library is loaded, so there are no native records to forward")
        return False
    callback = _abi.LogCallback(_forward)
    _retired.append(callback)  # before the install: the native side may call it as soon as it is set
    _set_sink(library, callback, level)
    if not _exit_hook_registered:
        atexit.register(_disable_at_exit)
        _exit_hook_registered = True
    _installed[:] = [entry for entry in _installed if entry[0] is not library] + [(library, callback)]
    return True


def disable_logging() -> None:
    """Restore the native default sink on every library a sink was installed on and release the Python callbacks."""
    # Restore first; the callbacks stay in _retired because a call already in flight may still be running one.
    # WHY every library is tried before an error is raised: one library failing to detach must not leave the others
    # forwarding. A library that failed stays tracked so a later disable_logging retries it.
    failed: list[tuple[ct.CDLL, object]] = []
    first_error: Exception | None = None
    for entry in _installed:
        try:
            _set_sink(entry[0], None, LOG_WARN)  # level is only a hint to the stderr default; WARN is its usual floor
        except (SeabraidError, RuntimeError) as error:
            failed.append(entry)
            first_error = first_error or error
    _installed[:] = failed
    if first_error is not None:
        raise first_error


def _disable_at_exit() -> None:
    # A native thread logging into a finalising interpreter would crash, so detach; report, never raise, at exit.
    try:
        disable_logging()
    except (SeabraidError, RuntimeError) as error:
        _LOG.error("could not restore the native log sink at exit: %s", error)
