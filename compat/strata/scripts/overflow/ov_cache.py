# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Make the OS page cache honest before a start: evict the model files, then measure how much of them is still cached.

OWNER: this module owns "which files are the model" and "how much of them is in the page cache right now".
WHY it is needed even with a cgroup limit: a cgroup charges a cache page to the group that first read it.  Pages left in the cache
by an earlier run (a previous start, a helper that read its slots, a copy) are charged elsewhere, so the engine can read them
without any charge to its own limit: those "SSD" reads would be RAM reads.  Eviction before every start, and a measured residency
that must stay under a threshold, close that hole.  (The device-read counters in ov_gate.py are the second, independent witness.)
CONTRACT: ``model_files(native, pack) -> [path]``; ``evict(files) -> [error]`` (posix_fadvise DONTNEED on each file: it drops clean,
unmapped pages and needs no privilege); ``resident_bytes(files) -> int | None`` (util-linux ``fincore``; None when it cannot run);
``preflight(files, max_resident_mb) -> dict``.
FAILURE: eviction errors and a missing fincore are returned in the result, never swallowed; ov_gate.py decides what they mean.
CHECKER: compat/strata/tests/overflow/test_overflow_proc.py (monkeypatched fadvise and fincore runner).
"""

from __future__ import annotations

import glob
import os
import subprocess

DONTNEED = getattr(os, "POSIX_FADV_DONTNEED", 4)  # 4 on Linux; the attribute is missing off Linux, where fadvise is injected by the tests


def model_files(native: str, pack: str) -> list[str]:
    """Every GGUF shard next to the native file, plus the pack's dense and experts files when they exist (what the engine and helper read)."""
    files = sorted(glob.glob(os.path.join(os.path.dirname(native), "*.gguf"))) or [native]
    files += [p for p in (os.path.join(pack, "experts.bin"), os.path.join(pack, "dense.bin")) if os.path.isfile(p)]
    return files


def evict(files: list[str], fadvise=None) -> list[str]:
    """Ask the kernel to drop the cached pages of each file; returns one error string per file that could not be advised."""
    fadvise = fadvise or getattr(os, "posix_fadvise", None)
    if fadvise is None:
        return ["os.posix_fadvise is not available on this platform: the page cache was NOT evicted"]
    errors = []
    for path in files:
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError as exc:
            errors.append(f"{path}: {exc}")
            continue
        try:
            fadvise(fd, 0, 0, DONTNEED)
        except OSError as exc:
            errors.append(f"{path}: {exc}")
        finally:
            os.close(fd)
    return errors


def resident_bytes(files: list[str], run=subprocess.run) -> int | None:
    """Bytes of ``files`` currently in the page cache (``fincore --bytes --noheadings -o RES``); None if fincore is missing or fails."""
    try:
        p = run(["fincore", "--bytes", "--noheadings", "-o", "RES", *files], capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    if p.returncode != 0:
        return None
    try:
        return sum(int(x) for x in p.stdout.split())
    except ValueError:
        return None


def preflight(files: list[str], max_resident_mb: float, fadvise=None, run=subprocess.run) -> dict:
    """Evict, then measure.  ``ok`` is True only when fincore ran and the residency is at most ``max_resident_mb``; ``errors`` names every failure."""
    errors = evict(files, fadvise)
    res = resident_bytes(files, run)
    if res is None:
        errors.append("fincore could not measure the residency of the model files (the device-read witnesses stay as the only proof)")
    mb = None if res is None else res / 1e6
    return {"files": len(files), "resident_mb": mb, "max_resident_mb": max_resident_mb, "errors": errors, "ok": mb is not None and mb <= max_resident_mb and not errors}
