# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Host probes for :mod:`detect`: memory, filesystems, the GPU-mappable host
memory ceiling, the memlock limit and the kernel.

A sibling of :mod:`detect`, not a second owner: these return plain values and
``detect.Filesystem``, and define no type of their own.

Every parser takes text or a filesystem root as an argument, so the tests drive
them from output recorded on a real machine while running on one without a GPU.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Optional

from ._log import describe
from ._probe import _read

_log = logging.getLogger(__name__)

_PSEUDO = {
    "autofs", "bpf", "cgroup", "cgroup2", "configfs", "debugfs", "devpts",
    "devtmpfs", "efivarfs", "fuse.portal", "fusectl", "hugetlbfs", "mqueue",
    "proc", "pstore", "ramfs", "securityfs", "selinuxfs", "sysfs", "tracefs",
}


# --------------------------------------------------------------------- memory


def parse_meminfo(text: str) -> tuple[int, int]:
    """``/proc/meminfo`` -> (total bytes, available bytes).

    ``MemAvailable`` rather than ``MemFree``: free excludes reclaimable page
    cache, and sizing a RAM tier off it under-uses the machine by gigabytes.
    """
    vals: dict[str, int] = {}
    for line in text.splitlines():
        m = re.match(r"(\w+):\s+(\d+)\s*kB", line)
        if m:
            vals[m.group(1)] = int(m.group(2)) * 1024
    return vals.get("MemTotal", 0), vals.get("MemAvailable", vals.get("MemFree", 0))


def probe_ram() -> tuple[int, int]:
    text = _read("/proc/meminfo", strip=False)  # raw: parse_meminfo is line-anchored
    if text is None:
        from .detect import DetectionRefused  # noqa: PLC0415

        raise DetectionRefused(
            "/proc/meminfo could not be read, so this machine cannot be described at all. "
            "The seabraid installer targets Linux; on Windows or macOS, run it inside the "
            "Linux environment that will run the engine."
        )
    return parse_meminfo(text)


def probe_memlock() -> Optional[int]:
    """``ulimit -l`` in bytes; ``None`` means unlimited.

    It is reported because a RAM tier that pins its pages cannot exceed it, and
    the usual default is a few megabytes -- small enough that the failure looks
    like a mysterious allocation error rather than a limit.
    """
    try:
        import resource  # noqa: PLC0415 -- POSIX only, and this probe is POSIX only

        soft, _hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    except (ImportError, OSError, ValueError) as exc:
        _log.warning("memlock limit unreadable, reporting it as unlimited-or-unknown: %s", describe(exc))
        return None
    return None if soft == resource.RLIM_INFINITY else int(soft)


def probe_ttm_pages_limit(sys_root: str = "/sys") -> Optional[int]:
    """The system-wide ceiling on host memory any GPU may map, in bytes.

    TTM states it in pages, so it is multiplied by the page size here.  This is
    NOT the same number as a card's own ``mem_info_gtt_total``, even when they
    happen to be equal on a one-GPU machine -- the safety check for a RAM tier
    has to respect both, so both are detected.
    """
    text = _read(os.path.join(sys_root, "module/ttm/parameters/pages_limit"))
    if text is None:
        return None
    try:
        pages = int(text.strip())
    except ValueError:
        _log.warning("ttm pages_limit is not a number; the host-memory ceiling is unknown")
        return None
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError, OSError) as exc:
        _log.warning("page size unreadable (%s); assuming 4096", describe(exc))
        page_size = 4096
    return pages * page_size


# ---------------------------------------------------------------- filesystems


def parse_mounts(text: str) -> list[tuple[str, str, str]]:
    """``/proc/mounts`` -> [(device, mount, fstype)], real storage and tmpfs only.

    tmpfs is kept, not dropped, because ``/dev/shm`` is where the helper channel
    lives and its size is a real constraint; it is marked by its fstype so that
    nothing mistakes it for a place to put a model.

    READ-ONLY MOUNTS ARE DROPPED, which is what makes the report readable.  The
    first run listed dozens of filesystems, nearly all read-only
    squashfs images and credential mounts, each one printed as "UNKNOWN
    (cannot hold model files)".  A report nobody can scan cannot be checked by
    hand, and being checkable by hand is the whole job of stage 1.  "Read-only"
    is the honest rule rather than a filename filter: nothing can be installed
    onto a filesystem that cannot be written.
    """
    out: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        device, mount, fstype = parts[0], parts[1].replace("\\040", " "), parts[2]
        options = parts[3].split(",")
        if fstype in _PSEUDO or "ro" in options:
            continue
        if not (device.startswith("/dev/") or fstype == "tmpfs"):
            continue
        if mount in seen:
            continue
        seen.add(mount)
        out.append((device, mount, fstype))
    return out


def rotational_for(device: str, sys_root: str = "/sys") -> Optional[bool]:
    """Is this device backed by spinning media?  ``None`` when unresolvable.

    Resolves partition -> parent disk, because ``queue/rotational`` lives on the
    disk and not on the partition.  It deliberately gives up on anything it
    cannot follow (LVM, md, mapper, network and loop devices): an unresolved
    answer must stay unresolved, since the caller treats ``None`` as a refusal
    and a wrong ``False`` here puts model files on a hard disk.
    """
    if not device.startswith("/dev/"):
        return None
    name = device[len("/dev/"):]
    if name.startswith(("mapper/", "dm-", "md", "loop")):
        return None
    block = os.path.join(sys_root, "class/block", name)
    if not os.path.exists(block):
        return None
    real = os.path.realpath(block)
    if os.path.exists(os.path.join(real, "partition")):
        real = os.path.dirname(real)
    text = _read(os.path.join(real, "queue/rotational"))
    if text is None:
        return None
    value = text.strip()
    return True if value == "1" else (False if value == "0" else None)


def probe_filesystems(blockers: list, sys_root: str = "/sys") -> tuple:
    from .detect import Blocker, Filesystem  # noqa: PLC0415

    text = _read("/proc/mounts", strip=False)  # raw: parse_mounts is line-anchored
    if text is None:
        blockers.append(
            Blocker(
                where="fs",
                what="/proc/mounts could not be read, so no filesystem could be sized",
                fix="run the installer on the Linux machine that will run the engine",
            )
        )
        return ()

    out: list = []
    for device, mount, fstype in parse_mounts(text):
        try:
            st = os.statvfs(mount)
        except OSError as exc:
            _log.warning("filesystem %s could not be sized, left out of the report: %s", os.path.basename(mount) or "/", describe(exc))
            continue
        total = st.f_blocks * st.f_frsize
        if total <= 0:
            continue
        out.append(
            Filesystem(
                mount=mount,
                device=device,
                fstype=fstype,
                total_bytes=total,
                free_bytes=st.f_bavail * st.f_frsize,
                rotational=(False if fstype == "tmpfs" else rotational_for(device, sys_root)),
            )
        )
    out.sort(key=lambda fs: fs.mount)
    return tuple(out)


# --------------------------------------------------------------------- kernel


def parse_os_release(text: str) -> Optional[str]:
    m = re.search(r'^PRETTY_NAME="?([^"\n]+)"?', text, re.M)
    return m.group(1) if m else None


def probe_kernel() -> tuple[str, str, bool, Optional[str]]:
    """(kernel release, kernel cmdline, /dev/kfd present, OS pretty name).

    The cmdline is carried verbatim because the fix for a too-small GTT ceiling
    is a boot parameter, and the installer must be able to show what is already
    there before proposing a change it will never apply itself.
    """
    release = (_read("/proc/sys/kernel/osrelease") or "").strip()
    cmdline = (_read("/proc/cmdline") or "").strip()
    os_release = _read("/etc/os-release", strip=False)  # raw: PRETTY_NAME is ^-anchored
    pretty = parse_os_release(os_release) if os_release else None
    return release, cmdline, os.path.exists("/dev/kfd"), pretty
