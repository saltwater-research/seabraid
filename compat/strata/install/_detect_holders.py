# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""WHO is holding memory on a card, per vendor.

This exists so that "this card has no room right now" can be told apart from
"the sizing is wrong".  Without it, a 57 MiB free reading was forwarded as a
PLAN arithmetic bug when the truth was that two test arms held both cards.

Keyed per card, never by ordinal: ``--query-compute-apps`` numbers nothing, so
NVIDIA results are keyed by GPU UUID and the caller matches its own row's UUID,
and AMD is read from the per-``gpu_id`` file.  A process attributed to the wrong
card reads as evidence, which is worse than no answer at all.

Failure contract: fail-open and empty.  No holder list is a normal answer -- an
idle machine has none -- so a missing tool, a missing KFD or an unparsable line
yields ``{}`` and never an exception.  Imports only ``_probe``."""

from __future__ import annotations

import logging
import os
import re

from ._log import describe
from ._probe import _int, _read, _run

_log = logging.getLogger(__name__)


def _holder_tuple(holders: dict) -> tuple:
    """{pid: (bytes, name)} -> ((pid, bytes, name), ...), biggest first.

    One definition for both vendors: the report truncates to the top few, so the ordering is part of
    the contract and must not be re-stated per probe.
    """
    return tuple(sorted(((p, b, n) for p, (b, n) in holders.items()), key=lambda h: -h[1]))


def parse_compute_apps(text: str) -> dict:
    """--query-compute-apps=gpu_uuid,pid,used_memory,process_name -> {uuid: {pid: (bytes, name)}}.

    Keyed by UUID, never by index: this query numbers nothing, and attributing one card's processes
    to another is the same collision that once made the AMD card answer for the NVIDIA helper.
    Pure, so the tests drive it from recorded output.
    """
    out: dict = {}
    for line in text.strip().splitlines():
        # Bounded split: only the first three fields are positional.  A command line may contain
        # commas, and splitting on every one then re-joining the remainder rewrites the process name
        # (it cost the space in "--arg, with-comma").  The name is taken verbatim instead.
        parts = line.split(",", 3)
        if len(parts) < 4:
            continue
        uuid, pid = parts[0].strip(), _int(parts[1])
        m = re.match(r"(\d+)", parts[2].strip())
        if not uuid or uuid.startswith("[") or pid is None or not m:
            continue
        out.setdefault(uuid, {})[pid] = (int(m.group(1)) * 1024 * 1024, parts[3].strip())
    return out


def nvidia_holders() -> dict:
    text = _run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory,process_name",
                 "--format=csv,noheader"])
    return parse_compute_apps(text) if text else {}


def kfd_holders(gpu_id: int, sys_kfd_proc: str = "/sys/class/kfd/kfd/proc") -> dict:
    """{pid: (bytes, name)} holding memory on one AMD GPU, from the KFD's own per-process accounting.

    The kernel states this per (pid, gpu) in vram_<gpu_id>, which is the only place AMD attributes it
    to a process -- there is no --query-compute-apps equivalent.

    The value is BYTES, page-aligned.  Checked against the kernel source rather than assumed, because
    no HSA process was running to measure: kfd_process.c prints
    ``atomic64_read(&pdd->vram_usage)`` with no conversion, and kfd_chardev.c's alloc ioctl adds
    ``PAGE_ALIGN(args->size)`` to that counter.  The unit matters -- the report formats it as a size.
    """
    out: dict = {}
    try:
        pids = os.listdir(sys_kfd_proc)
    except OSError as exc:
        _log.debug("no KFD per-process accounting: %s", describe(exc))
        return out
    for entry in pids:
        pid = _int(entry)
        if pid is None:
            continue
        used = _int(_read(os.path.join(sys_kfd_proc, entry, "vram_%d" % gpu_id)))
        if not used:
            continue
        name = _read("/proc/%d/comm" % pid) or "pid %d" % pid
        out[pid] = (used, name)
    return out
