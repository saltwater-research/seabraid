# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Read-only witnesses of "the disk really was read": the model's disk counter, per-process read bytes, and the cgroup's own counters.

OWNER: this module owns every read of /proc, /sys/block and the cgroup tree.  It decides nothing (ov_gate.py judges the numbers)
and writes nothing.  The three roots are parameters so the tests can plant a fake tree and exercise the real parsing.
CONTRACT: ``Host(proc, sysfs, cgroot)``; ``Host.disk_for_dev(major, minor) -> Disk | None``; ``Host.cgroup_report(pids) -> [dict]``;
``Probe(host, disk, pids_fn).snap() -> Snap``; ``Host.cg_pids(cg)`` and ``Host.cg_kill(cg, own_cg)`` for a scope the runner started itself.  A value that cannot be read is None, never 0, so a missing witness cannot read as "no I/O".
THE WITNESSES, and what each one cannot say:
  * /proc/diskstats sectors read of the model's disk: the device really transferred them, but it is machine-wide (other processes
    on the same disk add to it), so it can only be trusted together with
  * /proc/<pid>/io read_bytes summed over the start's own processes: attributable to the engine, counts bytes fetched from the
    storage layer (page-cache hits are not counted), including mapped-file faults;
  * the cgroup's memory.stat ``pgmajfault`` (faults that needed I/O), memory.events ``max`` and ``oom_kill`` and memory.peak: the
    limit was reached, or not.  The io controller is NOT delegated to a user cgroup on the box, so there is no per-cgroup io.stat.
FAILURE: fail-open-loud: unreadable = None, and ov_gate turns a None witness into INVALID with the file named.
CHECKER: compat/strata/tests/overflow/test_overflow_proc.py (a planted procfs and cgroup tree).
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Disk:
    """The block device a model file lives on: ``name`` is its /proc/diskstats name, ``parent`` the whole disk (``rotational`` is read from it)."""

    name: str
    parent: str
    rotational: bool | None


@dataclass(frozen=True)
class Snap:
    dev_bytes: int | None
    proc_bytes: int | None
    majflt: int | None
    oom_kill: int | None
    max_hits: int | None
    peak: int | None


def _int(text: str | None) -> int | None:
    try:
        return int(text.split()[0]) if text else None
    except (ValueError, IndexError):
        return None


def _kv(text: str | None) -> dict[str, int]:
    out = {}
    for line in (text or "").splitlines():
        k, _, v = line.partition(" ")
        if v.strip().lstrip("-").isdigit():
            out[k] = int(v)
    return out


def _sum(vals: list[int | None]) -> int | None:
    return None if not vals or any(v is None for v in vals) else sum(vals)


class Host:
    def __init__(self, proc: str = "/proc", sysfs: str = "/sys", cgroot: str = "/sys/fs/cgroup"):
        self.proc, self.sysfs, self.cgroot = proc, sysfs, cgroot

    def _read(self, path: str) -> str | None:
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                return fh.read()
        except OSError:
            return None  # the caller turns a missing witness into a named INVALID reason; it is never read as zero

    def disk_by_name(self, name: str) -> Disk | None:
        """The disk (or partition) ``name`` with its whole-disk parent and rotational flag, from /sys/block (a partition is a child directory of its disk)."""
        blk = f"{self.sysfs}/block"
        try:
            parent = name if os.path.isdir(f"{blk}/{name}") else next((d for d in sorted(os.listdir(blk)) if os.path.isdir(f"{blk}/{d}/{name}")), None)
        except OSError:
            return None  # no /sys/block on this machine: the caller reports the unresolved disk by name
        if parent is None:
            return None
        rot = _int(self._read(f"{blk}/{parent}/queue/rotational"))
        return Disk(name, parent, None if rot is None else bool(rot))

    def disk_for_dev(self, major: int, minor: int) -> Disk | None:
        """The /proc/diskstats entry of device major:minor as a Disk; None when diskstats has no such line or sysfs does not know the name."""
        rows = map(str.split, (self._read(f"{self.proc}/diskstats") or "").splitlines())
        name = next((f[2] for f in rows if len(f) > 3 and f[:2] == [str(major), str(minor)]), None)
        return self.disk_by_name(name) if name else None

    def disk_read_bytes(self, name: str) -> int | None:
        for f in map(str.split, (self._read(f"{self.proc}/diskstats") or "").splitlines()):
            if len(f) > 5 and f[2] == name:
                return int(f[5]) * 512
        return None

    def proc_read_bytes(self, pid: int) -> int | None:
        return _kv((self._read(f"{self.proc}/{pid}/io") or "").replace(":", "")).get("read_bytes")

    def cgroup_of(self, pid: int) -> str | None:
        """The cgroup v2 path of a process (the ``0::`` line of /proc/<pid>/cgroup), relative to the cgroup root."""
        for line in (self._read(f"{self.proc}/{pid}/cgroup") or "").splitlines():
            if line.startswith("0::"):
                return line[3:]
        return None

    def cg_file(self, cg: str, name: str) -> str | None:
        return self._read(f"{self.cgroot}{cg}/{name}")

    def cg_pids(self, cg: str) -> list[int]:
        return [int(x) for x in (self.cg_file(cg, "cgroup.procs") or "").split()]

    def cg_kill(self, cg: str, own_cg: str | None) -> str | None:
        """SIGKILL every process of cgroup ``cg`` (cgroup.kill); returns an error text or None.  REFUSES the runner's own cgroup (the login session)."""
        name = cg.rsplit("/", 1)[-1]
        if cg == own_cg or not (name.startswith("run-") and name.endswith(".scope") and "/user@" in cg):
            return f"refusing to kill cgroup {cg!r}: not a transient run-*.scope of the user manager (the only kind systemd-run makes)"
        try:
            with open(f"{self.cgroot}{cg}/cgroup.kill", "w", encoding="utf-8") as fh:
                fh.write("1")
        except OSError as exc:
            return f"cgroup.kill of {cg} failed: {exc}"
        return None

    def comm(self, pid: int) -> str:
        return (self._read(f"{self.proc}/{pid}/comm") or "?").strip()

    def cgroup_report(self, pids: list[int]) -> list[dict]:
        """Per process: its cgroup and that cgroup's own memory.max / memory.swap.max (int bytes, the string ``max``, or None when unreadable)."""
        out = []
        for pid in pids:
            cg = self.cgroup_of(pid)
            lim = {k: (self.cg_file(cg, f).strip() if cg and self.cg_file(cg, f) else None) for k, f in (("memory_max", "memory.max"), ("swap_max", "memory.swap.max"))}
            out.append({"pid": pid, "comm": self.comm(pid), "cgroup": cg, **{k: (int(v) if v and v.isdigit() else v) for k, v in lim.items()}})
        return out

    def cg_counters(self, cgs: set[str]) -> tuple[int | None, int | None, int | None, int | None]:
        """(pgmajfault, oom_kill, ``max`` hits, peak bytes) summed (peak: maximum) over the given cgroups; None if any is unreadable or there is none."""
        stats = [_kv(self.cg_file(c, "memory.stat")) for c in cgs]
        events = [_kv(self.cg_file(c, "memory.events")) for c in cgs]
        peaks = [_int(self.cg_file(c, "memory.peak")) for c in cgs]
        majflt = _sum([s.get("pgmajfault") for s in stats])
        oom = _sum([e.get("oom_kill") for e in events])
        hits = _sum([e.get("max") for e in events])
        return majflt, oom, hits, (max(peaks) if peaks and None not in peaks else None)


class Probe:
    """Snapshots of every witness for one start.  ``pids_fn`` lists the start's own processes right now.

    WHY high-water marks per pid: a process that exits between two snapshots takes its /proc/<pid>/io with it; summing only the live
    ones would make the counter fall and a delta go negative.  Each pid keeps the largest value ever read, so the sum never falls."""

    def __init__(self, host: Host, disk: str, pids_fn):
        self.host, self.disk, self.pids_fn = host, disk, pids_fn
        self._hw: dict[int, int] = {}

    def _proc_bytes(self, pids: list[int]) -> int | None:
        for pid in pids:
            v = self.host.proc_read_bytes(pid)
            if v is not None:
                self._hw[pid] = max(self._hw.get(pid, 0), v)
        return sum(self._hw.values()) if self._hw else None

    def snap(self) -> Snap:
        pids = self.pids_fn()
        cgs = {c for c in map(self.host.cgroup_of, pids) if c}
        majflt, oom, hits, peak = self.host.cg_counters(cgs)
        return Snap(self.host.disk_read_bytes(self.disk), self._proc_bytes(pids), majflt, oom, hits, peak)
