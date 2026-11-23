# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""What this machine actually is: the probe stage of the seabraid installer.

OWNER: this module owns the concept "detected host".  ``HostReport`` and its parts
(``Gpu``, ``Toolkit``, ``Filesystem``, ``Blocker``) are defined here and nowhere
else; ``plan.py`` and ``config.py`` import them from here.  Nothing in seabraid is
imported back -- the dependency edge runs one way only:

    detect <- plan <- config <- build/services <- seabraid_setup

WHY NOT extend Strata's own probe: upstream ``setup.py`` has ``gpus()``, but it
shells ``nvidia-smi`` and nothing else, and it lives in an 11,000-line script we
fetch at build time and must not fork.  We shell out to that script for the model
download / pack / tokenizer -- the part worth reusing -- and detect hardware here.

FAILURE CONTRACT -- fail-open-LOUDLY, then refuse in one place.  A probe never
raises over a hardware or toolchain fact: it records a ``Blocker`` and the report
prints it, because a probe that raises cannot print the report that tells you how
to fix it.  ``HostReport.require_ok()`` is the single site that turns blockers
into ``DetectionRefused``, and the caller decides when that happens.  Cosmetic
fields (marketing name, PCIe link, driver version) degrade to ``None`` and print
as "unknown"; a ``None`` is never silently defaulted into a decision.

``DetectionRefused`` is also raised directly by a probe for a host we cannot
describe at all (``/proc/meminfo`` unreadable) -- that is not a finding about the
machine, it is a broken probe.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass
from typing import Optional

from ._detect_gpu import probe_gpus
from ._detect_toolkit import probe_toolkits
from ._detect_host import (
    probe_filesystems,
    probe_kernel,
    probe_memlock,
    probe_ram,
    probe_ttm_pages_limit,
)

__all__ = [
    "Blocker",
    "DetectionRefused",
    "Filesystem",
    "Gpu",
    "HostReport",
    "Toolkit",
    "detect_host",
    "format_report",
]


class DetectionRefused(Exception):
    """Raised by ``HostReport.require_ok()``, and by a probe that cannot describe
    the host at all.  Never raised merely because one fact was missing."""


@dataclass(frozen=True, slots=True)
class Gpu:
    """One GPU, as that vendor's own tooling numbers it.

    ``index`` is per-vendor: NVIDIA 0 and AMD 0 are two different cards.  PCIe is
    four fields because an idle link downclocks -- a card can report gen
    max 3 / current 1 and width max 16 / current 8 for a card with nothing running
    on it, so ranking cards by the current link would rank them by idleness.  Plan
    off the ``_max`` fields; the ``_current`` pair is for the printed report.
    """

    index: int
    vendor: str
    name: Optional[str]
    arch: Optional[str]
    vram_total_bytes: Optional[int]
    vram_free_bytes: Optional[int]
    gtt_total_bytes: Optional[int]
    pci_bus_id: Optional[str]
    pcie_gen_max: Optional[int]
    pcie_width_max: Optional[int]
    pcie_gen_current: Optional[int]
    pcie_width_current: Optional[int]
    driver_version: Optional[str]
    probe: str
    uuid: Optional[str] = None
    # (pid, bytes, name) for every process holding memory on this card, biggest first, read in the SAME
    # probe as vram_free_bytes.  It exists so that "this card has no room" can be told apart from "the
    # sizing is wrong" at a glance -- that ambiguity has already cost one wrong diagnosis.
    holders: tuple = ()


@dataclass(frozen=True, slots=True)
class Toolkit:
    """A compiler toolkit that can build for one vendor.

    ``host_compiler`` is populated ONLY by a compile that returned 0.  A version
    table would have said a recent gcc was fine on a host with CUDA 12.4, which
    rejects anything past gcc 13, so the table would have planned a build that
    cannot compile.  ``host_compiler_tried`` keeps the candidates so the report
    can show which were rejected.
    """

    vendor: str
    kind: str
    root: Optional[str]
    version: Optional[str]
    compiler: Optional[str]
    host_compiler: Optional[str] = None
    host_compiler_tried: tuple[str, ...] = ()
    lib_dirs: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Filesystem:
    """A mounted filesystem, considered as a place a model or a tier could live.

    ``rotational`` is ``None`` when it could not be resolved -- which is a refusal,
    not a default.  It resolves mount -> device -> parent block device, and that
    chain is what breaks on LVM and md, so ``device`` is carried too.  ``fstype``
    is carried because tmpfs mounts (``/dev/shm``, where the helper channel lives)
    must never be mistaken for storage.
    """

    mount: str
    device: Optional[str]
    fstype: str
    total_bytes: int
    free_bytes: int
    rotational: Optional[bool]


@dataclass(frozen=True, slots=True)
class Blocker:
    """Something that makes a path unplannable, with the action that clears it.

    ``where`` IS the structured key, and its grammar is ``<kind>:<ref>``:

        toolkit:cuda   toolkit:rocm   toolkit:sycl
        gpu:<vendor>:<index>
        fs:<mount>     fs            (the whole probe failed)

    It is one field rather than two because a second field holding the same
    information is a second definition, and the two drift.  ``kind`` and ``ref``
    read the parts, so no caller has to know where the colon is.

    ``fix`` is mandatory and must be actionable: a blocker the reader cannot act
    on is a crash with extra steps.
    """

    where: str
    what: str
    fix: str

    @property
    def kind(self) -> str:
        """The part before the first colon: what sort of thing is blocked."""
        return self.where.split(":", 1)[0]

    @property
    def ref(self) -> str:
        """Everything after the first colon, or "" when there is none."""
        parts = self.where.split(":", 1)
        return parts[1] if len(parts) == 2 else ""


@dataclass(frozen=True, slots=True)
class HostReport:
    """Everything detected, including what could not be."""

    gpus: tuple[Gpu, ...] = ()
    toolkits: tuple[Toolkit, ...] = ()
    filesystems: tuple[Filesystem, ...] = ()
    ram_total_bytes: int = 0
    ram_available_bytes: int = 0
    ttm_pages_limit_bytes: Optional[int] = None
    memlock_bytes: Optional[int] = None
    kernel_release: str = ""
    kernel_cmdline: str = ""
    kfd_present: bool = False
    os_pretty: Optional[str] = None
    blockers: tuple[Blocker, ...] = ()

    def require_ok(self) -> None:
        """The single place a blocker becomes a refusal.  Callers (PLAN, or the
        CLI once it has printed the report) decide when that happens."""
        if not self.blockers:
            return
        lines = [f"  - [{b.where}] {b.what}\n      fix: {b.fix}" for b in self.blockers]
        raise DetectionRefused(
            "this machine cannot be planned as it stands:\n" + "\n".join(lines)
        )

    def gpus_of(self, vendor: str) -> tuple[Gpu, ...]:
        return tuple(g for g in self.gpus if g.vendor == vendor)

    def toolkit_for(self, vendor: str) -> Optional[Toolkit]:
        for t in self.toolkits:
            if t.vendor == vendor:
                return t
        return None

    def filesystem_at(self, mount: str) -> Optional[Filesystem]:
        for fs in self.filesystems:
            if fs.mount == mount:
                return fs
        return None

    def filesystem_holding(self, path: str) -> Optional[Filesystem]:
        """The filesystem a path lives on: the longest mount point that prefixes
        it.  PLAN needs this to ask whether the model files sit on rotational
        storage, which is the real NVMe-tier question -- native packs read the
        GGUF in place, so what matters is where the pack already is."""
        best: Optional[Filesystem] = None
        for fs in self.filesystems:
            if path == fs.mount or path.startswith(fs.mount.rstrip("/") + "/"):
                if best is None or len(fs.mount) > len(best.mount):
                    best = fs
        return best


def detect_host(
    rocm_path: Optional[str] = None, cuda_path: Optional[str] = None
) -> HostReport:
    """Probe everything and return the report.  Never raises over a missing fact.

    ``rocm_path`` / ``cuda_path`` let the caller supply a root the search cannot
    guess -- ROCm may sit under a data mount, reachable from no
    standard location and absent from PATH.
    """
    blockers: list[Blocker] = []

    gpus = probe_gpus(blockers)
    toolkits = probe_toolkits(gpus, blockers, rocm_path=rocm_path, cuda_path=cuda_path)
    filesystems = probe_filesystems(blockers)
    ram_total, ram_avail = probe_ram()
    kernel_release, kernel_cmdline, kfd_present, os_pretty = probe_kernel()

    return HostReport(
        gpus=gpus,
        toolkits=toolkits,
        filesystems=filesystems,
        ram_total_bytes=ram_total,
        ram_available_bytes=ram_avail,
        ttm_pages_limit_bytes=probe_ttm_pages_limit(),
        memlock_bytes=probe_memlock(),
        kernel_release=kernel_release,
        kernel_cmdline=kernel_cmdline,
        kfd_present=kfd_present,
        os_pretty=os_pretty,
        blockers=tuple(blockers),
    )


# ------------------------------------------------------------------ the report


def _size(n: Optional[int]) -> str:
    """Bytes in the unit a reader can check against the machine.  A memlock of
    8 MiB printed as "0.01 GiB" is a number nobody can verify at a glance."""
    if n is None:
        return "unknown"
    if n >= 1024 ** 3:
        return "%.2f GiB" % (n / (1024.0 ** 3))
    if n >= 1024 ** 2:
        return "%.0f MiB" % (n / (1024.0 ** 2))
    return "%d B" % n


def _link(gen: Optional[int], width: Optional[int]) -> str:
    if gen is None and width is None:
        return "unknown"
    g = "?" if gen is None else str(gen)
    w = "?" if width is None else str(width)
    return "gen%s x%s" % (g, w)


def _gpu_lines(g: Gpu) -> list[str]:
    out = [
        "  [%s:%d] %s  (%s)" % (g.vendor, g.index, g.name or "unknown name", g.arch or "UNKNOWN ARCH"),
        "      VRAM      %s total, %s free" % (_size(g.vram_total_bytes), _size(g.vram_free_bytes)),
    ]
    if g.gtt_total_bytes is not None:
        out.append(
            "      GTT       %s (host memory this card may map)" % _size(g.gtt_total_bytes)
        )
    link_max = _link(g.pcie_gen_max, g.pcie_width_max)
    link_now = _link(g.pcie_gen_current, g.pcie_width_current)
    out.append(
        "      PCIe      %s" % link_max
        if link_now in (link_max, "unknown")
        else "      PCIe      %s max, %s now (links downclock when idle)" % (link_max, link_now)
    )
    out.append(
        "      bus       %s   driver %s"
        % (g.pci_bus_id or "unknown", g.driver_version or "unknown")
    )
    if g.holders:
        who = ", ".join("%s(%d) %s" % (n, p, _size(b)) for p, b, n in g.holders[:4])
        out.append("      in use by %s" % who)
    out.append("      detected by %s" % g.probe)
    return out


def _toolkit_lines(t: Toolkit) -> list[str]:
    out = ["  [%s] %s  root %s" % (t.kind, t.version or "unknown version", t.root or "NOT FOUND")]
    if t.compiler:
        out.append("      compiler       %s" % t.compiler)
    if t.kind == "cuda":
        if t.host_compiler:
            out.append(
                "      host compiler  %s  (verified: it compiled a test file)" % t.host_compiler
            )
        else:
            out.append("      host compiler  NONE WORKED")
        if t.host_compiler_tried:
            out.append("      tried          %s" % ", ".join(t.host_compiler_tried))
    for d in t.lib_dirs:
        out.append("      lib dir        %s" % d)
    return out


def _fs_line(fs: Filesystem) -> str:
    if fs.rotational is None:
        kind = "UNKNOWN (cannot hold model files)"
    elif fs.rotational:
        kind = "ROTATIONAL (cannot hold model files)"
    else:
        kind = "solid state"
    return "  %-24s %11s free of %-11s  %-8s %s" % (
        fs.mount,
        _size(fs.free_bytes),
        _size(fs.total_bytes),
        fs.fstype,
        kind,
    )


def _host_lines(r: HostReport) -> list:
    """The OS, kernel, RAM, memlock, ttm and KFD lines."""
    return [
        "OS       %s" % (r.os_pretty or platform.platform()),
        "Kernel   %s" % r.kernel_release,
        "RAM      %s total, %s available" % (_size(r.ram_total_bytes), _size(r.ram_available_bytes)),
        "memlock  %s   (ulimit -l; pinned host memory cannot exceed this)"
        % ("unlimited" if r.memlock_bytes is None else _size(r.memlock_bytes)),
        "ttm cap  %s   (system-wide ceiling on host memory any GPU may map)"
        % (_size(r.ttm_pages_limit_bytes) if r.ttm_pages_limit_bytes is not None else "unknown"),
        "KFD      %s   (/dev/kfd, the AMD compute device)" % ("present" if r.kfd_present else "absent"),
    ]


def _blocker_lines(r: HostReport) -> list:
    if not r.blockers:
        return ["No blockers."]
    out = ["BLOCKERS (%d) - these stop a plan, and each says how to clear it" % len(r.blockers)]
    for b in r.blockers:
        out += ["  [%s] %s" % (b.where, b.what), "      fix: %s" % b.fix]
    return out


def format_report(r: HostReport) -> str:
    """The plain report stage 1 prints.  Every unknown is shown as an unknown --
    the point of the report is that a human can check it line by line."""
    out: list[str] = ["", "=" * 74, "seabraid - what this machine is", "=" * 74, ""]
    out.extend(_host_lines(r))

    out.append("")
    out.append("GPUs (%d)" % len(r.gpus))
    if not r.gpus:
        out.append("  none found")
    for g in r.gpus:
        out.extend(_gpu_lines(g))

    out.append("")
    out.append("Toolkits (%d)" % len(r.toolkits))
    if not r.toolkits:
        out.append("  none found")
    for t in r.toolkits:
        out.extend(_toolkit_lines(t))

    out.append("")
    out.append("Filesystems (%d)" % len(r.filesystems))
    for fs in r.filesystems:
        out.append(_fs_line(fs))

    out.append("")
    out.extend(_blocker_lines(r))
    out.append("")
    return "\n".join(out)
