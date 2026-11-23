# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""GPU probes for :mod:`detect`.  A sibling, not a second owner: every function
here returns ``detect.Gpu`` values and defines no type of its own.

Each vendor is probed by the cheapest thing that is ACTUALLY PRESENT, which is
not the same as the obvious tool:

* NVIDIA -- ``nvidia-smi``, which ships with the driver, so if a card works the
  tool is there.
* AMD -- the kernel, via ``/sys/class/kfd`` and ``/sys/class/drm``.  NOT
  ``amd-smi`` or ``rocminfo``: on a host that runs an AMD card as
  its main engine, neither may be on PATH and there is no ``/opt/rocm`` at all.  A
  probe that needed them would have reported no AMD GPU on such a host.  They are enrichment (the marketing name), never a
  requirement.
* Intel -- ``sycl-ls`` when present; otherwise not reported.

Every parser is a pure function over text or over a filesystem root, so the
tests drive them from recorded output with no hardware present.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Optional

from ._detect_holders import _holder_tuple, kfd_holders, nvidia_holders
from ._detect_pci import (
    _link_from_sysfs,
    narrowest_link,
    normalise_bus_id,
    pci_chain_links,
    pci_device_name,
)
from ._log import describe
from ._probe import _int, _read, _run

_log = logging.getLogger(__name__)

_SYS_DRM = "/sys/class/drm"
_SYS_KFD = "/sys/class/kfd/kfd/topology/nodes"

# ------------------------------------------------------------------- NVIDIA

NVIDIA_QUERY = (
    "index,name,memory.total,memory.free,pci.bus_id,pcie.link.gen.max,"
    "pcie.link.gen.current,pcie.link.width.max,pcie.link.width.current,"
    "compute_cap,driver_version,uuid"
)

def parse_nvidia_smi(text: str) -> list[dict]:
    """Parse ``nvidia-smi --query-gpu=NVIDIA_QUERY --format=csv,noheader``.

    Pure, so the tests can feed it recorded output.  A field nvidia-smi prints
    as ``[N/A]`` or ``[Not Supported]`` becomes ``None`` rather than a guess.
    """
    rows: list[dict] = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 12:
            continue

        def val(i: int) -> Optional[str]:
            v = parts[i]
            return None if not v or v.startswith("[") else v

        def mib(i: int) -> Optional[int]:
            v = val(i)
            if v is None:
                return None
            m = re.match(r"(\d+)", v)
            return int(m.group(1)) * 1024 * 1024 if m else None

        idx = _int(parts[0])
        if idx is None:
            continue
        cc = val(9)
        rows.append(
            {
                "index": idx,
                "name": val(1),
                "vram_total_bytes": mib(2),
                "vram_free_bytes": mib(3),
                "pci_bus_id": normalise_bus_id(val(4)),
                "pcie_gen_max": _int(val(5)),
                "pcie_gen_current": _int(val(6)),
                "pcie_width_max": _int(val(7)),
                "pcie_width_current": _int(val(8)),
                "arch": ("sm_" + cc.replace(".", "")) if cc else None,
                "driver_version": val(10),
                # The UUID, not the ordinal, is the unambiguous way to pin a
                # CUDA process to a card: CUDA_VISIBLE_DEVICES accepts it, and
                # it is immune to CUDA_DEVICE_ORDER, which does NOT default to
                # PCI bus order.
                "uuid": val(11),
            }
        )
    return rows

def nvidia_gpus() -> list[dict]:
    text = _run(["nvidia-smi", "--query-gpu=" + NVIDIA_QUERY, "--format=csv,noheader"])
    rows = parse_nvidia_smi(text) if text else []
    # Read in the SAME probe as vram_free_bytes.  A holder list fetched a moment later can disagree with
    # the free figure it is meant to explain -- which is exactly how a busy machine was once mistaken for
    # an arithmetic bug.  Same moment, or the pair is not evidence.
    holders = nvidia_holders()
    for row in rows:
        # Matched on the row's own UUID.  A row without one gets NO holders rather than the machine's
        # whole list: a process blamed on the wrong card reads as evidence, so it is worse than silence.
        row["holders"] = _holder_tuple(holders.get(row.get("uuid") or "", {}))
        # nvidia-smi reports the card's own link, which overstates a card behind a switch exactly as the
        # AMD path did.  sysfs is the only place the hops above it are visible.
        bus = row.get("pci_bus_id")
        if not bus:
            continue
        dev = os.path.join("/sys/bus/pci/devices", bus)
        if not os.path.isdir(dev):
            continue
        hop = narrowest_link(pci_chain_links(dev))
        if hop is not None:
            row["pcie_gen_max"], row["pcie_width_max"], row["pcie_gen_current"], row["pcie_width_current"] = hop
    return rows

# ----------------------------------------------------------------------- AMD

def gfx_from_target_version(version: int) -> Optional[str]:
    """``gfx_target_version`` -> the gfx name the compiler wants.

    The encoding is major*10000 + minor*100 + step, and the STEP IS HEX in the
    name: 90010 is gfx90a, not gfx9010.  Getting this wrong produces a plausible
    string that no compiler accepts, so it is decoded here once and nowhere else.
    """
    if version <= 0:
        return None
    major, minor, step = version // 10000, (version // 100) % 100, version % 100
    if step > 15:
        return None
    return "gfx%d%d%x" % (major, minor, step)

def parse_kfd_properties(text: str) -> dict:
    """``/sys/class/kfd/kfd/topology/nodes/N/properties`` -> a dict of ints."""
    props: dict = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2:
            v = _int(parts[1])
            if v is not None:
                props[parts[0]] = v
    return props

def _kfd_nodes(sys_kfd: str):
    """The KFD node names in numeric order, or None when there is no KFD (no AMD compute stack)."""
    try:
        return sorted(os.listdir(sys_kfd), key=lambda n: _int(n) if _int(n) is not None else 0)
    except OSError as exc:
        _log.debug("no KFD nodes: %s", describe(exc))
        return None


def _gpu_node_props(sys_kfd: str, node: str):
    """The node's KFD properties when it is a GPU; None for the CPU node (simd_count 0) or an unreadable one."""
    text = _read(os.path.join(sys_kfd, node, "properties"))
    props = parse_kfd_properties(text) if text else None
    return props if props and props.get("simd_count", 0) > 0 else None


def _render_device(props: dict, sys_drm: str):
    """The DRM device directory tied to this KFD node through drm_render_minor, or None when it is absent."""
    minor = props.get("drm_render_minor")
    dev = os.path.join(sys_drm, "renderD%d" % minor, "device") if minor else None
    return dev if dev is not None and os.path.isdir(dev) else None


def _pci_bus_id(dev):
    if dev is None:
        return None
    real = os.path.realpath(dev)
    return os.path.basename(real) if re.search(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.\d$", real) else None


def _amd_link(dev) -> tuple:
    """(gen_max, width_max, gen_cur, width_cur) of the narrowest hop up to the root complex, not the card's own link: see pci_chain_links."""
    if not dev:
        return None, None, None, None
    gen_max, width_max = _link_from_sysfs(dev, "max")
    gen_cur, width_cur = _link_from_sysfs(dev, "current")
    hop = narrowest_link(pci_chain_links(dev))
    return hop if hop is not None else (gen_max, width_max, gen_cur, width_cur)


def _amd_row(index: int, props: dict, dev, gpu_id: int) -> dict:
    gen_max, width_max, gen_cur, width_cur = _amd_link(dev)
    sysfs = lambda name: _int(_read(os.path.join(dev, name))) if dev else None  # noqa: E731
    vram_total, vram_used = sysfs("mem_info_vram_total"), sysfs("mem_info_vram_used")
    return {
        "index": index,
        "name": amd_name(dev, props),
        "arch": gfx_from_target_version(props.get("gfx_target_version", 0)),
        "vram_total_bytes": vram_total,
        "vram_free_bytes": vram_total - vram_used if vram_total is not None and vram_used is not None else None,
        "gtt_total_bytes": sysfs("mem_info_gtt_total"),
        "pci_bus_id": _pci_bus_id(dev),
        "pcie_gen_max": gen_max,
        "pcie_width_max": width_max,
        "pcie_gen_current": gen_cur,
        "pcie_width_current": width_cur,
        "driver_version": _read("/sys/module/amdgpu/version"),
        "holders": _holder_tuple(kfd_holders(gpu_id)) if gpu_id else (),
    }


def amd_gpus(sys_drm: str = _SYS_DRM, sys_kfd: str = _SYS_KFD) -> list[dict]:
    """Every AMD compute GPU, from the kernel alone.

    Roots are arguments so the tests point them at a recorded ``/sys`` tree.
    A KFD node with ``simd_count`` 0 is the CPU node, not a GPU, and is skipped.
    The node is tied to its DRM device through ``drm_render_minor``, which is the
    only link that stays correct when card numbering and KFD node order disagree.
    """
    found: list[dict] = []
    for node in _kfd_nodes(sys_kfd) or ():
        props = _gpu_node_props(sys_kfd, node)
        if props is None:
            continue
        gpu_id = _int(_read(os.path.join(sys_kfd, node, "gpu_id"))) or 0
        found.append(_amd_row(len(found), props, _render_device(props, sys_drm), gpu_id))
    return found


# Where distributions keep the shared PCI id database.  Read rather than vendored: a table of our own
# would need maintaining for every card released after we wrote it, and would be wrong by default.

def amd_name(dev: Optional[str], props: dict) -> Optional[str]:
    """A human name for an AMD card, or ``None``.

    The KFD node's own ``name`` file says "ip discovery" on current kernels, which
    is not a product name, so it is not used.  Without ROCm there is no marketing
    name available at all; the PCI id is reported instead so the line still
    identifies the card, and PLAN never reads this field.
    """
    vendor_id = props.get("vendor_id")
    device_id = props.get("device_id")
    if dev is not None and not device_id:
        pci = _read(os.path.join(dev, "device"))
        if pci:
            try:
                device_id = int(pci, 16)
            except ValueError:
                _log.warning("pci device id is not hexadecimal; the card is reported without a name")
                device_id = None
    if not device_id:
        return None
    named = pci_device_name(vendor_id or 0x1002, device_id)
    # The id is kept even when a name was found: it is what identifies the card in a bug report, and the
    # marketing name alone has sent people to the wrong driver before.
    return "%s [%04x:%04x]" % (named, vendor_id or 0x1002, device_id) if named \
        else "AMD GPU %04x:%04x" % (vendor_id or 0x1002, device_id)

# --------------------------------------------------------------------- Intel

def parse_sycl_ls(text: str) -> list[dict]:
    """Parse ``sycl-ls``.  Only Level-Zero GPU devices count: the OpenCL and CPU
    backends list the same hardware again, and a duplicated card would be planned
    as two."""
    rows: list[dict] = []
    for line in text.strip().splitlines():
        if "level_zero:gpu" not in line:
            continue
        m = re.search(r"\]\s*(.+?)\s*(?:\d+\.\d+)?\s*$", line)
        rows.append({"index": len(rows), "name": m.group(1).strip() if m else None})
    return rows

def intel_gpus() -> list[dict]:
    text = _run(["sycl-ls"])
    return parse_sycl_ls(text) if text else []

# ------------------------------------------------------------------ assembly

def probe_gpus(blockers: list) -> tuple:
    """Every GPU of every vendor.  Appends a ``Blocker`` for a card detected
    without an arch, because an arch is what a build is parameterised by: an
    unknown arch is an unknown that feeds a decision, so it refuses its own
    path rather than letting a default be guessed."""
    from .detect import Blocker, Gpu  # noqa: PLC0415 -- cycle: detect owns the types

    out: list = []

    def add(vendor: str, probe: str, row: dict) -> None:
        g = Gpu(
            index=row["index"],
            vendor=vendor,
            name=row.get("name"),
            arch=row.get("arch"),
            vram_total_bytes=row.get("vram_total_bytes"),
            vram_free_bytes=row.get("vram_free_bytes"),
            gtt_total_bytes=row.get("gtt_total_bytes"),
            pci_bus_id=row.get("pci_bus_id"),
            pcie_gen_max=row.get("pcie_gen_max"),
            pcie_width_max=row.get("pcie_width_max"),
            pcie_gen_current=row.get("pcie_gen_current"),
            pcie_width_current=row.get("pcie_width_current"),
            driver_version=row.get("driver_version"),
            probe=probe,
            uuid=row.get("uuid"),
            holders=row.get("holders", ()),
        )
        out.append(g)
        if g.arch is None:
            blockers.append(
                Blocker(
                    where="gpu:%s:%d" % (vendor, g.index),
                    what="this %s GPU was found but its architecture could not be read, "
                    "and a build is compiled for an architecture" % vendor,
                    fix="install or update the vendor driver and toolkit, then retry detection",
                )
            )

    for row in nvidia_gpus():
        add("nvidia", "nvidia-smi", row)
    for row in amd_gpus():
        add("amd", "kfd-sysfs", row)
    for row in intel_gpus():
        add("intel", "sycl-ls", row)

    return tuple(out)
