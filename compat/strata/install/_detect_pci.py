# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""PCI facts that belong to no vendor: how a bus id is spelled, how fast a link
is, and what a device id is called.

Owner of two things that were each got wrong once:

* The LIMITING link.  A card's own ``max_link_speed`` describes its hop to
  whatever sits directly above it, which on some systems is the card's own
  internal switch -- gen5 x16 while the root port above it is gen4 x8.  Reading
  the card's own link overstated both cards 4x, and PLAN ranks cards on this
  field.  ``pci_chain_links`` walks to the root complex; ``narrowest_link``
  picks the hop by BANDWIDTH, since gen5 x8 carries more than gen3 x16.
* The NAME.  Read from the distribution's ``pci.ids`` rather than a table of our
  own, which would be wrong by default for every card released after we wrote
  it.

Failure contract: fail-open.  Every function returns ``None`` or an empty list
rather than raising; an unreadable sysfs attribute is a fact, not an error.
Imports only ``_probe``."""

from __future__ import annotations

import logging
import os
import re
from typing import Optional

from ._log import describe
from ._probe import _int, _read

_log = logging.getLogger(__name__)


def normalise_bus_id(raw: Optional[str]) -> Optional[str]:
    """PCI addresses to one spelling: ``0000:0e:00.0``.

    nvidia-smi prints an eight-digit domain (``00000000:0E:00.0``) and sysfs a
    four-digit one, so the same slot had two spellings and nothing downstream
    could compare an NVIDIA card to an AMD one by address.
    """
    if not raw:
        return None
    m = re.match(r"^\s*([0-9a-fA-F]+):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.(\d)\s*$", raw)
    if not m:
        return raw.strip().lower() or None
    return "%04x:%s:%s.%s" % (
        int(m.group(1), 16),
        m.group(2).lower(),
        m.group(3).lower(),
        m.group(4),
    )


# PCIe "GT/s" lane rate -> generation.  sysfs prints the rate and never the
# generation, so this mapping is ours and is the only place it is stated.
_GTS_TO_GEN = {"2.5": 1, "5.0": 2, "8.0": 3, "16.0": 4, "32.0": 5, "64.0": 6}


def _link_from_sysfs(dev: str, which: str) -> tuple[Optional[int], Optional[int]]:
    speed = _read(os.path.join(dev, "%s_link_speed" % which))
    width = _int(_read(os.path.join(dev, "%s_link_width" % which)))
    gen = None
    if speed:
        m = re.match(r"([\d.]+)\s*GT/s", speed)
        if m:
            gen = _GTS_TO_GEN.get(m.group(1))
    return gen, width


# GT/s per lane for each PCIe generation, used only to compare one hop against another.


_GEN_GTS = {1: 2.5, 2: 5.0, 3: 8.0, 4: 16.0, 5: 32.0, 6: 64.0}


def pci_chain_links(device_path: str) -> list:
    """Every PCIe link from a device up to the root complex, as (gen, width, cur_gen, cur_width).

    A card can sit behind its own switch.  The test system's R9700 reports gen5 x16 -- which is real, but
    it is the link to the card's INTERNAL switch; the root port above it is gen4 x8, and that is what
    actually limits traffic to host memory.  Reading only the device's own link overstates a card by 4x
    there, and card ranking is one of the things that reads it.
    """
    out: list = []
    path = os.path.realpath(device_path)
    seen = 0
    while seen < 16:   # the bus topology is shallow; the bound is only to stop a symlink cycle
        base = os.path.basename(path)
        if not re.match(r"^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.\d$", base):
            break
        gen, width = _link_from_sysfs(path, "max")
        cur_gen, cur_width = _link_from_sysfs(path, "current")
        if gen is not None or width is not None:
            out.append((gen, width, cur_gen, cur_width))
        path = os.path.dirname(path)
        seen += 1
    return out


def narrowest_link(links: list):
    """The hop with the least bandwidth, as (gen, width, cur_gen, cur_width), or None.

    The narrowest hop is what the card can actually achieve, so it is the honest value for a field named
    "max".  Hops that do not state both a generation and a width are skipped rather than assumed wide.
    """
    best = None
    best_bw = None
    for gen, width, cur_gen, cur_width in links:
        if gen is None or width is None:
            continue
        bw = _GEN_GTS.get(gen, 0.0) * width
        if best_bw is None or bw < best_bw:
            best_bw, best = bw, (gen, width, cur_gen, cur_width)
    return best


_PCI_IDS = ("/usr/share/hwdata/pci.ids", "/usr/share/misc/pci.ids")


def pci_ids_name(text: str, vendor: int, device: int) -> Optional[str]:
    """Look one device up in a pci.ids file.  Pure, so the tests drive it from the real format.

    The format nests by indentation: a vendor at column 0, its devices under one tab, and SUBSYSTEM
    entries under two.  Matching "one tab" exactly is what keeps a subsystem line -- which can repeat the
    same four hex digits with a different meaning -- from being read as the device.
    """
    want_vendor = "%04x" % vendor
    want_device = "%04x" % device
    in_vendor = False
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        if not line.startswith("\t"):
            if line.startswith("C "):     # the class section: no vendors below it
                break
            in_vendor = line[:4].lower() == want_vendor
            continue
        if not in_vendor:
            continue
        # Exactly one tab is stripped and the id is read at a FIXED offset.  That is what
        # excludes subsystems: their line keeps a second tab, so body[:4] is a tab plus three
        # digits and cannot match four hex digits.  An explicit two-tab test used to sit here
        # as well; it was redundant, and the redundancy MASKED both guards -- sabotaging
        # either one alone changed no behaviour because the other still held.  One guard, tested.
        body = line[1:]
        if body[:4].lower() == want_device:
            return body[4:].strip() or None
    return None


def pci_device_name(vendor: int, device: int, paths=_PCI_IDS) -> Optional[str]:
    """The marketing name for a PCI id, from whichever pci.ids this machine has."""
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                name = pci_ids_name(fh.read(), vendor, device)
        except OSError as exc:
            _log.debug("pci.ids %s not usable: %s", os.path.basename(path), describe(exc))
            continue
        if name:
            return name
    _log.debug("no pci.ids entry for %04x:%04x; reporting the id only", vendor, device)
    return None
