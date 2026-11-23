# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""The detect adapter: the cards and toolkits the planner may use, and the reason each excluded card is excluded.

OWNER     : this module owns the one reading of detect's HostReport (adapt_report: the ONLY code that touches its
            attributes), the card record (_Card), SUPPORTED (which vendors have a build) and the exclusion rules.
            plan.py owns every placement decision made from what this returns.
CONTRACT  : adapt_report(report) -> dict; usable_cards(cards, kits, blockers) -> (usable, [(CardKey, reason)]);
            candidates(report) -> the usable cards as Candidate rows; held_by / blocked_by / why format reasons.
            Imports: stdlib and _plan_opts only (one-way edge: plan imports this, never the reverse).
FAILURE   : an unknown that feeds a decision becomes an exclusion with a reason, never a guess.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

from ._plan_opts import GIB

CardKey = Tuple[str, int]     # (vendor, index): Gpu.index is a PER-VENDOR ordinal, so index alone collides (amd 0 vs nvidia 0)
SUPPORTED = {"nvidia": "cuda", "amd": "rocm"}   # vendor -> toolkit kind we can build for; others (intel/sycl) are refused with a reason
HOLDERS_SHOWN = 3


# ---- detect adapter: the ONLY code that reads the report's attributes (detect.py FINAL contract) ----------------
@dataclass(frozen=True)
class _Card:
    index: int
    vendor: str
    name: str
    arch: str
    free: Optional[int]
    total: Optional[int]
    gtt: Optional[int]
    pcie: int                       # gen_max * width_max (the _current fields downclock when idle: never used)
    uuid: Optional[str] = None
    holders: tuple = ()             # detect's Gpu.holders: ((pid, bytes, name), ...) biggest first, THIS card's only

    @property
    def key(self) -> CardKey:
        return (self.vendor, self.index)


def _adapt_cards(report: Any) -> list:
    g = getattr
    return [_Card(x.index, x.vendor, g(x, "name", None) or "unknown card", g(x, "arch", None) or "", g(x, "vram_free_bytes", None),
                  g(x, "vram_total_bytes", None), g(x, "gtt_total_bytes", None),
                  (g(x, "pcie_gen_max", None) or 0) * (g(x, "pcie_width_max", None) or 0), g(x, "uuid", None),
                  tuple(g(x, "holders", None) or ())) for x in g(report, "gpus", ())]


def _adapt_kits(report: Any) -> dict:
    """{vendor: toolkit}: per vendor, prefer a toolkit that can actually build."""
    g = getattr
    kits = {}
    for t in g(report, "toolkits", ()):
        ok = bool(t.root and t.compiler and (t.kind != "cuda" or g(t, "host_compiler", None)))
        if t.kind == SUPPORTED.get(t.vendor) and (t.vendor not in kits or (ok and not kits[t.vendor][1])):
            kits[t.vendor] = (t, ok)
    return {v: k[0] for v, k in kits.items()}


def adapt_report(report: Any) -> dict:
    """Duck-typed read of detect's HostReport (frozen tuples).  Missing optional fields become None = 'unknown',
    which the planner treats as a refusal for whatever depends on them.  The only function to edit if detect moves."""
    g = getattr
    fss = [(f.mount, g(f, "rotational", None), g(f, "free_bytes", None), g(f, "fstype", "")) for f in g(report, "filesystems", ())]
    return {"cards": _adapt_cards(report), "kits": _adapt_kits(report), "fs": fss, "ram_total": g(report, "ram_total_bytes", None),
            "ram_avail": g(report, "ram_available_bytes", None), "ttm": g(report, "ttm_pages_limit_bytes", None),
            "memlock": g(report, "memlock_bytes", None), "blockers": tuple(g(report, "blockers", ()))}


def held_by(holders: tuple) -> str:
    """Who occupies a card's VRAM, for an exclusion reason.  WHY: a bare "free VRAM too low" sends the user hunting; naming the
    top holders (detect lists them biggest first, per card, keyed by UUID/KFD id) tells him what to close.  Empty = the probe
    found none or could not attribute, and we say exactly that rather than guess."""
    if not holders:
        return "no process attributed"
    shown = ", ".join(f"{n}({p}) {b / GIB:.1f} GiB" for p, b, n in holders[:HOLDERS_SHOWN])
    more = len(holders) - HOLDERS_SHOWN
    return f"in use by {shown}" + (f", +{more} more" if more > 0 else "")


def blocked_by(blockers: tuple, kind: str, *refs: str) -> list:
    """Blockers of this kind whose ref is in refs (detect: where = '<kind>:<ref>'); an empty ref matches the whole kind."""
    return [b for b in blockers if b.kind == kind and (b.ref in refs or b.ref == "")]


def why(bs: list) -> str:
    return "; ".join(f"{b.what} (fix: {b.fix})" for b in bs)


def _toolkit_gap(c: _Card, t: Any) -> str:
    """Why `c`'s vendor toolkit cannot build for it ("" when it can)."""
    if t is None:
        return f"no {SUPPORTED[c.vendor]} toolkit detected, cannot build for it"
    if not (t.root and t.compiler):
        return f"{t.kind} toolkit has no root/compiler detected, cannot build for it"
    if t.kind == "cuda" and not getattr(t, "host_compiler", None):
        return f"no working CUDA host compiler (tried: {', '.join(getattr(t, 'host_compiler_tried', ()) or ()) or 'none'})"
    return ""


def _exclusion_reason(c: _Card, kits: dict, blockers: tuple) -> str:
    """Why `c` cannot be used ("" when it can); the first rule that applies is the one reported."""
    if c.vendor not in SUPPORTED:
        return f"vendor {c.vendor!r} has no seabraid build (supported: {', '.join(SUPPORTED)})"
    bl = blocked_by(blockers, "toolkit", SUPPORTED.get(c.vendor, "")) + blocked_by(blockers, "gpu", f"{c.vendor}:{c.index}")
    if bl:
        return f"blocked: {why(bl)}"
    gap = _toolkit_gap(c, kits.get(c.vendor))
    if gap:
        return gap
    if not c.arch:
        return "architecture unknown, cannot choose compile targets"
    return "free VRAM unknown, cannot size its share" if c.free is None else ""


def usable_cards(cards: list, kits: dict, blockers: tuple) -> tuple:
    """(usable cards, [(CardKey, reason)] for the rest)."""
    reasons = [(c, _exclusion_reason(c, kits, blockers)) for c in cards]
    return [c for c, r in reasons if not r], [(c.key, r) for c, r in reasons if r]


@dataclass(frozen=True)
class Candidate:
    """A card the planner may use, in the terms a measurement run needs (see _bench.measure)."""
    vendor: str
    index: int
    device_id: int
    uuid: Optional[str]
    name: str


def ordinals(cards: list) -> dict:
    return {c.key: sum(1 for o in cards if o.vendor == c.vendor and o.index < c.index) for c in cards}


def candidates(report: Any) -> tuple:
    """The usable cards, i.e. exactly the set make_plan ranks; _bench measures this set so no candidate lacks a number."""
    h = adapt_report(report)
    usable, _ = usable_cards(h["cards"], h["kits"], h["blockers"])
    ordinal = ordinals(h["cards"])
    return tuple(Candidate(c.vendor, c.index, ordinal[c.key], c.uuid, c.name) for c in usable)
