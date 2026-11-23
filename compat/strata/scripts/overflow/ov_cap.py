# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""The capacity arithmetic: how many experts each rung holds, how many are left for the SSD, and the memory limits that make it true.

OWNER: this module owns "which numbers make a model overflow".  Inputs are the ``capacity`` block of arms.json and each arm's
``tiers`` block; nothing is read from disk and no engine is consulted (the engine's own startup lines are compared with
these numbers by ov_gate.py after the fact, so a wrong input is caught, not trusted).
CONTRACT: ``build(cfg) -> {arm: ArmPlan}`` or ``qual_cfg.ConfigError``; ``ArmPlan.lines`` is the printed arithmetic.
THE RULE (the reason the harness exists): for a capped arm, main slots + helper slots + RAM-budget blobs must leave at least
``min_file_share`` of all expert blobs on the SSD, otherwise nothing overflows and the run would test nothing.  The share is
checked here, at plan time, in every arm, and the plan refuses to build when it is short.
The engine memory limit is RAM budget + engine overhead + page-cache allowance.  WHY the allowance is small: a cgroup v2
limit also charges page cache, so the cache the file tier can use is at most that allowance; any more would let the OS serve
"SSD" reads from RAM and the test would lie.  engine_overhead_gib is an ESTIMATE until a first run reports memory.peak.
FAILURE: fail-fast; every bad input raises ConfigError naming the key before anything starts.
CHECKER: compat/strata/tests/overflow/test_overflow_cap.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import ov_paths  # noqa: F401
import qual_cfg

GIB = 1 << 30


@dataclass(frozen=True)
class ArmPlan:
    """One arm's capacity.  For an uncapped arm (the reference) every number is None: it is all-VRAM by design."""

    arm: str
    capped: bool
    helpers: int
    ladder: bool
    main_slots: int | None
    helper_slots: tuple[int, ...]
    ram_gib: float | None
    ram_blobs: int | None
    card_blobs: int | None
    file_blobs: int | None
    file_gib: float | None
    file_share: float | None
    complement_gib: float | None
    engine_cap_bytes: int | None
    helper_cap_bytes: int | None
    lines: tuple[str, ...]


def _need(cond: bool, msg: str) -> None:
    if not cond:
        raise qual_cfg.ConfigError(msg)


def _check_block(cap: dict) -> None:
    for k in ("layers", "experts", "main_slots"):
        _need(isinstance(cap.get(k), int) and cap[k] > 0, f"capacity.{k} must be a positive int")
    for k in ("experts_gib", "ram_gib", "main_card_gib", "helper_card_gib", "engine_overhead_gib", "page_cache_gib", "helper_cap_gib"):
        _need(isinstance(cap.get(k), (int, float)) and cap[k] > 0, f"capacity.{k} must be a positive number")
    hs = cap.get("helper_slots")
    _need(isinstance(hs, list) and all(isinstance(n, int) and n > 0 for n in hs), "capacity.helper_slots must be a list of positive ints")
    _need(isinstance(cap.get("min_file_share"), (int, float)) and 0 < cap["min_file_share"] < 1, "capacity.min_file_share must be in (0, 1)")


def _uncapped(arm: str, helpers: int, total: int, cap: dict) -> ArmPlan:
    extra = " --expert-cache-device1 auto" if helpers else ""
    line = (f"{arm}: uncapped reference: --expert-cache auto{extra}, no file tier, no cgroup limit; "
            f"the experts ({cap['experts_gib']:.2f} GiB, {total} blobs) are expected to sit on the cards")
    return ArmPlan(arm, False, helpers, False, None, (), None, None, None, None, None, None, None, None, None, (line,))


def _lines(arm: str, p: dict, ladder: bool) -> tuple[str, ...]:
    h = " + ".join(f"helper{i + 1} {n}" for i, n in enumerate(p["helper_slots"]))
    cards = f"main {p['main_slots']}" + (f" + {h}" if h else "")
    margin = 100 * (p["file_share"] - p["min_share"])
    return (
        f"{arm}: {p['total']} expert blobs = {p['experts_gib']:.2f} GiB (blob {p['blob_gib'] * 1024:.2f} MiB, ESTIMATE from the config)",
        f"    cards: {cards} slots = {p['card_blobs']} blobs ({p['card_blobs'] * p['blob_gib']:.2f} GiB)",
        f"    RAM budget {p['ram_gib']:.2f} GiB = {p['ram_blobs']} blobs",
        f"    cards + RAM = {p['card_blobs'] + p['ram_blobs']} blobs; SSD = {p['file_blobs']} blobs = {p['file_gib']:.2f} GiB = "
        f"{100 * p['file_share']:.1f}% of all experts (rule: >= {100 * p['min_share']:.0f}%, margin {margin:.1f} points)",
        f"    cgroup MemoryMax: engine {p['engine_gib']:.2f} GiB = RAM {p['ram_gib']:.2f} + overhead {p['overhead']:.2f} (ESTIMATE) + page cache {p['cache']:.2f}; "
        f"helper {p['helper_gib']:.2f} GiB each; swap 0",
    ) + (("    ladder: live moves ON (flags and env come from install/_ladder.py)",) if ladder else ())


def _check_fits(arm: str, cap: dict, blob_gib: float, hs: tuple[int, ...], share: float) -> None:
    _need(cap["main_slots"] * blob_gib <= cap["main_card_gib"], f"arm {arm}: main slots need {cap['main_slots'] * blob_gib:.1f} GiB, the main card holds {cap['main_card_gib']} GiB of experts")
    _need(sum(hs) * blob_gib <= cap["helper_card_gib"], f"arm {arm}: helper slots need {sum(hs) * blob_gib:.1f} GiB, the helper card holds {cap['helper_card_gib']} GiB of experts")
    _need(share >= cap["min_file_share"], f"arm {arm}: only {100 * share:.1f}% of the experts would sit on the SSD (rule >= {100 * cap['min_file_share']:.0f}%): "
          "this cap does not overflow, it would test nothing")


def _capped(arm: str, helpers: int, ladder: bool, total: int, cap: dict) -> ArmPlan:
    _need(helpers <= len(cap["helper_slots"]), f"arm {arm}: {helpers} helpers but capacity.helper_slots lists {len(cap['helper_slots'])}")
    blob_gib = cap["experts_gib"] / total
    hs = tuple(cap["helper_slots"][:helpers])
    ram_blobs = math.floor(cap["ram_gib"] / blob_gib)
    cards = cap["main_slots"] + sum(hs)
    file_blobs = total - cards - ram_blobs
    share = file_blobs / total
    _check_fits(arm, cap, blob_gib, hs, share)
    engine_gib = cap["ram_gib"] + cap["engine_overhead_gib"] + cap["page_cache_gib"]
    shown = {"total": total, "experts_gib": cap["experts_gib"], "blob_gib": blob_gib, "main_slots": cap["main_slots"], "helper_slots": hs, "card_blobs": cards,
             "ram_gib": cap["ram_gib"], "ram_blobs": ram_blobs, "file_blobs": file_blobs, "file_gib": file_blobs * blob_gib, "file_share": share,
             "min_share": cap["min_file_share"], "engine_gib": engine_gib, "overhead": cap["engine_overhead_gib"], "cache": cap["page_cache_gib"],
             "helper_gib": cap["helper_cap_gib"]}
    return ArmPlan(arm, True, helpers, ladder, cap["main_slots"], hs, cap["ram_gib"], ram_blobs, cards, file_blobs, file_blobs * blob_gib, share,
                   cap["experts_gib"] - cards * blob_gib, round(engine_gib * GIB), round(cap["helper_cap_gib"] * GIB), _lines(arm, shown, ladder))


def _tiers(arm: str, spec: dict) -> dict:
    t = spec.get("tiers")
    ok = isinstance(t, dict) and isinstance(t.get("capped"), bool) and isinstance(t.get("helpers"), int) and 0 <= t["helpers"] <= 3
    _need(ok, f"arms.{arm}.tiers must be {{capped: bool, helpers: 0..3[, ladder: bool]}} (an arm must say what it is)")
    _need(isinstance(t.get("ladder", False), bool) and (not t.get("ladder") or (t["capped"] and t["helpers"] >= 1)), f"arms.{arm}.tiers.ladder needs a capped arm with a helper")
    return t


def build(cfg: dict) -> dict[str, ArmPlan]:
    """The plan of every arm in the config, or ConfigError."""
    cap = cfg.get("capacity", {})
    _check_block(cap)
    total = cap["layers"] * cap["experts"]
    out = {}
    for arm, spec in cfg["arms"].items():
        t = _tiers(arm, spec)
        out[arm] = _capped(arm, t["helpers"], t.get("ladder", False), total, cap) if t["capped"] else _uncapped(arm, t["helpers"], total, cap)
    return out
