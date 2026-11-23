# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""The engine startup lines the overflow harness needs beyond payoff's: the RAM budget placement and the file tier's read mode.

OWNER: this module owns "which startup line carries the RAM-budget and file-tier facts".  Every per-request line (prompt, tiers,
file tier I/O, decode waited) stays pay_parse's.  Formats read from Strata source, not guessed:
  * ``FileExpertSource: RAM budget B GiB: H of the C GiB of experts the GPU cache does not hold, by profile rank; the rest are read
    from the files`` (src/core/expert_source.cpp): H experts are copied to RAM, C GiB is the complement, so C minus H blobs sit on
    the SSD;
  * ``strata generate: the file tier reads unbuffered | through the file cache (why)`` (src/program/generate.cpp);
  * ``seabraid: helper <name> on channel <shm>, N GiB free``, ONE line per helper channel the engine opened (patches 0001 and 0017);
  * ``placement move L/E lower->upper src=card|fallback bytes=N verified=0|1``, ONE line per COMMITTED hop of the Seabraid placement driver
    (adapters/strata/placement_round_log.cpp; rung 0 is the main card, rung r the r-th helper, ``src=fallback`` = the lower rung is the RAM/SSD
    fallback), and the two stale-placement records: ``Seabraid Qwen layer L refused: stale placement epoch E`` (qwen_decode.cpp) and
    ``seabraid retier: helper residency is STALE after an applied round`` (patch 0017).
CONTRACT: ``startup(text) -> OvStartup`` (payoff's Startup fields plus those, and the helper channels in log order); ``moves(text) -> [dict]``;
``stale_count(text) -> int``.  A field whose line is absent is None (channels: empty tuple, moves: empty list).
FAILURE: pure; a missing line is None and ov_gate.py names it as a reason, so an engine that changed its wording is INVALID, not a pass.
CHECKER: compat/strata/tests/overflow/test_overflow_gate.py (synthetic lines in the formats above).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import ov_paths  # noqa: F401
import pay_parse

BUDGET = re.compile(r"RAM budget ([\d.]+) GiB: (\d+) of the ([\d.]+) GiB of experts the GPU cache does not hold")
FILE_MODE = re.compile(r"the file tier reads (unbuffered|through the file cache)")
CHANNEL = re.compile(r"seabraid: helper .+? on channel (\S+?),")
MOVE = re.compile(r"placement move (\d+)/(\d+) (\d+)->(\d+) src=(card|fallback) bytes=(\d+) verified=[01]")
STALE = re.compile(r"refused: stale placement epoch|helper residency is STALE")


@dataclass(frozen=True)
class OvStartup:
    cap_slots: int | None
    cap_gib: float | None
    helper_up: bool
    resident_gib: float | None
    ram_budget_gib: float | None
    ram_held_blobs: int | None
    complement_gib: float | None
    file_mode: str | None
    channels: tuple[str, ...] = ()

    @property
    def slot_gib(self) -> float | None:
        """VRAM GiB per main-cache slot, from the engine's own cache line (the measured blob size)."""
        return self.cap_gib / self.cap_slots if self.cap_slots and self.cap_gib else None


def _last(rx: re.Pattern, text: str):
    found = list(rx.finditer(text))
    return found[-1] if found else None


def startup(text: str) -> OvStartup:
    base = pay_parse.parse_startup(text)
    b, m = _last(BUDGET, text), _last(FILE_MODE, text)
    return OvStartup(base.cap_slots, base.cap_gib, base.helper_up, base.resident_gib,
                     float(b.group(1)) if b else None, int(b.group(2)) if b else None, float(b.group(3)) if b else None, m.group(1) if m else None,
                     tuple(CHANNEL.findall(text)))


def moves(text: str) -> list[dict]:
    """Every committed hop the placement driver logged: layer, expert, the lower and upper rung, whether it came from the fallback rung, and its bytes."""
    return [{"layer": int(g[0]), "expert": int(g[1]), "lower": int(g[2]), "upper": int(g[3]), "src": g[4], "bytes": int(g[5])} for g in MOVE.findall(text)]


def stale_count(text: str) -> int:
    """How many stale-placement records the engine printed (each is logged once per session, so any count above 0 is a refusal that happened)."""
    return len(STALE.findall(text))
