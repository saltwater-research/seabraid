# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""seabraid.ranking: which card is card 1, in what order the rest follow, and which slow cards are admitted.

OWNER   : the card-ordering rules, their code (rank, admit, layer_cost, rank_note, bandwidth_gbs) and the per-card
          memory-bandwidth table. It ships in the wheel, so the rule text lives here with the code; the compat add-on's
          installer only calls it and decides nothing itself. The rules:
          R1 (card 1): among the cards that pass the eligibility gate (`fits_main`: dense weights, KV cache and drafter
             fit), card 1 is the one with the lowest measured LEAD-JOB time per decode window at the configured
             concurrency. One regime per concurrency class (1 stream; >= 4 streams), never blended (`lead_class`).
             Without a complete lead measurement the proxy order is VRAM, then memory bandwidth, then architecture
             generation, then PCIe; the plan line says "proxy" and never "measured" (`rank_note`).
          R2 (cards 2..N): order by measured decode expert rows/s, ties by free VRAM. Rows/s is their job; the
             per-layer round trip is not an order key, it decides admission (R3).
          R3 (admission): card k is admitted only if (a) it would hold experts that otherwise fall to RAM or NVMe
             (the expert bytes exceed the VRAM budgets of cards 1..k-1) and (b) the layer time with the card is not
             longer than without it: with = host + max(t1, hop + tk), without = t1 + pull (`layer_cost`). A card that
             fails (b) but passes (a) is reported with the rule and the bytes it would have held, not dropped silently.
          force_main (the user's choice) always wins card 1 and skips the gate.
CONTRACT: rank(cards, bench_lead, bench_rows, fits_main, concurrency=1, force_main=None) -> (ordered, Basis);
          admit(slots, expert_bytes, floors=None) -> (admitted [Admission], excluded [(key, reason)]);
          layer_cost(Terms) -> LayerCost;  rank_note(...) -> str;  lead_class(streams) -> 1 | 4 | None.
          `cards` are duck-typed (key, name, arch, free, total, pcie).  Imports: stdlib only.
FAILURE : rank() raises TypeError for input that is not the documented shape (a card without the six attributes or with
          a wrong-typed one: arch must be str, pcie a number, free/total int or None, name str or None, key a hashable tuple;
          bench_lead/bench_rows neither a Mapping nor None; fits_main not callable; concurrency not an int) and ValueError
          for duplicate card keys or a force_main that names no card in `cards` (a silently ignored choice would rank
          without the user's card 1); an exception raised by fits_main itself propagates. It never raises for a MEASUREMENT:
          a missing/NaN/non-positive/non-numeric measurement for ANY card drops ALL cards to the proxy order for that job
          (no partial mix of measured and guessed). An unknown bandwidth/arch sorts last in its tier. admit() with no
          floors for a card admits it on R3a alone and SAYS R3b was not evaluated (fail open, loud). force_main, when it
          names a card, always wins card 1.
CHECKER : bindings/python/tests/test_ranking.py (each of R1, R2, R3a, R3b, force_main and the FAILURE contract has a
          test) and tests/test_public_api.py (import surface). The compat add-on's test_plan_rank.py exercises how the
          installer uses this module; it is not this module's checker.
STATUS  : UNTESTED-ON-HARDWARE.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping, NamedTuple, Optional, Tuple

GIB = 1 << 30
CardKey = Tuple[str, int]
LEAD_NOT_RUN = "the lead-job bench was not run (expert rows/s, if measured, orders cards 2..N only)"

# Last-resort tie-break only: architecture generation says little about how fast a card reads memory.
_ARCH_CLASS = (("sm_12", 3), ("gfx12", 3), ("sm_89", 2), ("sm_90", 2), ("gfx11", 2), ("sm_86", 1), ("sm_80", 1))  # seabraid-generic: vendor-wide taxonomy, not this machine

# Peak memory bandwidth, GB/s, keyed by a lower-case substring of the detected card NAME. The ONE owner of these figures:
# bandwidth_gbs() reads it and nothing else defines a bandwidth. Keyed by name, not arch, because bandwidth is per SKU
# (sm_86 spans a 3060 at 360 and a 3090 at 936). Each figure is bus width x data rate from the vendor's published spec,
# written from the spec sheets and NOT re-fetched in the session that wrote them. Insertion order matters: the first
# substring that matches wins, which is why "4060 ti" precedes "4060". Only a tie-break after VRAM, so a stale or missing
# row can only reorder two cards with the same VRAM. (Built from ordered pairs, one per line, so each row can carry its
# citation and the compat add-on's test can check that every row has one.)
_BANDWIDTH_GBS: Mapping[str, float] = MappingProxyType(dict((
    ("4060 ti", 288.0),     # NVIDIA GeForce RTX 4060 Ti: 128-bit GDDR6 18 Gbps
    ("4060", 272.0),        # NVIDIA GeForce RTX 4060: 128-bit GDDR6 17 Gbps
    ("3090", 936.0),        # NVIDIA GeForce RTX 3090: 384-bit GDDR6X 19.5 Gbps (3090 Ti is 1008; same tier here)
    ("3060 ti", 448.0),     # NVIDIA GeForce RTX 3060 Ti: 256-bit GDDR6 14 Gbps ("3060" below is a substring of it)
    ("3060", 360.0),        # NVIDIA GeForce RTX 3060 12 GB: 192-bit GDDR6 15 Gbps (the 8 GB variant is 240)
    ("4080 super", 736.0),  # NVIDIA GeForce RTX 4080 SUPER: 256-bit GDDR6X 23 Gbps
    ("4090", 1008.0),       # NVIDIA GeForce RTX 4090: 384-bit GDDR6X 21 Gbps
    ("5070 ti", 896.0),     # NVIDIA GeForce RTX 5070 Ti: 256-bit GDDR7 28 Gbps
    ("r9700", 640.0),       # AMD Radeon AI PRO R9700: 256-bit GDDR6 20 Gbps
)))


def bandwidth_gbs(name: str) -> float:
    n = (name or "").lower()
    return next((bw for sub, bw in _BANDWIDTH_GBS.items() if sub in n), 0.0)


def _arch_class(arch: str) -> int:
    return next((c for pre, c in _ARCH_CLASS if arch.startswith(pre)), 0)


def _vram(c) -> int:
    return c.total or c.free or 0


def _ok(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0


def _complete(bench: Optional[Mapping], cards: list) -> bool:
    """True only when EVERY card has a usable figure: a partial map would rank measured numbers against guesses."""
    return bool(cards) and bool(bench) and all(_ok(bench.get(c.key)) for c in cards)


def lead_class(streams: int) -> Optional[int]:
    """The lead-job regime a stream count belongs to: 1 or 4 (>= 4).  2 and 3 streams have no measured regime (None):
    blending the two would hide which one the number describes (same rule as card_bench.hpp for decode vs prefill)."""
    return 1 if streams == 1 else 4 if streams >= 4 else None


@dataclass(frozen=True)
class Basis:
    """What decided each job.  card1: 'user' | 'lead' | 'proxy'; rest: 'rows' | 'proxy'."""
    card1: str
    rest: str


def _proxy_key(c) -> tuple:
    return (-_vram(c), -bandwidth_gbs(c.name), -_arch_class(c.arch), -c.pcie, c.key)


def _card1_key(fits_main: Callable[[object], bool], bench_lead: Optional[Mapping]) -> Callable:
    """Sort key for card 1.  WHY module-level: it keeps rank() small enough to read; bench_lead=None selects the proxy order."""
    def key(c) -> tuple:
        gate = 0 if fits_main(c) else 1
        return (gate, bench_lead[c.key], c.key) if bench_lead is not None else (gate, *_proxy_key(c))
    return key


def _rest_key(bench_rows: Optional[Mapping]) -> Callable:
    """Sort key for cards 2..N: measured rows/s (then free VRAM) when bench_rows is given, else the proxy order."""
    if bench_rows is None:
        return _proxy_key
    return lambda c: (-bench_rows[c.key], -(c.free or 0), c.key)


def _forced_card(cards: list, force_main: Optional[CardKey]):
    return next((c for c in cards if c.key == force_main), None) if force_main is not None else None


_CARD_FIELDS = ("key", "name", "arch", "free", "total", "pcie")


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _check_card(c) -> None:
    """TypeError naming the field when a card is not what rank() reads; WHY here and not at the use sites: the failures
    otherwise surface as 'NoneType' errors from inside a sort key, far from the cause."""
    for field in _CARD_FIELDS:
        if not hasattr(c, field):
            raise TypeError(f"card {c!r} has no .{field} (cards need {', '.join(_CARD_FIELDS)})")
    wanted = (
        ("key", isinstance(c.key, tuple), "a tuple"), ("arch", isinstance(c.arch, str), "a str"),
        ("name", c.name is None or isinstance(c.name, str), "a str or None"),
        ("free", c.free is None or _is_int(c.free), "an int or None"),
        ("total", c.total is None or _is_int(c.total), "an int or None"),
        ("pcie", isinstance(c.pcie, (int, float)) and not isinstance(c.pcie, bool), "a number"),
    )
    for field, good, want in wanted:
        if not good:
            raise TypeError(f"card {c.key!r}: .{field} must be {want}")


def _check_cards(cards: list) -> list:
    """Validate every card and return the keys (a duplicate key would make the measurement maps ambiguous)."""
    for c in cards:
        _check_card(c)
    keys = [c.key for c in cards]
    if len(set(keys)) != len(keys):
        raise ValueError(f"duplicate card key in cards: {sorted(k for k in set(keys) if keys.count(k) > 1)}")
    return keys


def _check_rank_arguments(cards: list, bench_lead, bench_rows, fits_main, concurrency, force_main) -> None:
    keys = _check_cards(cards)
    for name, bench in (("bench_lead", bench_lead), ("bench_rows", bench_rows)):
        if bench is not None and not isinstance(bench, Mapping):
            raise TypeError(f"{name} must be a Mapping of card key to measurement, or None, not {type(bench).__name__}")
    if not callable(fits_main):
        raise TypeError("fits_main must be callable")
    if not _is_int(concurrency):
        raise TypeError(f"concurrency must be an int, not {type(concurrency).__name__}")
    if force_main is not None and force_main not in keys:
        raise ValueError(f"force_main {force_main!r} names no card in `cards` (keys: {keys})")


def _basis(forced, lead, rows) -> "Basis":
    return Basis("user" if forced else "lead" if lead else "proxy", "rows" if rows else "proxy")


def rank(cards: list, bench_lead: Optional[Mapping[CardKey, Optional[float]]], bench_rows: Optional[Mapping[CardKey, Optional[float]]],
         fits_main: Callable[[object], bool], concurrency: int = 1, force_main: Optional[CardKey] = None) -> tuple:
    """Order candidates: card 1 first, then cards 2..N.  bench_lead = {key: ms per decode window} (lower is better) for the
    regime of `concurrency`; bench_rows = {key: decode expert rows/s} (higher is better).

    WHY two jobs, two keys: expert rows/s ranked the R9700 over the 3090 as card 1 although the 3090 is the faster card 1 at one
    stream on the test system (measured by the add-on's installer bench), so rows/s only orders cards 2..N.  WHY capacity gates card 1: it must hold attention, the KV
    cache and the hottest experts, so a card that cannot is never card 1 however fast it benched; without a lead measurement the
    proxy is VRAM, bandwidth, generation, PCIe.  force_main (the user's choice) skips the gate: if it cannot hold card 1's reserve,
    plan.py's exclusion message says so, not a silent swap."""
    cards = list(cards)  # a non-iterable is the TypeError list() raises
    _check_rank_arguments(cards, bench_lead, bench_rows, fits_main, concurrency, force_main)
    if not cards:
        return [], Basis("proxy", "proxy")
    lead = bench_lead if lead_class(concurrency) is not None and _complete(bench_lead, cards) else None
    rows = bench_rows if _complete(bench_rows, cards) else None
    forced = _forced_card(cards, force_main)
    first = forced or sorted(cards, key=_card1_key(fits_main, lead))[0]
    others = sorted((c for c in cards if c is not first), key=_rest_key(rows))
    return [first, *others], _basis(forced, lead, rows)


def rank_note(main, other, how: Basis, bench_lead: Optional[Mapping], why: str, other_fits_main: bool,
              bench_rows: Optional[Mapping] = None, streams: int = 1) -> str:
    """One sentence saying why `main` is card 1, against the next candidate `other` (None = only one card).  Says
    'measured' only when the lead-job figure was; a rows/s figure is never described as the reason for card 1."""
    g = lambda c: f"{_vram(c) / GIB:.1f} GiB"  # noqa: E731
    if other is None:
        return f"main: {main.name} ({g(main)}; the only usable card)"
    rest = ""
    if how.rest == "rows" and bench_rows:
        rest = f"; cards after it ordered by measured expert rows/s ({bench_rows[other.key]:.0f} for {other.name})"
    if how.card1 == "user":
        return f"main: {main.name} (chosen by the user with --main-card, not ranked; {g(main)}){rest}"
    tail = f"; {other.name} is too small to hold main's resident set" if not other_fits_main else ""
    if how.card1 == "lead":
        return (f"main: {main.name} (measured lead job, {streams} stream{'s' * (streams != 1)}: {bench_lead[main.key]:.1f} ms/window, "
                f"{g(main)}) over {other.name} ({bench_lead[other.key]:.1f} ms/window, {g(other)}){tail}{rest}")
    if other_fits_main and _vram(main) == _vram(other):
        tail = "; equal VRAM, tie broken by memory bandwidth, then architecture, then PCIe"
    return (f"main: {main.name} by heuristic ({g(main)} vs {g(other)}, not measured: "
            f"{why or 'no lead-job measurement supplied'}){tail}{rest}")


# ------------------------------------------------------------------------------------------------------------- R3 admission


class Slot(NamedTuple):
    key: CardKey
    name: str
    budget_bytes: int            # expert bytes this card could hold after its reserve

class Admission(NamedTuple):
    slot: Slot
    held_bytes: int              # expert bytes that leave RAM/NVMe because this card exists
    note: str                    # the R3 terms behind the decision, printed in the plan


@dataclass(frozen=True)
class Terms:
    """Per-layer microseconds, from A/B 4's per-layer log.  host: card 1's serial work for this card (scan, stage, publish,
    merge); hop: the shared-memory round trip; t1: card 1's own expert time; tk: this card's expert time; pull: what the
    same experts cost per layer from RAM/NVMe without the card."""
    host_us: float
    hop_us: float
    t1_us: float
    tk_us: float
    pull_us: float


@dataclass(frozen=True)
class LayerCost:
    with_card_us: float
    without_card_us: float
    terms: str
    ok: bool                     # R3b: the card does not make the layer slower

def layer_cost(t: Terms) -> LayerCost:
    """host + max(t1, hop + tk) with the card, t1 + pull without it (R3b in the module docstring).  Pure; the weakest inference in
    the design, so the terms travel with the verdict.  NaN terms compare False: a broken floor is never 'ok'."""
    w = t.host_us + max(t.t1_us, t.hop_us + t.tk_us)
    wo = t.t1_us + t.pull_us
    s = (f"with {w:.0f} us = host {t.host_us:.0f} + max(t1 {t.t1_us:.0f}, hop {t.hop_us:.0f} + tk {t.tk_us:.0f}); "
         f"without {wo:.0f} us = t1 {t.t1_us:.0f} + pull {t.pull_us:.0f}")
    return LayerCost(w, wo, s, bool(w <= wo))


def admit(slots: list, expert_bytes: int, floors: Optional[Mapping[CardKey, Terms]] = None) -> tuple:
    """R3 on cards 2..N of `slots` (card 1 first, never tested): ([Admission] incl. card 1, [(key, reason)] naming the rule,
    the bytes it would have held and, for R3b, the floors)."""
    g = lambda b: f"{b / GIB:.1f} GiB"  # noqa: E731
    if not slots:
        return [], []
    admitted = [Admission(slots[0], min(expert_bytes, slots[0].budget_bytes), "card 1")]
    excluded, covered = [], slots[0].budget_bytes
    for s in slots[1:]:
        held = min(s.budget_bytes, max(0, expert_bytes - covered))
        if held <= 0:
            excluded.append((s.key, f"R3a: 0 bytes would leave VRAM (the {g(expert_bytes)} of experts already fit in the "
                                    f"{g(covered)} the cards before it hold; it would only add a per-layer round trip)"))
            continue
        t = (floors or {}).get(s.key)
        if t is None:
            note = f"R3a ok: would hold {g(held)} that otherwise sit in RAM/NVMe; R3b not evaluated (no per-layer floors measured)"
        else:
            c = layer_cost(t)
            if not c.ok:
                excluded.append((s.key, f"R3b: layer cost with the card {c.with_card_us:.0f} us > without {c.without_card_us:.0f} us "
                                        f"(floors: {c.terms}); it would have held {g(held)}"))
                continue
            note = f"R3a ok: would hold {g(held)}; R3b ok: {c.terms}"
        admitted.append(Admission(s, held, note))
        covered += s.budget_bytes
    return admitted, excluded
