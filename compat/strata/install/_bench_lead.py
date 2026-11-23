# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""seabraid installer, PLAN input: measure the LEAD JOB of each candidate card 1 by running the whole system.  UNTESTED-ON-HARDWARE.

OWNER     : this file is the ONE place that decides how a "lead-job ms per decode window" figure is produced: start the real
            engine once per candidate with that card FIRST (the planned placement for the other cards), drive a fixed greedy
            decode, and read ms/window from the engine's own `strata decode timing:` line.  It emits the figures
            _bench.BenchResult.lead_ms / _plan_rank.rank already consume; ranking stays in _plan_rank, parsing of the helper's
            own bench line stays in _bench.
CONTRACT  : measure_lead(candidates, plan, streams_classes, runner) -> {(vendor, index): LeadResult};
            to_lead_ms(results, candidates) -> ({1 | 4: {(vendor, index): ms}}, why);   parse_timing(text) -> (ms | None, why).
            `plan` is a LeadPlan (how to start one engine; `placement(card)` supplies cfg.json + the other cards for that card
            first).  `runner` has one method, probe(Probe) -> ProbeRun; this module never touches subprocess or the network.
            Imports: build (Cmd), services (card mask, fixed env).  Real runner: _bench_lead_run.py.
FAILURE   : fail-open-loud.  A candidate whose launch cannot be built, whose runner raises or reports a failed start, or whose
            round printed no usable timing line is recorded UNMEASURED WITH THE REASON; nothing raises.  to_lead_ms then omits any
            class a candidate lacks, and _plan_rank falls back to its heuristic and prints that reason.
CHECKER   : tests/install/test_bench_lead.py (fake runner, real-format timing lines, four sabotages).

WHY THE WHOLE SYSTEM AND NOT A SYNTHETIC KERNEL (the decision, with its defense): the lead job is attention, dense layers and
the drafter, whose cost depends on the real model, KV layout, drafter and how the other cards overlap with it; a stand-in kernel
would rank by a number nobody ever served.  The cost is one cold start per candidate (about 70-100 s each with this model),
so it is opt-in (--bench-lead) and capped at MAX_CANDIDATES (the caller orders them best-first by the heuristic).
WHY ONE START PER CANDIDATE FOR BOTH CLASSES: the server is started at --parallel max(class) and the rounds run in order
(warm-up, 1 stream, 4 streams).  A second start for the 4-stream class would double the install time; the 1-stream round on a
--parallel 4 server is what a 4-stream install serves anyway, and both candidates are measured the same way.
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional, Protocol, Sequence

from ._log import describe
from .build import Cmd
from .config import SERVER_ENV                        # the one owner of the server's fixed env
from .services import _visibility                      # one owner of "mask a process to one card"

_log = logging.getLogger(__name__)

MAX_CANDIDATES = 3
WARM_TOKENS = 400           # a 400-token run is ~140 windows (the 2026-11-07 ab1 engine logs)
MEASURE_TOKENS = 400
MIN_WINDOWS = 50            # fewer windows than this is a start-up artifact (the 1-window line read 356 ms)
READY_S = 1800.0            # model load + both caches; the engine is ready when /health answers
SHM_FMT = "/seabraid-lead{n}"    # private channel names: never collide with a running /seabraidN
TIMING = re.compile(r"strata decode timing: (\d+) windows,.*?(\d+(?:\.\d+)?) ms/window")
CardId = tuple


@dataclass(frozen=True)
class Other:
    """A card that is not card 1 in this launch, with the args/env the config gave it (visibility is added here)."""
    vendor: str
    index: int
    uuid: Optional[str]
    args: tuple
    env: Mapping = field(default_factory=dict)


@dataclass(frozen=True)
class Placement:
    cfg_json_path: str
    server_env: Mapping             # config's server env; the card-1 mask is applied OVER it in _server
    others: tuple                   # Other, in shm order
    log_path: str                   # where the engine writes its log (the timing lines)


class PlacementError(Exception):
    """The caller could not build a launch for this card 1 (plan not ready, config refused, no binary)."""


@dataclass(frozen=True)
class LeadPlan:
    python: str
    server_py: str
    port: int
    exes: Mapping[str, str]         # vendor -> seabraid-helper binary
    libs: Mapping[str, str]         # vendor -> LD_LIBRARY_PATH for the other-card processes
    placement: Callable[[Any], Placement]


@dataclass(frozen=True)
class Round:
    label: str                      # "warm" | "lead1" | "lead4"
    streams: int
    tokens: int


@dataclass(frozen=True)
class Probe:
    """Everything a runner needs; commands are plain data so a test reads which card was first."""
    others: tuple                   # Cmd per other card, started first
    shms: tuple                     # channel names to wait for before starting the server
    server: Cmd
    rounds: tuple
    log_path: str
    port: int
    ready_s: float = READY_S


@dataclass(frozen=True)
class ProbeRun:
    started: bool
    why: str = ""
    texts: Mapping[str, str] = field(default_factory=dict)     # round label -> engine log text written during that round


class LeadRunner(Protocol):
    def probe(self, probe: Probe) -> ProbeRun: ...


@dataclass(frozen=True)
class LeadResult:
    ms: Mapping[int, float]         # class -> ms/window, measured classes only
    why: Mapping[int, str]          # class -> reason, unmeasured classes only


def parse_timing(text: str) -> tuple:
    """ms/window from the LAST `strata decode timing:` line in `text`, or (None, why).  The last line is the cumulative one for
    the round; a line over fewer than MIN_WINDOWS windows is a start-up reading and is refused."""
    hits = [m for m in map(TIMING.search, text.splitlines()) if m]
    if not hits:
        return None, "no `strata decode timing:` line (is STRATA_DECODE_TIMING reaching the engine?)"
    windows, ms = int(hits[-1].group(1)), float(hits[-1].group(2))
    if windows < MIN_WINDOWS:
        return None, f"only {windows} decode windows timed (need {MIN_WINDOWS})"
    return (ms, "") if math.isfinite(ms) and ms > 0 else (None, "timing line has no usable ms/window")


def _others(card: Any, plan: LeadPlan, pl: Placement) -> tuple:
    cmds, shms = [], []
    for n, o in enumerate(pl.others):
        exe = plan.exes.get(o.vendor)
        if not exe:
            raise PlacementError(f"no seabraid-helper binary built for vendor {o.vendor!r}")
        kept = [a for i, a in enumerate(o.args) if a != "--device" and (i == 0 or o.args[i - 1] != "--device")]   # always 0: see mask
        shm = SHM_FMT.format(n=n)
        env = {**_visibility(o.vendor, o.index, o.uuid), **({"LD_LIBRARY_PATH": plan.libs[o.vendor]} if plan.libs.get(o.vendor) else {}), **o.env}
        cmds.append(Cmd((exe, *kept, "--shm", shm, "--device", "0"), env=env, label=f"lead probe: {o.vendor} gpu{o.index} (not card 1)"))
        shms.append(shm)
    return tuple(cmds), tuple(shms)


def _server(card: Any, plan: LeadPlan, pl: Placement, shms: Sequence[str], parallel: int) -> Cmd:
    # WHY the mask is applied LAST: this is the whole experiment.  Whatever the planned config's env says, card 1 for this probe is
    # the candidate, so its visibility variables win over any other value in the dict.
    env = {**pl.server_env, "STRATA_DECODE_TIMING": "1", **_visibility(card.vendor, card.index, card.uuid)}
    if shms:
        env.update(SEABRAID_STRATA_SHM=",".join(shms), **SERVER_ENV)
    argv = (plan.python, plan.server_py, "--engine", "strata", "--config", pl.cfg_json_path, "--host", "127.0.0.1",
            "--port", str(plan.port), "--no-api-key", "--parallel", str(parallel))
    return Cmd(argv, env=env, label=f"lead probe: {card.vendor} gpu{card.index} as card 1")


def build_probe(card: Any, plan: LeadPlan, classes: Sequence[int]) -> Probe:
    """The launch for `card` as card 1 plus the fixed rounds.  Raises PlacementError."""
    pl = plan.placement(card)
    others, shms = _others(card, plan, pl)
    rounds = (Round("warm", 1, WARM_TOKENS), *(Round(f"lead{k}", k, MEASURE_TOKENS) for k in classes))
    return Probe(others, shms, _server(card, plan, pl, shms, max(classes)), rounds, pl.log_path, plan.port)


def _measure_one(card: Any, plan: LeadPlan, classes: tuple, runner: LeadRunner) -> LeadResult:
    void = lambda why: LeadResult({}, {k: why for k in classes})  # noqa: E731
    try:
        probe = build_probe(card, plan, classes)
    except PlacementError as exc:
        _log.warning("lead probe: no launch for a candidate card: %s", describe(exc))
        return void(f"no launch could be built with {card.name} first: {exc}")
    try:
        run = runner.probe(probe)
    except Exception as exc:  # noqa: BLE001  WHY broad: the runner is the injected process boundary; ANY failure of it is data for the
        # installer's fallback, and raising here would abort an install over an optional measurement.
        _log.warning("lead probe: the runner crashed: %s", describe(exc))
        return void(f"the probe with {card.name} first crashed: {type(exc).__name__}: {exc}")
    if not run.started:
        _log.warning("lead probe: the engine did not start: %s", run.why or "no reason given")
        return void(f"the engine did not start with {card.name} first: {run.why or 'no reason given'}")
    ms, why = {}, {}
    for k in classes:
        val, reason = parse_timing(run.texts.get(f"lead{k}", ""))
        if val is None:
            why[k] = f"{card.name} first, {k} stream{'s' * (k != 1)}: {reason}" + (f" ({run.why})" if run.why else "")
        else:
            ms[k] = val
    return LeadResult(ms, why)


def measure_lead(candidates: Iterable, plan: LeadPlan, streams_classes: Iterable[int], runner: LeadRunner) -> dict:
    """One engine start per candidate (at most MAX_CANDIDATES, in the order given: best-first by the heuristic), measuring each
    requested class (1 and/or 4).  Returns {(vendor, index): LeadResult}; a candidate past the cap is simply absent."""
    classes = tuple(sorted({k for k in streams_classes if k in (1, 4)}))
    if not classes:
        return {}
    return {(c.vendor, c.index): _measure_one(c, plan, classes, runner) for c in tuple(candidates)[:MAX_CANDIDATES]}


def _complete_classes(results: Mapping[CardId, LeadResult], cands: tuple) -> dict:
    """{class: {card: ms}} for each class that EVERY candidate has (a partial map would rank measured numbers against guesses)."""
    out = {}
    for k in sorted({k for r in results.values() for k in (*r.ms, *r.why)}):
        got = {(c.vendor, c.index): results[(c.vendor, c.index)].ms[k] for c in cands
               if (c.vendor, c.index) in results and k in results[(c.vendor, c.index)].ms}
        if len(got) == len(cands):
            out[k] = got
    return out


def to_lead_ms(results: Mapping[CardId, LeadResult], candidates: Iterable) -> tuple:
    """({class: {card: ms}}, why) for _bench.BenchResult: a class appears only when EVERY candidate has it (a partial map would rank
    measured numbers against guesses); `why` lists what is missing, including candidates the cap left unmeasured."""
    cands = tuple(candidates)
    out, notes = _complete_classes(results, cands), []
    for r in results.values():
        notes += list(r.why.values())
    skipped = [c.name for c in cands if (c.vendor, c.index) not in results]
    if skipped:
        notes.append(f"not probed (only the top {MAX_CANDIDATES} are): {', '.join(skipped)}")
    return out, "; ".join(notes)
