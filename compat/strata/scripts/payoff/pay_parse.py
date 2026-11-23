# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Parsers for the engine and helper log lines the payoff harness reads (pure: text in, numbers out, no I/O).

OWNER: this module owns "which log line carries which number".  Every format below was read from source, not guessed:
  * ``strata serve: prompt ..``, ``decode expert cache hit rate``, ``expert tiers``, ``file tier I/O this request``,
    ``strata decode timing:`` and ``strata generate: expert cache N slots`` are Strata v0.2.15 src/program/generate.cpp
    (the ``--stats`` tiers are described in docs/DETAILS.md, "Reading the tiers");
  * ``strata serve: CUDA<r>: .. host B ms staging+launching, W ms waiting for it`` and ``strata serve: seabraid helper:``
    and ``prefill helper source`` are added by compat/strata/patches/strata/0001 (the ms_begin / ms_wait deltas and
    the helper's own service counters).
CONTRACT: ``parse_request(slice) -> RequestLog`` for the log text of ONE request; ``parse_startup(text) -> Startup``;
``last_tiers(text) -> Tiers | None``; ``components(log) -> Components`` (the explanation columns, see its docstring).
A field whose line is absent is None, never 0, so a missing measurement cannot read as "no waiting".
FAILURE: fail-open-loud for observability: an unmatched line is simply not counted; the caller flags a request whose
``prompt`` line is missing (``RequestLog.prompt is None``) so it is excluded rather than guessed.
CHECKER: compat/strata/tests/payoff/test_payoff_parse.py (synthetic lines built from the formats above).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_F = r"([\d.]+)"
_I = r"(\d+)"
PROMPT = re.compile(rf"strata serve: prompt {_I} tokens = {_I} reused \+ {_I}(?: of \d+)? read in {_I} ms \({_F} tok/s\), "
                    rf"{_I} generated in {_I} ms \({_F} tok/s\)")
HIT = re.compile(rf"decode expert cache hit rate: {_F}% \({_I} hits / {_I} lookups\)"
                 rf"(?:; {_I} more read by the GPU over PCIe or from another GPU \({_F}% of all {_I} routed\))?")
TIERS = re.compile(rf"expert tiers: GPU {_I} hits this request; since the start RAM {_I} blobs, files {_I} blobs {_F} MB read")
IO = re.compile(rf"file tier I/O this request: the OS read {_F} MB from storage \({_I} major faults\) for {_F} MB of expert reads")
IO_WAIT = re.compile(rf"decode waited {_F} ms in {_I} fetches")
REMOTE = re.compile(rf"strata serve: CUDA{_I}: {_I} expert entries, {_I} active layer launches, {_F} MiB returned "
                    rf"\({_F} MiB with full rows\) in this request; host {_F} ms staging\+launching, {_F} ms waiting for it")
HELPER = re.compile(rf"strata serve: seabraid helper: {_I} layer requests, {_F} us each on average in the helper "
                    rf"\({_F} us staging\+launching, {_F} us waiting for its GPU\)")
PREFILL_HELPER = re.compile(rf"prefill helper source: {_I} blobs / {_F} MiB")
# serial loop: "= adapt wait X + verify ..."; pipelined loop: "= verify ..." (docs/DETAILS.md: a script must accept both)
TIMING = re.compile(rf"strata decode timing: {_I} windows, avg T {_F}, {_F} tokens/window, {_F} ms/window = "
                    rf"(?:adapt wait {_F} \+ )?verify {_F} \(GPU-reach wait {_F} \+ per-layer host {_F} "
                    rf"\[plan {_F} actq {_F} jobs {_F} CPU {_F}\] \+ stage {_F}\).*?"
                    rf"per layer-window: CPU experts {_F} \({_F} entries\), VRAM hits {_F}, PCIe {_F}")
CAP = re.compile(rf"expert cache {_I} slots, {_F} GiB of VRAM")
HELPER_UP = re.compile(r"seabraid: helper .+ on channel")  # the helper name is a device string and may hold spaces
RESIDENT = re.compile(rf"resident RAM: {_F} GiB of experts in RAM")


@dataclass(frozen=True)
class Prompt:
    prompt_tokens: int
    reused: int
    read_tokens: int
    prefill_ms: float
    prefill_tps: float
    generated: int
    decode_ms: float
    decode_tps: float


PROMPT_KEYS = tuple(Prompt.__dataclass_fields__)


@dataclass(frozen=True)
class Timing:
    """The ``strata decode timing`` line: ms values are per decode window; the last four are per layer-window."""

    windows: int
    ms_per_window: float
    gpu_reach_wait: float
    host: float
    cpu: float
    stage: float
    cpu_experts: float
    vram_hits: float
    pcie: float


@dataclass(frozen=True)
class Remote:
    """One ``CUDA<r>`` line: card index ``card`` (1 = the first helper), request counts and the host's two times in ms."""

    card: int
    entries: int
    launches: int
    staging_ms: float
    wait_ms: float


@dataclass(frozen=True)
class Tiers:
    ram_blobs: int
    file_blobs: int
    file_mb: float


@dataclass
class RequestLog:
    prompt: Prompt | None = None
    timing: Timing | None = None
    hits: int | None = None
    lookups: int | None = None
    offload: int | None = None  # experts read by the GPU over PCIe or from another GPU, as the engine counts them
    tiers: Tiers | None = None
    os_read_mb: float | None = None
    io_wait_ms: float | None = None
    remote: list[Remote] = field(default_factory=list)
    helper_requests: int | None = None
    helper_us_each: float | None = None
    prefill_helper_mib: float | None = None


@dataclass(frozen=True)
class Startup:
    cap_slots: int | None
    cap_gib: float | None
    helper_up: bool
    resident_gib: float | None


@dataclass(frozen=True)
class Components:
    """Where a request's decode time went, in ms for the whole request; None = the engine printed nothing for it.

    staging: ``stage`` term of the decode timing (host work to stage experts for the GPU, PCIe copies included) plus the
      host's "staging+launching" ms toward a helper.  merge_wait: the host's "waiting for it" ms past its own work, i.e.
      the stall at the merge with the helper's answer (arm C only: A and B have no merge partner).  local_compute: the
      CPU-pool term (experts computed in place from RAM).  gpu_reach_wait: the engine's own "GPU-reach wait" term, kept
      apart because it is a different wait.  io_wait: decode time blocked on file-tier reads.  They overlap and need
      not add up to decode_ms: the table shows each beside decode_ms, never as a partition."""

    staging_ms: float | None
    merge_wait_ms: float | None
    local_compute_ms: float | None
    gpu_reach_wait_ms: float | None
    io_wait_ms: float | None


def _last(rx: re.Pattern, text: str):
    found = None
    for found in rx.finditer(text):
        pass
    return found


def _prompt(text: str) -> Prompt | None:
    m = _last(PROMPT, text)
    if not m:
        return None
    n, reused, read, p_ms, p_tps, gen, d_ms, d_tps = m.groups()
    return Prompt(int(n), int(reused), int(read), float(p_ms), float(p_tps), int(gen), float(d_ms), float(d_tps))


def _timing(text: str) -> Timing | None:
    m = _last(TIMING, text)
    if not m:
        return None
    g = m.groups()
    # groups: windows, avgT, tok/window, ms/window, adapt wait?, verify, reach, host, plan, actq, jobs, cpu, stage, 4 per-layer
    return Timing(int(g[0]), float(g[3]), float(g[6]), float(g[7]), float(g[11]), float(g[12]), float(g[13]), float(g[15]), float(g[16]))


def _hit(text: str, log: RequestLog) -> None:
    m = _last(HIT, text)
    if m:
        log.hits, log.lookups = int(m.group(2)), int(m.group(3))
        log.offload = int(m.group(4)) if m.group(4) else 0  # the engine omits the clause when nothing went over PCIe


def _tiers(text: str) -> Tiers | None:
    m = _last(TIERS, text)
    return Tiers(int(m.group(2)), int(m.group(3)), float(m.group(4))) if m else None


def _io(text: str, log: RequestLog) -> None:
    m = _last(IO, text)
    if m:
        log.os_read_mb = float(m.group(1))
    w = _last(IO_WAIT, text)
    if w:
        log.io_wait_ms = float(w.group(1))


def _helper(text: str, log: RequestLog) -> None:
    log.remote = [Remote(int(m[1]), int(m[2]), int(m[3]), float(m[6]), float(m[7])) for m in REMOTE.finditer(text)]
    m = _last(HELPER, text)
    if m:
        log.helper_requests, log.helper_us_each = int(m.group(1)), float(m.group(2))
    p = _last(PREFILL_HELPER, text)
    if p:
        log.prefill_helper_mib = float(p.group(2))


def parse_request(text: str) -> RequestLog:
    """Everything the engine printed for one request (the slice of the log between sending it and the log going quiet)."""
    log = RequestLog(prompt=_prompt(text), timing=_timing(text), tiers=_tiers(text))
    _hit(text, log)
    _io(text, log)
    _helper(text, log)
    return log


def last_tiers(text: str) -> Tiers | None:
    """The cumulative tiers line of the log so far: the baseline the per-request tier deltas are taken against."""
    return _tiers(text)


def parse_startup(text: str) -> Startup:
    """Facts printed once at start: the expert-cache slot count, whether the helper channel came up, the RAM copy size."""
    c, r = _last(CAP, text), _last(RESIDENT, text)
    return Startup(int(c.group(1)) if c else None, float(c.group(2)) if c else None, bool(HELPER_UP.search(text)),
                   float(r.group(1)) if r else None)


def _timing_parts(t: Timing | None) -> tuple[float | None, float | None, float | None]:
    """(stage, cpu, gpu-reach wait) in ms for the whole request: the per-window terms times the windows."""
    return (t.stage * t.windows, t.cpu * t.windows, t.gpu_reach_wait * t.windows) if t else (None, None, None)


def components(log: RequestLog) -> Components:
    """Per-request explanation columns (see ``Components``); per-window terms are multiplied by the request's windows."""
    stage, cpu, reach = _timing_parts(log.timing)
    remote_stage = sum(r.staging_ms for r in log.remote) if log.remote else None
    parts = [x for x in (stage, remote_stage) if x is not None]
    merge = sum(r.wait_ms for r in log.remote) if log.remote else None
    return Components(sum(parts) if parts else None, merge, cpu, reach, log.io_wait_ms)
