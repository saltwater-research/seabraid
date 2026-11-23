# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""seabraid installer, stage 2 (PLAN): who is main, who helps, where the experts live.  UNTESTED-ON-HARDWARE.

OWNER     : this file owns the types Plan, Role, Build, TierSizes, Blocked, ModelFacts and every placement decision
            (card ranking, VRAM/RAM/NVMe tier sizes, the AMD GTT safety check).  Nothing downstream re-decides.
CONTRACT  : make_plan(report, model, model_path, bench=None, opts=PlanOptions()) -> Plan;  read_model_facts(paths)
            -> ModelFacts (layers/experts/bytes read from GGUF headers, never hardcoded);  Plan.explain() -> str.
            Imports: stdlib only.  The detect report is taken DUCK-TYPED through adapt_report() (the single place
            that touches its attributes), so detect.py's real dataclasses swap in without edits here.
            Import direction: detect <- plan <- config <- build/services <- setup.
FAILURE   : an unknown that feeds a decision refuses that path with a reason (card excluded, tier refused, plan not
            ready).  Cosmetic unknowns pass through.  Nothing is guessed and nothing is applied to the host: the
            AMD GTT remedy is only OFFERED in Plan.blocked.  An unreadable GGUF raises ModelError.
CHECKER   : tests/install/test_plan.py (each rule has a test; one was sabotaged and seen to fail).
Linux first; Windows is out of scope.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, NamedTuple, Optional, Sequence

from ._plan_gguf import ModelError  # noqa: F401 (re-exported)
from ._plan_model import ModelFacts, facts_from_tensors, read_model_facts  # noqa: F401 (re-exported)
from ._plan_opts import GIB, PlanOptions, dense_on_card, helper_reserve, main_reserve  # noqa: F401 (re-exported)
from ._plan_cards import (  # noqa: F401 (the detect adapter lives in _plan_cards; re-exported so plan stays the one import)
    SUPPORTED, Candidate, CardKey, _Card, adapt_report, candidates, usable_cards)
from ._plan_cards import blocked_by as _blocked_by, held_by as _held_by, ordinals as _ordinals, why as _why
from ._plan_fs import move_advice, under
from seabraid.ranking import LEAD_NOT_RUN, Slot, admit, lead_class, rank, rank_note
from ._plan_shm import BACKEND_WINDOW_MAX_T, channel_bytes

CONTEXT_ALL_VRAM = 131072     # --max-context of configs/example-iq2xs-all-vram.json (the all-VRAM run behind main_overhead_bytes)
# Strata holds three remote expert tiers (`RemoteExperts* remote[3]`, expert_source.hpp), one per helper process, so a plan has at
# most this many helpers; the 4th card is refused with a reason, never silently dropped.  The one owner of the cap (config trusts
# Plan.ready).  Seabraid itself takes 63 rungs; the limit is Strata's.
MAX_HELPERS = 3
CONTEXT_TIERED = 32768        # --max-context of configs/example-iq3s-tiered.json; DESIGN-LOG: "IQ3_S at --max-context 32768: ~2 GB VRAM back"


@dataclass(frozen=True)
class Role:
    kind: str                       # "main" | "helper"
    gpu_index: int                  # detect's Gpu.index
    vendor: str
    name: str
    arch: str                       # as detected
    device_id: int                  # ordinal among ALL detected same-vendor cards, by index (the runtime's "device N")
    helper_index: Optional[int]     # N in SEABRAID_STRATA_SHM=/seabraidN; None for main
    vram_budget_bytes: int          # free VRAM minus this role's reserve
    uuid: Optional[str] = None      # NVIDIA UUID when detect knows it (pins the process more stably than an index)

    @property
    def key(self) -> CardKey:
        return (self.vendor, self.gpu_index)

    @property
    def shm(self) -> Optional[str]:
        return None if self.helper_index is None else f"/seabraid{self.helper_index}"


@dataclass(frozen=True)
class Build:
    vendor: str
    kind: str                       # "cuda" | "rocm"
    targets: tuple                  # "strata" (if main vendor) and/or "seabraid-helper"
    archs: tuple                    # distinct archs of the DETECTED cards of this vendor, as detected
    gpu_indices: tuple
    toolkit_root: str
    toolkit_version: str
    compiler: str
    host_compiler: Optional[str]
    lib_dirs: tuple
    host_compiler_tried: tuple = ()


@dataclass(frozen=True)
class TierSizes:
    vram_bytes: Mapping[CardKey, int]   # (vendor, index) -> expert bytes placed on that card
    ram_bytes: int
    nvme_bytes: int
    nvme_mount: Optional[str]       # filesystem holding the model, used only when nvme_bytes > 0


@dataclass(frozen=True)
class Blocked:
    tier: str
    wanted_bytes: int
    reason: str
    remedy: str                     # offered text; never applied by this tool


@dataclass(frozen=True)
class Plan:
    model: ModelFacts
    mode: str                       # "all-vram" | "tiered" | "infeasible"
    roles: tuple
    builds: tuple
    tiers: TierSizes
    blocked: tuple = ()
    refusals: tuple = ()            # reasons the plan cannot be satisfied
    excluded: tuple = ()            # (CardKey, reason)
    notes: tuple = ()
    ranking: str = "heuristic"      # what chose card 1: "bench" (measured lead job) | "user" | "heuristic" (proxy; NOT measured)
    shm_bytes_per_helper: int = 0
    window_max_t: int = 0           # resolved main value; -DSTRATA_WINDOW_MAX_T for EVERY build (0 = unset)
    kv_reserve_bytes: Mapping = field(default_factory=dict)   # CardKey -> VRAM held back for KV cache + compute (main excludes dense weights)
    assumed_max_context: int = CONTEXT_ALL_VRAM   # per mode (explicit PlanOptions value wins); config emits --max-context from it
    rank_note: str = ""            # why main is main (measured or heuristic, and why not measured)
    rest_ranking: str = "heuristic"  # what ordered cards 2..N: "bench" (measured expert rows/s) | "heuristic"

    @property
    def ready(self) -> bool:      # True only when config may turn this into args: feasible, nothing BLOCKED or refused
        return self.mode != "infeasible" and not self.blocked and not self.refusals and bool(self.roles)

    def main(self) -> Role:
        return next(r for r in self.roles if r.kind == "main")

    def helpers(self) -> tuple:
        return tuple(r for r in self.roles if r.kind == "helper")

    def explain(self) -> str:
        return "\n".join(_explain_head(self) + _explain_tail(self))


def _gib(b: int) -> str:
    return f"{b / GIB:.1f} GiB"


def _rung(r: "Role") -> str:
    """What ranking order means for a helper: its channel, the engine's cache flag and the Seabraid rung, all by the same number."""
    if r.helper_index is None:
        return " (rung 0)"
    n = r.helper_index
    return f"; {r.shm} = --expert-cache-device{n + 1} = rung {n + 1}"


def _explain_head(p: "Plan") -> list:
    """The title, the model, and one line per card."""
    out = [f"seabraid plan: {p.mode.upper()} ({'ready' if p.ready else 'NOT READY'}), card 1 ranked by {p.ranking}, "
           f"cards 2..N by {p.rest_ranking}",
           *([f"  {p.rank_note}"] if p.rank_note else []),
           f"Model: {p.model.layers} layers x {p.model.experts} experts, experts {_gib(p.model.expert_bytes)}, "
           f"dense on the main card {_gib(p.model.dense_bytes)} (+{_gib(p.model.ple_bytes)} PLE streamed from disk, not in VRAM)"]
    for r in p.roles:
        who = "MAIN   " if r.kind == "main" else f"HELPER{r.helper_index}"
        out.append(f"  {who} {r.vendor} gpu{r.gpu_index} {r.name} [{r.arch}] experts on card: "
                   f"{_gib(p.tiers.vram_bytes.get(r.key, 0))} of {_gib(r.vram_budget_bytes)} budget" + _rung(r))
    return out


def _explain_tail(p: "Plan") -> list:
    """The tiers, the builds, the channel size, then exclusions, notes, blocked tiers and refusals."""
    out = [f"  RAM tier {_gib(p.tiers.ram_bytes)}; NVMe tier {_gib(p.tiers.nvme_bytes)}"
           + (f" on {p.tiers.nvme_mount}" if p.tiers.nvme_bytes else "")]
    out += [f"  build: {b.kind} for {','.join(b.archs)} -> {' + '.join(b.targets)}" for b in p.builds]
    if p.shm_bytes_per_helper:
        out.append(f"  /dev/shm: {len(p.helpers())} channel(s) x {p.shm_bytes_per_helper:,} B")
    out += [f"  excluded {k[0]} gpu{k[1]}: {why}" for k, why in p.excluded]
    out += [f"  note: {n}" for n in p.notes]
    for b in p.blocked:
        out += [f"  BLOCKED {b.tier} tier ({_gib(b.wanted_bytes)}): {b.reason}", f"    offered remedy (not applied): {b.remedy}"]
    return out + [f"  REFUSED: {r}" for r in p.refusals]


# ---- the planner -----------------------------------------------------------------------------------------------
def _fs_for(path: str, fss: list) -> Optional[tuple]:
    best = None
    for mount, rot in fss:
        m = mount.rstrip("/") or "/"
        under = path == m or path.startswith(m.rstrip("/") + "/")
        if under and (best is None or len(m) > len(best[0])):
            best = (m, rot)
    return best


def _targets(vendor: str, roles: list, mine: list) -> tuple:
    """The make targets for one vendor: the engine if it holds main, and the helper when anything else needs it."""
    # FIX (2026-11-09 install): the helper binary was built only for vendors holding a helper card, so --bench (which times
    # EVERY card, main included, through seabraid-helper) had no binary for the main card's vendor.  With 2+ cards every
    # vendor present gets one (~10 s link); a one-card plan has nothing to rank, so it still builds the engine alone.
    want_helper = len(roles) > 1 or any(r.kind == "helper" for r in mine)
    return tuple((["strata"] if vendor == roles[0].vendor else []) + (["seabraid-helper"] if want_helper else []))


def _build_for(vendor: str, roles: list, kits: dict) -> Build:
    t, mine = kits[vendor], [r for r in roles if r.vendor == vendor]
    return Build(vendor, SUPPORTED[vendor], _targets(vendor, roles, mine), tuple(sorted({r.arch for r in mine})),
                 tuple(r.gpu_index for r in mine), str(t.root), str(t.version), str(t.compiler),
                 getattr(t, "host_compiler", None), tuple(getattr(t, "lib_dirs", ()) or ()),
                 tuple(getattr(t, "host_compiler_tried", ()) or ()))


def _builds(roles: list, kits: dict) -> tuple:
    return tuple(_build_for(v, roles, kits) for v in dict.fromkeys(r.vendor for r in roles))    # one build per vendor present among the roles


class _Tiering(NamedTuple):
    """Where the experts that do not fit VRAM go, and what that decided."""
    mode: str
    ram: int
    nvme: int
    mount: Optional[str]
    blocked: list
    refusals: list
    notes: list
    shm: int


def _rank_cards(usable: list, bench_lead, bench, model: ModelFacts, opts: PlanOptions) -> tuple:
    """(ranked cards, how they were ranked, fits_main).  fits_main(c): can c hold main's resident set at the SMALLER (tiered) context."""
    # WHY the smaller (tiered) context here: a card that cannot hold main's resident set even at the smaller context can
    # never be main; at the larger one the tiered retry would reconsider it anyway.
    ctx_gate = opts.assumed_max_context or CONTEXT_TIERED
    fits_main = lambda c: c.free - _main_reserve(opts, model, c.vendor, c.arch, ctx_gate) > 0  # noqa: E731
    # WHY filtered here: --main-card may name a card that is not usable (absent or excluded); rank() refuses a choice that
    # names no card, and this caller reports that case itself ("--main-card ... is not among the usable cards").
    forced = opts.main_card if opts.main_card in {c.key for c in usable} else None
    ranked, how = rank(usable, bench_lead, bench, fits_main, opts.streams, forced)
    return ranked, how, fits_main


def _assign_roles(ranked: list, ordinal: dict, model: ModelFacts, opts: PlanOptions) -> tuple:
    """(roles, exclusions, context): the all-VRAM context first; if the experts then do not fit, the smaller tiered one."""
    ctx = opts.assumed_max_context or CONTEXT_ALL_VRAM
    roles, excl = _roles(ranked, ordinal, model, opts, ctx)
    if not opts.assumed_max_context and sum(r.vram_budget_bytes for r in roles) < model.expert_bytes:
        ctx = CONTEXT_TIERED                          # tiered plans run the smaller context (the KV reserve follows it)
        roles, excl = _roles(ranked, ordinal, model, opts, ctx)
    return roles, excl, ctx


def _admission(roles: list, model: ModelFacts, opts: PlanOptions) -> tuple:
    """(roles, dropped, notes) after R3: a slow card must hold experts that would otherwise sit in RAM/NVMe."""
    if not (opts.admission and roles):
        return roles, [], []
    adm, drop = admit([Slot(r.key, r.name, r.vram_budget_bytes) for r in roles], model.expert_bytes, opts.floors)
    kept = [r for r in roles if r.key in {a.slot.key for a in adm}]
    kept = [r if r.kind == "main" else replace(r, helper_index=i - 1) for i, r in enumerate(kept)]   # /seabraidN stays 0..n-1
    return kept, drop, [f"{a.slot.name}: {a.note}" for a in adm[1:]]


def _fill_vram(roles: list, expert_bytes: int) -> tuple:
    """({card: expert bytes on it}, bytes left over): the hottest experts fill the fastest card first."""
    left, vram = expert_bytes, {}
    for r in roles:
        vram[r.key] = min(left, r.vram_budget_bytes)
        left -= vram[r.key]
    return vram, left


def _tiering(left: int, ranked: list, roles: list, h: dict, model: ModelFacts, model_path: str, opts: PlanOptions, window: int) -> _Tiering:
    """Place what did not fit VRAM (RAM, then the model disk), size the helper channel, and settle the mode."""
    ram = nvme = 0
    mount, blocked, refusals, notes = None, [], [], []
    mode = "all-vram"
    if left > 0:
        mode = "tiered"
        main_card = next(c for c in ranked if c.key == roles[0].key)
        ram, nvme, mount = _lower_tiers(left, main_card, h, model, model_path, opts, blocked, refusals, notes)
    if len(roles) - 1 > MAX_HELPERS:
        refusals.append(f"{len(roles) - 1} helpers planned ({', '.join(r.name for r in roles[1:])}), but Strata holds three remote tiers: "
                        f"at most {MAX_HELPERS} helpers. take a card out of the machine (or hide it from the driver) and rerun")
    shm = _shm(len(roles) - 1, model, h, refusals, window)
    if refusals:
        mode = "infeasible"
    notes += [f"mlock requested but the memlock limit is {h['memlock'] / 1024:.0f} KiB"] * bool(opts.mlock and h["memlock"] is not None)
    return _Tiering(mode, ram, nvme, mount, blocked, refusals, notes, shm)


def _main_note(ranked: list, roles: list, how, bench_lead, bench, fits_main, opts: PlanOptions, bench_why: str) -> str:
    """Why main is main: measured or heuristic, and why not measured."""
    pos = next(i for i, c in enumerate(ranked) if c.key == roles[0].key)
    other = ranked[pos + 1] if pos + 1 < len(ranked) else None
    why = bench_why or (f"no lead-job regime for {opts.streams} streams (measured at 1 or >= 4)" if lead_class(opts.streams) is None else LEAD_NOT_RUN)
    return rank_note(ranked[pos], other, how, bench_lead, why, bool(other) and fits_main(other), bench, opts.streams)


def make_plan(report: Any, model: ModelFacts, model_path: str, bench: Optional[Mapping[CardKey, Optional[float]]] = None,
              opts: PlanOptions = PlanOptions(), bench_why: str = "", bench_lead: Optional[Mapping[CardKey, Optional[float]]] = None) -> Plan:
    """Rank cards (bench = {(vendor, index): decode expert rows/s}, which orders cards 2..N; bench_lead = {(vendor, index): ms per
    decode window of the lead job at opts.streams}, which alone picks card 1; each used only when every candidate has one, else a
    heuristic), size each role's expert budget, then fill VRAM -> RAM -> model-disk tier.  model_path = where the model FILES are.
    bench_why = the reason there is no lead measurement, printed in the plan when the heuristic decided card 1."""
    h = adapt_report(report)
    usable, excluded = usable_cards(h["cards"], h["kits"], h["blockers"])
    ranked, how, fits_main = _rank_cards(usable, bench_lead, bench, model, opts)
    roles, excl, ctx = _assign_roles(ranked, _ordinals(h["cards"]), model, opts)
    roles, drop, adm_notes = _admission(roles, model, opts)
    excluded += excl + drop
    if not roles:
        return Plan(model, "infeasible", (), (), TierSizes({}, 0, 0, None), refusals=("no usable GPU",), excluded=tuple(excluded))
    vram, left = _fill_vram(roles, model.expert_bytes)
    window = opts.window_max_t or BACKEND_WINDOW_MAX_T[h["kits"][roles[0].vendor].kind]    # main's value, propagated to helpers
    t = _tiering(left, ranked, roles, h, model, model_path, opts, window)
    notes = t.notes
    if opts.main_card and how.card1 != "user":
        notes.append(f"--main-card {opts.main_card[0]}:{opts.main_card[1]} is not among the usable cards; ranked normally")
    notes += adm_notes
    return Plan(model, t.mode, tuple(roles), _builds(roles, h["kits"]), TierSizes(vram, t.ram, t.nvme, t.mount), tuple(t.blocked),
                tuple(t.refusals), tuple(excluded), tuple(notes), {"lead": "bench", "user": "user"}.get(how.card1, "heuristic"), t.shm, window,
                {r.key: _main_reserve(opts, model, r.vendor, r.arch, ctx) if r.kind == "main" else helper_reserve(opts) for r in roles},
                ctx, _main_note(ranked, roles, how, bench_lead, bench, fits_main, opts, bench_why), "bench" if how.rest == "rows" else "heuristic")


def _main_reserve(opts: PlanOptions, model: ModelFacts, vendor: str, arch: str, ctx: int) -> int:
    dense = dense_on_card(opts, model.hc_stored_bytes, model.hc_q8_bytes, model.dense_bytes, SUPPORTED[vendor], arch)
    return main_reserve(opts, dense, model.head_bytes, ctx)


def _roles(ranked: list, ordinal: dict, model: ModelFacts, opts: PlanOptions, ctx: int) -> tuple:
    """Main takes the full reserve; every helper only its small runtime reserve."""
    roles, excl = [], []
    for c in ranked:
        is_main = not roles
        reserve = _main_reserve(opts, model, c.vendor, c.arch, ctx) if is_main else helper_reserve(opts)
        if c.free - reserve <= 0:
            excl.append((c.key, f"free VRAM {c.free / GIB:.1f} GiB does not cover its {reserve / GIB:.1f} GiB reserve; {_held_by(c.holders)}"))
            continue
        roles.append(Role("main" if is_main else "helper", c.index, c.vendor, c.name, c.arch, ordinal[c.key],
                          None if is_main else len(roles) - 1, c.free - reserve, c.uuid))
    return roles, excl


def _shm(n_help: int, model: ModelFacts, h: dict, refusals: list, window: int) -> int:
    """Bytes per helper channel (patch 0001 Layout); refuses when /dev/shm free < the sum over helpers (layout is fixed: no shrinking)."""
    if n_help == 0:
        return 0
    per = channel_bytes(model.layers, model.experts, model.hidden, model.blob_max_bytes, window)
    if per is None:
        refusals.append("model hidden size or expert blob size unknown, cannot size the /dev/shm channel")
        return 0
    shm = [f for f in h["fs"] if f[0] == "/dev/shm"]
    if not shm or shm[0][2] is None:
        refusals.append("/dev/shm free space unknown, cannot size the helper channel")
        return 0
    if shm[0][2] < per * n_help:
        refusals.append(f"/dev/shm has {shm[0][2]:,} B free; {n_help} helper channel(s) need {per * n_help:,} B "
                        f"({per:,} B each); remount /dev/shm larger")
        return 0
    return per


def _ram_tier(left: int, main: _Card, h: dict, opts: PlanOptions, blocked: list, notes: list) -> int:
    """Bytes of the leftover that go to RAM: what is available minus the headroom; an AMD main also needs the GTT route."""
    total, avail = h["ram_total"], h["ram_avail"]
    if total is None or avail is None:
        notes.append("RAM size unknown: no RAM tier planned")
        return 0
    headroom = max(opts.ram_headroom_floor_bytes, int(total * opts.ram_headroom_fraction))
    cap = max(0, avail - headroom)
    ram = min(left, cap)                          # the larger tier wins when it fits (10 GiB beat 6 GiB, IQ3_S)
    notes.append(f"RAM tier cap {cap / GIB:.1f} GiB = available {avail / GIB:.1f} - headroom {headroom / GIB:.1f}")
    if ram and main.vendor == "amd":                  # KFD userptr eviction freeze unless the GTT route is on
        _gtt_check(ram, main, h["ttm"], opts, blocked)
    return ram


def _disk_refusal(rest: int, fs: Optional[tuple], h: dict, model: ModelFacts, opts: PlanOptions) -> str:
    """Why the experts that fit neither VRAM nor RAM cannot go to the model's disk, with the way out."""
    kind = "of unknown type" if fs is None or fs[1] is None else "rotational"
    bl = _blocked_by(h["blockers"], "fs", fs[0] if fs else "")
    return (f"{rest / GIB:.1f} GiB of experts fit neither VRAM nor RAM and the model files are on {fs[0] if fs else 'a filesystem'} "
            f"({kind}), where no tier is ever placed; " + move_advice(model.expert_bytes + model.dense_bytes, h["fs"], opts.model_move_headroom)
            + (f" [{_why(bl)}]" if bl else ""))


def _lower_tiers(left: int, main: _Card, h: dict, model: ModelFacts, model_path: str, opts: PlanOptions, blocked: list,
                 refusals: list, notes: list) -> tuple:
    ram = _ram_tier(left, main, h, opts, blocked, notes)
    rest = left - ram
    if rest <= 0:
        return ram, 0, None
    # Native packs read the GGUF IN PLACE: the "NVMe tier" is not extra space, it is the model files' own disk,
    # and is valid only when that disk is known non-rotational (longest-prefix mount match).
    fs = max((f for f in h["fs"] if under(model_path, f[0])), key=lambda f: len(f[0]), default=None)
    if fs is not None and fs[1] is False:
        return ram, rest, fs[0]
    refusals.append(_disk_refusal(rest, fs, h, model, opts))
    return ram, 0, None


def _gtt_check(ram: int, main: _Card, ttm: Optional[int], opts: PlanOptions, blocked: list) -> None:
    caps = [x for x in (main.gtt, ttm) if x is not None]     # both cap the GTT route: the smaller binds
    need = ram + opts.gtt_margin_bytes
    if len(caps) < 2 or min(caps) < need:
        have = "unknown (card GTT or ttm.pages_limit not detected)" if len(caps) < 2 else f"{min(caps) / GIB:.2f} GiB"
        fits = "" if len(caps) < 2 else f" (the current cap carries a RAM tier up to {max(0, min(caps) - opts.gtt_margin_bytes) / GIB:.1f} GiB)"
        blocked.append(Blocked("ram", ram, f"AMD main with a RAM tier needs the GTT route; GTT cap is {have}, need >= "
                               f"{need / GIB:.2f} GiB (tier + {opts.gtt_margin_bytes / GIB:.0f} GiB){fits}",
                               f"raise ttm.pages_limit to >= {need // 4096} pages ({need / GIB:.1f} GiB) on the kernel "
                               "command line and reboot, or pick a smaller RAM tier"))
