# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""seabraid installer, stage 3 (CONFIG): the knobs, with the evidence for each.

OWNER     : this file owns ConfigOutput, ModelPaths, ConfigOptions and the VALUE
            of every tuned knob.  Placement is not re-decided here -- stage 2
            owns that, and this stage refuses rather than second-guesses it.
CONTRACT  : make_config(report, plan, paths, opts=ConfigOptions()) -> ConfigOutput
            write_config(out, directory) -> path to the cfg.json actually written
            Import direction: detect <- plan <- config <- build/services <- setup.
FAILURE   : fail-fast.  A plan that is not ready raises ConfigRefused, because
            emitting a tier the planner declined would hand the engine a
            configuration nobody approved.  Everything softer is a KnobWarning
            carried on the output and printed; nothing here touches the host.
CHECKER   : tests/install/test_config.py, which asserts the generated cfg.json
            against the two configs in configs/ that were actually measured.
SIZE      : target ~330 lines, hard cap 500.

WHY THE VALUES ARE CONSTANTS HERE AND NOT DERIVATIONS.  Every tuned number
below came from an A/B run on one machine, and each carries the result that
justifies it.  Where no measurement exists, the knob is NOT invented: it is an
option with a stated default and a note saying so.  A derivation that looks
principled and is wrong is worse than a constant that admits what it is.
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional
from . import _file_moves, _ladder
from ._rocm import rocm_lib_dirs

GIB = 1 << 30


class ConfigRefused(Exception):
    """The plan cannot be turned into a launch configuration."""


EXPERT_PROFILE_MISSING = (
    "no static expert profile: expected <work-dir>/strata/data/expert-profile.bin (the Strata checkout's own file). The engine's "
    "--resident-experts refuses to start without one. Fetch Strata (a build does) or put the profile there, then rerun.")


# --------------------------------------------------------------- the knobs

# Measured: --draft-conf 0.35 at --spec 6 gave ~116-122 tok/s against ~106 at
# the default.  Pushing further LOST: spec 8 gave 113.2/118.0, conf 0.25 gave
# 107.4/112.3 and conf 0.15 gave 100.4/104.5, all below 0.35/spec 6 at
# 116.1/121.5.  So this is a measured optimum, not a direction to keep walking.
DRAFT_CONF = "0.35"
SPEC = "6"

# #5109: on a mixed-vendor host Strata's drafting.apply sizes the draft from the
# wrong card, so the value has to be re-asserted where the server cannot
# recompute it.  This is a workaround for an upstream bug, and it is the reason
# --draft-conf appears twice in a generated config.
USER_ARGS_REASSERTED = ("--draft-conf", DRAFT_CONF)

ENGINE_ENV: Mapping[str, str] = {
    # Measured: same speed, and byte-identical greedy output -- the tail skip is
    # an RDNA4 default that costs accuracy here for nothing.
    "STRATA_ROUTE_TAIL_SKIP": "0",
    # Carried from the configuration that produced the verified 116.6/121.1
    # tok/s run.  Its individual effect was measured once at +4.7% and then
    # RETRACTED as noise from a single pair of runs, so it is here because
    # deviating from a known-good configuration on unproven grounds is the
    # riskier move -- not because it is known to help.
    "STRATA_VERIFY_QDEDUP": "1",
}

# The ONE definition of the server's fixed environment (services and the lead bench render it; neither keeps a copy).
SERVER_ENV: Mapping[str, str] = {
    # The helper owns its own experts; counting them against the host RAM tier
    # double-books memory on a host whose failure mode is host-memory pressure
    # evicting GPU queues.
    "SEABRAID_STRATA_EXCLUDE_HELPER_RAM": "1",
    "STRATA_PREFILL_HELPER_SOURCE": "1",
    # Upstream, opt-in (#731, generate.cpp): the adaptive tiers leave an expert
    # a helper holds out of the primary's set.  Without it the two cards fight
    # over the same experts.
    "STRATA_DISJOINT_ADAPT": "1",
}

# AMD main + ANY host-RAM tier.  Without this route the KFD userptr path lets
# the MMU notifier evict the queues and the GPU freezes; RDNA has no XNACK to
# recover.  This is a hang, not a slowdown, which is why it is not optional.
GTT_ROUTE_ENV: Mapping[str, str] = {
    "HSA_USERPTR_FOR_PAGED_MEM": "0",
    "HSA_SVM_FOR_PAGED_MEM": "0",
    "STRATA_HIP_ARENA_NONCOHERENT": "1",
    "GPU_PINNED_MIN_XFER_SIZE": "1048576",
}

@dataclass(frozen=True, slots=True)
class KnobWarning:
    """Something the operator should see but that does not stop a launch."""
    knob: str
    what: str
    fix: str


@dataclass(frozen=True, slots=True)
class ModelPaths:
    """Where the model's files ended up.

    Produced by the CLI from Strata's own setup.py -- we do not reimplement the
    download, the pack or the tokenizer -- and consumed here.  Every one is
    absolute, because systemd has no working directory of ours.
    """

    exe: str
    cwd: str
    pack: str
    native: str
    tokenizer: str
    model_name: str
    log: str
    ple_gguf: Optional[str] = None
    draft_model: Optional[str] = None
    draft_vocab: Optional[str] = None
    expert_profile: Optional[str] = None
    helper_exe: Optional[str] = None


@dataclass(frozen=True, slots=True)
class ConfigOptions:
    port: int = 8080
    host: str = "127.0.0.1"
    # There is no context default here.  Stage 2 owns the number, because it is
    # the stage that held VRAM back for the KV cache; a default here would be a
    # second owner that silently disagrees with the tier sizing.
    max_context: Optional[int] = None
    kv: str = "int8"
    draft: str = "dflash2"
    extra_args: tuple = ()
    extra_env: Mapping[str, str] = field(default_factory=dict)
    # Live moves (_ladder.py), opt-in until measured; ladder_fallback = --retier-fallback; file_moves (_file_moves.py) needs the ladder.
    ladder: bool = False
    ladder_fallback: bool = True
    file_moves: bool = False


@dataclass(frozen=True, slots=True)
class ConfigOutput:
    """Everything stage 5 needs to launch, and nothing it has to recompute."""

    port: int
    cfg_json: Mapping[str, Any]
    cfg_json_path: str
    server_env: Mapping[str, str]
    helper_args: Mapping[int, tuple]
    helper_env: Mapping[int, Mapping[str, str]]
    notes: tuple = ()
    warnings: tuple = ()

    def explain(self) -> str:
        out = ["seabraid config: port %d, %d helper(s)" % (self.port, len(self.helper_args))]
        out += ["  note: %s" % n for n in self.notes]
        for w in self.warnings:
            out += ["  WARNING [%s] %s" % (w.knob, w.what), "    fix: %s" % w.fix]
        return "\n".join(out)


# ------------------------------------------------------------------ helpers


def _mask_env(role, gpus_by_key: Mapping[tuple, Any]) -> dict:
    """Pin one process to one card.

    NVIDIA is pinned BY UUID where one was detected: CUDA_DEVICE_ORDER does not
    default to PCI bus order, so an integer in CUDA_VISIBLE_DEVICES is read
    against an ordering that depends on which cards are present, and a UUID is
    not.  CUDA_DEVICE_ORDER is still set, so that the ordinal fallback is at
    least deterministic.
    """
    # Keyed by (vendor, index), never index alone: Gpu.index is a PER-VENDOR
    # ordinal, so on the two-vendor machine this project exists for, NVIDIA 0
    # and AMD 0 collide and one card silently answers for the other.
    gpu = gpus_by_key.get((role.vendor, role.gpu_index))
    if role.vendor == "nvidia":
        uuid = getattr(gpu, "uuid", None)
        return {
            "CUDA_VISIBLE_DEVICES": uuid or str(role.device_id),
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        }
    if role.vendor == "amd":
        return {"HIP_VISIBLE_DEVICES": str(role.device_id)}
    return {}


def _lib_dirs(report, vendor: str, isdir: Callable[[str], bool]) -> tuple:
    """The library directories a process of this vendor needs on its path.

    rocm_sysdeps is appended when it exists because a TheRock install keeps its
    own libstdc++ and friends there, and leaving it off produces a symbol error
    at engine start that looks nothing like a missing ROCm.
    """
    tk = report.toolkit_for(vendor)
    if tk is None:
        return ()
    # One owner for the sysdeps probe: the leaf _rocm (BUILD uses the same function; config must not import BUILD).
    return rocm_lib_dirs(tuple(tk.lib_dirs), isdir)


def _engine_args(plan, paths: ModelPaths, context: int, opts: ConfigOptions) -> list:
    """The engine command line.  Everything that can come from the pack does."""
    if not paths.expert_profile:
        # --resident-experts below makes the engine exit 2 ("requires --mmap-experts and a static --expert-profile") without one;
        # refusing here names the missing file at plan time instead of a cryptic engine exit at start.
        raise ConfigRefused(EXPERT_PROFILE_MISSING)
    args = ["--pack", paths.pack, "--native", paths.native]
    if paths.ple_gguf:
        args += ["--ple-gguf", paths.ple_gguf]
    args += ["--expert-profile", paths.expert_profile]
    args += [
        "--expert-cache", "auto",
        "--prefill", "auto",
        "--max-context", str(context),
        "--kv", opts.kv,
        "--resident-experts",
    ]
    if paths.draft_model:
        args += ["--draft", opts.draft, "--draft-model", paths.draft_model, "--draft-fallback", "none"]
        if paths.draft_vocab:
            args += ["--mtp-draft-vocab", paths.draft_vocab]
        args += ["--spec", SPEC, "--draft-conf", DRAFT_CONF]
    # One --expert-cache-deviceN per helper: the engine has to be told that a
    # second cache exists at all, and "auto" lets it size from what the helper
    # reports rather than from a number we would have to keep in step.  N is the
    # card id (helper_index + 1), so the flag order is the rung order.
    k = len(plan.helpers())
    for n in range(1, k + 1):
        args += ["--expert-cache-device%d" % n, "auto"]
    args += _ladder.rendered(k, opts.ladder, opts.ladder_fallback, _file_moves.ENGINE_ARGS if opts.file_moves else ())
    if plan.tiers.ram_bytes > 0:
        args += ["--resident-budget-gib", "%g" % round(plan.tiers.ram_bytes / GIB, 2)]
    args += list(opts.extra_args)
    return args


# -------------------------------------------------------------------- stage


def _resolve_context(plan, opts: ConfigOptions, notes: list, warnings: list) -> int:
    """Context precedence, most specific first.  Stage 2 owns the number when it
    has one, because it is the stage that held VRAM back for the KV cache;
    CONFIG must not pick a context the tier sizing did not budget for."""
    planned = getattr(plan, "assumed_max_context", None)
    if opts.max_context is not None:
        context = opts.max_context
        notes.append("max-context %d was given explicitly" % context)
        if planned and context > planned:
            warnings.append(
                KnobWarning(
                    knob="max-context",
                    what="the requested context %d is larger than the %d the tier sizing "
                    "reserved VRAM for, so the engine may run out of room on the main card"
                    % (context, planned),
                    fix="re-plan with this context, or drop --max-context to use %d" % planned,
                )
            )
        return context
    if planned:
        notes.append("max-context %d comes from the plan, which sized the KV reserve for it" % int(planned))
        return int(planned)
    raise ConfigRefused(
        "the plan carries no assumed_max_context, and this stage will not invent one: "
        "the context has to match the VRAM the tier sizing reserved for the KV cache. "
        "Re-plan, or pass --max-context explicitly."
    )


def _engine_env(plan, main, opts: ConfigOptions, notes: list) -> dict:
    engine_env = dict(ENGINE_ENV)
    if opts.ladder:
        engine_env.update(_ladder.LADDER_ENGINE_ENV)
    if plan.tiers.ram_bytes > 0 and main.vendor == "amd":
        engine_env.update(GTT_ROUTE_ENV)
        notes.append(
            "the GTT route is on: an AMD main card with a host RAM tier needs it, or the "
            "GPU can freeze rather than merely slow down"
        )
    engine_env.update(opts.extra_env)
    return engine_env


def _helper_launch(report, plan, paths: ModelPaths, gpus_by_key: dict, isdir: Callable[[str], bool], server_env: dict, file_moves: bool = False) -> tuple:
    """(helper_args, helper_env) per helper index; sets SEABRAID_STRATA_SHM in `server_env` to every channel, in rung order."""
    helper_args: dict = {}
    helper_env: dict = {}
    helpers = sorted(plan.helpers(), key=lambda r: r.helper_index)
    if [r.helper_index for r in helpers] != list(range(len(helpers))):
        # Position in SEABRAID_STRATA_SHM IS the card id, so a gap or a repeat would send a helper's experts to another card's channel.
        raise ConfigRefused("helper indices must be 0..%d with none missing, got %s" % (len(helpers) - 1, [r.helper_index for r in helpers]))
    # WHY here and only here: this is the one owner of the server's SEABRAID_STRATA_SHM.  config.server_env is what services renders and
    # what the lead bench launches with, so it is the complete truth (the engine finds channel N at list position N).
    server_env["SEABRAID_STRATA_SHM"] = ",".join(r.shm or "" for r in helpers)
    for role in helpers:
        shm = role.shm
        if shm is None:
            raise ConfigRefused("helper on gpu%d has no shared-memory name" % role.gpu_index)
        # The helper takes the channel name as --shm, because helper.cpp reads no environment at all.
        helper_args[role.helper_index] = (
            "--device", "0",
            "--shm", shm,
            "--pack", paths.pack,
            "--native", paths.native,
            "--layers", str(plan.model.layers),
            "--experts", str(plan.model.experts),
        ) + (_file_moves.HELPER_ARGS if file_moves else ())
        env = _mask_env(role, gpus_by_key)
        libs = _lib_dirs(report, role.vendor, isdir)
        if libs:
            env["LD_LIBRARY_PATH"] = ":".join(libs)
        helper_env[role.helper_index] = env
    return helper_args, helper_env


def make_config(report, plan, paths: ModelPaths, opts: ConfigOptions = ConfigOptions(),
                isdir: Callable[[str], bool] = os.path.isdir) -> ConfigOutput:
    """Turn an approved plan into the exact launch surface, or refuse."""
    if not plan.ready:
        raise ConfigRefused(
            "the plan is not ready, so there is nothing to configure:\n" + plan.explain()
        )

    k, fm = len(plan.helpers()), (_file_moves.ENGINE_ARGS if opts.file_moves else ())
    why = _ladder.refusal(k, opts.ladder, opts.extra_args, opts.extra_env, _ladder.rendered(k, opts.ladder, opts.ladder_fallback, fm)) or (opts.file_moves and _file_moves.refusal(opts.ladder, opts.extra_args))
    if why:
        raise ConfigRefused(why)
    notes: list[str] = []
    warnings: list[KnobWarning] = []
    gpus_by_key = {(g.vendor, g.index): g for g in report.gpus}
    main = plan.main()
    context = _resolve_context(plan, opts, notes, warnings)
    main_mask = _mask_env(main, gpus_by_key)
    # Strata applies cfg.env after its ordinal mask, so preserve the planned GPU here.
    engine_env = {**_engine_env(plan, main, opts, notes), **main_mask}
    main_libs = _lib_dirs(report, main.vendor, isdir)
    cfg_json = {
        "exe": paths.exe,
        "cwd": paths.cwd,
        "args": _engine_args(plan, paths, context, opts),
        # The same flag again, deliberately: see USER_ARGS_REASSERTED.
        "user_args": list(USER_ARGS_REASSERTED),
        "env": engine_env,
        "tokenizer": paths.tokenizer,
        "model_name": paths.model_name,
        "log": paths.log,
        "lib_dirs": list(main_libs),
        "port": opts.port,
        "backend": {"amd": "hip", "nvidia": "cuda"}.get(main.vendor),
        "gpu": [main.device_id],
    }

    server_env = dict(SERVER_ENV)
    server_env.update(main_mask)
    if main_libs:
        server_env["LD_LIBRARY_PATH"] = ":".join(main_libs)

    helper_args, helper_env = _helper_launch(report, plan, paths, gpus_by_key, isdir, server_env, opts.file_moves)

    if len(plan.helpers()) > 1:
        notes.append("SEABRAID_STRATA_SHM=%s (channel N = rung N+1 = --expert-cache-device(N+1))" % server_env["SEABRAID_STRATA_SHM"])

    return ConfigOutput(
        port=opts.port,
        cfg_json=cfg_json,
        cfg_json_path="",
        server_env=server_env,
        helper_args=helper_args,
        helper_env=helper_env,
        notes=tuple(notes),
        warnings=tuple(warnings),
    )


def config_path(directory: str, name: str = "seabraid-engine.json") -> str:
    """Where cfg.json goes.  Absolute, because systemd has no working directory of ours.

    Separate from write_config so a dry run can report the path it WOULD write without writing it.  The
    first dry run on the test system died here: nothing was written, cfg_json_path stayed empty, and services
    refused an empty path.  A path is not a side effect, so computing it is always safe.
    """
    return os.path.abspath(os.path.join(directory, name))


def write_config(out: ConfigOutput, directory: str, name: str = "seabraid-engine.json") -> ConfigOutput:
    """Write cfg.json and return the output carrying the path it went to."""
    os.makedirs(directory, exist_ok=True)
    path = config_path(directory, name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out.cfg_json, fh, indent=1, sort_keys=False)
        fh.write("\n")
    # replace(), not a field-by-field rebuild: a rebuild silently drops any field added later.
    return dataclasses.replace(out, cfg_json_path=path)
