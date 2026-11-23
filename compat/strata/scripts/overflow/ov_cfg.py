# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Load arms.json, turn each arm's capacity plan into engine flags and a memory-limited process start, order the starts, estimate the window.

OWNER: this module owns the overflow config's expansion and the run plan; ov_cap.py owns the arithmetic, qual_cfg the shared schema,
placement-qual/arms.json the helper command and the KL thresholds (read from there, never repeated here).
CONTRACT: ``load(path, overrides, dry, sweep) -> cfg``; ``plan(cfg, arms, kl) -> [qual_cfg.Run]`` (workload ``speed`` or ``w7``);
``engine_command(cfg, run, rundir) -> (argv, env)``; ``helper_specs(cfg, run) -> [spec]``; ``capture_command(cfg, run, rundir) -> argv``;
``describe(cfg, run) -> [line]`` (what --dry-run prints); ``estimate(cfg, runs) -> (low_s, high_s)``.
THE CGROUP MECHANISM: a capped arm's engine, its helpers and its W7 capture are started under
``systemd-run --user --scope -p MemoryMax=<bytes> -p MemorySwapMax=0``.  WHY this and not a hand-made cgroup: it is the one
unprivileged route (the user manager owns a delegated memory controller), it execs the command in place so the pid and process group
the runner signals stay the engine's own, and it creates and removes only a transient scope of that one process tree.  It is not a
service-manager command on anything the runner does not own.  The limit is in bytes, not ``G``, so the readback is an exact equality.
WHY the ladder arm imports its flags: install/_ladder.py owns which engine flags and env turn the live pyramid on (and the refusals that
keep one publisher per rung); this file calls ``ladder_args``, ``LADDER_ENGINE_ENV`` and ``refusal`` and spells none of them.
FAILURE: fail-fast.  Any bad value raises ConfigError before an engine starts (the CLI maps it to exit 2).  With ``dry`` an UNSET
machine var becomes ``<name>`` so the plan prints on a machine that has none of the paths.
CHECKER: compat/strata/tests/overflow/test_overflow_cfg.py.
"""

from __future__ import annotations

import copy
import json
import os
import shlex

import ov_cap
import ov_paths
import qual_cfg
import qual_engine
from install import _ladder

SCOPE = ["systemd-run", "--user", "--scope", "--quiet"]


def _unset_names(path: str) -> list[str]:
    with open(path, encoding="utf-8") as fh:
        return [k for k, v in json.load(fh).get("vars", {}).items() if str(v).startswith("UNSET")]


def _need(cond: bool, msg: str) -> None:
    if not cond:
        raise qual_cfg.ConfigError(msg)


def _placement() -> dict:
    """The blocks placement-qual owns and this config borrows: the named helper command and the KL thresholds."""
    try:
        with open(os.path.join(ov_paths.PLACEMENT_DIR, "arms.json"), encoding="utf-8") as fh:
            d = json.load(fh)
        return {"helpers": d["helpers"], "thresholds": {k: d["thresholds"][k] for k in ("kl_max", "top1_min")}}
    except (OSError, ValueError, KeyError) as exc:
        raise qual_cfg.ConfigError(f"cannot read helpers/thresholds of placement-qual/arms.json: {exc}") from exc


def _check_sweep(sw: dict, cfg: dict) -> None:
    _need(bool(sw.get("arms")) and all(a in cfg["arms"] for a in sw["arms"]), "sweep.arms must name arms of the config")
    _need(sw.get("reference_arm") in cfg["arms"] and not cfg["arms"][sw["reference_arm"]]["tiers"]["capped"], "sweep.reference_arm must name the uncapped arm")
    for k in ("n_requests", "warmup", "decode_tokens", "prefill_tokens", "ref_repeats"):
        floor = 0 if k == "warmup" else 1
        _need(isinstance(sw.get(k), int) and sw[k] >= floor, f"sweep.{k} must be an int >= {floor}")


def _check(cfg: dict) -> None:
    gates = cfg.get("gates", {})
    _check_sweep(cfg.get("sweep", {}), cfg)
    for k in ("min_device_frac", "min_file_mb", "max_resident_mb", "complement_tol_frac", "min_file_plan_frac", "reference_max_file_mb", "w7_min_device_mb",
              "prefill_reuse_max_frac", "settle_s", "settle_timeout_s"):
        _need(isinstance(gates.get(k), (int, float)) and gates[k] >= 0, f"gates.{k} must be a number >= 0")
    _need(isinstance(gates.get("max_consecutive_failures"), int) and gates["max_consecutive_failures"] >= 1, "gates.max_consecutive_failures must be an int >= 1")
    _need(isinstance(cfg.get("w7", {}).get("argv"), list), "w7.argv must be a list")


def default_xdg() -> str:
    uid = os.getuid() if hasattr(os, "getuid") else 1000
    return os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{uid}"


def scope_prefix(limit_bytes: int) -> list[str]:
    """argv prefix that starts the rest of the command in its own transient scope with a hard memory limit and no swap."""
    return [*SCOPE, "-p", f"MemoryMax={limit_bytes}", "-p", "MemorySwapMax=0", "--"]


def ladder_flags(k: int) -> tuple[list[str], dict[str, str]]:
    """The live-move flags and engine env of a k-helper ladder arm: the installer's own set (install/_ladder.py), never a copy of it."""
    return _ladder.ladder_args(k, True), dict(_ladder.LADDER_ENGINE_ENV)


def _arm_args(plan: ov_cap.ArmPlan, v: dict) -> tuple[list[str], dict[str, str]]:
    """The capacity flags of a capped arm, the ladder set for a ladder arm, and the optional unbuffered hook (flags, env)."""
    args = ["--expert-cache", str(plan.main_slots), "--resident-budget-gib", f"{plan.ram_gib:g}"]
    env: dict[str, str] = {}
    for i, n in enumerate(plan.helper_slots, 1):
        args += [f"--expert-cache-device{i}", str(n)]
    if plan.ladder:
        flags, lenv = ladder_flags(plan.helpers)
        args += flags + shlex.split(v.get("ladder_extra_args", ""))
        env.update(lenv)
    if v.get("unbuffered"):
        env["STRATA_UNBUFFERED_LOAD"] = str(v["unbuffered"])
    why = _ladder.refusal(plan.helpers, plan.ladder, args, env)
    _need(why is None, f"arm {plan.arm}: {why}")
    return args, env


def _helper_spec(base: dict, idx: int, plan: ov_cap.ArmPlan) -> dict:
    """The helper command of helper ``idx`` (0-based): channel {shm}/{shm2}, card {gpu}/{gpu2}, started in its own memory-limited scope if the arm is capped."""
    spec = copy.deepcopy(base)
    if idx:
        spec["argv"] = [a.replace("{shm}", "{shm2}") for a in spec["argv"]]
        spec["env"] = {k: v.replace("{gpu}", "{gpu2}") for k, v in spec.get("env", {}).items()}
        spec["ready_path"] = spec["ready_path"].replace("{shm}", "{shm2}")
    if plan.capped:
        spec["argv"] = scope_prefix(plan.helper_cap_bytes) + spec["argv"]
    return spec


def _expand_arms(cfg: dict, plans: dict[str, ov_cap.ArmPlan]) -> None:
    for arm, spec in cfg["arms"].items():
        p = plans[arm]
        if p.capped:
            args, env = _arm_args(p, cfg["vars"])
            spec["args"] = spec.get("args", []) + args
            spec["env"] = {**spec.get("env", {}), **env}
        spec["helper_specs"] = [_helper_spec(cfg["helpers"]["named"], i, p) for i in range(p.helpers)]
        if p.helpers:
            spec["helper_start"] = spec["helper_specs"][0]  # qual_engine.Engine starts this one; OvEngine starts the rest


def _expand_workloads(cfg: dict) -> None:
    arms = {a: ["base"] for a in cfg["sweep"]["arms"]}
    common = {"n": 1, "arms": arms, "cap": 0, "extra_args": [], "env": {}, "phases": []}
    cfg["workloads"] = {"speed": {**common, "kind": "decode", "mode": "speed"},
                        "w7": {**common, "kind": "forced", "mode": "forced", "capture": cfg["w7"]}}


def load(path: str, overrides: dict[str, str] | None = None, dry: bool = False, sweep: dict | None = None) -> dict:
    """Read, validate and expand.  ``sweep`` replaces sweep keys (CLI --arms, --kl, --ref-repeats)."""
    over = {**({n: f"<{n}>" for n in _unset_names(path)} if dry else {}), **(overrides or {})}
    cfg = qual_cfg.load(path, over)
    cfg["vars"].setdefault("xdg_runtime_dir", default_xdg())
    cfg["vars"]["gpu2"] = cfg["vars"].get("gpu2") or cfg["vars"]["gpu"]
    cfg["sweep"].update(sweep or {})
    cfg.update(_placement())
    plans = ov_cap.build(cfg)
    _check(cfg)
    cfg["plans"] = plans
    cfg["_dir"] = ov_paths.PLACEMENT_DIR  # WHY: the prompts have one owner, placement-qual/prompts (qual_work.prompt_text reads cfg["_dir"])
    cfg["_overflow_dir"] = os.path.dirname(os.path.abspath(path))
    _expand_arms(cfg, plans)
    _expand_workloads(cfg)
    return cfg


def _capped_runs(arms: list[str], kl: bool) -> list[qual_cfg.Run]:
    out: list[qual_cfg.Run] = []
    for a in arms:
        out.append(qual_cfg.Run("speed", a, "base", 0))
        if kl:
            out.append(qual_cfg.Run("w7", a, "base", 0))
    return out


def plan(cfg: dict, arms: list[str] | None = None, kl: bool | None = None) -> list[qual_cfg.Run]:
    """Starts in execution order: the reference's W7 first (every gate needs it), each capped arm's speed then W7 start in capacity order,
    then the reference's repeats (the noise floor) and its speed start, so a window cut short loses the least useful starts."""
    sw = cfg["sweep"]
    arms = arms or sw["arms"]
    ref, kl = sw["reference_arm"], sw["kl"] if kl is None else kl
    out = [qual_cfg.Run("w7", ref, "base", 0)] if kl and ref in arms else []
    out += _capped_runs([a for a in arms if a != ref], kl)
    if ref in arms:
        out += [qual_cfg.Run("w7", ref, "base", i) for i in range(1, sw["ref_repeats"]) if kl]
        out.append(qual_cfg.Run("speed", ref, "base", 0))
    return out


def engine_command(cfg: dict, run: qual_cfg.Run, rundir: str) -> tuple[list[str], dict[str, str]]:
    """(argv, env) of one speed start: the placement-qual command, inside a memory-limited scope for a capped arm."""
    argv, env = qual_engine.build_command(cfg, run, rundir)
    p = cfg["plans"][run.arm]
    return (scope_prefix(p.engine_cap_bytes) + argv if p.capped else argv), env


def helper_specs(cfg: dict, run: qual_cfg.Run, rundir: str = "<rundir>") -> list[tuple[list[str], dict[str, str], str]]:
    """(argv, env, ready path) of every helper process this start needs, placeholders filled; empty for a one-card arm."""
    vars_ = {**cfg["vars"], "rundir": rundir}
    out = []
    for spec in cfg["arms"][run.arm]["helper_specs"]:
        env = {k: v for k, v in os.environ.items() if k not in qual_engine.UNSET}
        env.update(qual_cfg.subst(spec.get("env", {}), vars_))
        out.append((qual_cfg.subst(spec["argv"], vars_), env, qual_cfg.subst(spec["ready_path"], vars_)))
    return out


def capture_command(cfg: dict, run: qual_cfg.Run, rundir: str) -> list[str]:
    """The W7 forced-decode capture of one arm, in the arm's memory-limited scope when capped (its engine starts as the capture's child)."""
    vars_ = {**cfg["vars"], "rundir": rundir, "forced_cfg": cfg["vars"][f"forced_cfg_{run.arm}"]}
    argv = qual_cfg.subst(cfg["w7"]["argv"], vars_)
    p = cfg["plans"][run.arm]
    return scope_prefix(p.engine_cap_bytes) + argv if p.capped else argv


def describe(cfg: dict, run: qual_cfg.Run, rundir: str = "<rundir>") -> list[str]:
    """The exact start of one run as lines: for a speed start the engine argv, the env keys the config sets and each helper; for W7 the capture."""
    if run.workload == "w7":
        return [f"  capture: {' '.join(capture_command(cfg, run, rundir))}"]
    argv, env = engine_command(cfg, run, rundir)
    set_env = {k: v for k, v in env.items() if os.environ.get(k) != v}
    lines = [f"  engine: {' '.join(argv)}", "  env:    " + " ".join(f"{k}={v}" for k, v in sorted(set_env.items()))]
    lines += [f"  helper {i}: {' '.join(a)}  (channel file {ready})" for i, (a, _e, ready) in enumerate(helper_specs(cfg, run, rundir), 1)]
    return lines


def estimate(cfg: dict, runs: list[qual_cfg.Run]) -> tuple[float, float]:
    """(low, high) seconds of exclusive GPU window for ``runs`` from the config's ESTIMATE block (nothing here is measured)."""
    sw, est = cfg["sweep"], cfg["estimate"]
    rounds = sw["n_requests"] + sw["warmup"]
    totals = []
    for side in (0, 1):  # 0 = optimistic, 1 = pessimistic
        speed = rounds * (sw["decode_tokens"] / est["decode_tps"][1 - side] + sw["prefill_tokens"] / est["prefill_tps"][1 - side])
        w7 = est["w7_tokens"] / est["decode_tps"][1 - side]
        totals.append(sum(est["load_s"][side] + est["stop_s"] + (speed if r.workload == "speed" else w7) for r in runs))
    return totals[0], totals[1]
