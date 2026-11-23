# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Load, validate and expand arms.json into an ordered run plan for the placement qualification runner.

OWNER: this module owns the config schema and the plan (which (workload, arm, variant, start) runs, in which order).
CONTRACT: ``load(path, overrides) -> dict`` or ``ConfigError``; ``plan(cfg, arms, n_cap) -> list[Run]``;
``subst(obj, vars)`` fills {name} placeholders.  Pure; reads only the config file.
FAILURE: fail fast.  A bad config raises ConfigError before any engine starts (the CLI maps it to exit 2).  Engine
commands are argv lists run directly (never a shell); an argv whose program is a service manager or a GPU reset tool
is refused, because the runner may only start and stop its own engine processes.
CHECKER: <repo>/tests/test_placement_qual.py (dry run) and its bad-config cases; tests/test_placement_qual_decode.py (W7 config).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

FORBIDDEN_PROGRAMS = ("systemctl", "gpubreak", "service", "wsl", "docker")
_HOLE = re.compile(r"\{(\w+)\}")
RUNTIME_VARS = ("rundir",)


class ConfigError(Exception):
    """The config cannot be run; message names the key."""


@dataclass(frozen=True)
class Run:
    workload: str
    arm: str
    variant: str
    idx: int

    @property
    def key(self) -> str:
        return self.arm if self.variant == "base" else f"{self.arm}/{self.variant}"

    @property
    def tag(self) -> str:
        return f"{self.workload}-{self.key.replace('/', '-')}-s{self.idx}"


def subst(obj, vars_: dict[str, str]):
    """Fill {name} in every string of a JSON-like value; an unknown name is a ConfigError (never left in an argv)."""
    if isinstance(obj, str):
        return _HOLE.sub(lambda m: _lookup(m.group(1), vars_), obj)
    if isinstance(obj, list):
        return [subst(x, vars_) for x in obj]
    if isinstance(obj, dict):
        return {k: subst(v, vars_) for k, v in obj.items()}
    return obj


def _lookup(name: str, vars_: dict[str, str]) -> str:
    if name not in vars_:
        raise ConfigError(f"unknown placeholder {{{name}}}")
    if str(vars_[name]).startswith("UNSET"):  # machine paths are not kept in the repo (checks/check_no_local_refs.py): they come from --var
        raise ConfigError(f"var {name!r} has no value: {vars_[name]}")
    return str(vars_[name])


def _need(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(msg)


def _check_engine(cfg: dict) -> None:
    eng = cfg.get("engine", {})
    argv = eng.get("argv")
    _need(isinstance(argv, list) and argv and all(isinstance(a, str) for a in argv), "engine.argv must be a non-empty list of strings")
    prog = os.path.basename(subst(argv[0], {**cfg["vars"], **dict.fromkeys(RUNTIME_VARS, "x")}))
    _need(prog not in FORBIDDEN_PROGRAMS, f"engine.argv[0] {prog!r} is not an engine process")
    for k in ("ready_s", "stop_grace_s"):
        _need(isinstance(eng.get(k), (int, float)) and eng[k] > 0, f"engine.{k} must be a positive number")


def _check_phases(name: str, wl: dict) -> None:
    _need(bool(wl.get("phases")), f"workloads.{name}.phases is empty")
    for ph in wl["phases"]:
        _need(("prompt" in ph) != ("tokens" in ph), f"workloads.{name} phase {ph.get('name')!r} needs exactly one of prompt/tokens")
        _need(ph.get("count", 0) >= 1 and ph.get("max_tokens", 0) >= 1, f"workloads.{name} phase {ph.get('name')!r} count/max_tokens")


def _check_forced(cfg: dict, name: str, wl: dict) -> None:
    """A forced-decode workload (W7) has no HTTP phases: it runs one external capture program per start, with a per-arm engine config."""
    argv = wl.get("capture", {}).get("argv")
    _need(isinstance(argv, list) and argv and all(isinstance(a, str) for a in argv), f"workloads.{name}.capture.argv must be a non-empty list of strings")
    prog = os.path.basename(subst(argv[0], {**cfg["vars"], **dict.fromkeys(RUNTIME_VARS, "x")}))
    _need(prog not in FORBIDDEN_PROGRAMS, f"workloads.{name}.capture.argv[0] {prog!r} is not a capture tool")
    _need("M" in wl["arms"], f"workloads.{name} needs the main-card-alone arm M as the reference")
    for arm in wl["arms"]:
        _need(bool(cfg["arms"][arm].get("forced_cfg")), f"arms.{arm}.forced_cfg is missing (workloads.{name} needs an engine config per arm)")


def _check_workloads(cfg: dict) -> None:
    for name, wl in cfg.get("workloads", {}).items():
        _need(wl.get("kind") in ("smoke", "decode", "shift", "prefill", "kl", "fault", "forced"), f"workloads.{name}.kind unknown")
        _need(isinstance(wl.get("n"), int) and wl["n"] >= 1, f"workloads.{name}.n must be an int >= 1")
        _need(bool(wl.get("arms")), f"workloads.{name}.arms is empty")
        for arm, variants in wl["arms"].items():
            _need(arm in cfg["arms"], f"workloads.{name} names unknown arm {arm!r}")
            for v in variants:
                _need(v in cfg["variants"], f"workloads.{name}.arms.{arm} names unknown variant {v!r}")
        if wl["kind"] == "forced":
            _check_forced(cfg, name, wl)
        else:
            _check_phases(name, wl)


def load(path: str, overrides: dict[str, str] | None = None) -> dict:
    """Read and validate the config; ``overrides`` replace entries of ``vars``."""
    try:
        with open(path, encoding="utf-8") as fh:
            cfg = json.load(fh)
    except (OSError, ValueError) as exc:
        raise ConfigError(f"cannot read config {path}: {exc}") from exc
    for key in ("vars", "engine", "arms", "workloads", "thresholds", "variants", "limits"):
        _need(key in cfg, f"missing key {key!r}")
    cfg["vars"] = {**cfg["vars"], **(overrides or {})}
    _check_engine(cfg)
    _check_workloads(cfg)
    cfg["_dir"] = os.path.dirname(os.path.abspath(path))
    return cfg


def policy_args(cfg: dict, arm: dict) -> list[str]:
    """The explicit ``--retier-*`` flags of a promotion arm: protocol policy with the arm's overrides (none for static arms)."""
    if "policy" not in arm:
        return []
    merged = {**cfg.get("policy", {}), **arm["policy"]}
    return [x for k, v in merged.items() for x in (f"--retier-{k}", str(v))]


def plan(cfg: dict, arms: list[str] | None, n_cap: int | None = None) -> list[Run]:
    """Runs in execution order: per workload, start i runs every arm once, the arm order rotated by i (interleaved)."""
    runs: list[Run] = []
    for name, wl in cfg["workloads"].items():
        combos = [(a, v) for a, vs in wl["arms"].items() if arms is None or a in arms for v in vs]
        for i in range(min(wl["n"], n_cap) if n_cap else wl["n"]):
            rot = i % len(combos) if combos else 0
            runs += [Run(name, a, v, i) for a, v in combos[rot:] + combos[:rot]]
    return runs
