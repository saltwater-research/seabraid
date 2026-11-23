# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Load arms.json, expand the sweep into placement-qual workloads, order the engine starts, estimate the GPU window.

OWNER: this module owns the payoff config's extra keys (sweep, gates, estimate) and the run plan.  The shared keys (vars,
engine, arms, variants, limits) are placement-qual's schema and are validated by qual_cfg.load; the helper command line
(``helpers``) is read from placement-qual's arms.json, so it has one owner.
CONTRACT: ``load(path, overrides, dry) -> dict`` or ``qual_cfg.ConfigError``; ``plan(cfg, starts) -> list[qual_cfg.Run]``;
``describe(cfg, run) -> list[str]`` (the exact engine command and env of one start, for --dry-run);
``estimate(cfg, runs) -> (low_s, high_s)``.  A start is a ``qual_cfg.Run`` whose ``workload`` is ``cap<N>`` (speed) or
``cap<N>kl`` (quality gate), ``arm`` is A/B/C and ``idx`` the repeat number.
WHY the cap lives in the workload and not in a variant: qual_engine.build_command appends a workload's ``extra_args`` and
``env`` but a variant's ``env`` only, and the cap is an argument.
FAILURE: fail-fast.  Any bad value raises ConfigError before an engine starts (the CLI maps it to exit 2).  With ``dry``
an UNSET machine var becomes ``<name>`` so the plan prints on a machine that has none of the paths.
CHECKER: compat/strata/tests/payoff/test_payoff_cfg.py.
"""

from __future__ import annotations

import copy
import json
import os

import pay_paths
import qual_cfg
import qual_engine

KL_ARGS = ["--short-read", "1000"]
KL_ENV = {"STRATA_LOGPOS": "{rundir}/rows.tsv", "STRATA_LOGPOS_TOPK": "32", "STRATA_ROUTE_TAIL_SKIP": "0"}


def _unset_names(path: str) -> list[str]:
    with open(path, encoding="utf-8") as fh:
        return [k for k, v in json.load(fh).get("vars", {}).items() if str(v).startswith("UNSET")]


def _need(cond: bool, msg: str) -> None:
    if not cond:
        raise qual_cfg.ConfigError(msg)


def _check_arms(sw: dict, cfg: dict) -> None:
    _need(bool(sw.get("arms")) and all(a in cfg["arms"] for a in sw["arms"]), "sweep.arms must name arms of the config")
    _need(sw.get("baseline_arm") in sw.get("arms", []), "sweep.baseline_arm must be one of sweep.arms")
    _need(sw.get("second_gpu_arm") in sw.get("arms", []) and len(sw["arms"]) >= 2, "sweep.second_gpu_arm must be one of sweep.arms (and there must be a rival arm)")


def _check_counts(sw: dict) -> None:
    caps = sw.get("caps", [])
    _need(len(caps) >= 3 and all(isinstance(c, int) and c > 0 for c in caps) and len(set(caps)) == len(caps),
          "sweep.caps needs >= 3 distinct positive integers (the crossover is located between measured points)")
    _need(isinstance(sw.get("n_requests"), int) and sw["n_requests"] >= 6, "sweep.n_requests must be an int >= 6")
    _need(isinstance(sw.get("warmup"), int) and sw["warmup"] >= 0, "sweep.warmup must be an int >= 0")
    for k in ("decode_tokens", "prefill_tokens", "baseline_repeats"):
        _need(isinstance(sw.get(k), int) and sw[k] >= 1, f"sweep.{k} must be an int >= 1")


def _check_gates(gates: dict) -> None:
    for k in ("margin_pct", "b_pcie_min_share", "a_pcie_max_share", "prefill_reuse_max_frac", "settle_s", "settle_timeout_s"):
        _need(isinstance(gates.get(k), (int, float)) and gates[k] >= 0, f"gates.{k} must be a number >= 0")
    _need(isinstance(gates.get("max_consecutive_failures"), int) and gates["max_consecutive_failures"] >= 1,
          "gates.max_consecutive_failures must be an int >= 1")


def _check_sweep(cfg: dict) -> None:
    _check_arms(cfg.get("sweep", {}), cfg)
    _check_counts(cfg["sweep"])
    _check_gates(cfg.get("gates", {}))


def _expand(cfg: dict) -> None:
    """One workload per cap point (speed) and, when sweep.kl, one per cap point for the teacher-forced quality start."""
    sw = cfg["sweep"]
    arms = {a: ["base"] for a in sw["arms"]}
    flat: dict[str, dict] = {}
    for cap in sw["caps"]:
        base = {"n": 1, "arms": arms, "cap": cap, "extra_args": ["--expert-cache", str(cap)], "phases": []}
        flat[f"cap{cap}"] = {**copy.deepcopy(base), "kind": "decode", "mode": "speed", "env": {}}
        if sw["kl"]:
            flat[f"cap{cap}kl"] = {**copy.deepcopy(base), "kind": "kl", "mode": "kl", "texts": list(sw["kl_texts"]),
                                   "extra_args": base["extra_args"] + KL_ARGS, "env": dict(KL_ENV)}
    cfg["workloads"] = flat


def _placement_helpers() -> dict:
    """The named-channel helper command, read from placement-qual's config (its one owner); a missing block is a ConfigError."""
    try:
        with open(os.path.join(pay_paths.PLACEMENT_DIR, "arms.json"), encoding="utf-8") as fh:
            return json.load(fh)["helpers"]
    except (OSError, ValueError, KeyError) as exc:
        raise qual_cfg.ConfigError(f"cannot read the helpers block of placement-qual/arms.json: {exc}") from exc


def load(path: str, overrides: dict[str, str] | None = None, dry: bool = False, sweep: dict | None = None) -> dict:
    """Read, validate and expand the payoff config.  ``dry`` fills UNSET vars with ``<name>`` placeholders; ``sweep`` replaces sweep keys (CLI --caps, --arms, --kl)."""
    over = {**({n: f"<{n}>" for n in _unset_names(path)} if dry else {}), **(overrides or {})}
    cfg = qual_cfg.load(path, over)
    cfg["sweep"].update(sweep or {})
    _check_sweep(cfg)
    cfg["helpers"] = _placement_helpers()
    cfg["_dir"] = pay_paths.PLACEMENT_DIR  # WHY: the prompts have one owner, placement-qual/prompts (qual_work.prompt_text reads cfg["_dir"])
    cfg["_payoff_dir"] = os.path.dirname(os.path.abspath(path))
    _expand(cfg)
    return cfg


def _rotated(items: list[str], i: int) -> list[str]:
    k = i % len(items)
    return items[k:] + items[:k]


def plan(cfg: dict, starts: int = 1) -> list[qual_cfg.Run]:
    """Engine starts in execution order: caps from fits-on-main down to overflow, the arm order rotated by one per cap
    point (A,B,C then B,C,A then C,A,B ..) so slow drift (heat, page cache) cannot line up with one arm; per cap point
    the speed starts come first, then the quality starts, and the baseline arm repeats its quality start for the noise floor."""
    sw = cfg["sweep"]
    caps = sorted(sw["caps"], reverse=True)
    runs: list[qual_cfg.Run] = []
    for ci, cap in enumerate(caps):
        for suffix, n in (("", starts), ("kl", 1)):
            if suffix and not sw["kl"]:
                continue
            for i in range(n):
                for arm in _rotated(sw["arms"], ci + i):
                    runs.append(qual_cfg.Run(f"cap{cap}{suffix}", arm, "base", i))
    if sw["kl"]:
        top = f"cap{caps[0]}kl"
        runs += [qual_cfg.Run(top, sw["baseline_arm"], "base", i) for i in range(1, sw["baseline_repeats"])]
    return runs


def describe(cfg: dict, run: qual_cfg.Run, rundir: str = "<rundir>") -> list[str]:
    """The exact start of one run, as lines: engine argv, the env keys the config sets, and the helper command if any."""
    argv, env = qual_engine.build_command(cfg, run, rundir)
    set_env = {k: v for k, v in env.items() if os.environ.get(k) != v}
    lines = [f"  engine: {' '.join(argv)}", "  env:    " + " ".join(f"{k}={v}" for k, v in sorted(set_env.items()))]
    removed = [k for k in qual_engine.UNSET if k not in env]
    lines.append("  env removed: " + (" ".join(removed) if removed else "(none)"))
    helper = qual_engine.build_helper(cfg, run, rundir)
    if helper:
        lines.append(f"  helper: {' '.join(helper[0])}  (channel file {helper[2]})")
    return lines


def request_count(cfg: dict) -> int:
    sw = cfg["sweep"]
    return sw["n_requests"] + sw["warmup"]


def estimate(cfg: dict, runs: list[qual_cfg.Run]) -> tuple[float, float]:
    """(low, high) seconds of exclusive GPU window for ``runs``, from the config's ESTIMATE block (nothing here is measured)."""
    sw, est = cfg["sweep"], cfg["estimate"]
    n = request_count(cfg)
    totals = []
    for side in (0, 1):  # 0 = optimistic, 1 = pessimistic
        per = {"speed": n * (sw["decode_tokens"] / est["decode_tps"][1 - side] + sw["prefill_tokens"] / est["prefill_tps"][1 - side]),
               "kl": len(sw["kl_texts"]) * est["kl_text_s"][side]}
        totals.append(sum(est["load_s"][side] + est["stop_s"] + per[cfg["workloads"][r.workload]["mode"]] for r in runs))
    return totals[0], totals[1]
