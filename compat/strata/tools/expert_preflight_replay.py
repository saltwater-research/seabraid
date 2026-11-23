# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Replay prompt-start cache hints using Strata's per-request route histograms.

This offline CLI owns prediction and scoring. route_hist_report owns the
histogram format. The manifest supplies preflight facts and fixed residency.
Invalid input fails before any report is emitted. Tests pin hand-counted loads
and train/test isolation. See docs/design/N-CARD-LADDER.md for the assumptions.
"""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys

from route_hist_report import load_one

Pair = tuple[int, int]
StagePlans = dict[str, tuple[set[Pair], bool]]


@dataclass(frozen=True)
class Layer:
    expert_bytes: int
    resident: frozenset[int]


@dataclass(frozen=True)
class Request:
    family: str
    split: str
    task: str
    effort: str
    prefill_tokens: int
    picks: Counter[Pair]


@dataclass(frozen=True)
class Dataset:
    model: str
    experts: int
    picks_per_token: int
    layers: tuple[Layer, ...]
    requests: tuple[Request, ...]


@dataclass
class Totals:
    requests: int = 0
    staged_bytes: int = 0
    useful_bytes: int = 0
    wasted_bytes: int = 0
    on_demand_bytes: int = 0
    on_demand_loads: int = 0
    layers_with_misses: int = 0
    fallback_requests: int = 0

    def record(self, misses: set[Pair], staged: set[Pair], layers: tuple[Layer, ...], fallback: bool) -> None:
        def byte_size(pairs: set[Pair]) -> int:
            return sum(layers[layer].expert_bytes for layer, _ in pairs)

        needed = misses - staged
        self.requests += 1
        self.staged_bytes += byte_size(staged)
        self.useful_bytes += byte_size(staged & misses)
        self.wasted_bytes += byte_size(staged - misses)
        self.on_demand_bytes += byte_size(needed)
        self.on_demand_loads += len(needed)
        self.layers_with_misses += len({layer for layer, _ in needed})
        self.fallback_requests += fallback


def positive_int(value: object, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def label(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value.strip()


def read_layer(item: dict, experts: int) -> Layer:
    resident = item["resident"]
    if not isinstance(resident, list) or any(type(e) is not int or not 0 <= e < experts for e in resident):
        raise ValueError("resident expert ID out of range")
    if len(set(resident)) != len(resident):
        raise ValueError("duplicate resident expert")
    return Layer(positive_int(item["expert_bytes"], "expert_bytes"), frozenset(resident))


def read_request(path: Path, row: dict, shape: tuple[int, int], k: int) -> Request:
    hist = load_one(path)
    tokens = positive_int(row["prompt_tokens"], "prompt_tokens")
    # Strata serve routes the last prompt token after switching the histogram to decode.
    prefill = positive_int(row.get("prefill_tokens", tokens - 1), "prefill_tokens")
    if prefill != tokens - 1:
        raise ValueError(f"{path}: expected Strata serve's prompt length minus the final token")
    if (hist["layers"], hist["experts"]) != shape or hist["prompt_tokens"] != tokens:
        raise ValueError(f"{path}: routing geometry or prompt length does not match manifest")
    picks: Counter[Pair] = Counter()
    for layer, counts in enumerate(hist["prompt"]):
        # Require the full fresh prompt so a KV-cache hit cannot look like a good hint.
        if sum(counts.values()) != prefill * k:
            raise ValueError(f"{path}: incomplete prompt routing at layer {layer}")
        for expert, count in counts.items():
            if type(expert) is not int or type(count) is not int:
                raise ValueError(f"{path}: expert IDs and counts must be integers")
            picks[layer, expert] = count
    split = row["split"]
    if split not in ("train", "test"):
        raise ValueError("split must be train or test")
    return Request(label(row["family"], "family"), split, label(row["task"], "task"),
                   label(row.get("effort", "default"), "effort"), prefill, picks)


def read_dataset(path: Path) -> Dataset:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if type(raw["version"]) is not int or raw["version"] != 1:
        raise ValueError("manifest version must be 1")
    if raw["placement"] != "frozen":
        raise ValueError("manifest must declare frozen starting placement")
    experts = positive_int(raw["experts"], "experts")
    k = positive_int(raw["picks_per_token"], "picks_per_token")
    layers = tuple(read_layer(row, experts) for row in raw["layers"])
    if not layers or k > experts:
        raise ValueError("invalid model geometry")
    rows = raw["requests"]
    paths = [(path.parent / row["histogram"]).resolve() for row in rows]
    if len(set(paths)) != len(paths):
        raise ValueError("each histogram must appear only once")
    requests = tuple(read_request(p, row, (len(layers), experts), k) for p, row in zip(paths, rows))
    check_split(requests)
    return Dataset(label(raw["model"], "model"), experts, k, layers, requests)


def check_split(requests: tuple[Request, ...]) -> None:
    train = {r.family for r in requests if r.split == "train"}
    test = {r.family for r in requests if r.split == "test"}
    if not train or not test:
        raise ValueError("need both training and test families")
    if train & test:
        raise ValueError("prompt families overlap train and test")


def fit_profile(rows: list[Request], *, presence: bool = False) -> dict[Pair, float]:
    """Give each independent prompt family equal weight across repeats and lengths."""
    by_family: dict[str, Counter[Pair]] = {}
    repeats: Counter[str] = Counter()
    for row in rows:
        acc = by_family.setdefault(row.family, Counter())
        for pair, count in row.picks.items():
            # A replay miss costs one load per request, even when the gate picks the expert repeatedly.
            acc[pair] += 1 if presence else count / row.prefill_tokens
        repeats[row.family] += 1
    combined: Counter[Pair] = Counter()
    for family, counts in by_family.items():
        for pair, count in counts.items():
            combined[pair] += count / repeats[family] / len(by_family)
    return dict(combined)


def log_miss(share: float, picks: int) -> float:
    return -math.inf if share >= 1.0 else picks * math.log1p(-share)


def select(scores: dict[Pair, float], data: Dataset, budget: int,
           miss: dict[Pair, float] | None = None, threshold: float = 0.0) -> set[Pair]:
    selected = set()
    ranks = {} if miss is None else miss
    max_miss = -math.inf if threshold == 1.0 else math.log1p(-threshold)
    # Equal benefit per avoided load. Ranking divides by bytes so large weights must justify their space.
    # The log miss value preserves the order when long prompts make use probabilities round to 1.0.
    rank = sorted(scores, key=lambda pair: (-scores[pair] / data.layers[pair[0]].expert_bytes,
                                            ranks.get(pair, 0.0), pair))
    for layer, expert in rank:
        size = data.layers[layer].expert_bytes
        if ranks.get((layer, expert), 0.0) > max_miss or expert in data.layers[layer].resident or size > budget:
            continue
        selected.add((layer, expert))
        budget -= size
    return selected


def conditioned(train: list[Request], task: str, effort: str | None = None) -> dict[Pair, float] | None:
    rows = [r for r in train if r.task == task and (effort is None or r.effort == effort)]
    # A single prompt family cannot establish a task profile.
    return fit_profile(rows) if len({r.family for r in rows}) >= 2 else None


def select_for_length(profile: dict[Pair, float], prefill_tokens: int, data: Dataset,
                      budget: int, threshold: float) -> set[Pair]:
    k = data.picks_per_token
    miss = {pair: log_miss(min(1.0, rate / k), prefill_tokens * k) for pair, rate in profile.items()}
    scores = {pair: -math.expm1(value) for pair, value in miss.items()}
    return select(scores, data, budget, miss, threshold)


def plan_requests(data: Dataset, train: list[Request], test: list[Request],
                  budget: int, threshold: float) -> Iterator[tuple[Request, StagePlans]]:
    """Return stage plans using only training routes and preflight metadata."""
    global_profile = fit_profile(train)
    task_profiles = {task: conditioned(train, task) for task in {r.task for r in test}}
    effort_profiles = {key: conditioned(train, *key) for key in {(r.task, r.effort) for r in test}}
    static = select(global_profile, data, budget)
    static_presence = select(fit_profile(train, presence=True), data, budget)
    for row in test:
        task = task_profiles[row.task]
        effort = effort_profiles[row.task, row.effort]
        choices = (("length", global_profile, False), ("task", task or global_profile, task is None),
                   ("task_effort", effort or task or global_profile, effort is None))
        plans = {"no_hint": (set(), False), "static": (static, False),
                 "static_presence": (static_presence, False)}
        for name, profile, fallback in choices:
            staged = select_for_length(profile, row.prefill_tokens, data, budget, threshold)
            plans[name] = (staged, fallback)
        yield row, plans


def summarize(totals: dict[str, Totals]) -> dict:
    baseline = totals["no_hint"]
    return {name: {**asdict(stats),
                   "avoided_on_demand_loads": baseline.on_demand_loads - stats.on_demand_loads,
                   "avoided_layers_with_misses": baseline.layers_with_misses - stats.layers_with_misses,
                   "total_load_bytes": stats.staged_bytes + stats.on_demand_bytes}
            for name, stats in totals.items()}


def _accumulate(data: Dataset, train: list, test: list, budget: int, threshold: float):
    """-> (totals per policy, totals per policy per request family) over the test requests' planned staging."""
    totals: dict[str, Totals] = {}
    families: dict[str, dict[str, Totals]] = {}
    for row, plans in plan_requests(data, train, test, budget, threshold):
        misses = {(layer, e) for layer, e in row.picks if e not in data.layers[layer].resident}
        by_policy = families.setdefault(row.family, {})
        for name, (staged, fallback) in plans.items():
            totals.setdefault(name, Totals()).record(misses, staged, data.layers, fallback)
            by_policy.setdefault(name, Totals()).record(misses, staged, data.layers, fallback)
    return totals, families


def evaluate(data: Dataset, budget: int, threshold: float) -> dict:
    train = [r for r in data.requests if r.split == "train"]
    test = [r for r in data.requests if r.split == "test"]
    totals, families = _accumulate(data, train, test, budget, threshold)
    return {"model": data.model, "budget_bytes_per_request": budget, "probability_threshold": threshold,
            "train_families": len({r.family for r in train}), "test_families": len({r.family for r in test}),
            "policies": summarize(totals), "by_family": {family: summarize(stats) for family, stats in families.items()}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--budget-bytes", type=int, required=True)
    parser.add_argument("--probability", type=float, default=0.9)
    args = parser.parse_args(argv)
    try:
        budget = positive_int(args.budget_bytes, "budget_bytes")
        if not 0 < args.probability <= 1:
            raise ValueError("probability must be in (0, 1]")
        report = evaluate(read_dataset(args.manifest), budget, args.probability)
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        print(f"invalid replay: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
