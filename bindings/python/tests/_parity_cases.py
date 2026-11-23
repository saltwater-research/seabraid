# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Seeded, deterministic case generator for the reference-vs-native parity fuzz (stdlib `random` only).

OWNER   : what a fuzz case is (a runtime declaration plus a script of weave steps) and how one seed becomes one case.
CONTRACT: make_case(seed) -> Case; the same seed always gives the same Case on any platform and Python version (the
          generator draws from random.Random(seed) in a fixed order and uses no set/dict ordering). Operand specs are
          (kind, values) pairs that build() turns into a FRESH container for each run, so two sessions never share one.
          Nothing here imports the package: the runner (test_parity_fuzz.py) owns running a case.
SIZE    : under 280 lines.
FAILURE : none; this module only builds data.
CHECKER : tests/test_parity_fuzz.py (determinism, reference-only invariants, and the native comparison).
FINISH  : the same seed twice gives equal cases; every fault kind and container kind below is drawn by some seed.
"""

from __future__ import annotations

import random
from array import array
from dataclasses import dataclass, field

# Activation values may be anything (the engine never inspects them); weights must be finite.
ACTIVATION_POOL = [0.0, -0.0, 1.0, -1.0, 0.5, 2.5, 1e-45, -1e-45, 1.1754944e-38, 3.4028235e38, -3.4028235e38,
                   1e30, float("inf"), float("-inf"), float("nan"), 1e39, 0.1, -7.25]
WEIGHT_POOL = [0.0, -0.0, 0.25, 0.5, 1.0, -1.5, 0.1, 1e-45, 1.1754944e-38, 3.4028235e38, 1e30, -1e30, 7.0]
BAD_WEIGHTS = [float("nan"), float("inf"), float("-inf"), 1e39]  # 1e39 rounds to inf in float32
FAULTS = ["duplicate", "id_high", "id_negative", "id_huge", "skip_without_partial", "skip_weighted", "skip_nan",
          "bad_weight"]
MODES = ["ok"] * 10 + ["raise", "nan", "half"]  # what the hosted provider does on a step
FORMATS_F32, FORMATS_F16, FORMATS_BF16 = 1, 3, 4


@dataclass
class Step:
    activations: tuple  # (kind, values) for F32, (kind, bytes) for F16/BF16
    expert_ids: tuple
    weights: tuple
    epoch: int
    activation_format: int
    operation: int
    origins: int
    out: tuple | None  # (kind, element count) of an `out` buffer, or None
    provider_mode: str


@dataclass
class Case:
    seed: int
    shape: dict
    builtin: bool  # True: the built-in reference transform; False: a hosted Python callable
    partial: bool
    compact: bool
    accepts: tuple
    operations: int
    bad_prepare: dict | None  # a shape the engine must refuse, tried before the real one
    steps: list = field(default_factory=list)


def _pick(rng: random.Random, pool: list) -> object:
    return pool[rng.randrange(len(pool))]


def _shape(rng: random.Random) -> dict:
    experts = rng.randint(1, 5)
    return {"hidden": rng.randint(1, 4), "experts": experts, "max_tokens": rng.randint(1, 4),
            "top_k": rng.randint(1, experts)}


def _bad_shape(rng: random.Random) -> dict | None:
    kind = rng.randrange(8)
    if kind == 0:
        return {"hidden": 2, "experts": 2, "max_tokens": 2, "top_k": 3}  # top_k above experts
    if kind == 1:
        return {"hidden": (1 << 32) - 1, "experts": 4, "max_tokens": (1 << 32) - 1, "top_k": 2}  # unaddressable
    if kind == 2:
        return {"hidden": 0, "experts": 4, "max_tokens": 2, "top_k": 2}  # Python refuses before any engine call
    return None


def _clean_picks(rng: random.Random, case: Case, tokens: int) -> tuple[list, list]:
    """Valid ids and weights for `tokens` rows: distinct real ids per row, skips only with partial routes."""
    k, experts = case.shape["top_k"], case.shape["experts"]
    ids: list = []
    weights: list = []
    for _ in range(tokens):
        row = rng.sample(range(experts), k)
        for index in range(k):
            if case.partial and rng.random() < 0.35:
                row[index] = -1
        ids.extend(row)
        weights.extend(_pick(rng, [0.0, -0.0]) if i == -1 else _pick(rng, WEIGHT_POOL) for i in row)
    return ids, weights


def _inject(rng: random.Random, case: Case, ids: list, weights: list) -> str:
    """Corrupt exactly one pick (or return "" when the fault cannot apply to this shape)."""
    if not ids:
        return ""
    fault = _pick(rng, FAULTS)
    index = rng.randrange(len(ids))
    k = case.shape["top_k"]
    row_start = index - index % k
    if fault == "duplicate":
        other = row_start + (index - row_start + 1) % k
        if k < 2 or ids[other] == -1:
            return ""
        ids[index], weights[index] = ids[other], 0.5
    elif fault == "id_high":
        ids[index] = case.shape["experts"] + rng.randint(0, 1)
    elif fault == "id_negative":
        ids[index] = -2
    elif fault == "id_huge":
        ids[index] = 2**31 - 1
    elif fault == "skip_without_partial":
        if case.partial:
            return ""
        ids[index], weights[index] = -1, 0.0
    elif fault == "skip_weighted":
        ids[index], weights[index] = -1, _pick(rng, [0.5, -1.0, 1e-45])
    elif fault == "skip_nan":
        ids[index], weights[index] = -1, float("nan")
    else:
        weights[index] = _pick(rng, BAD_WEIGHTS)
        if ids[index] == -1:
            ids[index] = 0  # a bad weight on a skip is the skip_weighted fault; keep the faults distinct
    return fault


def _container(rng: random.Random, values: list, integers: bool) -> tuple:
    kinds = ["list", "list", "tuple", "array", "view", "misaligned"]
    kind = _pick(rng, kinds)
    if integers and any(abs(v) >= 2**31 for v in values):
        kind = "list"  # array('i') cannot hold it; the plain-sequence path must refuse it with ValueError
    return (kind, list(values))


def _activations(rng: random.Random, case: Case, count: int, fmt: int) -> tuple:
    if fmt == FORMATS_F32:
        return _container(rng, [_pick(rng, ACTIVATION_POOL) if rng.random() < 0.5 else rng.uniform(-9, 9)
                                for _ in range(count)], False)
    raw = bytes(rng.randrange(256) for _ in range(2 * count))  # every 16-bit pattern: NaN, inf, subnormal, -0
    return (_pick(rng, ["bytes", "bytearray", "half_array", "byte_view"]), raw)


def _epoch(rng: random.Random, policy: str, previous: int) -> int:
    if policy == "none":
        return 0
    if rng.random() < 0.04:
        return _pick(rng, [2**64 - 1, 2**64, -1])  # the last two are refused by Python before any engine call
    if policy == "monotone":
        return previous + rng.randint(0, 2)
    return rng.randint(0, 4)


def _format(rng: random.Random, case: Case) -> int:
    if rng.random() < 0.04:
        return _pick(rng, [0, 2, 5, 99, 2**32 - 1, 2**32])  # unknown ids; the last is out of range for Python
    declared = [FORMATS_F32] * 3 + [f for f in (FORMATS_F16, FORMATS_BF16) if f in case.accepts]
    if rng.random() < 0.06:
        return _pick(rng, [FORMATS_F16, FORMATS_BF16])  # declared or not: undeclared must be refused on both
    return _pick(rng, declared)


def _out(rng: random.Random, count: int) -> tuple | None:
    if rng.random() < 0.65:
        return None
    kinds = [("array", count)] * 4 + [("view", count)] * 2 + [("bytearray", count * 4)] * 2
    kinds += [("misaligned", count), ("array", count + 1), ("array", max(count - 1, 0)), ("readonly", count), ("list", count),
              ("bytearray", count * 4 + 1)]
    return _pick(rng, kinds)


def _step(rng: random.Random, case: Case, policy: str, previous_epoch: int) -> Step:
    shape = case.shape
    fmt = _format(rng, case)
    # Mostly valid token counts; 0 and max_tokens + 1 are the two refusals at the edges.
    tokens = _pick(rng, list(range(1, shape["max_tokens"] + 1)) * 12 + [0, shape["max_tokens"] + 1])
    ids, weights = _clean_picks(rng, case, tokens)
    if rng.random() < 0.12:
        _inject(rng, case, ids, weights)
    element_count = tokens * shape["hidden"]
    if rng.random() < 0.03:
        element_count += _pick(rng, [-1, 1])  # ragged rows or a count the shape refuses
    if rng.random() < 0.02 and ids:
        ids.pop()
    out_count = element_count if tokens else 0
    return Step(
        _activations(rng, case, max(element_count, 0), fmt), _container(rng, ids, True),
        _container(rng, weights, False), _epoch(rng, policy, previous_epoch), fmt,
        _pick(rng, [0] * 10 + list(range(case.operations)) * 2 + [case.operations]),
        _pick(rng, [0] * 6 + [1, 2, 4, 8, 16, 31] + [32] * (rng.random() < 0.06)), _out(rng, max(out_count, 0)), _pick(rng, MODES),
    )


def make_case(seed: int) -> Case:
    rng = random.Random(seed)
    shape = _shape(rng)
    partial = rng.random() < 0.55
    # compact without partial is the construction refusal; draw it rarely so it is still covered.
    compact = rng.random() < (0.5 if partial else 0.08)
    builtin = rng.random() < 0.4
    accepts = () if builtin else tuple(f for f in (FORMATS_F16, FORMATS_BF16) if rng.random() < 0.5)
    operations = 1 if builtin else rng.randint(1, 3)
    case = Case(seed, shape, builtin, partial, compact, accepts, operations, _bad_shape(rng))
    policy = _pick(rng, ["none", "monotone", "monotone", "chaos"])
    previous = 0
    for _ in range(rng.randint(2, 6)):
        step = _step(rng, case, policy, previous)
        if isinstance(step.epoch, int) and 0 <= step.epoch < 2**64:
            previous = max(previous, step.epoch)
        case.steps.append(step)
    return case


def _misaligned(code: str, values: list) -> memoryview:
    """A writable, correctly typed buffer one byte past a 4-byte boundary (the engine refuses such a pointer)."""
    data = array(code, values).tobytes()
    raw = bytearray(1 + len(data))
    raw[1:] = data
    return memoryview(raw)[1:].cast(code)


def build(spec: tuple, integers: bool = False) -> object:
    """A fresh container for one operand spec."""
    kind, values = spec
    code = "i" if integers else "f"
    if kind == "list":
        return list(values)
    if kind == "tuple":
        return tuple(values)
    if kind == "array":
        return array(code, values)
    if kind == "view":
        return memoryview(array(code, values))
    if kind == "misaligned":
        return _misaligned(code, values)
    if kind == "bytes":
        return bytes(values)
    if kind == "bytearray":
        return bytearray(values)
    if kind == "half_array":
        return array("H", bytes(values))  # the generator always draws whole 16-bit elements
    return memoryview(bytes(values))  # byte_view


def build_out(spec: tuple | None) -> object | None:
    """A fresh `out` target pre-filled with a sentinel, so an untouched buffer is visible in the comparison."""
    if spec is None:
        return None
    kind, size = spec
    if kind == "array":
        return array("f", [123.0] * size)
    if kind == "view":
        return memoryview(array("f", [123.0] * size))
    if kind == "bytearray":
        return bytearray(b"\xaa" * size)
    if kind == "misaligned":
        return _misaligned("f", [123.0] * size)
    if kind == "readonly":
        return bytes(4 * size)
    return [123.0] * size  # list: not a buffer, must be refused
