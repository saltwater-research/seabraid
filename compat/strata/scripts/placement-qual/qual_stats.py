# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Pure statistics and engine-log parsing for the placement qualification runner (no I/O, no network).

OWNER: this module owns "how a number is computed from raw samples" (percentiles, Welch lower bound, time-to-effect,
round-line and file-tier-I/O parsing).  The pass thresholds live in arms.json and the judging in qual_judge.py.
CONTRACT: every function is pure; an empty input gives ``None`` (never 0), so a missing measurement cannot pass a gate.
FAILURE: a malformed number in a log line drops that token and is counted by the caller via ``len(rounds)``; it never
raises, because one garbled log line must not erase a whole run's evidence.
"""

from __future__ import annotations

import math
import re

_KV = re.compile(r"([A-Za-z_][\w]*)=([^\s,;]+)")
_Z95 = 1.6448536269514722


def percentile(xs: list[float], q: float) -> float | None:
    """Linear-interpolated percentile, q in [0, 100]."""
    if not xs:
        return None
    s = sorted(xs)
    pos = (len(s) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _var(xs: list[float]) -> float:
    m = sum(xs) / len(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)


def _lentz(aa: float, c: float, d: float) -> tuple[float, float, float]:
    """One modified-Lentz step: returns (new c, new d, factor d*c)."""
    tiny = 1e-300
    d = 1.0 + aa * d
    c = 1.0 + aa / c
    d = 1.0 / (d if abs(d) > tiny else tiny)
    c = c if abs(c) > tiny else tiny
    return c, d, d * c


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction of the regularized incomplete beta (Numerical Recipes betacf)."""
    c, d = 1.0, 1.0 / (1.0 - (a + b) * x / (a + 1.0))
    h = d
    for m in range(1, 2000):
        c, d, f = _lentz(m * (b - m) * x / ((a - 1.0 + 2 * m) * (a + 2 * m)), c, d)
        h *= f
        c, d, f = _lentz(-(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 1.0 + 2 * m)), c, d)
        h *= f
        if abs(f - 1.0) < 1e-15:
            break
    return h


def _ibeta(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0 or x >= 1.0:
        return 0.0 if x <= 0.0 else 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _t_tail(t: float, df: float) -> float:
    """P(T > t) for t >= 0 and Student t with ``df`` degrees of freedom."""
    return 0.5 * _ibeta(df / 2.0, 0.5, df / (df + t * t))


def t95(df: float) -> float:
    """One-sided 95% Student t quantile for any real df >= 1 (numerical inversion of the incomplete beta; stdlib only).

    WHY exact: the earlier Cornish-Fisher expansion is wrong below df 3 (df 1 gave 3.8 against 6.3138), and Welch's
    effective df is as low as 1.005 with two starts per arm, so a small-sample approximation would pass a failing arm.
    Above df 10000 the normal quantile with the first Cornish-Fisher terms is exact to 1e-10.  df < 1 or NaN raises."""
    if not df >= 1.0:
        raise ValueError(f"t95 needs df >= 1, got {df}")
    if df > 1e4:
        z = _Z95
        return z + (z**3 + z) / (4 * df)
    lo, hi = 0.0, 8.0  # t(0.95, 1) = 6.3138 is the largest quantile for df >= 1
    for _ in range(70):
        mid = (lo + hi) / 2.0
        lo, hi = (mid, hi) if _t_tail(mid, df) > 0.05 else (lo, mid)
    return (lo + hi) / 2.0


def welch_bounds(arm: list[float], ref: list[float]) -> tuple[float, float, float] | None:
    """One-sided 95% (lower, upper) bounds and the point estimate of (arm - ref), each as percent of mean(ref); None if n < 2 or the effective df is below 1."""
    if len(arm) < 2 or len(ref) < 2 or not mean(ref):
        return None
    va, vr = _var(arm) / len(arm), _var(ref) / len(ref)
    diff, se = mean(arm) - mean(ref), math.sqrt(va + vr)
    k = 0.0
    if se != 0:
        df = (va + vr) ** 2 / (va**2 / (len(arm) - 1) + vr**2 / (len(ref) - 1))
        if not df >= 1.0:
            return None  # decline rather than approximate
        k = t95(df)
    scale = 100.0 / abs(mean(ref))
    return ((diff - k * se) * scale, (diff + k * se) * scale, diff * scale)


_FLOAT = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
# WHY strings: a 16-digit hex hash that happens to be all digits (or digits and one e) would parse as a float and lose bits.
_STRING_KEYS = ("qmask", "table_hash")
# The legacy ladder's record (compat/strata/strata-ext/src/_retier_ladder.cpp RetierLoop::apply_ladder + describe_ladder_round, plus
# the STALE / "held by both cards" tail the engine appends in patch 0012's generate.cpp round hook): no hash, no helper_share.
_LEGACY = re.compile(r"retier round (\d+) \(window (\d+), promote\).*?committed (\d+), failed (\d+).*?transition ([0-9.]+) ms"
                     r"(?:.*?STALE (\d+), held by both cards (\d+), card 2 (open|CLOSED))?")


def _num(tok: str) -> float | str:
    return float(tok) if _FLOAT.fullmatch(tok) else tok


def _legacy_row(line: str) -> dict | None:
    m = _LEGACY.search(line)
    if not m:
        return None
    row = {"round": float(m[1]), "window": float(m[2]), "committed": float(m[3]), "failed": float(m[4]), "fence_us": float(m[5]) * 1000.0, "legacy": 1.0}
    if m[6] is not None:
        row.update(stale=float(m[6]), both=float(m[7]), qmask="0x2" if m[8] == "CLOSED" else "0x0")
    return row


def parse_round_lines(text: str, marker: str = "placement round=") -> list[dict]:
    """Every seabraid round line (``marker`` plus ``fence_us=``) and every legacy ladder round line, as {key: number-or-string}."""
    out = []
    for line in text.splitlines():
        if marker in line and "fence_us=" in line:
            out.append({k: (v if k in _STRING_KEYS else _num(v)) for k, v in _KV.findall(line)})
        elif "retier round " in line and (row := _legacy_row(line)):
            out.append(row)
    return out


# One record per failure (the driver's FAILURE CONTRACT: adapters/strata/placement_driver.cpp PlacementDriver::apply, ::verify,
# ::note_degraded, ::die and ::skip, and placement_driver_open.cpp): these texts ARE the "disagreements"
# of criterion C3, because the round line has no such field.  A legacy STALE or CLOSED shows in the parsed row instead.
DRIVER_ERRORS = ("placement lead disagrees", "placement degraded", "placement driver dead", "placement lead rung quarantined",
                 "ERR placement", "ERR retier promote", "placement open refused", "serve: placement: ", "placement round skipped")


def driver_errors(text: str) -> list[str]:
    """Log lines that report a placement fault; ``placement round skipped`` is a WARN the driver prints once per 16 repeats."""
    return [ln.strip()[:200] for ln in text.splitlines() if any(k in ln for k in DRIVER_ERRORS) and "placement round=" not in ln]


def column(rounds: list[dict], key: str) -> list[float]:
    return [r[key] for r in rounds if isinstance(r.get(key), float)]


def quarantined(rounds: list[dict]) -> bool:
    """True when any round line reports a non-zero quarantine mask (``qmask=0x..`` hex)."""
    return any(_mask(r) != 0 for r in rounds)


def _mask(r: dict) -> int:
    try:
        return int(str(r.get("qmask", "0x0")), 16)
    except ValueError:
        return 1  # an unreadable mask is a fault, never a pass


def committed(rounds: list[dict]) -> float:
    """Total pairs committed over the round lines."""
    return sum(column(rounds, "committed"))


def latency_table(rounds: list[dict], keys: tuple[str, ...]) -> dict[str, dict]:
    """Per key: p50, p95, max, count over all round lines (the swap-latency table of the protocol)."""
    out = {}
    for k in keys:
        xs = column(rounds, k)
        out[k] = {"p50": percentile(xs, 50), "p95": percentile(xs, 95), "max": max(xs) if xs else None, "n": len(xs)}
    return out


def time_to_effect(shares: list[float], tol: float = 0.10, win: int = 3) -> int | None:
    """Rounds until the ``win``-round moving mean stays within ``tol`` (relative) of the plateau (mean of the last ``win``).

    ``shares`` starts at the first round after the shift.  None when there are too few rounds or the plateau is zero.
    """
    if len(shares) < 2 * win:
        return None
    plateau = sum(shares[-win:]) / win
    if plateau == 0:
        return None
    ma = [sum(shares[i : i + win]) / win for i in range(len(shares) - win + 1)]
    ok = [abs(m - plateau) <= tol * abs(plateau) for m in ma]
    first = len(ok)
    while first > 0 and ok[first - 1]:
        first -= 1
    return None if first == len(ok) else first


def commit_effect(per_round: list[float], frac: float = 0.9) -> int | None:
    """Rounds until ``frac`` of the commits of ``per_round`` (committed pairs per round after the shift) have landed.

    No longer used by C4 (see ``settle``: 90% of an unbounded total cannot tell churn from convergence); kept for its test.
    None when nothing was committed (never 0: a run that moved nothing has no time-to-effect)."""
    total = sum(per_round)
    if total <= 0:
        return None
    done = 0.0
    for i, c in enumerate(per_round, 1):
        done += c
        if done >= frac * total:
            return i
    return None


def settle(per_round: list[float], horizon: int, tail: int) -> tuple[str, int | None, float | None]:
    """Did the commit stream settle inside a FIXED horizon?  -> (status, round of the last commit, stable share).

    ``per_round`` is committed pairs per round from the shift on.  Only the first ``horizon`` rounds count, and the last ``tail``
    of them must be quiet, so continuous churn is "churn", never convergence (the 90%-of-the-eventual-count rule it replaces
    scored a run that never stops moving as converged at round 0.9 N).  stable share = quiet rounds / horizon.
    Status: "settled", "churn" (still moving in the tail), "none" (nothing committed: no effect), "short" (fewer rounds than the horizon)."""
    if len(per_round) < horizon or not 1 <= tail < horizon:
        return "short", None, None
    moved = [i for i, c in enumerate(per_round[:horizon], 1) if c > 0]
    share = 1.0 - len(moved) / horizon
    if not moved:
        return "none", None, share
    if moved[-1] > horizon - tail:
        return "churn", None, share
    return "settled", moved[-1], share


def parse_io(text: str, patterns: dict[str, str]) -> dict[str, list[float]]:
    """Per metric name, every float captured by its regex (group 1) in the log text, in order."""
    out: dict[str, list[float]] = {}
    for name, rx in patterns.items():
        out[name] = [float(m.group(1)) for m in re.finditer(rx, text) if _FLOAT.fullmatch(m.group(1))]
    return out


def last_segment(rows: list[str]) -> list[str]:
    """Rows of the last position-restart segment (column 0 = position): the rows of the most recent request only."""
    start = 0
    prev = None
    for i, row in enumerate(rows):
        pos = _num(row.split("\t", 1)[0])
        if prev is not None and isinstance(pos, float) and pos <= prev:
            start = i
        prev = pos if isinstance(pos, float) else prev
    return rows[start:]
