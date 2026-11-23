#!/usr/bin/env python3
# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""kl.py A B                                  (report only: always exits 0)
kl.py --baseline TSV --arm TSV [--max-kl X] [--min-top1 Y]   (gate: exit 0 PASS, 1 FAIL, 2 INVALID)

Compares two STRATA_LOGPOS files (per row: pos, target, logp_target, top, logp_top, top==target, extra, without,
then STRATA_LOGPOS_TOPK id:logprob pairs). Prints over the positions present in both files: mean and p99
|d logp(target)|, top-1 agreement, and KL(A||B) over A's top-K (B's logprob of a token outside its own top-K
is floored at B's K-th value).
Arguments:
  A   reference LOGPOS file
  B   test LOGPOS file
Gate mode prints ONE line: verdict, the reason, and the numbers it judged (mean KL, top-1, paired signed dlogp of
logp_target = arm - baseline, with its t statistic and n). dlogp/t are reported, not gated: the bound is on KL and top-1.
No environment variables.
"""
import argparse
import math
import sys

# Gate defaults. 0.03 mean KL is the threshold Strata's own MMQ and tail-skip defaults accept, so an arm that stays
# inside it is no worse than changes the engine already ships; it also sits inside the warm off-vs-off noise band
# (0.021-0.078) from below, so it does not fail an arm for noise the design already tolerates in its low half.
# 0.95 top-1: at most 1 position in 20 may change its argmax; a distribution can keep a low mean KL while flipping
# the greedy token, and the greedy token is what the user reads.
MAX_KL = 0.03
MIN_TOP1 = 0.95


class Invalid(Exception):
    """The data cannot be judged (missing, empty, NaN, misaligned); the gate must not pass it."""


def _parse_line(p, n, line, strict):
    """One LOGPOS row -> (pos, (logp_target, top, {id: logprob}, target)); strict maps a bad row to Invalid, else it raises.
    WHY target is kept: the teacher-forced token at a position is part of the row's identity, and a pair of runs that forced
    different tokens measures nothing (two files can agree on every position and still be different texts)."""
    try:
        f = line.split("	")
        tk = {}
        for x in f[8:]:
            if ":" in x:
                i, v = x.split(":")
                tk[int(i)] = float(v)
        return int(f[0]), (float(f[2]), int(f[3]), tk, int(f[1]))
    except (ValueError, IndexError):
        if strict:
            raise Invalid(f"{p} line {n} is malformed")
        raise


def _check_row(p, n, pos, row, rows):
    """Gate-mode row checks: finite numbers, a non-empty top-K, no repeated position."""
    if not math.isfinite(row[0]) or not all(math.isfinite(v) for v in row[2].values()):
        raise Invalid(f"{p} line {n} has NaN/inf")
    if not row[2]:
        raise Invalid(f"{p} line {n} has no top-K pairs")
    if pos in rows:
        raise Invalid(f"{p} repeats position {pos}")


def load(p, strict=False):
    """strict (gate mode): raise Invalid on unreadable/short/NaN/duplicate rows instead of trusting them."""
    rows = {}
    try:
        lines = open(p).read().splitlines()
    except OSError as e:
        raise Invalid(f"cannot read {p}: {e.strerror}")
    for n, line in enumerate(lines, 1):
        pos, row = _parse_line(p, n, line, strict)
        if strict:
            _check_row(p, n, pos, row, rows)
        rows[pos] = row
    return rows


def kl_row(ta, tb):
    """KL(A||B) over A's top-K; B's logprob for a token outside its own top-K is floored at B's K-th value."""
    floor = min(tb.values())
    return sum(math.exp(la) * (la - tb.get(i, floor)) for i, la in ta.items())


def _require_aligned(base, arm):
    """Alignment is exact: same positions AND the same teacher-forced target at each position, or INVALID."""
    if not base or not arm:
        raise Invalid(f"empty file (baseline {len(base)} rows, arm {len(arm)} rows)")
    if set(base) != set(arm):
        only = sorted(set(base) ^ set(arm))
        raise Invalid(f"positions differ: baseline {len(base)} rows, arm {len(arm)} rows, "
                      f"{len(only)} unpaired (first {only[:3]})")
    diff = [p for p in sorted(base) if base[p][3] != arm[p][3]]
    if diff:
        raise Invalid(f"target ids differ at {len(diff)} of {len(base)} positions (first position {diff[0]}: "
                      f"baseline {base[diff[0]][3]}, arm {arm[diff[0]][3]}): the runs did not teacher-force the same text")


def _paired_t(dl):
    """-> (mean, t) of the paired differences; zero spread gives t 0 for a zero mean, else signed inf."""
    n = len(dl)
    mean = sum(dl) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in dl) / (n - 1)) if n > 1 else 0.0
    if sd > 0:
        return mean, mean / (sd / math.sqrt(n))
    return mean, (0.0 if mean == 0 else math.copysign(float("inf"), mean))


def gate(base, arm, max_kl, min_top1):
    """Return (exit_code, one summary line). Alignment is exact: same positions and targets in both files or INVALID."""
    _require_aligned(base, arm)
    pos = sorted(base)
    n = len(pos)
    kl = sum(kl_row(base[p][2], arm[p][2]) for p in pos) / n
    top1 = sum(base[p][1] == arm[p][1] for p in pos) / n
    mean, t = _paired_t([arm[p][0] - base[p][0] for p in pos])
    if not all(math.isfinite(v) for v in (kl, top1, mean)):
        raise Invalid(f"non-finite statistic (KL {kl}, top-1 {top1}, dlogp {mean})")
    why = []
    if kl > max_kl:
        why.append(f"mean KL {kl:.5f} > max {max_kl}")
    if top1 < min_top1:
        why.append(f"top-1 {top1:.4f} < min {min_top1}")
    nums = f"n={n} KL {kl:.5f} (max {max_kl}) top-1 {top1:.4f} (min {min_top1}) dlogp {mean:+.4f} t {t:+.2f}"
    return (1, f"KL-GATE FAIL: {'; '.join(why)} | {nums}") if why else (0, f"KL-GATE PASS: {nums}")


def _run_gate(ap, args):
    """Gate mode: print one line and exit 0 PASS / 1 FAIL / 2 INVALID."""
    if not (args.baseline and args.arm) or args.a or args.b:
        ap.error("gate mode needs both --baseline and --arm (and no positional A B)")
    try:
        code, line = gate(load(args.baseline, True), load(args.arm, True), args.max_kl, args.min_top1)
    except Invalid as e:
        code, line = 2, f"KL-GATE INVALID: {e}"
    print(line)
    sys.exit(code)


def _report(a, b):
    """Report mode: print the one-line comparison of the positions present in both files."""
    pos = sorted(set(a) & set(b))
    d = [abs(a[p][0] - b[p][0]) for p in pos]
    agree = sum(a[p][1] == b[p][1] for p in pos)
    kls = sorted(kl_row(a[p][2], b[p][2]) for p in pos if a[p][2] and b[p][2])
    d.sort()
    q = lambda v, f: v[min(len(v) - 1, int(f * len(v)))] if v else float("nan")
    print(f"positions {len(pos)}  |dlogp| mean {sum(d)/len(d):.4f} p99 {q(d,.99):.4f}  top-1 agree {100*agree/len(pos):.1f}%"
          f"  KL(top32) mean {sum(kls)/max(1,len(kls)):.5f} p99 {q(kls,.99):.5f} max {kls[-1] if kls else float('nan'):.5f}")


def main():
    ap = argparse.ArgumentParser(description="Compare two STRATA_LOGPOS files: logp drift, top-1 agreement, KL(top-K).")
    ap.add_argument("a", metavar="A", nargs="?", help="reference LOGPOS file")
    ap.add_argument("b", metavar="B", nargs="?", help="test LOGPOS file")
    ap.add_argument("--baseline", help="gate mode: baseline LOGPOS file")
    ap.add_argument("--arm", help="gate mode: arm LOGPOS file")
    ap.add_argument("--max-kl", type=float, default=MAX_KL, help=f"gate: fail above this mean KL (default {MAX_KL})")
    ap.add_argument("--min-top1", type=float, default=MIN_TOP1, help=f"gate: fail below this top-1 agreement (default {MIN_TOP1})")
    args = ap.parse_args()
    if args.baseline or args.arm:
        _run_gate(ap, args)
    if not (args.a and args.b):
        ap.error("give A B, or --baseline and --arm")
    _report(load(args.a), load(args.b))


if __name__ == "__main__":
    main()
