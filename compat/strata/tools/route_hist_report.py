#!/usr/bin/env python3
# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""route_hist_report.py DIR [--log ENGINE_LOG]: read the per-request routing histograms Strata writes when
STRATA_ROUTE_HIST=DIR is set (one req-NNNNNN.json per request) and answer, with data:

  1. Prompt-conditioned promotion: if the prompt's top-k experts of a layer were kept in fast memory, what share of the
     layer's DECODE work would they cover?  (k = 16/32/64, per layer; overall = routed pairs, pooled over requests.)
     Printed beside two yardsticks: chance (k / experts) and the oracle (the decode phase's own top-k, the ceiling).
  2. Tierability: the share of expert work carried by the top 10/25/50% of experts (ranked per layer, since tiers are
     per layer), for the decode phase, the prompt phase and both, pooled over requests and as a per-request mean.
  3. Which card held the decode work (the holder map at the end of each request).
  4. With --log: card 1's own expert work per layer request and the stall past it, from the decode timing lines.

Stdlib only.  File format (strata include/strata/core/route_hist.hpp, version 1): {"format","version","request","layers",
"experts","prompt_tokens","decode_tokens","prompt":[layer -> [[expert,count],...]],"decode":[...],"holder":[layer -> str]}.
FAILURE: a file that is not a version-1 dump is skipped with a warning; no valid file at all is exit 2, never an empty table.
"""
import argparse
import glob
import json
import math
import os
import re
import sys

FORMAT, VERSION = "strata-route-hist", 1
KS = (16, 32, 64)
FRACS = (0.10, 0.25, 0.50)


def _read_phase(d, phase, n_l, n_e):
    """One phase's per-layer {expert: count} dicts, validated against the header; ValueError says what is wrong."""
    rows = d[phase]
    if len(rows) != n_l:
        raise ValueError("%s has %d layers, header says %d" % (phase, len(rows), n_l))
    layers = [dict((e, c) for e, c in row) for row in rows]
    if any(len(counts) != len(row) for counts, row in zip(layers, rows)):
        raise ValueError("%s contains duplicate expert IDs" % phase)
    for row in layers:
        if any(not 0 <= e < n_e or c <= 0 for e, c in row.items()):
            raise ValueError("%s has an expert id out of range or a non-positive count" % phase)
    return layers


def load_one(path):
    """Parse and validate one dump; ValueError says what is wrong."""
    with open(path) as fh:
        d = json.load(fh)
    if d.get("format") != FORMAT or d.get("version") != VERSION:
        raise ValueError("not a %s v%d file" % (FORMAT, VERSION))
    n_l, n_e = d["layers"], d["experts"]
    req = {"path": path, "layers": n_l, "experts": n_e, "prompt_tokens": d.get("prompt_tokens", 0),
           "decode_tokens": d.get("decode_tokens", 0), "holder": d["holder"]}
    for phase in ("prompt", "decode"):
        req[phase] = _read_phase(d, phase, n_l, n_e)
    if len(d["holder"]) != n_l or any(len(h) != n_e for h in d["holder"]):
        raise ValueError("holder map does not match layers x experts")
    return req


def load_dir(path):
    reqs, bad = [], 0
    for f in sorted(glob.glob(os.path.join(path, "req-*.json"))):
        try:
            reqs.append(load_one(f))
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
            bad += 1
            print("warning: skipping %s: %s" % (f, e), file=sys.stderr)
    return reqs, bad


def top_k(counts, k):
    """The k most-routed experts, ties broken by expert id so the answer does not depend on dict order."""
    return [e for e, _ in sorted(counts.items(), key=lambda ec: (-ec[1], ec[0]))[:k]]


def coverage(reqs, k):
    """Per layer, pooled over requests: (covered, total, oracle) decode pairs.  covered = decode pairs on the prompt's
    top-k experts of that layer; oracle = decode pairs on the decode phase's own top-k.  A request with no prompt
    pairs (its prompt came from a cache) or no decode pairs cannot be scored and is left out; returns (rows, used)."""
    n_l = reqs[0]["layers"]
    rows = [[0, 0, 0] for _ in range(n_l)]
    used = 0
    for r in reqs:
        if not any(r["prompt"]) or not any(r["decode"]):
            continue
        used += 1
        for l in range(n_l):
            dec = r["decode"][l]
            keep = top_k(r["prompt"][l], k)
            rows[l][0] += sum(dec.get(e, 0) for e in keep)
            rows[l][1] += sum(dec.values())
            rows[l][2] += sum(dec[e] for e in top_k(dec, k))
    return rows, used


def _share(rows, frac, n_e):
    """Overall share of the work on each layer's top ceil(frac * n_e) experts (rows: per-layer {expert: count})."""
    n = max(1, math.ceil(frac * n_e - 1e-9))
    top = sum(sum(c for e, c in sorted(r.items(), key=lambda ec: -ec[1])[:n]) for r in rows)
    tot = sum(sum(r.values()) for r in rows)
    return top / tot if tot else None


def _merge(reqs, phases):
    n_l = reqs[0]["layers"]
    out = [dict() for _ in range(n_l)]
    for r in reqs:
        for p in phases:
            for l in range(n_l):
                for e, c in r[p][l].items():
                    out[l][e] = out[l].get(e, 0) + c
    return out


def concentration(reqs, phases, frac):
    """(pooled, per-request mean) share of work on the top `frac` of experts for the phases named."""
    n_e = reqs[0]["experts"]
    pooled = _share(_merge(reqs, phases), frac, n_e)
    each = [s for s in (_share(_merge([r], phases), frac, n_e) for r in reqs) if s is not None]
    return pooled, (sum(each) / len(each) if each else None)


def holder_share(reqs):
    """Share of decode pairs by the tier that held the expert at the end of its request."""
    tot, by = 0, {}
    for r in reqs:
        for l, row in enumerate(r["decode"]):
            for e, c in row.items():
                h = r["holder"][l][e]
                by[h] = by.get(h, 0) + c
                tot += c
    return dict((h, c / tot) for h, c in by.items()) if tot else {}


TIMING = re.compile(r"strata decode timing:.*?; card (\d+): (\d+) layer requests, own expert work ([\d.]+) us "
                    r"\+ waited past it ([\d.]+) us each")


def timing_from_log(path):
    """Per card, request-weighted: (requests, own us, waited us) over every decode timing line that carries the split."""
    acc = {}
    with open(path, errors="replace") as fh:
        for line in fh:
            for m in TIMING.finditer(line):
                card, n = int(m.group(1)), int(m.group(2))
                a = acc.setdefault(card, [0, 0.0, 0.0])
                a[0] += n
                a[1] += n * float(m.group(3))
                a[2] += n * float(m.group(4))
    return dict((c, (n, o / n, w / n)) for c, (n, o, w) in acc.items() if n)


def pct(x):
    return "   n/a" if x is None else "%5.1f%%" % (100 * x)


def _layer_cell(rows, l):
    """One layer's covered / total decode pairs as a percentage cell (n/a with no decode work)."""
    covered, total, _ = rows[l]
    return pct(covered / total if total else None)


def _pooled_cell(rows, i):
    """Column i (0 covered, 2 oracle) pooled over layers, as a share of all decode pairs."""
    return pct(sum(r[i] for r in rows) / max(1, sum(r[1] for r in rows)))


def _promotion_section(reqs, p):
    """Section 1: prompt-conditioned promotion, per layer, with the chance and oracle yardsticks."""
    n_l, n_e = reqs[0]["layers"], reqs[0]["experts"]
    p("1. PROMPT-CONDITIONED PROMOTION: share of a layer's decode work on the prompt's own top-k experts")
    res = {k: coverage(reqs, k)[0] for k in KS}
    used = coverage(reqs, KS[0])[1]
    p("   scored %d of %d requests (the rest have no prompt or no decode routing)" % (used, len(reqs)))
    if not used:
        return
    p("   layer " + "".join("  prompt-top%-3d" % k for k in KS))
    for l in range(n_l):
        p("   %5d " % l + "".join("  %11s " % _layer_cell(res[k], l) for k in KS))
    p("   ALL   " + "".join("  %11s " % _pooled_cell(res[k], 0) for k in KS))
    p("   chance" + "".join("  %11s " % pct(k / n_e) for k in KS))
    p("   oracle" + "".join("  %11s " % _pooled_cell(res[k], 2) for k in KS)
      + "   (the decode phase's own top-k: the ceiling for any predictor)")


def _tierability_section(reqs, p):
    """Section 2: share of expert work carried by the top x% of experts, per phase set."""
    p("2. TIERABILITY: share of expert work carried by the top x% of experts (per-layer ranking)")
    p("   phase            " + "".join("   top %2d%% pooled / per-request mean" % round(100 * f) for f in FRACS))
    for name, phases in (("decode", ("decode",)), ("prompt", ("prompt",)), ("prompt+decode", ("prompt", "decode"))):
        cells = []
        for f in FRACS:
            a, b = concentration(reqs, phases, f)
            cells.append("   %s / %s     " % (pct(a), pct(b)))
        p("   %-15s" % name + "".join(cells))


def report(reqs, out):
    n_l, n_e = reqs[0]["layers"], reqs[0]["experts"]
    p = lambda *a: print(*a, file=out)
    pt = sum(r["prompt_tokens"] for r in reqs)
    dt = sum(r["decode_tokens"] for r in reqs)
    p("%d requests, %d layers x %d experts, %d prompt tokens, %d decoded tokens" % (len(reqs), n_l, n_e, pt, dt))
    p("")
    _promotion_section(reqs, p)
    p("")
    _tierability_section(reqs, p)
    p("")
    p("3. WHO HELD THE DECODE WORK (tier at the end of each request; 1 = card 1 VRAM, 2 = card 2, p = peer, 0 = host memory)")
    for h, s in sorted(holder_share(reqs).items()):
        p("   %s  %s" % (h, pct(s)))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dir", help="the STRATA_ROUTE_HIST directory")
    ap.add_argument("--log", help="engine log with the decode timing lines (STRATA_DECODE_TIMING=1 and STRATA_ROUTE_HIST set)")
    a = ap.parse_args(argv)
    reqs, bad = load_dir(a.dir)
    if not reqs:
        print("error: no valid req-*.json in %s (%d skipped)" % (a.dir, bad), file=sys.stderr)
        return 2
    report(reqs, sys.stdout)
    if a.log:
        print("")
        print("4. CARD 1 OWN WORK vs WAIT, per layer request (from %s)" % a.log)
        t = timing_from_log(a.log)
        if not t:
            print("   no decode timing line carries the split (needs STRATA_DECODE_TIMING=1, STRATA_ROUTE_HIST and a remote card)")
        for card, (n, own, wait) in sorted(t.items()):
            print("   card %d: %d requests; card 1 did %.1f us of its own work meanwhile, then stalled %.1f us "
                  "(stall = %.0f%% of own+wait; wait 0 means the card answered inside card 1's own work)"
                  % (card, n, own, wait, 100 * wait / (own + wait) if own + wait else 0.0))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
