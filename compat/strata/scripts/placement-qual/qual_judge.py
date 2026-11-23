# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Pass criteria C1 to C9 of the placement qualification (DESIGN.md section 5.4), decided from collected results (C10, the decode path, is qual_judge_decode.py).

OWNER: this module owns the criteria logic; every number it compares to lives in arms.json ``thresholds``.
CONTRACT: ``judge(cfg, results, kl) -> list[verdict]`` where a verdict is {id, title, status, detail}; status is
PASS, FAIL, or WARN.  WARN means "not decidable from this data" (missing arm, too few starts, no round lines, no
reference, no pooled KL record for a required start): a criterion never passes on absence of evidence, and
tests/test_placement_qual_verdicts.py removes each criterion's evidence in turn to prove it.  A KL checker that did not
return one of its documented exit codes (0 PASS, 1 FAIL, 2 INVALID) is a FAIL.  Non-inferiority uses the Welch one-sided
95% Student-t bound from qual_stats (declined, hence WARN, when the effective df is below 1).
PROVISIONAL: the fence budget (``fence_p95_ms`` 100 ms) is the maintainers' to change; C1 is the real arbiter.
FAILURE: pure, no I/O; a missing key is treated as missing data.
"""

from __future__ import annotations

import statistics

import qual_stats as st


def _v(cid: str, title: str, status: str, detail: str) -> dict:
    return {"id": cid, "title": title, "status": status, "detail": detail}


def _runs(results: list[dict], wl: str, key: str) -> list[dict]:
    return [r for r in results if r["workload"] == wl and r["key"] == key and r.get("ok")]


def _decode(r: dict, phase: str = "decode", skip: int = 1) -> float | None:
    xs = [q["decode_tps"] for q in r["phases"].get(phase, [])[skip:] if q.get("decode_tps") is not None]
    return st.mean(xs)


def _series(results, wl, key, fn) -> list[float]:
    return [x for x in (fn(r) for r in _runs(results, wl, key)) if x is not None]


def _noninferior(cid: str, title: str, arm: list[float], ref: list[float], margin: float, names: str) -> dict:
    b = st.welch_bounds(arm, ref)
    if b is None:
        return _v(cid, title, "WARN", f"{names}: need >= 2 starts each (have {len(arm)}/{len(ref)})")
    ok = b[0] > -margin
    return _v(cid, title, "PASS" if ok else "FAIL", f"{names}: diff {b[2]:+.2f}%, lower 95% bound {b[0]:+.2f}% (need > -{margin}%), n={len(arm)}/{len(ref)}")


def c1(cfg, res) -> list[dict]:
    m = cfg["thresholds"]["c1_margin_pct"]
    return [_noninferior("C1", "Decode non-inferior", _series(res, "W1", a, _decode), _series(res, "W1", r, _decode), m, f"{a} vs {r}")
            for a, r in (("P2", "P0"), ("N2", "N1"))]


def c2(cfg, res) -> list[dict]:
    p0, z = _runs(res, "W1", "P0"), _runs(res, "W1", "P2z")
    shas = lambda rs: {tuple(q.get("sha") for q in r["phases"]["decode"]) for r in rs}  # noqa: E731
    out = [_noninferior("C2", "A/A decode", _series(res, "W1", "P2z", _decode), _series(res, "W1", "P0", _decode), cfg["thresholds"]["c2_margin_pct"], "P2z vs P0")]
    if not p0 or not z:
        return out + [_v("C2", "A/A token streams", "WARN", "need P0 and P2z starts")]
    same = shas(z) <= shas(p0) and len(shas(p0)) == 1
    return out + [_v("C2", "A/A token streams", "PASS" if same else "FAIL", f"P0 distinct streams {len(shas(p0))}, P2z streams {len(shas(z))}, P2z subset of P0: {shas(z) <= shas(p0)}")]


def _c3_scan(res: list[dict]) -> tuple[int, list[str], list[str]]:
    """(round lines seen, starts with a driver fault, non-W5 starts that saw a quarantine).

    WHY driver_errors: the round line has no disagreements field; a disagreement is the driver's own ERROR record
    (st.DRIVER_ERRORS), and a legacy STALE count or CLOSED card shows up as a parsed field."""
    seen, bad, quarantines = 0, [], []
    for r in (x for x in res if x["workload"] != "W5"):
        who = f"{r['workload']}/{r['key']}/s{r['idx']}"
        seen += len(r.get("rounds", []))
        stale = any(rd.get("stale", 0) > 0 for rd in r.get("rounds", []))
        bad += [who] if r.get("driver_errors") or stale else []
        quarantines += [who] if r.get("quarantine_seen") else []
    return seen, bad, quarantines


def _hash_diverged(res: list[dict]) -> list[int]:
    """Starts whose first-round table hash differs between N1 and N2.  The legacy ladder prints no hash, so with a missing side nothing is compared."""
    h = {k: {r["idx"]: r.get("table_hash_first") for r in _runs(res, "W1", k)} for k in ("N1", "N2")}
    return [i for i in h["N1"] if i in h["N2"] and None not in (h["N1"][i], h["N2"][i]) and h["N1"][i] != h["N2"][i]]


def c3(cfg, res) -> dict:
    seen, bad, quarantines = _c3_scan(res)
    diverged = _hash_diverged(res)
    if seen == 0:
        return _v("C3", "Invariants", "WARN", "no round line in any start; nothing to check (the seabraid round line is INFO: it is invisible unless the engine installs a log sink at INFO)")
    status = "FAIL" if (bad or quarantines or diverged) else "PASS"
    return _v("C3", "Invariants", status, f"rounds checked {seen}; driver faults in {bad or 'none'}; quarantine outside W5 {quarantines or 'none'}; N1/N2 round-1 hash diverged at starts {diverged or 'none'} (legacy prints no hash: only compared when both exist)")


def _expects_commit(cfg: dict, key: str) -> bool:
    """A promotion arm with a pair budget (moves >= 2: the planner moves max_moves/2 pairs) must commit in the smoke; static and A/A arms must not fault."""
    arm = cfg["arms"][key.split("/")[0]]
    return "policy" in arm and {**cfg["policy"], **arm["policy"]}["moves"] >= 2


def w0(cfg, res) -> dict:
    """Smoke: every W0 start served and logged no driver fault; promotion arms also produced a round line with a committed pair."""
    runs = [r for r in res if r["workload"] == "W0"]
    if not runs:
        return _v("W0", "Smoke", "WARN", "no W0 starts")
    rounds = lambda r: r.get("rounds", [])  # noqa: E731
    served = {f"{r['workload']}-{r['key'].replace('/', '-')}-s{r['idx']}" for r in runs}
    # a start that never served is not in results: it must still fail the smoke (a failed first attempt that a retry then served is only noted)
    dead = [n for n in cfg.get("_failed_starts", []) if "W0-" in n and n.rsplit("-", 1)[0].replace("-a2", "") not in served]
    bad = [f"start failed: {n}" for n in dead] + [f"{r['key']}/s{r['idx']}: ok={r['ok']} rounds={len(rounds(r))} committed={st.committed(rounds(r))} errors={len(r.get('driver_errors', []))} quarantine={r.get('quarantine_seen')}"
           for r in runs if not (r["ok"] and not r.get("driver_errors") and not r.get("quarantine_seen")
                                 and (st.committed(rounds(r)) >= 1 or not _expects_commit(cfg, r["key"])))]
    return _v("W0", "Smoke", "FAIL" if bad else "PASS", f"{len(runs) - len(bad)} of {len(runs)} starts OK" + (f"; failed: {bad}" if bad else ""))


def _settle(r: dict, t: dict) -> tuple[str, int | None, float | None]:
    """Settling of one W2 start over the fixed horizon that begins at the first round after the A phase."""
    a = r["phases"].get("A", [])
    k0 = a[-1]["rounds_after"] if a else 0
    return st.settle(st.column(r["rounds"], "committed")[k0:], t["c4_horizon_rounds"], t["c4_tail_rounds"])


def _med(xs: list) -> float | None:
    ys = [x for x in xs if x is not None]
    return statistics.median(ys) if ys else None


def _slow(per: dict, lim: float) -> dict:
    return {k: _med(per[k]) for k in ("P2", "N2") if _med(per[k]) is not None and _med(per[k]) > lim}


def _c4_problems(got: dict) -> tuple[list[str], list[str]]:
    """(fails, warns) of the settle statuses: P2/N2 churning or moving nothing FAIL; a short record or any unsettled N1 (the reference) is indeterminate."""
    fails, warns = [], []
    for arm, runs in got.items():
        for status in (x[0] for x in runs if x[0] != "settled"):
            (fails if arm != "N1" and status in ("churn", "none") else warns).append(f"{arm} {status}")
    return fails, warns


def _b_tail(r: dict) -> float | None:
    return st.mean([q["decode_tps"] for q in r["phases"].get("B", [])[-3:] if q.get("decode_tps")])


def _post_shift(cfg, res) -> tuple[dict, list[str]]:
    """(arms whose post-shift tok/s fell below the reference, comparisons that could not be made): a missing side is not a pass."""
    below, missing = {}, []
    for arm, ref in (("P2", "P0"), ("N1", "N0"), ("N2", "N0")):
        got, base = st.mean(_series(res, "W2", arm, _b_tail)), st.mean(_series(res, "W2", ref, _b_tail))
        if got is None or not base:
            missing.append(f"{arm} vs {ref}")
        elif got < base * (1 - cfg["thresholds"]["c4_post_margin_pct"] / 100):
            below[arm] = round(100 * (got / base - 1), 1)
    return below, missing


def _legacy_share(res) -> float | None:
    """Plateau helper_share of the legacy publisher (N1) in the B phase.  The legacy round line prints none, so None until it does."""
    xs = [st.mean(st.column(r["rounds"], "helper_share")[-3:]) for r in _runs(res, "W2", "N1")]
    return _med([x for x in xs if x is not None])


def _c4_reference(t: dict, got: dict) -> tuple[dict, float | None]:
    """(last-commit round per arm and start, limit for P2/N2).  The limit is None, never inf, when N1 (the reference) did not settle in every start."""
    per = {k: [x[1] for x in v] for k, v in got.items()}
    if any(x[0] != "settled" for x in got["N1"]):
        return per, None
    return per, t["c4_ratio"] * _med(per["N1"]) + t["c4_extra_rounds"]


def _c4_gaps(lim: float | None, missing: list[str], share: float | None) -> list[str]:
    """Reasons C4 cannot be decided: no reference, a post-shift comparison without data, no legacy-publisher share."""
    gaps = ["no reference: N1 did not settle in every start"] if lim is None else []
    gaps += [f"post-shift comparison missing: {m}" for m in missing]
    return gaps + ([] if share is not None else ["legacy-publisher helper_share not measured (the legacy round line prints none)"])


def _c4_detail(t: dict, got: dict, per: dict, lim: float | None, found: dict) -> str:
    shares = {k: [None if x[2] is None else round(x[2], 2) for x in v] for k, v in got.items()}
    limit = "n/a" if lim is None else f"{lim:.1f}"
    return (f"last-commit round per start {per} (settled = quiet in the last {t['c4_tail_rounds']} of {t['c4_horizon_rounds']} rounds), stable share {shares}; limit {limit}; "
            + "; ".join(f"{k} {v or 'none'}" for k, v in found.items()))


def c4(cfg, res) -> dict:
    t = cfg["thresholds"]
    # WHY commits: the legacy ladder (N1) prints no helper_share, so the cross-arm comparison uses the one measure all three print
    got = {k: [_settle(r, t) for r in _runs(res, "W2", k)] for k in ("P2", "N1", "N2")}
    if not all(got.values()):
        return _v("C4", "Convergence", "WARN", f"missing W2 starts: { {k: len(v) for k, v in got.items()} }")
    fails, warns = _c4_problems(got)
    per, lim = _c4_reference(t, got)
    slow = {} if lim is None else _slow(per, lim)
    post, missing = _post_shift(cfg, res)
    share = _legacy_share(res)
    warns += _c4_gaps(lim, missing, share)
    status = "FAIL" if fails or slow or post else ("WARN" if warns else "PASS")
    found = {"slow": slow, "not settled": fails, "post-shift tok/s below reference": post, "legacy share": share, "indeterminate": warns}
    return _v("C4", "Convergence", status, _c4_detail(t, got, per, lim, found))


def _lat(res, key: str, field: str) -> dict:
    rounds = [rd for r in res if r["workload"] in ("W1", "W2", "W3") and r["key"] == key for rd in r.get("rounds", [])]
    return st.latency_table(rounds, (field,))[field]


def c5(cfg, res) -> dict:
    t, notes, fail = cfg["thresholds"], [], False
    p95 = {k: _lat(res, k, "fence_us") for k in ("P2", "N1", "N2")}
    for k in ("P2", "N2"):
        if p95[k]["n"] == 0:
            return _v("C5", "Swap latency", "WARN", f"no fence_us round lines for {k}")
        ms95, msmax = p95[k]["p95"] / 1000, p95[k]["max"] / 1000
        fail |= ms95 > t["fence_p95_ms"] or msmax > t["fence_max_ms"]
        notes.append(f"{k} fence p95 {ms95:.1f} ms (budget {t['fence_p95_ms']} ms PROVISIONAL) max {msmax:.1f} ms (limit {t['fence_max_ms']})")
    if p95["N1"]["n"] and p95["N2"]["p95"] > t["n2_vs_n1_fence_ratio"] * p95["N1"]["p95"]:
        fail = True
        notes.append("N2 fence p95 above 1.2 x N1")
    plan = _lat(res, "P2", "plan_us")
    if plan["n"] and plan["p95"] > t["plan_p95_ms"] * 1000:
        notes.append(f"plan p95 {plan['p95'] / 1000:.2f} ms > {t['plan_p95_ms']} ms (planner-thread change, not a fail)")
    return _v("C5", "Swap latency", "FAIL" if fail else "PASS", "; ".join(notes))


def _p8k_ttft(r: dict) -> float | None:
    q = r["phases"].get("p8k", [{}])[0]
    return q.get("ttft_ms")


def _os_read(r: dict) -> float | None:
    xs = r.get("io", {}).get("os_read_mb", [])
    return sum(xs) if xs else None


def _upper_ok(arm: list[float], ref: list[float], pct: float) -> tuple[bool | None, str]:
    b = st.welch_bounds(arm, ref)
    if b is None:
        return None, f"need >= 2 starts each (have {len(arm)}/{len(ref)})"
    return b[1] <= pct, f"diff {b[2]:+.1f}%, upper 95% bound {b[1]:+.1f}% (limit +{pct}%)"


def _pair_verdict(cid: str, title: str, checks: list[tuple[str, tuple[bool | None, str]]]) -> dict:
    parts = [f"{n}: {d}" for n, (_, d) in checks]
    oks = [ok for n, (ok, _) in checks]
    status = "WARN" if None in oks else ("PASS" if all(oks) else "FAIL")
    return _v(cid, title, status, "; ".join(parts))


def c6(cfg, res) -> dict:
    p = cfg["thresholds"]["c6_pct"]
    g = lambda wl, k, fn: _series(res, wl, k, fn)  # noqa: E731
    return _pair_verdict("C6", "Reads", [
        ("N2 vs N1 TTFT", _upper_ok(g("W3", "N2/off", _p8k_ttft), g("W3", "N1/off", _p8k_ttft), p)),
        ("N2 vs N1 OS-read MB", _upper_ok(g("W3", "N2/off", _os_read), g("W3", "N1/off", _os_read), p)),
        ("P2 vs P0 OS-read MB", _upper_ok(g("W3", "P2/off", _os_read), g("W3", "P0/off", _os_read), p))])


def c7(cfg, res) -> dict:
    d4 = {k: [x for r in _runs(res, "W3", k) for x in r.get("io", {}).get("d4_chunks", [])] for k in ("N1/on", "N2/on")}
    engaged = {k: bool(v) and max(v) > 0 for k, v in d4.items()}
    ttft = _upper_ok(_series(res, "W3", "N2/on", _p8k_ttft), _series(res, "W3", "N1/on", _p8k_ttft), cfg["thresholds"]["c7_pct"])
    return _pair_verdict("C7", "D4", [("D4 engages N1", (engaged["N1/on"] or (None if not _runs(res, "W3", "N1/on") else False), f"COMPUTE chunks seen {d4['N1/on'][:3]}")),
                                      ("D4 engages N2", (engaged["N2/on"] or (None if not _runs(res, "W3", "N2/on") else False), f"COMPUTE chunks seen {d4['N2/on'][:3]}")),
                                      ("N2 vs N1 TTFT (D4 on)", ttft)])


_KL_RC = {0: "PASS", 1: "FAIL", 2: "WARN"}  # kl.py gate mode: exit 0 PASS, 1 FAIL, 2 INVALID (tools/kl.py)
_RANK = {"PASS": 0, "WARN": 1, "FAIL": 2}


def _kl_status(k: dict) -> str:
    """One kl record to a status.  An exit code outside the documented three (a signal such as -9, 137, None for a run that never finished) is a checker failure: FAIL."""
    rc = k.get("rc")
    status = _KL_RC.get(rc) if type(rc) is int else None
    if status is None:
        return "FAIL"
    return "WARN" if status == "PASS" and "KL-GATE PASS" not in str(k.get("line", "")) else status  # exit 0 must come with the gate's own PASS line


def _kl_start(kl: list[dict], idx: int) -> tuple[str, str]:
    """(status, note) of one required P2 start: the worst of its pooled records; no pooled record is not decidable."""
    recs = [k for k in kl if k.get("text") == "pooled" and str(k.get("pair", "")).startswith(f"P2-s{idx}-vs-")]
    if not recs:
        return "WARN", f"s{idx}: no pooled comparison record"
    worst = max((_kl_status(k) for k in recs), key=_RANK.get)
    return worst, f"s{idx}: rc {[k.get('rc') for k in recs]} {worst}"


def _worst(statuses: list[str], too_few: bool) -> str:
    """The worst status; fewer starts than configured is at best WARN (a missing start is not a pass)."""
    return max(statuses + (["WARN"] if too_few else []), key=_RANK.get)


def _per_text(kl: list[dict]) -> str:
    per = [f"{k['pair']} {k['text']}: rc {k['rc']}" for k in kl if k.get("text") != "pooled"]
    return f"; per-text (diagnostic): {per}" if per else ""


C8_SCOPE = "PREFILL ONLY (the helper tier does not run in prefill, so this does not cover it; the decode path is C10): "


def c8(cfg, res, kl: list[dict]) -> dict:
    """The criterion is the POOLED gate (KL and top-1 over all texts of a start, DESIGN.md 5.4), required for EVERY W4 P2 start; per-text gates are shown, never decide."""
    starts = sorted({r["idx"] for r in res if r["workload"] == "W4" and r["arm"] == "P2"})
    want = cfg["workloads"].get("W4", {}).get("n", len(starts))
    if not kl or not starts:
        return _v("C8", "Quality (KL)", "WARN", C8_SCOPE + f"no kl.py comparisons ran or no W4 P2 start to compare (comparisons {len(kl)}, starts {len(starts)})")
    got = [_kl_start(kl, i) for i in starts]
    status = _worst([g[0] for g in got], len(starts) < want)
    th, pooled = cfg["thresholds"], [k for k in kl if k.get("text") == "pooled"]
    return _v("C8", "Quality (KL)", status, C8_SCOPE + f"required starts {len(starts)} of {want}: {'; '.join(g[1] for g in got)} (max KL {th['kl_max']}, top-1 >= {th['top1_min']}): "
              + " | ".join(str(k["line"]) for k in pooled)[:600] + _per_text(kl))


def _fault_ok(f: dict, lim: float, rec_lim: float) -> bool:
    after, rec = f.get("fail_after_s"), f.get("recovery_s")
    return after is not None and after < lim and bool(f.get("quarantined")) and bool(f.get("next_served")) and f.get("zombies") == 0 and rec is not None and rec < rec_lim


def c9(cfg, res) -> dict:
    runs = [r for r in res if r["workload"] == "W5" and "fault" in r]
    if not runs or all(r["fault"].get("dry") for r in runs):
        return _v("C9", "Helper-loss recovery", "WARN", "no live W5 starts (dry run or not selected)")
    lim = cfg["thresholds"]["fault_fail_s"]
    good = [r for r in runs if _fault_ok(r["fault"], lim, cfg["thresholds"]["fault_recover_s"])]
    ok = len(good) == len(runs) == cfg["workloads"]["W5"]["n"]
    return _v("C9", "Helper-loss recovery", "PASS" if ok else "FAIL", f"{len(good)} of {len(runs)} starts met all of: request fails < {lim} s, rung 1 quarantined, next request served within {cfg['thresholds']['fault_recover_s']} s of the kill, no zombie")


def judge(cfg: dict, results: list[dict], kl: list[dict]) -> list[dict]:
    return [w0(cfg, results), *c1(cfg, results), *c2(cfg, results), c3(cfg, results), c4(cfg, results), c5(cfg, results),
            c6(cfg, results), c7(cfg, results), c8(cfg, results, kl), c9(cfg, results)]
