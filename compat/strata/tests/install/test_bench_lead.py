# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""_bench_lead: the whole-system lead-job probe, against a fake runner and the engine's real timing-line format."""
from __future__ import annotations

import socket
import sys
from dataclasses import dataclass
from types import SimpleNamespace

from install import _bench_lead as BL
from install._bench_lead_run import ProcessLeadRunner
from seabraid.ranking import rank

# Real engine line (a 2026-11-07 ab1 engine log); __MS__ is substituted per scenario.
REAL = ("strata decode timing: 715 windows, avg T 4.30, 2.80 tokens/window, __MS__ ms/window = adapt wait 0.00 + verify 23.89 "
        "(GPU-reach wait 18.42 + per-layer host 3.88 [plan 0.25 actq 0.00 jobs 0.06 CPU 3.23] + stage 0.23) + commit/emit 0.03 + "
        "draft 2.29 + adapt join 0.00 + other 0.05; per layer-window: CPU experts 0.00 (0.00 entries), VRAM hits 37.22, PCIe 0.00; "
        "0.0 swaps/window")
EARLY = REAL.replace("715 windows", "1 windows")


def line(ms: float) -> str:
    return REAL.replace("__MS__", f"{ms:.2f}")


@dataclass(frozen=True)
class Cand:
    vendor: str
    index: int
    uuid: object
    name: str


R9700, RTX = Cand("amd", 0, None, "R9700"), Cand("nvidia", 0, "GPU-abc", "RTX 3090")


def placement(card):
    other = RTX if card.vendor == "amd" else R9700
    # a stale mask from whatever the config was written for: the probe must override it with the candidate
    stale = {"HIP_VISIBLE_DEVICES": "9", "CUDA_VISIBLE_DEVICES": "stale"}
    return BL.Placement("/w/cfg.json", {**stale, "STRATA_X": "1"},
                        (BL.Other(other.vendor, other.index, other.uuid, ("--device", "5", "--pack", "/p"), {}),), "/w/engine.log")


PLAN = BL.LeadPlan("/usr/bin/python3", "/w/strata/serve/server.py", 8097, {"amd": "/b/amd/h", "nvidia": "/b/nv/h"}, {"amd": "/rocm/lib"}, placement)


class Fake:
    """Replays a scripted ProbeRun (or raises) per card that was FIRST in the server env; records every Probe."""

    def __init__(self, by_first):
        self.by_first, self.probes = by_first, []

    def probe(self, p):
        self.probes.append(p)
        first = "nvidia" if p.server.env["CUDA_VISIBLE_DEVICES"] == "GPU-abc" else "amd"
        r = self.by_first[first]
        if isinstance(r, Exception):
            raise r
        return r


def runs(ms1, ms4):
    return BL.ProbeRun(True, "", {"warm": EARLY, "lead1": line(ms1), "lead4": line(ms4)})


def test_picks_the_faster_window_per_class_and_the_ranking_consumes_it():
    # 1 stream: 3090 first 24.0 vs R9700 first 26.7.  4 streams: R9700 first wins on aggregate (lower ms here for the same reason).
    res = BL.measure_lead([RTX, R9700], PLAN, [1, 4], Fake({"nvidia": runs(24.0, 26.0), "amd": runs(26.7, 24.0)}))
    assert res[("nvidia", 0)].ms == {1: 24.0, 4: 26.0} and res[("amd", 0)].ms == {1: 26.7, 4: 24.0}
    lead, why = BL.to_lead_ms(res, [RTX, R9700])
    assert why == "" and set(lead) == {1, 4}
    cards = [SimpleNamespace(key=("amd", 0), name="R9700", vendor="amd", arch="gfx1201", free=32 << 30, total=32 << 30, pcie=64),
             SimpleNamespace(key=("nvidia", 0), name="RTX 3090", vendor="nvidia", arch="sm_86", free=24 << 30, total=24 << 30, pcie=64)]
    one, how1 = rank(cards, lead[1], None, lambda c: True, 1)
    four, how4 = rank(cards, lead[4], None, lambda c: True, 4)
    assert (one[0].name, how1.card1) == ("RTX 3090", "lead") and (four[0].name, how4.card1) == ("R9700", "lead")


def test_a_missing_timing_line_marks_the_class_unmeasured_with_a_reason():
    res = BL.measure_lead([RTX, R9700], PLAN, [1], Fake({"nvidia": BL.ProbeRun(True, "", {"lead1": "engine up\nno timing here"}),
                                                          "amd": runs(26.7, 0)}))
    why1 = res[("nvidia", 0)].why[1]
    assert res[("nvidia", 0)].ms == {} and "RTX 3090 first, 1 stream" in why1 and "no `strata decode timing:`" in why1
    lead, why = BL.to_lead_ms(res, [RTX, R9700])
    assert lead == {} and "RTX 3090 first" in why          # a partial class is never offered to the ranking


def test_the_last_timing_line_of_a_round_is_the_one_read():
    text = chr(10).join([EARLY.replace("__MS__", "356.12"), line(31.5), "log noise", line(24.02)])
    assert BL.parse_timing(text) == (24.02, "")


def test_a_start_line_over_too_few_windows_is_refused():
    ms, why = BL.parse_timing(EARLY.replace("__MS__", "356.12"))
    assert ms is None and "only 1 decode windows" in why


def test_a_crashing_or_failed_start_marks_unmeasured_and_the_other_card_is_still_measured():
    res = BL.measure_lead([RTX, R9700], PLAN, [1], Fake({"nvidia": RuntimeError("popen exploded"),
                                                          "amd": BL.ProbeRun(False, "engine exited (see x.log)")}))
    assert "crashed: RuntimeError: popen exploded" in res[("nvidia", 0)].why[1]
    assert "did not start" in res[("amd", 0)].why[1] and "engine exited" in res[("amd", 0)].why[1]
    assert BL.to_lead_ms(res, [RTX, R9700])[0] == {}


def test_the_start_command_really_puts_the_candidate_first_and_the_other_card_behind_it():
    f = Fake({"nvidia": runs(24.0, 26.0), "amd": runs(26.7, 24.0)})
    BL.measure_lead([RTX, R9700], PLAN, [1, 4], f)
    nv, amd = f.probes
    assert nv.server.env["CUDA_VISIBLE_DEVICES"] == "GPU-abc" and amd.server.env["HIP_VISIBLE_DEVICES"] == "0"
    assert nv.others[0].env["HIP_VISIBLE_DEVICES"] == "0" and amd.others[0].env["CUDA_VISIBLE_DEVICES"] == "GPU-abc"
    assert nv.others[0].argv == ("/b/amd/h", "--pack", "/p", "--shm", "/seabraid-lead0", "--device", "0")   # planned --device dropped
    assert nv.others[0].env["LD_LIBRARY_PATH"] == "/rocm/lib"
    assert nv.server.env["SEABRAID_STRATA_SHM"] == "/seabraid-lead0" and nv.server.env["STRATA_DECODE_TIMING"] == "1" and nv.shms == ("/seabraid-lead0",)
    assert nv.server.argv[-2:] == ("--parallel", "4") and [r.label for r in nv.rounds] == ["warm", "lead1", "lead4"]
    assert nv.server.argv[nv.server.argv.index("--config") + 1] == "/w/cfg.json"


def test_one_stream_only_starts_at_parallel_1_and_the_cap_bounds_the_install_time():
    f = Fake({"nvidia": runs(24.0, 0), "amd": runs(26.7, 0)})
    many = [RTX, R9700, Cand("amd", 1, None, "C"), Cand("nvidia", 1, "GPU-d", "D")]
    res = BL.measure_lead(many, PLAN, [1], f)
    assert f.probes[0].server.argv[-2:] == ("--parallel", "1") and len(f.probes) == BL.MAX_CANDIDATES == len(res)
    lead, why = BL.to_lead_ms(res, many)
    assert lead == {} and "not probed" in why and "D" in why     # a card past the cap leaves the class incomplete -> heuristic


def test_no_launch_for_a_card_is_unmeasured_not_a_crash():
    def refusing(card):
        raise BL.PlacementError("the plan with this card first is not usable")
    res = BL.measure_lead([RTX], BL.LeadPlan("p", "s", 1, {}, {}, refusing), [1], Fake({}))
    assert "no launch could be built with RTX 3090 first" in res[("nvidia", 0)].why[1]


SERVER = r"""
import http.server, sys
port, log, line = int(sys.argv[1]), sys.argv[2], sys.argv[3]
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        open(log, "a").write(line + "\n")
        self.send_response(200); self.end_headers(); self.wfile.write(b"{}")
    def log_message(self, *a): pass
http.server.ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
"""


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_the_real_runner_starts_loads_reads_the_log_per_round_and_stops_the_child(tmp_path):
    port, log = _free_port(), str(tmp_path / "engine.log")
    cmd = BL.Cmd((sys.executable, "-I", "-c", SERVER, str(port), log, line(24.0)), label="fake engine")
    rounds = (BL.Round("warm", 1, 5), BL.Round("lead1", 1, 5))
    kept = []
    runner = ProcessLeadRunner(sleep=lambda s: None, settle_s=0, echo=lambda s: None)
    orig = runner.popen
    runner.popen = lambda *a, **k: kept.append(orig(*a, **k)) or kept[-1]
    run = runner.probe(BL.Probe((), (), cmd, rounds, log, port, ready_s=60))
    assert run.started and BL.parse_timing(run.texts["lead1"]) == (24.0, "") and kept[0].poll() is not None   # child stopped
    dead = runner.probe(BL.Probe((), (), BL.Cmd((sys.executable, "-I", "-c", "raise SystemExit(3)")), rounds, log, port, ready_s=60))
    assert not dead.started and "exited" in dead.why
    missing = runner.probe(BL.Probe((), (), BL.Cmd((str(tmp_path / "nope"),)), rounds, log, port, ready_s=60))
    assert not missing.started and "could not start" in missing.why
