# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Synthetic inputs for the overflow tests: a planted procfs/sysfs/cgroup tree, engine log text, and a start record that passes the gate.

OWNER: this file owns the test-side copy of the formats the overflow harness reads (ov_proc.py and ov_parse.py own the readers; the
per-request engine lines are payoff_synth's, imported, not copied).  Nothing here starts a process or touches a real /proc.
Formats: procfs diskstats and io, cgroup v2 files (memory.max, memory.swap.max, memory.stat, memory.events, memory.peak, cgroup.procs),
Strata's ``FileExpertSource: RAM budget`` and ``the file tier reads`` startup lines (source named in ov_parse.py).
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.normpath(os.path.join(HERE, "..", "..", "scripts", "overflow"))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "payoff")))
sys.path.insert(0, SCRIPTS)
import ov_cfg  # noqa: E402
import ov_paths  # noqa: E402,F401
import ov_proc  # noqa: E402
import payoff_synth as P  # noqa: E402

CG = "/user.slice/user-1000.slice/user@1000.service/app.slice/run-p1-i1.scope"
HCG = "/user.slice/user-1000.slice/user@1000.service/app.slice/run-p2-i2.scope"
ENGINE_PID, HELPER_PID, LAUNCHER_PID = 100, 200, 99
DISK, PARENT = "nvme0n1p2", "nvme0n1"
CHANNELS = ["/seabraid-ovf1", "/seabraid-ovf2"]  # the shipped arms.json shm and shm2 vars


def load_cfg(**over) -> dict:
    """The shipped arms.json, dry (placeholders for the machine paths), with ``over`` var overrides."""
    return ov_cfg.load(os.path.join(SCRIPTS, "arms.json"), {k: str(v) for k, v in over.items()}, True)


def _w(path, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


class Tree:
    """A planted /proc, /sys and cgroup root.  ``host()`` reads it with the real ov_proc code; the ``bump_*`` methods are the engine doing I/O."""

    def __init__(self, root, plan=None, rotational: int = 0, engine_limit=None, swap: int = 0):
        self.root = str(root)
        self.proc, self.sysfs, self.cgroot = (os.path.join(self.root, d) for d in ("proc", "sys", "cg"))
        self.sectors, self.proc_bytes, self.helper_bytes = 1000, 0, 0
        self.plan = plan
        _w(f"{self.sysfs}/block/{PARENT}/queue/rotational", f"{rotational}\n")
        os.makedirs(f"{self.sysfs}/block/{PARENT}/{DISK}", exist_ok=True)
        limit = plan.engine_cap_bytes if engine_limit is None and plan and plan.capped else engine_limit
        for pid, comm, cg in ((LAUNCHER_PID, "bash", CG), (ENGINE_PID, "strata", CG), (HELPER_PID, "helper-cuda", HCG)):
            _w(f"{self.proc}/{pid}/comm", comm + "\n")
            _w(f"{self.proc}/{pid}/cgroup", f"0::{cg}\n")
        self.set_limit(CG, "max" if limit is None else limit, swap)
        self.set_limit(HCG, plan.helper_cap_bytes if plan and plan.capped else "max", 0)
        self.set_events(CG, 0, 0)
        self.set_events(HCG, 0, 0)
        self._flush()

    def set_limit(self, cg: str, limit, swap) -> None:
        _w(f"{self.cgroot}{cg}/memory.max", f"{limit}\n")
        _w(f"{self.cgroot}{cg}/memory.swap.max", f"{swap}\n")
        _w(f"{self.cgroot}{cg}/memory.peak", f"{9 << 30}\n")
        _w(f"{self.cgroot}{cg}/cgroup.procs", "99\n100\n" if cg == CG else "200\n")

    def set_events(self, cg: str, oom: int, hits: int, majflt: int = 0) -> None:
        _w(f"{self.cgroot}{cg}/memory.events", f"low 0\nhigh 0\nmax {hits}\noom 0\noom_kill {oom}\n")
        _w(f"{self.cgroot}{cg}/memory.stat", f"anon 1\nfile 2\npgmajfault {majflt}\n")

    def _flush(self) -> None:
        _w(f"{self.proc}/diskstats", f" 259 2 {DISK} 1 0 {self.sectors} 0 0 0 0 0 0 0 0\n 8 1 sda1 1 0 5 0 0 0 0 0 0 0 0\n")
        _w(f"{self.proc}/{ENGINE_PID}/io", f"rchar: 1\nread_bytes: {self.proc_bytes}\nwrite_bytes: 0\n")
        _w(f"{self.proc}/{HELPER_PID}/io", f"read_bytes: {self.helper_bytes}\n")
        _w(f"{self.proc}/{LAUNCHER_PID}/io", "read_bytes: 0\n")

    def bump(self, disk_bytes: int, proc_bytes: int) -> None:
        """The disk transferred ``disk_bytes`` (machine-wide) and the engine caused ``proc_bytes`` of storage reads."""
        self.sectors += disk_bytes // 512
        self.proc_bytes += proc_bytes
        self._flush()

    def host(self) -> ov_proc.Host:
        return ov_proc.Host(self.proc, self.sysfs, self.cgroot)

    def pids(self, with_helper: bool = True) -> list[int]:
        return [LAUNCHER_PID, ENGINE_PID] + ([HELPER_PID] if with_helper else [])


def budget_text(plan, slots: int | None = None) -> str:
    """Engine startup lines of a capped arm, consistent with ``plan``: the slot line, the RAM budget line, the file tier mode and the helper channel."""
    slots = plan.main_slots if slots is None else slots
    blob = 47.0 / 24576
    held = int(plan.ram_gib / blob)
    out = f"strata generate: expert cache {slots} slots, {slots * blob:.2f} GiB of VRAM; policy is\n"
    for ch in CHANNELS[: plan.helpers]:
        out += f"seabraid: helper NVIDIA GeForce RTX 3090 on channel {ch}, 22.10 GiB free\n"
    out += (f"FileExpertSource: RAM budget {plan.ram_gib:.2f} GiB: {held} of the {plan.complement_gib:.2f} GiB of experts the GPU cache does not hold, "
            "by profile rank; the rest are read from the files\n")
    return out + "strata generate: the file tier reads through the file cache (the free RAM holds a cache a twentieth of the file experts)\n"


def decode_log(file_mb_cum: float, os_read_mb: float, decode_tps: float = 8.0) -> str:
    """One decode request's log block with the file tier counters (cumulative MB read as the engine prints it)."""
    text = P.timing_line() + P.prompt_line(12, 0, 600.0, 200, decode_tps) + P.hit_line() + P.tiers_line(3000, int(file_mb_cum / 2), file_mb_cum)
    return text + (f"strata serve: file tier I/O this request: the OS read {os_read_mb:.1f} MB from storage (30 major faults) for 400.0 MB of expert reads; "
                   "io prefetch: 5 read ahead (1.0 MB by pread), 4 used, 1 unused, 0 skipped (cached), 0 dropped; decode waited 90.0 ms in 12 fetches\n")


def prefill_log(file_mb_cum: float, prefill_tps: float = 120.0) -> str:
    return P.prompt_line(3500, 5, prefill_tps, 1, 8.0) + P.hit_line() + P.tiers_line(3000, int(file_mb_cum / 2), file_mb_cum)


def request(phase: str, k: int, counted: bool, file_mb=400.0, dev=450.0, proc=440.0) -> dict:
    return {"phase": phase, "k": k, "counted": counted, "ok": True, "usable": counted, "problems": [], "file_mb": file_mb, "dev_read_mb": dev,
            "proc_read_mb": proc, "os_read_mb": 430.0, "io_wait_ms": 90.0, "majflt": 30, "decode_tps": 8.0, "prefill_tps": 120.0}


def good_start(cfg: dict, arm: str = "O2") -> dict:
    """A speed start that passes every check of the data gate for ``arm``; the sabotage tests edit one thing."""
    plan = cfg["plans"][arm]
    if not plan.capped:
        return reference_start(arm)
    blob = 47.0 / 24576
    reqs = [request(ph, k, k >= 1) for k in range(4) for ph in ("decode", "prefill")]
    startup = {"cap_slots": plan.main_slots, "cap_gib": plan.main_slots * blob, "helper_up": bool(plan.helpers), "channels": tuple(CHANNELS[: plan.helpers]), "resident_gib": None, "ram_budget_gib": plan.ram_gib,
               "ram_held_blobs": int(plan.ram_gib / blob), "complement_gib": plan.complement_gib, "file_mode": "through the file cache"}
    moves = ladder_moves(plan.helpers) if plan.ladder else []
    rows = [{"pid": ENGINE_PID, "comm": "strata", "cgroup": CG, "memory_max": plan.engine_cap_bytes, "swap_max": 0},
            {"pid": LAUNCHER_PID, "comm": "bash", "cgroup": CG, "memory_max": plan.engine_cap_bytes, "swap_max": 0}]
    if plan.helpers:
        rows.append({"pid": HELPER_PID, "comm": "helper-cuda", "cgroup": HCG, "memory_max": plan.helper_cap_bytes, "swap_max": 0})
    return {"workload": "speed", "cap": plan.card_blobs, "mode": "speed", "arm": arm, "idx": 0, "tag": f"speed-{arm}-s0", "startup": startup, "requests": reqs,
            "kl_files": {}, "aborted": False, "error": None, "moves": moves, "stale": 0, "cgroup": rows, "preflight": {"resident_mb": 20.0, "max_resident_mb": 512.0, "errors": [], "ok": True, "files": 3},
            "disk": {"name": DISK, "parent": PARENT, "rotational": False}, "memory": {"oom_kill": 0, "max_hits": 4, "peak": 14 << 30, "dev_mb": 3000.0, "proc_mb": 2900.0, "majflt": 100}}


def ladder_moves(k: int) -> list[dict]:
    """Committed hops of a k-helper pyramid as the adapter logs them: fallback -> top helper, then each helper one rung up (rung 0 = main)."""
    hops = [(k + 1, k, "fallback")] + [(r + 1, r, "card") for r in range(k - 1, -1, -1)]
    return [{"layer": 3, "expert": 10 + i, "lower": lo, "upper": up, "src": src, "bytes": 2005000} for i, (lo, up, src) in enumerate(hops)]


def move_lines(k: int) -> str:
    """The same hops as engine log text (placement_round_log.cpp format)."""
    return "".join(f"placement move {m['layer']}/{m['expert']} {m['lower']}->{m['upper']} src={m['src']} bytes={m['bytes']} verified=1\n" for m in ladder_moves(k))


def reference_start(arm: str) -> dict:
    """The uncapped reference: all experts on the cards, nothing read from the files, no cgroup, no preflight."""
    startup = {"cap_slots": 12000, "cap_gib": 22.9, "helper_up": True, "channels": (CHANNELS[0],), "resident_gib": None, "ram_budget_gib": None, "ram_held_blobs": None,
               "complement_gib": None, "file_mode": None}
    reqs = [request(ph, k, k >= 1, file_mb=0.0, dev=0.0, proc=0.0) for k in range(4) for ph in ("decode", "prefill")]
    return {"workload": "speed", "cap": 0, "mode": "speed", "arm": arm, "idx": 0, "tag": f"speed-{arm}-s0", "startup": startup, "requests": reqs, "kl_files": {},
            "aborted": False, "error": None, "moves": [], "stale": 0, "cgroup": [], "preflight": None, "disk": {"name": DISK, "parent": PARENT, "rotational": False}, "memory": {}}


def write_forced(rundir: str, seed: int, noise: float = 0.0) -> None:
    """The files force_check ``score`` writes (forced.tsv + .meta.json), so qual_decode.pool runs on them for real."""
    os.makedirs(rundir, exist_ok=True)
    tsv = os.path.join(rundir, "forced.tsv")
    P.write_logpos(tsv, seed=seed, noise=noise)
    with open(tsv + ".meta.json", "w", encoding="utf-8") as fh:
        json.dump({"items": [{"first_line": 0, "end_line": 60}]}, fh)
