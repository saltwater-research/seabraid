# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""One engine process per start: launch it as its own process group, wait until it serves, stop that exact group.

OWNER: this module owns engine lifetime for the qualification runner.  CONTRACT: ``Engine(cfg, run, rundir, dry)``;
``start()``, ``wait_ready()``, ``stop() -> leftover count``, ``helper_pid()``, ``zombies()``, ``log_text()``.
SAFETY: the engine is started with ``start_new_session`` so its pgid is its own pid; stop signals ONLY that pgid after
checking it is still the child's own group.  No service manager, no GPU reset, no shell string: argv lists only.
READY means "answers a one-token chat request" (the same post_json the workloads use), not a health URL, so there is one
HTTP client in the runner and it is the shipped loopback one that never touches a proxy.
FAILURE: a start that dies or times out raises EngineError with the log tail (the caller records a FAIL step and moves
on to the next run).  In dry mode nothing is launched and the canned log (if any) stands in for the engine log.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
from _loopback import post_json  # noqa: E402

import qual_cfg  # noqa: E402

PROBE = {"model": "x", "messages": [{"role": "user", "content": "ok"}], "max_tokens": 1, "temperature": 0}
UNSET = ("SEABRAID_QWEN_HELPER", "SEABRAID_STRATA_SHM")


class EngineError(Exception):
    """The engine did not start, died, or would not stop."""


def build_command(cfg: dict, run: qual_cfg.Run, rundir: str) -> tuple[list[str], dict[str, str]]:
    """(argv, env) for this run: protocol flags first, then the arm's, the policy, and the workload's extras."""
    arm, wl = cfg["arms"][run.arm], cfg["workloads"][run.workload]
    vars_ = {**cfg["vars"], "rundir": rundir}
    env = {k: v for k, v in os.environ.items() if k not in UNSET}
    for layer in (cfg["engine"].get("env", {}), arm.get("env", {}), cfg["variants"][run.variant].get("env", {}), wl.get("env", {})):
        env.update(qual_cfg.subst(layer, vars_))
    argv = cfg["engine"]["argv"] + arm.get("args", []) + qual_cfg.policy_args(cfg, arm) + cfg["variants"][run.variant].get("args", []) + wl.get("extra_args", [])
    return qual_cfg.subst(argv, vars_), env


def build_helper(cfg: dict, run: qual_cfg.Run, rundir: str) -> tuple[list[str], dict[str, str], str] | None:
    """(argv, env, ready path) of the named-channel helper process this arm needs, or None (private helper: the engine spawns it)."""
    spec = cfg["arms"][run.arm].get("helper_start")
    if not spec:
        return None
    spec = cfg["helpers"][spec] if isinstance(spec, str) else spec
    vars_ = {**cfg["vars"], "rundir": rundir}
    env = {k: v for k, v in os.environ.items() if k not in UNSET}
    env.update(qual_cfg.subst(spec.get("env", {}), vars_))
    return qual_cfg.subst(spec["argv"], vars_), env, qual_cfg.subst(spec["ready_path"], vars_)


class Engine:
    def __init__(self, cfg: dict, run: qual_cfg.Run, rundir: str, url: str, dry: bool):
        self.cfg, self.run, self.rundir, self.url, self.dry = cfg, run, rundir, url, dry
        self.proc: subprocess.Popen | None = None
        self.hproc: subprocess.Popen | None = None
        # WHY two names: the launcher's own stdout (server banner) and the engine's log (cfg.log: round lines, I/O lines) are different files
        self.log_path = os.path.join(rundir, cfg["engine"].get("log_name", "engine.log"))
        self.out_path = os.path.join(rundir, cfg["engine"].get("stdout_name", "engine.log"))
        self.argv, self.env = build_command(cfg, run, rundir)
        self.helper = build_helper(cfg, run, rundir)
        self.tag = f"{os.getpid()}.{run.tag}"
        self.env["QUAL_RUN_TAG"] = self.tag
        if self.helper:
            self.helper[1]["QUAL_RUN_TAG"] = self.tag

    def start(self) -> str:
        os.makedirs(self.rundir, exist_ok=True)
        text = "DRY-RUN would start: " + " ".join(self.argv)
        if self.dry:
            canned = self.cfg.get("dry_log")
            if canned:
                shutil.copyfile(os.path.join(self.cfg["_dir"], canned), self.log_path)
            return text
        note = self._start_helper()
        with open(self.out_path, "wb") as log:
            self.proc = subprocess.Popen(self.argv, env=self.env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        return f"{note}started engine pid/pgid {self.proc.pid}: " + " ".join(self.argv)

    def _start_helper(self) -> str:
        """Named channel: start the helper process (its own pgid) and wait for the channel file; the engine finds it by SEABRAID_STRATA_SHM."""
        if not self.helper:
            return ""
        argv, env, ready = self.helper
        hlog = os.path.join(self.rundir, "helper.log")
        with open(hlog, "wb") as log:
            self.hproc = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        t0 = time.time()
        while not os.path.exists(ready):
            if self.hproc.poll() is not None or time.time() - t0 > 300:
                raise EngineError(f"helper did not open its channel {ready} (rc {self.hproc.poll()}): {_read(hlog)[-400:]}")
            time.sleep(1)
        return f"helper pid/pgid {self.hproc.pid} ready after {time.time() - t0:.0f} s; "

    def wait_ready(self) -> str:
        """Ready = the server printed its ready line, then one one-token request is answered.

        WHY poll the launcher output: serve/server.py restarts a crashed engine by itself, so a blocking probe never errors
        (a crash-looping engine held the first live probe for the whole ready_s); the crash text is checked every 2 s."""
        if self.dry:
            return "dry: no engine, requests go to the stub server"
        t0, limit = time.time(), self.cfg["engine"]["ready_s"]
        while time.time() - t0 < limit:
            out = _read(self.out_path)
            if self.proc.poll() is not None or "stopped unexpectedly" in out:
                raise EngineError(f"engine died during startup (rc={self.proc.poll()}) after {time.time() - t0:.0f} s; log tail:\n{self.log_tail()}\n{out[-500:]}")
            if "ready: http" in out:
                break
            time.sleep(2)
        else:
            raise EngineError(f"not ready after {limit} s; log tail:\n{self.log_tail()}")
        return self._probe(t0)

    def _probe(self, t0: float) -> str:
        try:
            post_json(self.url, PROBE, 300)
        except (OSError, ValueError) as exc:
            raise EngineError(f"the server printed ready but the probe request failed: {exc}; log tail:\n{self.log_tail()}\n{_read(self.out_path)[-500:]}") from exc
        return f"ready after {time.time() - t0:.0f} s"

    def log_text(self) -> str:
        try:
            with open(self.log_path, encoding="utf-8", errors="replace") as fh:
                return fh.read()
        except OSError as exc:
            return f"[engine.log unreadable: {exc}]"

    def log_tail(self, n: int = 12) -> str:
        return "\n".join(self.log_text().splitlines()[-n:])

    def _group(self) -> list[int]:
        """Every live process of this start: the ones carrying its QUAL_RUN_TAG in their environment plus all their descendants.

        WHY not a pgid: serve/server.py gives the engine its own process group, so the launcher's pgid holds only the server; the
        tag survives that, and the ppid closure catches the private helper (its environment is allow-listed, so it lacks the tag)."""
        if not self.tag or not os.path.isdir("/proc"):
            return []
        needle = f"QUAL_RUN_TAG={self.tag}".encode() + b"\0"
        ppid = {int(p): _ppid(int(p)) for p in os.listdir("/proc") if p.isdigit()}
        mine = {p for p in ppid if needle in _raw(f"/proc/{p}/environ")}
        kids = {p for p, pp in ppid.items() if pp in mine} - mine
        while kids:
            mine |= kids
            kids = {p for p, pp in ppid.items() if pp in mine} - mine
        return sorted(mine)

    def _live(self) -> list[int]:
        return [p for p in self._group() if _state(p) not in ("Z", "?")]

    def helper_pid(self) -> int | None:
        """The helper process of this start: the named helper we started, else the family member whose command line holds ``engine.helper_match``."""
        if self.hproc and self.hproc.poll() is None:
            return self.hproc.pid
        match = qual_cfg.subst(self.cfg["engine"].get("helper_match", ""), self.cfg["vars"])
        for pid in self._group():
            if match and match in _cmdline(pid):
                return pid
        return None

    def zombies(self) -> int:
        return sum(1 for pid in self._group() if _state(pid) == "Z")

    def _wait_gone(self, seconds: float) -> None:
        t0 = time.time()
        while time.time() - t0 < seconds and self._live():
            time.sleep(0.5)

    def stop(self) -> int:
        """TERM the launcher's own pgid (the server then stops its engine), KILL the exact pids still alive after the grace; the survivor count (0 is clean)."""
        if self.dry or not self.proc:
            return 0
        for proc in (self.proc, self.hproc):
            if proc and proc.poll() is None:
                if os.getpgid(proc.pid) != proc.pid:
                    raise EngineError(f"refusing to signal: pid {proc.pid} is not its own group leader")
                os.killpg(proc.pid, signal.SIGTERM)
        self._wait_gone(self.cfg["engine"]["stop_grace_s"])
        for pid in self._live():
            os.kill(pid, signal.SIGKILL)  # exact pids of this start's family, never a pattern
        self._wait_gone(15)
        for proc in (self.proc, self.hproc):
            if proc:
                proc.wait(timeout=30)
        if self.helper and os.path.exists(self.helper[2]):
            os.unlink(self.helper[2])  # the exact channel file this start created, left behind by a killed helper
        return len(self._live())


def _read(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError as exc:
        return f"[unreadable {path}: {exc}]"


def _cmdline(pid: int) -> str:
    return _read(f"/proc/{pid}/cmdline").replace("\0", " ")


def _stat_fields(pid: int) -> list[str]:
    raw = _read(f"/proc/{pid}/stat")
    return raw[raw.rfind(")") + 2 :].split() if ")" in raw else []


def _state(pid: int) -> str:
    f = _stat_fields(pid)
    return f[0] if f else "?"


def _ppid(pid: int) -> int:
    f = _stat_fields(pid)
    return int(f[1]) if len(f) > 1 else -1


def _raw(path: str) -> bytes:
    try:
        with open(path, "rb") as fh:
            return fh.read()
    except OSError:
        return b""  # not ours, or gone: it cannot carry our tag
