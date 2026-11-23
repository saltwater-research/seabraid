# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""seabraid installer: the real process runner for the lead-job probe (_bench_lead.LeadRunner).  UNTESTED-ON-HARDWARE.

OWNER     : the only code that starts the engine and the other cards' processes for a probe, sends the fixed greedy load and
            reads back the engine log.  What to start and which rounds to run is decided in _bench_lead and arrives as a Probe.
CONTRACT  : ProcessLeadRunner().probe(Probe) -> ProbeRun.  stdlib only.  Everything it starts is stopped before it returns,
            on every path (the card must be free for the next candidate and for the real services).
FAILURE   : never raises on the engine's behaviour: a process that dies, a /health that never answers within probe.ready_s or a
            request that fails comes back as ProbeRun(started=False | texts for the rounds that finished, why=reason).
CHECKER   : tests/install/test_bench_lead.py runs this against a real python child that serves /health and logs a timing line.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
import urllib.request
from typing import Callable, Optional

from ._bench_lead import Probe, ProbeRun
from ._log import describe
from ._http import loopback_open
from .build import Cmd

_log = logging.getLogger(__name__)

PROMPT = ("Explain in detail how a mixture-of-experts transformer routes each token to its experts, "
          "and why expert placement across memory tiers affects decode speed.")      # same prompt as tools/greedy.py
REQUEST_TIMEOUT_S = 900.0
FLUSH_S = 2.0
SETTLE_S = 8.0              # arms.sh waits the same after a stop: VRAM is returned lazily


def _env(cmd: Cmd) -> dict:
    return {**os.environ, **{k: str(v) for k, v in cmd.env.items()}}


class ProcessLeadRunner:
    def __init__(self, popen: Callable = subprocess.Popen, urlopen: Callable = loopback_open,
                 sleep: Callable = time.sleep, clock: Callable = time.monotonic, settle_s: float = SETTLE_S,
                 echo: Callable = print):
        self.popen, self.urlopen, self.sleep, self.clock, self.settle_s, self.echo = popen, urlopen, sleep, clock, settle_s, echo

    def probe(self, p: Probe) -> ProbeRun:
        procs: list = []
        try:
            return self._run(p, procs)
        except OSError as exc:      # a binary that is missing or not executable: the one start failure Popen raises
            _log.warning("lead probe could not start a process: %s", describe(exc))
            return ProbeRun(False, f"could not start a process: {exc}")
        finally:
            for pr in reversed(procs):
                self._stop(pr)
            self.sleep(self.settle_s)

    def _stop(self, pr) -> None:
        if pr.poll() is None:
            pr.terminate()
            try:
                pr.wait(timeout=30)
            except subprocess.TimeoutExpired:
                _log.warning("lead probe process ignored SIGTERM for 30 s; killing it")
                pr.kill()
                pr.wait()

    def _start(self, cmd: Cmd, procs: list, out_path: str):
        self.echo(f"[run] {cmd}")
        out = open(out_path, "ab")  # noqa: SIM115  closed by the OS when the child exits; the file outlives it for the user to read
        pr = self.popen(list(cmd.argv), env=_env(cmd), cwd=cmd.cwd, stdout=out, stderr=subprocess.STDOUT)
        procs.append(pr)
        return pr

    def _await_channels(self, p: Probe, others: list, deadline: float, base: str) -> Optional[ProbeRun]:
        """Wait for every helper's shm channel (the ready signal; arms.sh: "holding ... GiB" comes only after the server's plan)."""
        for shm in p.shms:
            while not os.path.exists("/dev/shm" + shm):
                if any(o.poll() is not None for o in others) or self.clock() > deadline:
                    return ProbeRun(False, f"the process for {shm} did not come up (see {base}-other*.out)")
                self.sleep(2)
        return None

    def _await_server(self, p: Probe, server, deadline: float, base: str) -> Optional[ProbeRun]:
        while not self._healthy(p.port):
            if server.poll() is not None or self.clock() > deadline:
                return ProbeRun(False, f"the engine {'exited' if server.poll() is not None else 'was not ready'} (see {base}-server.out and {p.log_path})")
            self.sleep(3)
        return None

    def _rounds(self, p: Probe) -> ProbeRun:
        """Run each load round and keep the part of the engine log it wrote."""
        texts = {}
        for r in p.rounds:
            start = os.path.getsize(p.log_path) if os.path.exists(p.log_path) else 0
            err = self._load(p.port, r.streams, r.tokens)
            if err:
                return ProbeRun(True, f"round {r.label} failed: {err}", texts)
            self.sleep(FLUSH_S)     # the engine logs its timing line just after the last token is returned
            with open(p.log_path, "rb") as fh:
                fh.seek(start)
                texts[r.label] = fh.read().decode("utf-8", "replace")
        return ProbeRun(True, "", texts)

    def _run(self, p: Probe, procs: list) -> ProbeRun:
        base = os.path.splitext(p.log_path)[0]
        others = [self._start(c, procs, f"{base}-other{n}.out") for n, c in enumerate(p.others)]
        deadline = self.clock() + p.ready_s
        refused = self._await_channels(p, others, deadline, base)
        if refused is not None:
            return refused
        server = self._start(p.server, procs, f"{base}-server.out")
        refused = self._await_server(p, server, deadline, base)
        return refused if refused is not None else self._rounds(p)

    def _healthy(self, port: int) -> bool:
        try:
            return self.urlopen(f"http://127.0.0.1:{port}/health", timeout=3).status == 200
        except OSError as exc:
            _log.debug("engine on the probe port not healthy yet: %s", describe(exc))
            return False

    def _load(self, port: int, streams: int, tokens: int) -> Optional[str]:
        """`streams` concurrent greedy requests of `tokens` tokens; None on success, else the first error."""
        errs: list = []

        def one(i: int) -> None:
            # a distinct first line per stream, so four streams are four sequences and not one shared prefix
            body = {"model": "x", "messages": [{"role": "user", "content": f"Stream {i}. {PROMPT}"}], "max_tokens": tokens,
                    "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
            req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
            try:
                self.urlopen(req, timeout=REQUEST_TIMEOUT_S).read()
            except OSError as exc:
                _log.warning("lead probe stream %d failed: %s", i, describe(exc))
                errs.append(f"stream {i}: {exc}")
        ts = [threading.Thread(target=one, args=(i,)) for i in range(streams)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        return errs[0] if errs else None
