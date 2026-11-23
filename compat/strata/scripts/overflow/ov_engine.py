# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""The engine process of one overflow start: placement-qual's Engine, started inside its memory-limited scope, with any number of helpers.

OWNER: this module owns "how an overflow start's processes are launched and found"; qual_engine.Engine still owns readiness, the log,
stop-by-tag and the survivor count.  Differences from it, each forced by this test:
  * the engine argv carries the ``systemd-run --user --scope`` prefix for a capped arm (ov_cfg.engine_command), so the limit is the
    engine's own from its first instruction;
  * helpers are started from ov_cfg.helper_specs (each inside its own scope), and more than one is supported (arm O3);
  * ``pids()`` is public: the sampler needs the start's processes and must not reach into a private method.
CONTRACT: ``OvEngine(cfg, run, rundir, url, dry)`` quacks like qual_engine.Engine; ``sampled_sends(probe, store)`` records the probe
snapshots around every ``qual_work.send`` while the context is open.
SAFETY: only argv lists; stop signals the exact process groups of the processes this object started, then the survivors by tag (Engine.stop).
FAILURE: a helper that does not open its channel raises EngineError (the caller records a FAIL step and the start is INVALID).
CHECKER: compat/strata/tests/overflow/test_overflow_run.py (sampler on a fake send) and test_overflow_cfg.py (the commands).
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import time

import ov_cfg
import ov_paths  # noqa: F401
import qual_engine
import qual_work


class OvEngine(qual_engine.Engine):
    def __init__(self, cfg: dict, run, rundir: str, url: str, dry: bool):
        super().__init__(cfg, run, rundir, url, dry)
        argv, env = ov_cfg.engine_command(cfg, run, rundir)
        self.argv, self.env = argv, {**env, "QUAL_RUN_TAG": self.tag}
        self.more = [(a, {**e, "QUAL_RUN_TAG": self.tag}, ready) for a, e, ready in ov_cfg.helper_specs(cfg, run, rundir)[1:]]
        self.more_procs: list[subprocess.Popen] = []

    def pids(self) -> list[int]:
        """Every live process of this start (tag in its environment, plus descendants)."""
        return self._group()

    def _start_helper(self) -> str:
        note = super()._start_helper()
        for i, (argv, env, ready) in enumerate(self.more, 2):
            log = os.path.join(self.rundir, f"helper{i}.log")
            with open(log, "wb") as fh:
                proc = subprocess.Popen(argv, env=env, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
            self.more_procs.append(proc)
            t0 = time.time()
            while not os.path.exists(ready):
                if proc.poll() is not None or time.time() - t0 > 300:
                    raise qual_engine.EngineError(f"helper {i} did not open its channel {ready} (rc {proc.poll()}): {qual_engine._read(log)[-400:]}")
                time.sleep(1)
            note += f"helper {i} pid/pgid {proc.pid} ready after {time.time() - t0:.0f} s; "
        return note

    def stop(self) -> int:
        """TERM the extra helpers first (the base stop only knows the first), then the base stop; wait for them and remove their channel files."""
        for proc in self.more_procs:
            if proc.poll() is None and os.getpgid(proc.pid) == proc.pid:
                os.killpg(proc.pid, signal.SIGTERM)
        left = super().stop()
        for proc in self.more_procs:
            proc.wait(timeout=30)
        for _a, _e, ready in self.more:
            if os.path.exists(ready):
                os.unlink(ready)
        return left


@contextlib.contextmanager
def sampled_sends(probe, store: list):
    """While open, every ``qual_work.send`` is bracketed by two probe snapshots, appended to ``store`` as (before, after) in call order.

    WHY a wrapper and not a timer thread: the request loop (pay_measure) is payoff's and sends exactly one request per record; bracketing
    the call gives each request its own counter delta with no clock matching.  The pairing is by order, so ov_measure refuses a start
    whose store and request list differ in length."""
    real = qual_work.send

    def wrapped(ctx, prompt, max_tokens):
        before = probe.snap()
        resp = real(ctx, prompt, max_tokens)
        store.append((before, probe.snap()))
        return resp

    qual_work.send = wrapped
    try:
        yield
    finally:
        qual_work.send = real
