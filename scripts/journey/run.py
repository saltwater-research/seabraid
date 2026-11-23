# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""Replay the documented first-time-user journey (steps.toml) in a fresh tree and record every step.

OWNER: steps.toml owns WHAT runs (the README and CONTRIBUTING commands); this file owns HOW: a clean environment, one
bash per step, a verdict per step, and the record.  CONTRACT: `python scripts/journey/run.py --tree DIR --out DIR
[--offline]`; stdlib only; exit 0 when no step is an unexpected FAIL, 1 when one is, 2 when the arguments are unusable.
FAILURE: fail-open-loud for the documented gaps (SKIP and WARN are printed and recorded), fail-fast for everything else.
CHECKER: tests/test_journey_steps.py (schema, doc drift, a toy replay); CI job `journey` runs it on a `git archive`.
WHY a clean environment: a person's shell state is the thing under test, so each step gets a fresh PATH plus only the
state lines the docs say they kept (venv activation, the cargo PATH line), recorded in the step file.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
import tomllib
from pathlib import Path

STEPS_FILE = Path(__file__).resolve().with_name("steps.toml")
SYSTEM_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
KEYS = {"id", "doc", "command", "state", "when", "status", "output", "warn_output", "env_prefix", "substitute", "harness"}
STATUSES = ("pass", "skip-needs-network", "warn-expected")
WHENS = ("always", "online", "offline")


def load(path: Path = STEPS_FILE) -> tuple[dict, list[dict]]:
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    return data.get("state", {}), data.get("step", [])


def patterns(step: dict) -> list[str]:
    out = step.get("output", [])
    return [out] if isinstance(out, str) else list(out)


def _enum_problems(step: dict) -> list[str]:
    errs = []
    if step.get("status", "pass") not in STATUSES:
        errs.append(f"bad status {step['status']!r}")
    if step.get("when", "always") not in WHENS:
        errs.append(f"bad when {step['when']!r}")
    if step.get("status") == "warn-expected" and not step.get("warn_output"):
        errs.append("warn-expected needs warn_output")
    return errs


def _problems(step: dict, state: dict) -> list[str]:
    errs = [f"missing {k}" for k in ("id", "doc", "command") if not step.get(k)]
    errs += [f"unknown key {k}" for k in step if k not in KEYS]
    errs += [f"unknown state {n}" for n in step.get("state", []) if n not in state]
    errs += [f"placeholder {k!r} not in command" for k in step.get("substitute", {}) if k not in step.get("command", "")]
    return errs + _regex_problems(step) + _enum_problems(step)


def _regex_problems(step: dict) -> list[str]:
    errs = []
    for rx in patterns(step) + [step.get("warn_output", "")]:
        try:
            re.compile(rx)
        except re.error as exc:
            errs.append(f"bad regex {rx!r}: {exc}")
    return errs


def validate(state: dict, steps: list[dict]) -> list[str]:
    """Every schema problem, as `id: problem` lines (empty list = valid)."""
    errs, seen = [], set()
    for step in steps:
        sid = step.get("id", "?")
        errs += [f"{sid}: {p}" for p in _problems(step, state)]
        if sid in seen:
            errs.append(f"{sid}: duplicate id")
        seen.add(sid)
    return errs


def skip_reason(step: dict, offline: bool) -> str | None:
    when = step.get("when", "always")
    if when == "online" and offline:
        return "online recipe; the offline recipe is a separate step"
    if when == "offline" and not offline:
        return "offline recipe; the online recipe is a separate step"
    if step.get("status") == "skip-needs-network" and offline:
        return "needs network (--offline)"
    return None


def script_for(step: dict, state: dict) -> tuple[list[str], str]:
    """(state lines, command) exactly as bash will see them, placeholders substituted and env_prefix applied."""
    cmd = step["command"]
    for placeholder, value in step.get("substitute", {}).items():
        cmd = cmd.replace(placeholder, value)
    if step.get("env_prefix"):
        cmd = f"{step['env_prefix']} {cmd}"
    return [state[n] for n in step.get("state", [])], cmd


def clean_env(work: Path) -> dict:
    """A fresh shell's environment: the interpreter under test first on PATH, system dirs after, nothing else inherited."""
    env = {"PATH": os.path.dirname(sys.executable) + ":" + SYSTEM_PATH, "HOME": os.environ.get("HOME", str(work)), "LANG": "C.UTF-8", "TERM": "dumb"}
    env.update({k: os.environ[k] for k in ("CARGO_HOME", "RUSTUP_HOME") if k in os.environ})
    return {**env, "CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "JOURNEY_WORK": str(work)}


def verdict(step: dict, rc: int, text: str) -> str:
    if rc == 0 and all(re.search(rx, text, re.M) for rx in patterns(step)):
        return "PASS"
    if step.get("status") == "warn-expected" and rc != 0 and re.search(step["warn_output"], text, re.M):
        return "WARN"
    return "FAIL"


def execute(step: dict, state: dict, tree: Path, env: dict, timeout: int) -> dict:
    pre, cmd = script_for(step, state)
    began = time.monotonic()
    try:
        done = subprocess.run(["bash", "-e", "-o", "pipefail", "-c", "\n".join(pre + [cmd])], cwd=tree, env=env, text=True, errors="replace", timeout=timeout, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        rc, text = done.returncode, done.stdout
    except subprocess.TimeoutExpired as exc:
        rc, text = 124, f"TIMEOUT after {timeout}s\n{exc.stdout or ''}"
    return {"rc": rc, "text": text, "secs": time.monotonic() - began, "pre": pre, "cmd": cmd, "status": verdict(step, rc, text)}


def write_step(out: Path, n: int, step: dict, rec: dict) -> None:
    head = f"STEP: {n:02d} {step['id']}\nCOMMAND: {rec['cmd']}\n" + "".join(f"SHELL-STATE: {line}\n" for line in rec["pre"])
    tail = f"----- END -----\nEXIT: {rec['rc']}\nSECONDS: {rec['secs']:.1f}\nSTATUS: {rec['status']}\n"
    (out / "steps" / f"{n:02d}-{step['id']}.txt").write_text(head + "----- OUTPUT -----\n" + rec["text"] + tail, encoding="utf-8")


def report(out: Path, rows: list[tuple], meta: str) -> None:
    counts = {s: sum(r[4] == s for r in rows) for s in ("PASS", "WARN", "SKIP", "FAIL")}
    lines = [f"# Journey report\n\n{meta}\n\n" + ", ".join(f"{v} {k}" for k, v in counts.items()) + "\n", "| # | Step | Seconds | Exit | Status | Last lines |", "|---|---|---|---|---|---|"]
    for n, sid, secs, rc, status, last in rows:
        lines.append(f"| {n:02d} | {sid} | {secs:.1f} | {rc} | {status} | {last.replace('|', '/')} |")
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "index.tsv").write_text("".join(f"{n:02d}\t{sid}\t{secs:.1f}\t{rc}\t{status}\n" for n, sid, secs, rc, status, _ in rows), encoding="utf-8")


def run_all(state: dict, steps: list[dict], tree: Path, out: Path, offline: bool, timeout: int) -> int:
    (out / "steps").mkdir(parents=True, exist_ok=True)
    work = out / "work"
    work.mkdir(exist_ok=True)
    env, rows = clean_env(work), []
    for n, step in enumerate(steps, 1):
        reason = skip_reason(step, offline)
        rec = {"rc": "-", "text": f"SKIPPED: {reason}\n", "secs": 0.0, "pre": [], "cmd": step["command"], "status": "SKIP"} if reason else execute(step, state, tree, env, timeout)
        write_step(out, n, step, rec)
        last = " / ".join(rec["text"].strip().splitlines()[-3:])[-200:]
        rows.append((n, step["id"], rec["secs"], rec["rc"], rec["status"], last))
        print(f"[{n:02d} {step['id']}] {rec['status']} rc={rec['rc']} {rec['secs']:.1f}s", flush=True)
    notes = "".join(f"\n- `{s['id']}` is not in the docs: {s['harness']}" for s in steps if "harness" in s)  # never hide a step the docs lack
    report(out, rows, f"mode: {'offline' if offline else 'online'}; python {sys.version.split()[0]}; tree: {tree}\n{notes}")
    return 1 if any(r[4] == "FAIL" for r in rows) else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tree", required=True, type=Path, help="fresh tree to run in (no .venv yet)")
    ap.add_argument("--out", required=True, type=Path, help="record directory, outside the tree")
    ap.add_argument("--steps", type=Path, default=STEPS_FILE)
    ap.add_argument("--offline", action="store_true", help="record network steps as SKIP; use the offline recipes")
    ap.add_argument("--timeout", type=int, default=1500, help="seconds per step")
    args = ap.parse_args(argv)
    tree, out = args.tree.resolve(), args.out.resolve()
    state, steps = load(args.steps)
    problems = validate(state, steps)
    nested = tree in (out, *out.parents) or out in tree.parents  # the record must not land in the tree the checkers scan
    if not tree.is_dir() or (tree / ".venv").exists() or nested:
        problems.append("--tree must be an existing fresh tree without .venv, and --out a directory outside it")
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 2
    return run_all(state, steps, tree, out, args.offline, args.timeout)


if __name__ == "__main__":
    sys.exit(main())
