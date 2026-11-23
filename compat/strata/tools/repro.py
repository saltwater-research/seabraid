#!/usr/bin/env python3
# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""repro.py URL OUTDIR IDX[:MAXTOK][,IDX[:MAXTOK]...] SERVER_LOG [options]

Sends short classification prompts one request at a time to a chat-completions endpoint. Each prompt's max_tokens
is 8 unless given as IDX:MAXTOK. While a request is open, the server
progress counter in SERVER_LOG is watched; if it has not moved for 25 s the engine's thread stacks are dumped
into OUTDIR/stack_<tag>.txt (a second dump follows at +90 s), using rocgdb. rocgdb must be allowed to attach to the engine (run as the engine's parent, as root, or with
kernel.yama.ptrace_scope lowered by the administrator; seabraid itself no longer widens ptrace access).
Prints wall time and outcome per request.
Arguments:
  URL              chat-completions endpoint URL
  OUTDIR           directory for stack dumps (created if missing)
  IDX[:MAXTOK]     comma list of prompt indices, each optionally with its own max_tokens
  SERVER_LOG       server log file to read progress from
Options:
  --prompts FILE   JSON list of {"title": ..., "text": ...} objects (default: a small built-in set)
  --engine-pattern pgrep -f pattern for the engine (default: "build-hip/strata --serve")
  --rocm-home DIR  ROCm root with bin/rocgdb and lib/ (env ROCM_HOME; default: rocgdb from PATH)
"""
import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time

from _loopback import post_json

LABELS = ["positive", "negative", "neutral", "question", "statement"]
DEFAULT_PROMPTS = [
    {"title": "Review", "text": "The new release is fast and the installer finally just works."},
    {"title": "Complaint", "text": "The update broke my configuration and support never answered."},
    {"title": "Ask", "text": "Does the helper card need to be the same vendor as the main card?"},
    {"title": "Note", "text": "The server listens on port 8080 unless told otherwise."},
]


def parse_args():
    ap = argparse.ArgumentParser(description="Send classification prompts one at a time; dump engine stacks on stalls.")
    ap.add_argument("url", help="chat-completions endpoint URL")
    ap.add_argument("outdir", help="directory for stack dumps")
    ap.add_argument("idxs", help="comma list of IDX or IDX:MAXTOK")
    ap.add_argument("server_log", help="server log file to read progress from")
    ap.add_argument("--prompts", default=None, help="JSON list of {title, text} objects (default: a small built-in set)")
    ap.add_argument("--engine-pattern", default="build-hip/strata --serve", help="pgrep -f pattern for the engine")
    ap.add_argument("--rocm-home", default=os.environ.get("ROCM_HOME"), help="ROCm root with bin/rocgdb (env ROCM_HOME)")
    a = ap.parse_args()
    a.idxs = [(int(x.split(":")[0]), int(x.split(":")[1]) if ":" in x else 8) for x in a.idxs.split(",")]
    return a


A = parse_args()
url, out, srv = A.url, A.outdir, A.server_log
os.makedirs(out, exist_ok=True)
T = json.load(open(A.prompts)) if A.prompts else DEFAULT_PROMPTS


def prompt(e):
    return (f"Classify the text below with exactly one label from: {', '.join(LABELS)}.\n"
            'Reply with JSON only: {"label": "..."}\n\n'
            f"TITLE: {e['title']}\n\n{e['text']}")


VMSTAT_KEYS = ("compact_", "pgmigrate", "thp_migration", "numa_pages_migrated", "pgsteal", "pgscan", "allocstall")
PROGRESS = re.compile(r"[0-9,]+ of [0-9,]+ tokens|thinking: [0-9]+|answering: [0-9]+")


def first_pid(pattern):
    """First pid whose command line matches `pattern` (pgrep -f), or ''.  An argument vector: the pattern is never parsed by a shell."""
    r = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
    return r.stdout.split()[0] if r.stdout.split() else ""


def vmstat_line():
    with open("/proc/vmstat") as fh:
        return " ".join(ln.strip() for ln in fh if ln.startswith(VMSTAT_KEYS))


def server_progress(log_path):
    """The last progress marker in the server log ('' when none): what the old grep|tail|grep -o pipeline returned."""
    last = ""
    with open(log_path, errors="replace") as fh:
        for ln in fh:
            if "reading the prompt:" in ln or "thinking:" in ln or "answering:" in ln:
                m = PROGRESS.search(ln)
                last = m.group(0) if m else ""
    return last


def gpu_lines():
    try:
        r = subprocess.run(["rocm-smi", "-d", "0", "--showuse", "--showpower", "--showmemuse"], capture_output=True, text=True)
    except OSError as ex:   # WHY not fatal: the GPU sample is context for the stack dump, which must still be written
        print(f"  rocm-smi unavailable ({ex.__class__.__name__}); GPU sample skipped", file=sys.stderr, flush=True)
        return []
    return [ln for ln in r.stdout.splitlines() if re.search(r"GPU use|Power|Memory Allocated|VRAM%", ln)]


def dump(tag):
    pid = first_pid(A.engine_pattern)
    if not pid:
        print(f"  no process matches the engine pattern; no stack dump for {tag}", file=sys.stderr, flush=True)
        return
    f = f"{out}/stack_{tag}.txt"
    env = dict(os.environ)
    if A.rocm_home:
        R = A.rocm_home
        env["LD_LIBRARY_PATH"] = f"{R}/lib:{R}/lib/rocm_sysdeps/lib"
        gdb = f"{R}/bin/rocgdb"
    else:
        gdb = "rocgdb"
    with open(f, "w") as fh:
        subprocess.run(["timeout", "180", gdb, "-p", pid, "-batch", "-ex", "set pagination off",
                        "-ex", "info agents", "-ex", "info queues", "-ex", "info dispatches", "-ex", "info threads",
                        "-ex", "thread apply all bt 12"], stdout=fh, stderr=subprocess.STDOUT, env=env)
    with open(f, "a") as fh:
        for _ in range(2):
            fh.write("--- vmstat\n" + vmstat_line() + "\n")
            fh.flush()
            time.sleep(5)
        for _ in range(3):
            fh.write("--- gpu\n" + "\n".join(gpu_lines()) + "\n")
            fh.flush()
            time.sleep(1)
    print("  dumped", f, flush=True)


for n, (i, mt) in enumerate(A.idxs):
    body = {"model": "x", "messages": [{"role": "user", "content": prompt(T[i])}], "max_tokens": mt, "temperature": 0.2 if mt > 8 else 0}
    done = threading.Event()

    def watch(tag=f"{n}_prompt{i}"):
        # a long generation is not a stall: watch the server's progress instead of the request's age
        last, since, k = None, time.time(), 0
        while not done.wait(5):
            cur = server_progress(srv)
            if cur != last:
                last, since, k = cur, time.time(), 0
            elif time.time() - since > (25 if k == 0 else 90) and k < 2:
                dump(f"{tag}_{k}")
                k += 1

    th = threading.Thread(target=watch, daemon=True)
    th.start()
    t = time.time()
    try:
        d = post_json(url, body, 900)
        r = f"ok {d.get('usage',{}).get('prompt_tokens')} prompt tokens"
    except Exception as ex:
        r = f"ERR {ex}"
    done.set()
    th.join()
    print(f"req {n} prompt {i}: {time.time()-t:.1f} s {r}", flush=True)
