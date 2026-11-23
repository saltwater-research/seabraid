#!/bin/bash
# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
# stallwatch.sh SERVER_OUT OUTDIR
#   Watches the engine server log SERVER_OUT. When the progress counter ("N of M tokens", "thinking: N",
#   "answering: N") stops moving for 20 s, dumps every thread's stack of the running engine with gdb
#   (once; a second dump follows if it is still stuck at 90 s). Loops until killed.
#   Needs permission to ptrace the engine (same user as its parent, root, or a ptrace_scope lowered by the administrator;
#   seabraid itself no longer widens ptrace access).
# Arguments:
#   SERVER_OUT   server log file to watch (required)
#   OUTDIR       directory for stack_<HHMMSS>.txt dumps (required; created if missing)
# Environment (optional):
#   ENGINE_PATTERN  pgrep -f pattern that finds the engine process (default: "build-hip/strata --serve")
# Usage: stallwatch.sh SERVER_OUT OUTDIR
if [ $# -lt 2 ] || [ -z "$1" ] || [ -z "$2" ]; then
  echo "usage: $0 SERVER_OUT OUTDIR   (env ENGINE_PATTERN=\"build-hip/strata --serve\" optional)" >&2
  exit 2
fi
OUT=$1; DIR=$2; mkdir -p "$DIR"; last=""; since=$(date +%s); dumped=0
ENGINE=${ENGINE_PATTERN:-build-hip/strata --serve}
while true; do
  cur=$(grep -E "reading the prompt:|thinking:|answering:" "$OUT" 2>/dev/null | tail -1 | grep -oE "[0-9,]+ of [0-9,]+ tokens|thinking: [0-9]+|answering: [0-9]+")
  now=$(date +%s)
  if [ "$cur" != "$last" ]; then last="$cur"; since=$now; dumped=0; fi
  el=$((now - since)); if [ -n "$cur" ] && { { [ $el -ge 20 ] && [ $dumped = 0 ]; } || { [ $el -ge 90 ] && [ $dumped = 1 ]; }; }; then
    pid=$(pgrep -f "$ENGINE" | head -1)
    if [ -n "$pid" ]; then
      f=$DIR/stack_$(date +%H%M%S).txt
      echo "stall: '$cur' for $((now-since)) s, pid $pid" > "$f"
      timeout 120 gdb -p "$pid" -batch -ex "set pagination off" -ex "thread apply all bt 25" >> "$f" 2>&1
      echo "dumped $f"
    fi
    dumped=$((dumped+1))
  fi
  sleep 5
done
