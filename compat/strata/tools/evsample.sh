#!/bin/bash
# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
# evsample.sh OUT
#   Every 2 s appends one sample line to OUT: time, engine pid, KFD evicted_ms, KFD per-process counters,
#   selected /proc/vmstat counters (compaction, migration, reclaim), and the newest server log's last progress line.
#   Runs until killed.
# Arguments:
#   OUT                 file to append samples to (required)
# Environment (required):
#   SERVER_LOG_DIR      directory holding the engine server logs; the newest match is read
# Environment (optional):
#   ENGINE_PATTERN      pgrep -f pattern that finds the engine process (default: "build-hip/strata --serve")
#   SERVER_LOG_GLOB     log-name glob inside SERVER_LOG_DIR (default: server_*.out)
# Usage: SERVER_LOG_DIR=/path/to/logs evsample.sh OUT
OUT=$1
if [ -z "${OUT:-}" ] || [ -z "${SERVER_LOG_DIR:-}" ]; then
  echo "usage: SERVER_LOG_DIR=DIR $0 OUT   (env ENGINE_PATTERN, SERVER_LOG_GLOB optional)" >&2
  exit 2
fi
ENGINE=${ENGINE_PATTERN:-build-hip/strata --serve}
GLOB=${SERVER_LOG_GLOB:-server_*.out}
while true; do
  pid=$(pgrep -f "$ENGINE" | head -1)
  ev=""; ct=""
  if [ -n "$pid" ]; then
    ev=$(cat /sys/class/kfd/kfd/proc/$pid/stats_*/evicted_ms 2>/dev/null)
    ct=$(for f in /sys/class/kfd/kfd/proc/$pid/counters_*/*; do printf "%s=%s " "$(basename "$f")" "$(cat "$f")"; done 2>/dev/null)
  fi
  vm=$(grep -E '^(compact_stall|compact_migrate_scanned|compact_isolated|pgmigrate_success|thp_migration_success|allocstall_normal|pgsteal_kswapd|pgsteal_direct)' /proc/vmstat | awk '{printf "%s=%s ",$1,$2}')
  pr=$(grep -E "reading the prompt|thinking:|answering:|done:" "$(ls -t "$SERVER_LOG_DIR"/$GLOB | head -1)" | tail -1 | cut -c1-70)
  echo "$(date +%H:%M:%S) pid=$pid evicted_ms=$ev $ct| $vm| $pr"
  sleep 2
done >> "$OUT"
