#!/bin/bash
# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
# kl_gate.sh [OUT]
#   seabraid KL gate (teacher-forced). The same ~430-token text (send.py's text file) is read through decode
#   windows (--short-read 1000, the path where the helper computes experts) with STRATA_LOGPOS set.
#   Arms: stock, stock again (noise floor), seabraid. Then kl.py compares them.
#   OUT, the config paths and TEXT_FILE are resolved against this script's directory; an absolute path is used as given.
#   The three inputs are NOT shipped: see "The KL gate" in compat/strata/README.md for how to produce them. A missing
#   input stops the run before anything starts, and an arm that logs no rows stops it with a nonzero exit.
# Arguments:
#   OUT                  output directory (default: run)
# Required environment:
#   SEABRAID_STRATA_DIR          seabraid checkout root (contains serve/server.py and the helper build)
#   STRATA_DATA          Strata-data root (contains packs/)
#   NATIVE_GGUF          native GGUF model file the helper reads
#   PACK_NAME            pack directory under STRATA_DATA/packs
#   HELPER_BIN           helper binary (the vendor build of seabraid-helper)
#   LAYERS / EXPERTS     layer / expert counts of the pack (read them from the pack; never assume them)
#   PORT                 server port on 127.0.0.1
#   SHM                  helper shared-memory name (e.g. the one the installer assigned)
# Optional environment (default):
#   PY                   python interpreter for server, send.py and kl.py (python3)
#   GPU                  device index for CUDA and HIP visibility (0)
#   ROCM_LIB             ROCm lib dir added to LD_LIBRARY_PATH (default: unset)
#   STOCK_CFG            stock server config, relative to this script (cfg_stock.json)
#   SEABRAID_STRATA_CFG          seabraid server config, relative to this script (cfg_seabraid.json)
#   TEXT_FILE            the fixed ~440-position text sent in every arm, relative to this script (text.txt)
#   POLL_SECS            seconds between /health polls while a server loads (3)
#   HELPER_WAIT_SECS     seconds the helper gets to attach before the seabraid arm starts (6)
# Usage: SEABRAID_STRATA_DIR=DIR STRATA_DATA=DIR NATIVE_GGUF=FILE ./kl_gate.sh [OUT]
usage() {
  echo "usage: SEABRAID_STRATA_DIR=DIR STRATA_DATA=DIR NATIVE_GGUF=FILE PACK_NAME=N HELPER_BIN=FILE LAYERS=N EXPERTS=N PORT=N SHM=NAME [optional env, see header] $0 [OUT]" >&2
  exit 2
}
# Model geometry, port and shm name are REQUIRED: a default here is one machine's value and silently wrong on the next.
for v in SEABRAID_STRATA_DIR STRATA_DATA NATIVE_GGUF PACK_NAME HELPER_BIN LAYERS EXPERTS PORT SHM; do
  if [ -z "${!v:-}" ]; then echo "error: $v is required" >&2; usage; fi
done
cd "$(dirname "$0")" || exit 1
OUT=${1:-run}
# OUT becomes absolute here: the server gets it as STRATA_LOGPOS, so a prefix of $PWD would double an absolute OUT.
mkdir -p "$OUT" && OUT=$(cd "$OUT" && pwd) || { echo "error: cannot create the output directory" >&2; exit 1; }
PY=${PY:-python3}
PACK=$STRATA_DATA/packs/$PACK_NAME
HELPER=$HELPER_BIN
GPU=${GPU:-0}
STOCK_CFG=${STOCK_CFG:-cfg_stock.json}; SEABRAID_STRATA_CFG=${SEABRAID_STRATA_CFG:-cfg_seabraid.json}; TEXT_FILE=${TEXT_FILE:-text.txt}
POLL_SECS=${POLL_SECS:-3}; HELPER_WAIT_SECS=${HELPER_WAIT_SECS:-6}

# Inputs the tree does not ship. Name each missing one and how to make it, so the run never starts half-armed.
missing=0
need() { [ -f "$2" ] && return 0; echo "error: $1 not found: $3" >&2; missing=1; }
need STOCK_CFG "$STOCK_CFG" "the stock arm's engine config: your engine config WITHOUT --expert-cache-device1 (set STOCK_CFG)"
need SEABRAID_STRATA_CFG "$SEABRAID_STRATA_CFG" "the seabraid arm's engine config: the same config WITH --expert-cache-device1 (set SEABRAID_STRATA_CFG)"
need TEXT_FILE "$TEXT_FILE" "any fixed text of about 440 token positions that you supply; it is not shipped (set TEXT_FILE)"
[ "$missing" -eq 0 ] || exit 2

if [ -n "${ROCM_LIB:-}" ]; then
  export LD_LIBRARY_PATH="$ROCM_LIB:$ROCM_LIB/rocm_sysdeps/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export HIP_VISIBLE_DEVICES=$GPU
export STRATA_ROUTE_TAIL_SKIP=0 STRATA_LOGPOS_TOPK=32
HP=
trap '[ -z "$HP" ] || kill "$HP" 2>/dev/null' EXIT

# One arm: start the server, wait for /health, send the text, stop it, and require logged rows (kl.py cannot judge none).
arm() { local name=$1 cfg=$2 rows; echo "== $name"; rm -f "${OUT:?}/${name:?}.tsv"
  STRATA_LOGPOS=$OUT/$name.tsv "$PY" "$SEABRAID_STRATA_DIR/serve/server.py" --engine strata --config "$cfg" --host 127.0.0.1 --port "$PORT" \
    --no-api-key --parallel 1 > "$OUT/server_$name.out" 2>&1 & local sp=$!
  for i in $(seq 1 200); do curl --noproxy '*' -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null && break
    kill -0 $sp 2>/dev/null || { echo "error: arm $name: the server died before it was healthy (see server_$name.out in OUT)" >&2; return 1; }
    sleep "$POLL_SECS"; done
  "$PY" send.py "http://127.0.0.1:$PORT/v1/chat/completions" --text "$TEXT_FILE" \
    || { echo "error: arm $name: sending the text failed" >&2; kill $sp; wait $sp 2>/dev/null; return 1; }
  kill $sp; wait $sp 2>/dev/null
  rows=0; [ -s "$OUT/$name.tsv" ] && rows=$(wc -l < "$OUT/$name.tsv"); echo "rows $rows"
  [ "$rows" -gt 0 ] || { echo "error: arm $name logged no rows: the engine did not write STRATA_LOGPOS (does this engine build support it?)" >&2; return 1; }; }
arm stock1 "$STOCK_CFG" || exit 1
arm stock2 "$STOCK_CFG" || exit 1
CUDA_VISIBLE_DEVICES=$GPU "$HELPER" --shm "$SHM" --pack "$PACK" \
  --native "$NATIVE_GGUF" --layers "$LAYERS" --experts "$EXPERTS" --device "$GPU" > "$OUT/helper.log" 2>&1 &
HP=$!; sleep "$HELPER_WAIT_SECS"
SEABRAID_STRATA_SHM=$SHM SEABRAID_STRATA_EXCLUDE_HELPER_RAM=1 STRATA_PREFILL_HELPER_SOURCE=1 STRATA_DISJOINT_ADAPT=1 arm seabraid "$SEABRAID_STRATA_CFG" || exit 1
kill $HP; wait $HP 2>/dev/null; HP=
echo "noise floor (stock vs stock):"; "$PY" kl.py "$OUT/stock1.tsv" "$OUT/stock2.tsv" || exit $?
echo "seabraid vs stock:";             "$PY" kl.py "$OUT/stock1.tsv" "$OUT/seabraid.tsv"
