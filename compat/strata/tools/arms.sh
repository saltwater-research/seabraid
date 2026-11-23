#!/bin/bash
# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
# arms.sh RUN_DIR ARMS_FILE
#   Live A/B runner for seabraid on a two-card machine: one main engine (server.py + cfg.json) and one helper process.
#   Each arm is a cold start (helper + server), its measurement, then a full stop, so arms never share warm state.
#   Replaces the per-experiment copies (ab.sh, ab1.sh) the 2026-11-07 runs used; one runner, arms are data.
#
# ARMS_FILE: one arm per line, '|'-separated, '#' comments:
#     NAME | MODE | PARALLEL | ENV (space-separated K=V, or -) | JQ filter applied to the base cfg.json (or -)
#   MODE  speed : 400-token warm-up, 3 x 2,000-token greedy (quote run 2/3, never run 1), one ~3.5k-token prompt read
#         conc  : PARALLEL concurrent greedy requests x 2 rounds of 1,000 tokens (aggregate tok/s), server --parallel N
#         kl    : teacher-forced text through decode windows (--short-read 1000), STRATA_LOGPOS top-32 to kl_NAME.tsv
#         prefill: STRATA_PREFILL_TIMING=1; prompts of KL_TEXT x 2 / 8 / 32 (~0.9k / 3.5k / 14k tokens), each with a distinct
#                 first line so no prefix is reused; then one 400-token greedy decode
#   hist  : STRATA_ROUTE_HIST=RUN_DIR/hist_NAME (set by the runner) + STRATA_DECODE_TIMING; four long prompts of different
#           kinds (prose summary, prose translation, source code, arithmetic), 600 greedy tokens each -> one req-*.json per
#           request, then tools/route_hist_report.py writes report_NAME.txt (prompt-conditioned promotion, tierability,
#           card 1 own work vs wait)
#   ENV token NOHELPER=1: no helper process and SEABRAID_STRATA_SHM unset (a stock single-card arm; drop --expert-cache-device1 in JQ)
# Required environment:
#   BASE_CFG     base server cfg.json (as the installer writes it; port/log are rewritten per arm)
#   SERVER_PY    serve/server.py of the build under test;  PY: interpreter for it (default python3)
#   HELPER_CMD   full helper command line WITHOUT --shm (the runner adds a private shm name)
#   HELPER_ENV   visibility/lib env for the helper (e.g. "CUDA_VISIBLE_DEVICES=GPU-..."), may be empty
#   MAIN_ENV     visibility/lib env for the main engine (e.g. "HIP_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=...")
#   KL_TEXT      teacher-forced text for kl arms;  TOOLS: this repo's tools/ (greedy.py, conc.py, send.py)
# Optional: KL_BASELINE (baseline kl TSV: each kl arm is gated against it by tools/kl.py; a FAIL or INVALID verdict marks the
#   arm INVALID and the run ends rc=1 after the remaining arms), KL_MAX / KL_MIN_TOP1 (override kl.py's bounds), PORT (8080), NV_STOP_C (80), AMD_STOP_C (95, junction)
#
# FAILURE: fail-fast per arm and loud. A dead helper/server, a timeout, or the heat guard marks the arm INVALID in
#   RESULT.txt and the run stops; numbers from an INVALID arm are never written as results. Heat guard: hard stop (kill
#   both processes) at NV_STOP_C on the NVIDIA core or AMD_STOP_C on the AMD junction, checked every 3 s.
# CHECKER: the run's RESULT.txt names every arm VALID or INVALID; sabotage = set NV_STOP_C below idle (arm goes INVALID).
set -u
R=${1:?run dir}; ARMS=${2:?arms file}; mkdir -p "$R"
for v in BASE_CFG SERVER_PY HELPER_CMD MAIN_ENV KL_TEXT TOOLS; do [ -n "${!v:-}" ] || { echo "error: $v is required" >&2; exit 2; }; done
GATE_RC=0; PY=${PY:-python3}; PORT=${PORT:-8080}; NV_STOP_C=${NV_STOP_C:-80}; AMD_STOP_C=${AMD_STOP_C:-95}
URL=http://127.0.0.1:$PORT/v1/chat/completions; SHM=/seabraid-arms
export STRATA_ENGINE_READY_S=${STRATA_ENGINE_READY_S:-2400}
res(){ echo "$(date +%T) $*" | tee -a "$R/RESULT.txt"; }
amd=$(dirname "$(grep -lx amdgpu /sys/class/hwmon/hwmon*/name | head -1)")
HP=; SP=
( while [ ! -f "$R/DONE" ]; do nv=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')
    j=$(( $(cat "$amd/temp2_input" 2>/dev/null || echo 0) / 1000 ))
    if [ "${nv:-0}" -ge "$NV_STOP_C" ] || [ "$j" -ge "$AMD_STOP_C" ]; then
      echo "heat stop: NVIDIA $nv C, AMD junction $j C" > "$R/ABORT"; pkill -f -- "--shm $SHM"; pkill -f "port $PORT"; fi
    sleep 3; done ) & TW=$!
start_helper(){ env $HELPER_ENV $HELPER_CMD --shm $SHM > "$R/helper_$1.log" 2>&1 & HP=$!
  # The channel existing is the ready signal; "holding ... GiB" is printed only after MAIN sends its plan.
  for i in $(seq 1 300); do [ -e /dev/shm${SHM} ] && return 0
    kill -0 $HP 2>/dev/null || return 1; sleep 2; done; return 1; }
start_server(){ local tag=$1 cfg=$2 par=$3; shift 3
  # An empty SHM means a stock arm: SEABRAID_STRATA_SHM is removed from the environment, never set to "".
  if [ -n "$SHM" ]; then set -- SEABRAID_STRATA_SHM=$SHM "$@"; fi
  env -u SEABRAID_STRATA_SHM $MAIN_ENV "$@" $PY "$SERVER_PY" --engine strata --config "$cfg" --host 127.0.0.1 --port $PORT \
    --no-api-key --parallel "$par" > "$R/server_$tag.out" 2>&1 & SP=$!
  for i in $(seq 1 800); do curl --noproxy '*' -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null && return 0
    kill -0 $SP 2>/dev/null || return 1; [ -f "$R/ABORT" ] && return 1; sleep 3; done; return 1; }
stop_all(){ [ -n "$SP" ] && { kill $SP 2>/dev/null; wait $SP 2>/dev/null; }; [ -n "$HP" ] && { kill $HP 2>/dev/null; wait $HP 2>/dev/null; }
  SP=; HP=; pkill -f "port $PORT" 2>/dev/null; pkill -f "^$EXE " 2>/dev/null; pkill -f -- "--shm $SHM" 2>/dev/null; sleep 8; }
finish(){ stop_all; touch "$R/DONE"; wait $TW 2>/dev/null; res "end rc=$1"; exit "$1"; }
trap 'res "INTERRUPTED"; finish 130' INT TERM
speed(){ $PY -c "import json,sys;d=json.load(open(sys.argv[1]));print(d['tokens'],round(d['secs'],2),round(d['tokens']/d['secs'],1))" "$1"; }
LONG=$($PY -c "import json,sys;t=open(sys.argv[1]).read();print(json.dumps({'model':'x','max_tokens':1,'temperature':0,'messages':[{'role':'user','content':(t+'\n\n')*8+'Summarize.'}]}))" "$KL_TEXT")
bad(){ res "$1 INVALID: $2"; finish 1; }
# kl_gate NAME: when KL_BASELINE names a baseline TSV, judge kl_NAME.tsv against it with tools/kl.py (exit 0 PASS, 1 FAIL,
# 2 INVALID). Anything but PASS makes the arm INVALID (kl.py's one summary line says why); with no KL_BASELINE it is a no-op.
# Defense: the verdict comes from the one KL implementation in kl.py, so this script holds no thresholds of its own.
kl_gate(){ [ -n "${KL_BASELINE:-}" ] || return 0
  local line; line=$($PY "$TOOLS/kl.py" --baseline "$KL_BASELINE" --arm "$R/kl_$1.tsv" ${KL_MAX:+--max-kl $KL_MAX} ${KL_MIN_TOP1:+--min-top1 $KL_MIN_TOP1})
  local rc=$?; res "$1 $line"; return $rc; }
EXE=$(jq -r .exe "$BASE_CFG")
# fd 3, not a pipe: a pipe would run the loop in a subshell (finish/exit would not end the run) and curl/python would eat stdin.
while IFS='|' read -r NAME MODE PAR ENVS JQF <&3; do
  NAME=$(echo $NAME); MODE=$(echo $MODE); PAR=$(echo $PAR); ENVS=$(echo $ENVS); JQF=$(echo $JQF)
  [ "$ENVS" = "-" ] && ENVS=; [ "$JQF" = "-" ] && JQF=.
  [ "$MODE" = kl ] && JQF="$JQF | .args += [\"--short-read\",\"1000\"]"
  jq --arg l "$R/engine_$NAME.log" --argjson p $PORT "($JQF) | .log=\$l | .port=\$p" "$BASE_CFG" > "$R/cfg_$NAME.json" || bad $NAME "jq"
  extra="STRATA_DECODE_TIMING=1"; [ "$MODE" = kl ] && extra="STRATA_LOGPOS=$R/kl_$NAME.tsv STRATA_LOGPOS_TOPK=32 STRATA_ROUTE_TAIL_SKIP=0"
  [ "$MODE" = hist ] && extra="STRATA_DECODE_TIMING=1 STRATA_ROUTE_HIST=$R/hist_$NAME"
  [ "$MODE" = prefill ] && extra="STRATA_DECODE_TIMING=1 STRATA_PREFILL_TIMING=1"
  nohelper=; case " $ENVS " in *" NOHELPER=1 "*) nohelper=1; ENVS=$(echo " $ENVS " | sed 's/ NOHELPER=1 / /');; esac
  t0=$(date +%s); VERDICT=VALID
  if [ -n "$nohelper" ]; then echo "no helper (stock single-card arm)" > "$R/helper_$NAME.log"
    SHM_SAVED=$SHM; SHM=; start_server $NAME "$R/cfg_$NAME.json" "$PAR" $extra $ENVS || bad $NAME "server did not come up"; SHM=$SHM_SAVED
  else start_helper $NAME || bad $NAME "helper did not come up ($(tail -1 $R/helper_$NAME.log))"
    start_server $NAME "$R/cfg_$NAME.json" "$PAR" $extra $ENVS || bad $NAME "server did not come up"; fi
  res "$NAME [$MODE par=$PAR env='$ENVS' jq='$JQF'] up in $(( $(date +%s)-t0 ))s; helper: $(grep -o 'holding.*GiB' $R/helper_$NAME.log | head -1)"
  case $MODE in
    speed) $PY "$TOOLS/greedy.py" "$URL" "$R/${NAME}_warm.json" --max-tokens 400 --timeout 900 || bad $NAME "warm-up"
      for N in 1 2 3; do [ -f "$R/ABORT" ] && bad $NAME "$(cat $R/ABORT)"
        $PY "$TOOLS/greedy.py" "$URL" "$R/${NAME}_r$N.json" --max-tokens 2000 --timeout 900 || bad $NAME "run $N"
        res "$NAME run $N tokens/secs/tok_s $(speed $R/${NAME}_r$N.json)"; done
      curl --noproxy '*' -s -m 900 "$URL" -H 'content-type: application/json' -d "$LONG" > /dev/null
      res "$NAME long prompt: $(grep -E 'strata serve: prompt' $R/engine_$NAME.log | tail -1 | grep -oE 'prompt [0-9]+ tokens.*read in [0-9]+ ms \([0-9.]+ tok/s\)')" ;;
    conc) for N in 1 2; do [ -f "$R/ABORT" ] && bad $NAME "$(cat $R/ABORT)"
        res "$NAME conc round $N: $($PY "$TOOLS/conc.py" "$URL" "$PAR" 1000 --timeout 1800 2>&1 | tr '\n' ' ' | cut -c1-400)"; done ;;
    prefill) for k in 2 8 32; do [ -f "$R/ABORT" ] && bad $NAME "$(cat $R/ABORT)"
        body=$($PY -c "import json,sys;t=open(sys.argv[1]).read();k=int(sys.argv[2]);print(json.dumps({'model':'x','max_tokens':1,'temperature':0,'messages':[{'role':'user','content':'Prompt-length probe %d.\n\n'%k+(t+'\n\n')*k+'Summarize.'}]}))" "$KL_TEXT" $k)
        curl --noproxy '*' -s -m 1800 "$URL" -H 'content-type: application/json' -d "$body" > /dev/null || bad $NAME "prompt x$k"
        res "$NAME prompt x$k: $(grep -E 'strata serve: prompt' $R/engine_$NAME.log | tail -1 | grep -oE 'prompt [0-9]+ tokens.*read in [0-9]+ ms \([0-9.]+ tok/s\)')"
        res "$NAME prefill timing x$k: $(grep -aiE 'prefill.*(timing|ms)' $R/engine_$NAME.log | tail -1 | cut -c1-300)"; done
      $PY "$TOOLS/greedy.py" "$URL" "$R/${NAME}_dec.json" --max-tokens 400 --timeout 900 || bad $NAME "decode"
      res "$NAME decode 400: $(speed $R/${NAME}_dec.json)" ;;
    hist) for k in sum fr code math; do [ -f "$R/ABORT" ] && bad $NAME "$(cat $R/ABORT)"
        body=$($PY -c "
import glob,json,sys
t=open(sys.argv[1]).read(); k=sys.argv[2]
code=''.join(open(f).read() for f in sorted(glob.glob(sys.argv[3]+'/*.py')))[:14000]
math=''.join('Compute %d * %d + %d.\n'%(100+7*i,300+13*i,i) for i in range(300))
u={'sum':(t+'\n\n')*3+'Summarize the above in five bullet points.',
   'fr':(t+'\n\n')*3+'Translate the first paragraph into French, then into German.',
   'code':code+'\n\nExplain what this code does, then write unit tests for it.',
   'math':math+'Solve each one, showing the steps.'}[k]
u='Request %s.\n\n'%k+u   # distinct first line: no prefix is reused, so every prompt is routed in full
print(json.dumps({'model':'x','max_tokens':600,'temperature':0,'messages':[{'role':'user','content':u}]}))" "$KL_TEXT" $k "$TOOLS")
        curl --noproxy '*' -s -m 1800 "$URL" -H 'content-type: application/json' -d "$body" > "$R/${NAME}_$k.json" || bad $NAME "hist prompt $k"
        res "$NAME hist $k: $(grep -E 'strata serve: prompt' $R/engine_$NAME.log | tail -1 | grep -oE 'prompt [0-9]+ tokens.*read in [0-9]+ ms \([0-9.]+ tok/s\)')"; done
      [ "$(ls "$R/hist_$NAME"/req-*.json 2>/dev/null | wc -l)" -ge 4 ] || bad $NAME "fewer than 4 route-hist files in $R/hist_$NAME (engine without STRATA_ROUTE_HIST?)"
      $PY "$TOOLS/route_hist_report.py" "$R/hist_$NAME" --log "$R/engine_$NAME.log" > "$R/report_$NAME.txt" 2>&1
      res "$NAME report: rc=$? $R/report_$NAME.txt" ;;
    kl) $PY "$TOOLS/send.py" "$URL" --text "$KL_TEXT" >> "$R/RESULT.txt" 2>&1 || bad $NAME "send"
      res "$NAME kl rows $(wc -l < $R/kl_$NAME.tsv)"; kl_gate $NAME || VERDICT="INVALID: KL gate (see KL-GATE line above)" ;;
  esac
  [ -f "$R/ABORT" ] && bad $NAME "$(cat $R/ABORT)"
  res "$NAME timing: $(grep 'strata decode timing' $R/engine_$NAME.log | tail -2 | cut -c1-260 | tr '\n' ' ')"
  res "$NAME helper: $(grep 'seabraid helper:' $R/engine_$NAME.log | tail -2 | cut -c1-200 | tr '\n' ' ')"
  res "$NAME $VERDICT"; [ "$VERDICT" = VALID ] || GATE_RC=1; stop_all
done 3< <(grep -vE '^\s*(#|$)' "$ARMS")
finish $GATE_RC
