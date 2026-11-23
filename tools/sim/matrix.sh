#!/usr/bin/env bash
# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
# Sweeps seabraid_sim over a configuration grid and prints one JSON line per run. WHY bash: it only loops over
# flags; every number and every rule comes from the simulator. Exits nonzero if any run fails (the failing line is still
# printed); exits 64 (usage) for a grid the simulator would refuse, before any cell runs.
# Usage: matrix.sh [--bin PATH-or-BUILD-DIR] [--quick] [--hidden "a b"] [--experts "a b"] [--top-k "a b"] [--tokens "a b"]
#                  [--devices "a b"] [--kernel "a b"] [--steps N] [--layers N] [--extra "sim flags"]
# Every grid value is a plain decimal integer (no sign, no leading zeros) within the simulator's own caps, which this
# script reads with `seabraid_sim --print-caps` and never copies; --steps and --layers take exactly one.
set -u
# WHY noglob: `--extra` and the grid lists are split into words on purpose (`for word in $extra`, `$extra` in the command
# line), and an unquoted expansion is also pathname-expanded: `--extra "*"` became the file names of the current directory
# (a file called --print-caps then reached the simulator). Nothing here wants a glob; every path is spelled out.
set -f
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# WHY a search: the documented build dir is build/seabraid (README, CONTRIBUTING), so the simulator is
# build/seabraid/tools/sim/seabraid_sim; --bin and SEABRAID_SIM may name the binary or any build dir that holds it.
find_sim() {
    if [ -d "$1" ]; then
        for c in "$1/tools/sim/seabraid_sim" "$1/seabraid_sim" "$1/seabraid/tools/sim/seabraid_sim"; do
            [ -f "$c" ] && [ -x "$c" ] && { echo "$c"; return; }
        done
    fi
    echo "$1"
}
bin=$(find_sim "${SEABRAID_SIM:-$here/../../build/seabraid}")
hidden="512 2048 4096"; experts="64 128"; topk="2 8"; tokens="1 16 128"; devices="1 2 4 8"
kernel="scalar unrolled"; steps=20; layers=""; extra=""

# WHY validated here: a grid value like "5x" or "" reached `[ "$d" -le "$e" ]`, whose error was swallowed by `||
# continue`, so the sweep printed "0 runs, 0 failed" and exited 0.
# WHY the caps are read from the simulator: a second copy here drifted silently (a too-low copy here would still pass a
# test that only checks the refusal text). The simulator is the one owner (sim_cli.c); this just asks it.
cap_of() {
    local value
    value=$(printf '%s\n' "$caps" | while read -r key number; do [ "$key" = "$1" ] && echo "$number"; done)
    [ -n "$value" ] || { echo "matrix.sh: the simulator reported no cap for $1 (--print-caps)" >&2; exit 64; }
    echo "$value"
}
# plain_integer <flag> <word> <cap> <minimum>: prints the word, or says why not and returns 1.
# WHY no leading zeros or padding: "00000000008" is eleven characters and used to be called "above the cap", and " 2"
# passed here and then failed every cell inside the simulator; both are now refused with their own message.
plain_integer() {
    local flag=$1 word=$2 cap=$3 min=$4
    case "$word" in ''|*[!0-9]*) echo "matrix.sh: $flag value is not a non-negative integer: '$word'" >&2; return 1 ;; esac
    if [ ${#word} -gt 1 ] && [ "${word#0}" != "$word" ]; then
        echo "matrix.sh: $flag value has leading zeros (write it as a plain decimal integer): $word" >&2; return 1
    fi
    { [ ${#word} -le 10 ] && [ "$word" -le "$cap" ]; } ||
        { echo "matrix.sh: $flag value $word is above the simulator's cap of $cap" >&2; return 1; }
    [ "$word" -ge "$min" ] || { echo "matrix.sh: $flag value $word must be at least $min" >&2; return 1; }
}
need_integers() {
    local flag=$1 value=$2 cap=$3 min=$4 word
    [ -n "${value//[[:space:]]/}" ] || { echo "matrix.sh: $flag needs at least one integer" >&2; exit 64; }
    for word in $value; do plain_integer "$flag" "$word" "$cap" "$min" || exit 64; done
}
# One value, exactly as typed: a list, or a value with spaces around it, is not a single integer.
need_one() {
    local flag=$1 value=$2 cap=$3 min=$4
    case "$value" in *[[:space:]]*) echo "matrix.sh: $flag takes a single integer, not a list or a padded value: '$value'" >&2; exit 64 ;; esac
    plain_integer "$flag" "$value" "$cap" "$min" || exit 64
}

while [ $# -gt 0 ]; do
    # WHY this check: every value flag shifts 2, and `shift 2` with one argument left fails without consuming anything,
    # so a trailing `--bin` used to loop forever. It names the flag and exits 64 (usage); bash 3.2 safe (no `;;&`).
    case "$1" in
        --bin|--hidden|--experts|--top-k|--tokens|--devices|--kernel|--steps|--layers|--extra)
            [ $# -ge 2 ] || { echo "matrix.sh: $1 needs a value" >&2; exit 64; } ;;
    esac
    case "$1" in
        --quick) hidden="512"; experts="64"; topk="2 8"; tokens="1 16"; devices="1 4"; steps=8; shift ;;
        --bin) bin=$(find_sim "$2"); shift 2 ;;
        --hidden) hidden=$2; shift 2 ;;
        --experts) experts=$2; shift 2 ;;
        --top-k) topk=$2; shift 2 ;;
        --tokens) tokens=$2; shift 2 ;;
        --devices) devices=$2; shift 2 ;;
        --kernel) kernel=$2; shift 2 ;;
        --steps) steps=$2; shift 2 ;;
        --layers) layers=$2; shift 2 ;;
        --extra) extra=$2; shift 2 ;;
        *) echo "matrix.sh: unknown flag $1" >&2; exit 64 ;;
    esac
done
# WHY these words are refused in --extra: each makes the simulator print no run line and exit 0 (--validate-only checks
# and stops, --help and --print-caps print text), so every cell "passed" having run nothing: "16 runs, 0 failed", exit 0.
for word in $extra; do
    case "$word" in
        --validate-only|--help|-h|--print-caps)
            echo "matrix.sh: --extra may not contain $word: the simulator would print no run line and every cell would pass unrun" >&2
            exit 64 ;;
    esac
done
# WHY -f: `[ -x ]` is true for a directory, so `--bin some/dir` passed here and failed on every cell later.
{ [ -f "$bin" ] && [ -x "$bin" ]; } || { echo "matrix.sh: simulator is not an executable regular file: $bin (build with -DSEABRAID_BUILD_TOOLS=ON, or pass --bin)" >&2; exit 64; }
caps=$("$bin" --print-caps) || { echo "matrix.sh: $bin --print-caps failed (is it seabraid_sim?)" >&2; exit 64; }
need_integers --hidden "$hidden" "$(cap_of hidden)" 1; need_integers --experts "$experts" "$(cap_of experts)" 1
need_integers --top-k "$topk" "$(cap_of top-k)" 1; need_integers --tokens "$tokens" "$(cap_of tokens)" 1
need_integers --devices "$devices" "$(cap_of devices)" 1; need_one --steps "$steps" "$(cap_of steps)" 1
if [ -n "$layers" ]; then need_one --layers "$layers" "$(cap_of layers)" 1; fi
[ -n "${kernel//[[:space:]]/}" ] || { echo "matrix.sh: --kernel needs at least one name" >&2; exit 64; }

# Walks every runnable cell of the grid and calls "$1 h e k t d kernel" for each. A device must own at least one expert
# and top-k cannot exceed the experts; those cells are skipped, never run.
walk() {
    local fn=$1 h e k t d kn
    for h in $hidden; do for e in $experts; do for k in $topk; do for t in $tokens; do
    for d in $devices; do for kn in $kernel; do
        [ "$d" -le "$e" ] && [ "$k" -le "$e" ] || continue
        "$fn" "$h" "$e" "$k" "$t" "$d" "$kn"
    done; done; done; done; done; done
}
# WHY a dry run of every cell through the simulator's own validation (--validate-only): the kernel list and the joint
# limits (verifiable shape, combined memory) belong to the simulator, and a refused cell used to be found only when its
# turn came, after earlier cells had run for minutes. The first refusal ends the sweep with the simulator's own words.
# shellcheck disable=SC2086
validate_cell() {
    local out
    out=$("$bin" --validate-only --hidden "$1" --experts "$2" --top-k "$3" --tokens "$4" --devices "$5" --kernel "$6" \
        --steps "$steps" ${layers:+--layers "$layers"} $extra 2>&1) ||
        { echo "matrix.sh: the simulator refuses h=$1 e=$2 k=$3 t=$4 d=$5 $6: ${out#seabraid_sim: }" >&2; exit 64; }
}
runs=0; bad=0
# shellcheck disable=SC2086
run_cell() {
    local line rc
    runs=$((runs + 1))
    line=$("$bin" --hidden "$1" --experts "$2" --top-k "$3" --tokens "$4" --devices "$5" --kernel "$6" \
        --steps "$steps" ${layers:+--layers "$layers"} $extra); rc=$?
    printf '%s\n' "$line"
    if [ $rc -ne 0 ]; then bad=$((bad + 1)); echo "matrix.sh: exit $rc for h=$1 e=$2 k=$3 t=$4 d=$5 $6" >&2
    else
        # WHY a second test: exit 0 does not mean the cell ran; a run prints exactly one JSON line with an "ok" member.
        case "$line" in *'"ok":'*) ;; *) bad=$((bad + 1)); echo "matrix.sh: no \"ok\": run line for h=$1 e=$2 k=$3 t=$4 d=$5 $6 (counted as failed)" >&2 ;; esac
    fi
}
# Every kernel name first, on a minimal shape: a misspelt kernel is refused even when no cell would have run it.
for kn in $kernel; do
    out=$("$bin" --validate-only --kernel "$kn" 2>&1) || { echo "matrix.sh: ${out#seabraid_sim: }" >&2; exit 64; }
done
walk validate_cell
walk run_cell
echo "matrix.sh: $runs runs, $bad failed" >&2
# WHY: a grid whose every cell is impossible (more devices than experts, top-k above experts) ran nothing and must not pass.
[ $runs -gt 0 ] || { echo "matrix.sh: no runnable cell in the grid (each needs devices <= experts and top-k <= experts)" >&2; exit 64; }
[ $bad -eq 0 ]
