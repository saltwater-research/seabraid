#!/usr/bin/env bash
# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
# Command-line contract checks for seabraid_sim and matrix.sh: each case names the exit code and a message fragment that
# must appear. WHY a script: these are refusals (exit != 0), which check_output.cmake cannot assert.
# Usage: check_cli.sh <seabraid_sim> <matrix.sh>
set -u
sim=$1; matrix=$2   # matrix.sh is run through bash: the file is not committed executable
bad=0
# Bounds a hang (the old `matrix.sh --bin` looped forever). WHY CPU seconds and not wall seconds: the latency-log probe
# below is 500000 submit/collect hand-offs between two threads, about 4.6 CPU seconds however the box is loaded but 11 s
# of wall time idle and 24 s with eight busy loops on its cores (measured 2026-11-20), so a 20 s wall timeout killed it
# whenever the machine was busy and the case then read "estimated none bytes". A runaway loop burns CPU and is still
# stopped at 60 CPU seconds (13 times the slowest case here); the 150 s wall `timeout` is only the backstop for a hang
# that burns none, and ctest's own TIMEOUT backs both up. Without `ulimit -t` or `timeout` the bound just drops away.
limit() { ( ulimit -t 60 2>/dev/null; if command -v timeout >/dev/null 2>&1; then exec timeout 150 "$@"; else exec "$@"; fi ); }
# expect <name> <exit code> <fragment> <command...>
expect() {
    local name=$1 want=$2 fragment=$3 out rc
    shift 3
    out=$(limit "$@" 2>&1); rc=$?
    if [ "$rc" -ne "$want" ] || ! printf '%s' "$out" | grep -qF -- "$fragment"; then
        echo "FAIL $name: exit $rc (want $want), wanted \"$fragment\" in: $out" >&2
        bad=$((bad + 1))
    else
        echo "ok   $name"
    fi
}
# The out-of-memory case runs the simulator through `bash -c 'ulimit -v 1500000; exec "$@"'`: a 1.5 GB address-space
# limit in a child shell, so the allocation failure is real and cannot touch the host's memory.

expect "warmup+steps wrap refused"   1 "at most 1000000" "$sim" --warmup 4294967295 --steps 1
expect "seed leading space + minus"  1 "bad value"       "$sim" --seed " -1"
expect "seed plus sign"              1 "bad value"       "$sim" --seed "+5"
expect "hidden leading space"        1 "bad value"       "$sim" --hidden " 8"
expect "fail-rate leading space"     1 "bad value"       "$sim" --fail-rate " 0.5"
expect "fail-rate nan"               1 "bad value"       "$sim" --fail-rate nan
# --fail-rate is decimal only: a leading digit used to let C99 hex floats through strtod.
expect "fail-rate hex fraction"      1 "bad value"       "$sim" --fail-rate 0x0.8
expect "fail-rate hex exponent"      1 "bad value"       "$sim" --fail-rate 0X1p-1
expect "fail-rate hex integer"       1 "bad value"       "$sim" --fail-rate 0x1
expect "fail-rate decimal exponent"  0 '"ok":true'       "$sim" --fail-rate 0e0 --steps 2 --warmup 0
# Caps: every per-flag cap comes from `seabraid_sim --print-caps` (the one owner, sim_cli.c); each value one past it is
# refused by the simulator and, for the grid flags, by matrix.sh before any cell runs, naming the same number.
cap_of() { "$sim" --print-caps | awk -v key="$1" '$1 == key { print $2 }'; }
for flag in hidden experts top-k tokens devices layers steps warmup placement-rounds; do
    cap=$(cap_of "$flag")
    if [ -z "$cap" ]; then echo "FAIL --print-caps has no line for $flag" >&2; bad=$((bad + 1)); continue; fi
    expect "$flag cap ($cap) refused" 1 "--$flag must be at most $cap" "$sim" "--$flag" $((cap + 1))
done
expect "nothing verified is a fail"  2 "no layer was verified" \
    "$sim" --layers 1 --steps 3 --warmup 0 --fail-rate 0.99999999
expect "fail-rate 1 is allowed"      0 '"verified_layers":0' "$sim" --layers 1 --steps 3 --warmup 0 --fail-rate 1
# WHY two probes before the out-of-memory case: it allocates for real, so it runs only where `ulimit -v` demonstrably holds.
# (1) A sanitizer build reserves terabytes of shadow address space and dies under `ulimit -v` before main, and
# `ldd | grep libasan` misses clang, -static-libasan and macOS: look for the runtime's init symbol with nm (any link
# style) or for the dylib with otool -L (macOS), and treat an unreadable binary as sanitized, never as clean.
sanitized() {
    command -v nm >/dev/null 2>&1 || command -v otool >/dev/null 2>&1 || return 0 # nothing to look with: skip, never allocate blind
    if command -v nm >/dev/null 2>&1; then
        nm "$sim" 2>/dev/null | grep -Eq '__(asan|tsan|msan)_init' && return 0
        nm -D "$sim" 2>/dev/null | grep -Eq '__(asan|tsan|msan)_init' && return 0
    fi
    if command -v otool >/dev/null 2>&1; then
        otool -L "$sim" 2>/dev/null | grep -Eqi 'asan|tsan|msan' && return 0
    fi
    if command -v ldd >/dev/null 2>&1; then
        ldd "$sim" 2>/dev/null | grep -Eq 'lib(asan|tsan|msan)' && return 0
    fi
    return 1
}
# (2) Some systems accept `ulimit -v` and never enforce it (macOS). Probe: under a 256 MB limit a 256 MB string must fail
# to build, and a trivial awk must still run (so the failure is the limit, not awk). The probe's own allocation is
# bounded (it doubles a string to 2^28 bytes, under 1 GB peak even when the limit is ignored).
ulimit_enforced() {
    command -v awk >/dev/null 2>&1 || return 1
    bash -c 'ulimit -v 262144 2>/dev/null || exit 99; awk "BEGIN{print 1}" >/dev/null' || return 1
    ! bash -c 'ulimit -v 262144 2>/dev/null || exit 99; exec awk "BEGIN{s=\"x\"; for(i=0;i<28;i++) s=s s; print length(s)}"' >/dev/null 2>&1
}
if sanitized; then
    echo "skip out of memory is reported (sanitizer build or none detectable: ulimit -v is incompatible)"
elif ! ulimit_enforced; then
    echo "skip out of memory is reported (ulimit -v cannot be set or is not enforced here)"
else
    expect "out of memory is reported"   1 "out of memory" \
        bash -c 'ulimit -v 1500000 || exit 99; exec "$@"' _ "$sim" --hidden 1048576 --tokens 64
fi
expect "a large expert count passes" 0 '"ok":true' \
    "$sim" --experts 4096 --hidden 64 --tokens 8 --steps 4 --warmup 1
# The verifier must fail wrong results, not only pass right ones. nan, inf and off_by_one are kernels that are wrong on
# purpose (sim_providers.c): a NaN or infinite output, and an expert multiplier of id + 2 instead of id + 1.
small="--hidden 16 --tokens 4 --steps 2 --warmup 0 --layers 1"
for kernel in nan inf off_by_one; do
    # shellcheck disable=SC2086
    expect "kernel $kernel fails the run"   2 '"ok":false' "$sim" --kernel $kernel --experts 262144 $small
    # shellcheck disable=SC2086
    expect "kernel $kernel fails at 64"     2 '"ok":false' "$sim" --kernel $kernel --experts 64 $small
done
# shellcheck disable=SC2086
expect "nan on 2 devices fails"      2 '"ok":false' "$sim" --kernel nan --experts 64 --devices 2 $small
# The float32 rounding bound can see the off-by-one multiplier only while a JOINT limit holds (sim_cli.c,
# verifiable_problem): (experts + 1) * (top-k + devices + 3) * sqrt(top-k) <= 2^22. That leaves 838859 experts at most
# (top-k 1, one device); the --experts cap itself (1048576) is past it for every top-k and device count.
# shellcheck disable=SC2086
expect "off_by_one fails at the largest verifiable expert count" 2 '"ok":false' "$sim" --kernel off_by_one --experts 838859 --top-k 1 $small
# shellcheck disable=SC2086
expect "refused: the experts cap itself cannot be verified" 1 "cannot be verified" "$sim" --experts 1048576 --top-k 1 $small
# shellcheck disable=SC2086
expect "refused: one expert past the largest verifiable count" 1 "cannot be verified" "$sim" --experts 838860 --top-k 1 $small
for experts in 64 1024 65536 262144; do
    # shellcheck disable=SC2086
    expect "a real run passes at $experts experts" 0 '"ok":true' "$sim" --experts $experts $small
    # shellcheck disable=SC2086
    expect "a real unrolled run passes at $experts experts" 0 '"ok":true' "$sim" --kernel unrolled --experts $experts $small
done
# shellcheck disable=SC2086
expect "a real run passes at the largest verifiable expert count" 0 '"ok":true' "$sim" --experts 838859 --top-k 1 $small
# shellcheck disable=SC2086
expect "real run, 8 picks on 4 devices" 0 '"ok":true' "$sim" --experts 65536 --top-k 8 --devices 4 $small
# The three shapes where the old bound let off_by_one through are now refused as unverifiable (it used to print ok:true).
# shellcheck disable=SC2086
expect "refused: 1048576 experts, top-k 32" 1 "cannot be verified" "$sim" --kernel off_by_one --experts 1048576 --top-k 32 $small
# shellcheck disable=SC2086
expect "refused: 8192 experts, top-k 4096"  1 "cannot be verified" "$sim" --kernel off_by_one --experts 8192 --top-k 4096 $small
# shellcheck disable=SC2086
expect "refused: 262144 experts, top-k 64, 8 devices" 1 "cannot be verified" \
    "$sim" --kernel off_by_one --experts 262144 --top-k 64 --devices 8 $small
# shellcheck disable=SC2086
expect "the refusal names the joint limit" 1 "(experts + 1) * (top-k + devices + 3) * sqrt(top-k) at most 4194304" \
    "$sim" --experts 1048576 --top-k 32 $small
# shellcheck disable=SC2086
expect "a real run is refused too (it could not be verified either)" 1 "cannot be verified" \
    "$sim" --experts 1048576 --top-k 32 $small

# SWEEP (experts x top-k x devices): on every shape the simulator accepts, off_by_one must fail with exit 2 and the real
# kernel must pass; every other shape must be refused. The sweep is the checker for verifiable_problem; seeds vary the data.
sweep_accepted=0; sweep_refused=0; sweep_bad=0
one="--hidden 1 --tokens 1 --layers 1 --steps 1 --warmup 0"
for experts in 64 1024 16384 65536 262144 1048576; do
for topk in 1 2 8 64 512 4096; do
for devs in 1 2 8 64; do
    [ "$topk" -le "$experts" ] && [ "$devs" -le "$experts" ] || continue
    if "$sim" --validate-only --experts $experts --top-k $topk --devices $devs >/dev/null 2>&1; then
        sweep_accepted=$((sweep_accepted + 1))
        # Each shape is run on the 64-element run AND on a ONE-element run (hidden = tokens = layers = steps = 1): with one
        # element nothing averages out, which is where a margin of 2 let off_by_one through 7 times in 1000 seeds.
        for shape in "$small" "$one"; do
        for seed in 1 2 3; do
            # shellcheck disable=SC2086
            "$sim" $shape --experts $experts --top-k $topk --devices $devs --seed $seed --kernel off_by_one >/dev/null 2>&1; wrong=$?
            # shellcheck disable=SC2086
            "$sim" $shape --experts $experts --top-k $topk --devices $devs --seed $seed >/dev/null 2>&1; right=$?
            if [ "$wrong" -ne 2 ] || [ "$right" -ne 0 ]; then
                echo "FAIL sweep ($shape) experts=$experts top-k=$topk devices=$devs seed=$seed: off_by_one exit $wrong (want 2), real exit $right (want 0)" >&2
                sweep_bad=$((sweep_bad + 1))
            fi
        done; done
    else
        sweep_refused=$((sweep_refused + 1))
    fi
done; done; done
if [ "$sweep_bad" -eq 0 ] && [ "$sweep_accepted" -gt 0 ] && [ "$sweep_refused" -gt 0 ]; then
    echo "ok   sweep: $sweep_accepted shapes accepted (off_by_one fails, real passes; 64 elements and 1 element), $sweep_refused refused as unverifiable"
else
    echo "FAIL sweep: accepted $sweep_accepted, refused $sweep_refused, bad $sweep_bad" >&2
    bad=$((bad + 1))
fi
# The EDGE of the limit, one element, 100 seeds each: the shapes where cost is within a few units of 2^22, top-k 1, 2 and
# 8 on 1 to 64 devices (experts = the largest count the limit accepts). At 2^23 the top-k 1 edges missed 2 to 7 in 1000.
edge_bad=0
for shape in "838859 1 1" "524287 1 4" "349524 1 8" "209714 1 16" "61679 1 64" "494302 2 1" "228139 2 8" "19771 8 64"; do
    set -- $shape
    "$sim" --validate-only --experts "$1" --top-k "$2" --devices "$3" >/dev/null 2>&1 || { echo "FAIL edge $shape is refused" >&2; edge_bad=$((edge_bad + 1)); continue; }
    for seed in $(seq 1 100); do
        # shellcheck disable=SC2086
        "$sim" $one --experts "$1" --top-k "$2" --devices "$3" --seed $seed --kernel off_by_one >/dev/null 2>&1; wrong=$?
        [ "$wrong" -eq 2 ] || { echo "FAIL edge experts=$1 top-k=$2 devices=$3 seed=$seed: off_by_one exit $wrong (want 2)" >&2; edge_bad=$((edge_bad + 1)); }
    done
done
if [ "$edge_bad" -eq 0 ]; then echo "ok   edge of the joint limit: off_by_one fails on one element, 8 shapes x 100 seeds"; else bad=$((bad + 1)); fi
expect "refused: just past the top-k 1 edge (4 devices)" 1 "cannot be verified" "$sim" --experts 524288 --top-k 1 --devices 4

# Combined memory cap: each flag is within its cap and the product is not.
expect "memory: tokens x hidden"  1 "combined limit" "$sim" --tokens 1048576 --hidden 4096
expect "memory: layers x steps"   1 "combined limit" "$sim" --layers 65536 --steps 1000000
expect "memory: devices x cells"  1 "combined limit" "$sim" --tokens 1048576 --hidden 1024 --devices 1024 --experts 1024
# Combined work cap: every flag inside its cap, and the run would never finish. --validate-only must refuse each of
# these (it is what matrix.sh asks), and a shape just inside the limit must be accepted.
expect "work: warmup x layers"    1 "the run would take" "$sim" --validate-only --warmup 1000000 --layers 65536
expect "work: steps x layers x cells" 1 "the run would take" "$sim" --validate-only --steps 100000 --layers 1000
expect "work: placement rounds x cells" 1 "the run would take" \
    "$sim" --validate-only --placement-rounds 100000 --layers 1000 --experts 4096
expect "work: rounds cap is refused alone" 1 "--placement-rounds must be at most 100000" "$sim" --placement-rounds 4294967295
expect "work: a long but finite run is accepted" 0 "accepted" \
    bash -c '"$@" && echo accepted' _ "$sim" --validate-only --warmup 1000000 --steps 1000 --layers 4
expect "work: --print-caps lists work" 0 "work 600000000000" "$sim" --print-caps
# R6b #17: the pick term. This shape was accepted (and took about an hour, against a ten-minute cap) while the estimate
# counted no per-pick cost; a tenth of its layers fits the cap, and so does a shape with the same picks on more devices.
expect "work: the hour-long picks-heavy shape is refused" 1 "the run would take" \
    "$sim" --validate-only --hidden 1 --tokens 262144 --experts 4096 --top-k 64 --devices 1 --layers 1000 --steps 15 --warmup 1
expect "work: a tenth of its layers is accepted" 0 "accepted" \
    bash -c '"$@" && echo accepted' _ "$sim" --validate-only --hidden 1 --tokens 262144 --experts 4096 --top-k 64 --devices 1 --layers 50 --steps 15 --warmup 1
expect "work: the same picks on 8 devices are refused" 1 "the run would take" \
    "$sim" --validate-only --hidden 1 --tokens 65536 --experts 4096 --top-k 64 --devices 8 --layers 1000 --steps 15 --warmup 1
# The memory estimate is a BOUND on peak RSS (sim_estimate_bytes): run a handful of shapes under GNU time and require the
# JSON line's estimated_bytes to be at least the measured peak. Skipped (said so) without /usr/bin/time -f, and on a sanitizer
# build, whose shadow memory inflates RSS. Sequential, and the largest shape stays near 0.6 GB. --compact-rows has its own
# shapes: its per-cell copy is a term of its own, and the 1-device run was the one that measured over the estimate (R6b #19).
rss_probe=$(mktemp)
if sanitized; then
    echo "skip memory estimate bounds peak RSS (sanitizer build or none detectable: RSS is inflated)"
elif ! { [ -x /usr/bin/time ] && /usr/bin/time -f %M -o "$rss_probe" true >/dev/null 2>&1 && [ -s "$rss_probe" ]; }; then
    echo "skip memory estimate bounds peak RSS (no GNU /usr/bin/time -f here)"
else
    rss_bad=0
    # label|flags: a one-cell run, the activation and per-device terms, the pick term, the latency log and the placement buffers.
    for shape in "one cell|--hidden 1 --tokens 1 --experts 8 --top-k 1" \
                 "64 devices|--hidden 1 --tokens 1 --experts 64 --top-k 1 --devices 64" \
                 "cells x 4 devices|--hidden 4096 --tokens 1024 --experts 64 --top-k 2 --devices 4" \
                 "cells x 64 devices|--hidden 512 --tokens 256 --experts 128 --top-k 2 --devices 64" \
                 "cells x 1 device|--hidden 4096 --tokens 2048 --experts 64 --top-k 2 --devices 1" \
                 "compact rows x 1 device|--hidden 4096 --tokens 2048 --experts 64 --top-k 2 --devices 1 --compact-rows" \
                 "compact rows x 4 devices|--hidden 4096 --tokens 1024 --experts 64 --top-k 2 --devices 4 --compact-rows" \
                 "picks x 8 devices|--hidden 1 --tokens 65536 --experts 4096 --top-k 64 --devices 8" \
                 "latency log|--hidden 1 --tokens 1 --experts 2 --top-k 1 --layers 100 --steps 5000 --warmup 0" \
                 "placement|--hidden 1 --tokens 1 --experts 4096 --top-k 1 --layers 1024 --placement-rounds 2 --devices 4"; do
        label=${shape%%|*}; flags=${shape#*|}
        # shellcheck disable=SC2086
        json=$(limit /usr/bin/time -f %M -o "$rss_probe" "$sim" --steps 1 --warmup 0 --layers 1 $flags 2>/dev/null)
        peak_kb=$(tr -d '[:space:]' < "$rss_probe")
        estimate=$(printf '%s' "$json" | sed -n 's/.*"estimated_bytes":\([0-9]*\).*/\1/p')
        if [ -z "$estimate" ] || [ -z "$peak_kb" ] || [ "$estimate" -lt $((peak_kb * 1024)) ]; then
            echo "FAIL memory estimate ($label): estimated ${estimate:-none} bytes, peak RSS ${peak_kb:-none} KiB" \
                "(a run killed by the CPU or wall bound leaves both empty; time said: $(head -n 1 "$rss_probe"))" >&2
            rss_bad=$((rss_bad + 1))
        else
            echo "ok   memory estimate ($label): ${estimate} bytes >= peak RSS $((peak_kb * 1024)) bytes ($((estimate * 100 / (peak_kb * 1024))) percent)"
        fi
    done
    [ "$rss_bad" -eq 0 ] || bad=$((bad + 1))
fi
rm -f "$rss_probe"
# One owner of the kernel list: an unknown name is refused with the table's own list, and --help prints the same table.
expect "unknown kernel"           1 "--kernel must be one of: scalar unrolled nan inf off_by_one" "$sim" --kernel bogus
expect "help lists the kernels"   0 "off_by_one" "$sim" --help
expect "--print-caps lists bytes" 0 "bytes 4294967296" "$sim" --print-caps

expect "matrix: --bin without value" 64 "--bin needs a value" bash "$matrix" --bin
for flag in --hidden --experts --top-k --tokens --devices --kernel --steps --layers --extra; do
    expect "matrix: $flag without value" 64 "$flag needs a value" bash "$matrix" --bin "$sim" "$flag"
done
expect "matrix: zero runs fails" 64 "no runnable cell" bash "$matrix" --bin "$sim" --devices 64 --experts 2 --top-k 1
expect "matrix: non-integer grid value" 64 "not a non-negative integer" bash "$matrix" --bin "$sim" --hidden 5x
expect "matrix: empty grid value" 64 "needs at least one integer" bash "$matrix" --bin "$sim" --experts ""
expect "matrix: --bin is a directory" 64 "not an executable regular file" bash "$matrix" --bin "$(dirname "$matrix")"
expect "matrix: --steps takes one value" 64 "takes a single integer" bash "$matrix" --bin "$sim" --steps "2 3"
expect "matrix: --layers takes one value" 64 "takes a single integer" bash "$matrix" --bin "$sim" --layers "1 2"
expect "matrix: 20-digit value"      64 "above the simulator's cap" bash "$matrix" --bin "$sim" --tokens 99999999999999999999
# Each cap + 1 is refused up front with the number the simulator itself reports (--print-caps): matrix.sh keeps no copy.
for flag in hidden experts top-k tokens devices; do
    cap=$(cap_of "$flag")
    expect "matrix: $flag cap"   64 "cap of $cap" bash "$matrix" --bin "$sim" "--$flag" "8 $((cap + 1))"
done
for flag in steps layers; do
    cap=$(cap_of "$flag")
    expect "matrix: $flag cap"   64 "cap of $cap" bash "$matrix" --bin "$sim" "--$flag" $((cap + 1))
done
expect "matrix: --steps 0"       64 "must be at least 1" bash "$matrix" --bin "$sim" --steps 0
expect "matrix: --hidden 0"      64 "must be at least 1" bash "$matrix" --bin "$sim" --hidden 0
expect "matrix: --steps padded"  64 "takes a single integer, not a list or a padded value" bash "$matrix" --bin "$sim" --steps " 2"
expect "matrix: --steps zero-padded" 64 "leading zeros" bash "$matrix" --bin "$sim" --steps 02
expect "matrix: 11-char zero-padded value" 64 "leading zeros" bash "$matrix" --bin "$sim" --tokens 00000000008
expect "matrix: unknown kernel is refused up front" 64 "--kernel must be one of: scalar unrolled nan inf off_by_one" \
    bash "$matrix" --bin "$sim" --kernel "scalar bogus" --hidden 16 --experts 8 --top-k 2 --tokens 2 --devices 1
expect "matrix: an unverifiable cell is refused up front" 64 "cannot be verified" \
    bash "$matrix" --bin "$sim" --hidden 16 --experts 1048576 --top-k 32 --tokens 2 --devices 1 --kernel scalar
expect "matrix: a cell over the memory cap is refused up front" 64 "combined limit" \
    bash "$matrix" --bin "$sim" --hidden 4096 --experts 64 --top-k 2 --tokens 1048576 --devices 1 --kernel scalar
expect "matrix: --layers works"      0 '"layers":2'      bash "$matrix" --bin "$sim" --hidden 16 --experts 8 --top-k 2 \
    --tokens 2 --devices 1 --kernel scalar --steps 2 --layers 2
# --extra words after which the simulator prints no run line (exit 0): every cell would "pass" having run nothing.
for word in --validate-only --help -h --print-caps; do
    expect "matrix: --extra $word is refused" 64 "--extra may not contain $word" \
        bash "$matrix" --bin "$sim" --hidden 16 --experts 8 --top-k 2 --tokens 2 --devices 1 --kernel scalar --extra "--seed 3 $word"
done
# A cell whose simulator exits 0 but prints no "ok" run line counts as failed. The stand-in answers --print-caps and
# --validate-only like the real one (so matrix.sh gets past its checks) and runs nothing: exit 0, no output.
fake=$(mktemp)
cat > "$fake" <<EOF
#!/bin/sh
[ "\$1" = --print-caps ] && exec "$sim" --print-caps
exit 0
EOF
chmod +x "$fake"
expect "matrix: a cell with no run line is a failure" 1 "no \"ok\": run line" \
    bash "$matrix" --bin "$fake" --hidden 16 --experts 8 --top-k 2 --tokens 2 --devices 1 --kernel scalar --steps 2
expect "matrix: ... and the summary counts it" 1 "1 runs, 1 failed" \
    bash "$matrix" --bin "$fake" --hidden 16 --experts 8 --top-k 2 --tokens 2 --devices 1 --kernel scalar --steps 2
rm -f "$fake"
# R6b #22: --extra is split into words on purpose, never pathname-expanded. In a directory holding a file called
# --print-caps, an unquoted `*` became that word (refused by the --extra check) instead of reaching the simulator as a
# literal `*`, which the simulator refuses by its own name.
absolute() { printf '%s/%s' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
glob_dir=$(mktemp -d)
: > "$glob_dir/--print-caps"
expect "matrix: --extra '*' is not glob-expanded" 64 "matrix.sh: the simulator refuses" \
    bash -c 'cd "$1" && shift && exec bash "$@"' _ "$glob_dir" "$(absolute "$matrix")" --bin "$(absolute "$sim")" \
    --hidden 16 --experts 8 --top-k 2 --tokens 2 --devices 1 --kernel scalar --extra "*"
rm -rf "$glob_dir"
expect "matrix: unknown flag" 64 "unknown flag" bash "$matrix" --bin "$sim" --frobnicate 1
expect "matrix: a real sweep passes" 0 '"ok":true' bash "$matrix" --bin "$sim" --hidden 64 --experts 8 --top-k 2 \
    --tokens 2 --devices 2 --kernel scalar --steps 2

[ "$bad" -eq 0 ] || { echo "check_cli.sh: $bad case(s) failed" >&2; exit 1; }
echo "check_cli.sh: all cases passed"
