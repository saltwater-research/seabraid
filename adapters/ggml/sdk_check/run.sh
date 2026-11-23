#!/bin/sh
# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
# Standalone SDK link check for adapters/ggml, offline: install the root project to a prefix, configure the
# adapter (tests OFF) against that prefix with the engine's ggml, link one program, run it.
# Usage: adapters/ggml/sdk_check/run.sh <ggml source dir> [work dir]
# WHY a script and not a ctest entry: the check needs the whole root project installed first, which no in-tree
# CTest target may do to itself; CI runs this script in the sdk job. Needs cmake, cargo (the root build), a C compiler.
set -eu
GGML_SOURCE=${1:?usage: run.sh <ggml source dir> [work dir]}
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
WORK=${2:-"$ROOT/build-ggml-sdk-check"}
mkdir -p "$WORK"
cmake -S "$ROOT" -B "$WORK/sdk-build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$WORK/sdk-build" --parallel
cmake --install "$WORK/sdk-build" --prefix "$WORK/prefix"
cmake -S "$HERE" -B "$WORK/consumer" -DCMAKE_BUILD_TYPE=Release -DGGML_SOURCE="$GGML_SOURCE" \
    -DCMAKE_PREFIX_PATH="$WORK/prefix"
cmake --build "$WORK/consumer" --parallel --target seabraid_ggml_sdk_link
ctest --test-dir "$WORK/consumer" --output-on-failure -R '^seabraid_ggml_sdk_link$'
# The adapter must also refuse loudly, not link silently, when no SDK is on the path.
if cmake -S "$HERE" -B "$WORK/consumer-nosdk" -DGGML_SOURCE="$GGML_SOURCE" >"$WORK/nosdk.log" 2>&1; then
    echo "FAIL: adapters/ggml configured without an installed SDK" >&2
    exit 1
fi
grep -q "seabraid_ggml needs the Seabraid runtime" "$WORK/nosdk.log" || {
    echo "FAIL: the missing-SDK refusal did not name the runtime" >&2
    exit 1
}
echo "adapters/ggml standalone SDK link check passed"
