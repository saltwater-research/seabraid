#!/bin/sh
# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
# Build and run the adapters/ggml fixtures (SEABRAID_GGML_TESTS=ON) against a ggml source tree, offline, CPU-only.
# Usage: scripts/ggml_fixtures/run.sh <ggml source dir> [work dir]
# WHY a script: the ci.yml `sdk` job and a local run use the same three commands; it fails if CTest finds no fixture,
# so a configure that silently dropped them cannot pass.  Needs cmake and a C and C++ compiler (no cargo).
set -eu
GGML_SOURCE=${1:?usage: run.sh <ggml source dir> [work dir]}
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK=${2:-"$ROOT/build-ggml-fixtures"}
cmake -S "$HERE" -B "$WORK" -DCMAKE_BUILD_TYPE=Release -DGGML_SOURCE="$GGML_SOURCE"
cmake --build "$WORK" --parallel
ctest --test-dir "$WORK" --output-on-failure --no-tests=error
echo "adapters/ggml fixtures passed"
