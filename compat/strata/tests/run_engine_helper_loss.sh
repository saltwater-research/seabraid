#!/bin/sh
# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
# Build and run the helper-loss tests from a patched Strata tree with the project's own CMake targets.
set -eu

source=${1:?patched Strata source directory required}
build=${2:?build directory required}
here=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)

# The fixtures use host memory, but helper loss links the real engine. One CUDA architecture and MMQ off minimize its build.
cmake -S "$source" -B "$build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DSTRATA_BUILD_TESTS=ON \
  -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86-real -DSTRATA_MMQ=OFF \
  -DSEABRAID_WIRE_INCLUDE="$here/../../../adapters/strata/wire"

set -- seabraid_helper_loss_test lost_rows_test
cmake --build "$build" --parallel 2 --target "$@"
export CUDA_VISIBLE_DEVICES=-1
for test_name in "$@"; do
  # Run each by name so CTest refuses if either registration is missing.
  ctest --test-dir "$build" -R "^${test_name}$" --no-tests=error --output-on-failure
done
