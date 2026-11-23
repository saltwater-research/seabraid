# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
# Sourced (never run) by check_sdk.sh and check_layouts.sh: makes SEABRAID_CMAKE, when set, the cmake AND ctest they use.
# OWNER of the SEABRAID_CMAKE contract. Why it exists: the README promises CMake 3.22, and a CI image carries one newer
# cmake, so the floor is only ever tested by pointing these scripts (and tests/*.py, which read the same variable) at an
# older binary, e.g. one from `pip install cmake==3.22.6` in a venv.
#   SEABRAID_CMAKE=/abs/or/relative/path/to/cmake   must be an executable file named `cmake`; the `cmake`, `ctest` and
#                                                   `cpack` beside it become the ones the script finds first on PATH.
# Unset or empty: nothing changes. A set value that is not an executable `cmake` stops the caller at once (fail-fast: a typo
# must not silently test the system cmake and report the floor as checked).
# Two steps, because the shim needs the caller's scratch directory (which the caller already deletes on exit):
#   . use_cmake.sh                 validates SEABRAID_CMAKE (before anything is created)
#   use_seabraid_cmake "$scratch"  links the three tools into $scratch/cmake-shim and puts it first on PATH
# WHY a shim directory and not the binary's own directory on PATH: that directory is often /usr/bin, and putting it first
# would also shadow every other tool a caller (or a test's stub) placed ahead of it.
# CHECKER: tests/test_build_config.py::test_seabraid_cmake_is_honoured_or_refused.
if [ -n "${SEABRAID_CMAKE:-}" ]; then
    if [ ! -x "$SEABRAID_CMAKE" ] || [ -d "$SEABRAID_CMAKE" ] || [ "$(basename "$SEABRAID_CMAKE")" != cmake ]; then
        echo "SEABRAID_CMAKE='$SEABRAID_CMAKE' is not an executable file named cmake" >&2
        exit 1
    fi
    seabraid_cmake_dir=$(cd "$(dirname "$SEABRAID_CMAKE")" && pwd)
fi
use_seabraid_cmake() {
    [ -n "${SEABRAID_CMAKE:-}" ] || return 0
    mkdir "$1/cmake-shim"
    for tool in cmake ctest cpack; do
        if [ -x "$seabraid_cmake_dir/$tool" ]; then ln -s "$seabraid_cmake_dir/$tool" "$1/cmake-shim/$tool"; fi
    done
    PATH="$1/cmake-shim:$PATH"
    export PATH
    echo "using $seabraid_cmake_dir/cmake: $(cmake --version | head -n 1)"
}
