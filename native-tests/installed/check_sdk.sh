#!/bin/sh
# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
# Proves an installed SDK is what a downstream user needs, not just that it was copied somewhere.
# Usage: sh native-tests/installed/check_sdk.sh <install-prefix> [<cmake-build-dir>]
#   <install-prefix>   the directory `cmake --install --prefix` wrote (Linux; needs nm, cmake, a C compiler)
#   <cmake-build-dir>  optional: the build tree it came from; enables the install_manifest.txt checks
# Checks, each one fatal:
#   1. the library is executable (mode) and libseabraid.so is a symlink that resolves;
#   2. the exported dynamic symbols are exactly the functions the INSTALLED header declares (a leaked Rust or
#      libc symbol would become ABI, a missing one would be a link error for a consumer);
#   3. with a build dir, every installed file and symlink is listed in install_manifest.txt;
#   4. the prefix is MOVED to another directory (so no path of the build or the original prefix is baked in), then
#      native-tests/installed is configured against the moved prefix with find_package(Seabraid <major.minor>),
#      built, and run: the config relocates, the version check accepts the same minor and refuses the next one,
#      Seabraid_ABI_EPOCH equals the header and the library, and the consumer starts through the moved RUNPATH.
# The prefix is moved back on exit: success, failure, or INT/TERM (which exit 130/143); if the move back itself fails the
# scratch directory holding the only copy is kept and named on stderr. WHY a move and not a copy:
# a copy leaves the original in place, which would hide a reference to it.
# CHECKER: this is the checker (CI job `sdk`, step "Check the installed SDK"); run it locally the same way.
# TESTING THE CMAKE FLOOR (README: CMake 3.22): SEABRAID_CMAKE=/path/to/cmake makes this script use that cmake and the
# ctest beside it (see use_cmake.sh). The same variable steers tests/test_native_checks.py and tests/test_build_config.py:
#   python3 -m venv v && v/bin/pip install cmake==3.22.6
#   export SEABRAID_CMAKE=$PWD/v/bin/cmake
#   $SEABRAID_CMAKE -S . -B b -DCMAKE_BUILD_TYPE=Release -DSEABRAID_BUILD_NATIVE_TESTS=ON && $SEABRAID_CMAKE --build b --parallel
#   v/bin/ctest --test-dir b --output-on-failure && $SEABRAID_CMAKE --install b --prefix sdk
#   sh native-tests/installed/check_sdk.sh sdk b && sh native-tests/installed/check_layouts.sh
#   python3 -m pytest -q tests/test_native_checks.py tests/test_build_config.py
set -eu

prefix=${1:?usage: check_sdk.sh <install-prefix> [<cmake-build-dir>]}
build=${2:-}
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
prefix=$(cd "$prefix" && pwd)
fail() { echo "check_sdk: FAIL: $*" >&2; exit 1; }
. "$here/use_cmake.sh"

lib=$(find "$prefix" -name 'libseabraid.so.*' -type f | head -n 1)
[ -n "$lib" ] || fail "no libseabraid.so.MAJOR.MINOR under $prefix"
libdir=$(dirname "$lib")

[ -x "$lib" ] || fail "$lib is not executable (mode $(stat -c %a "$lib"))"
[ -L "$libdir/libseabraid.so" ] || fail "$libdir/libseabraid.so is not a symlink"
[ -e "$libdir/libseabraid.so" ] || fail "$libdir/libseabraid.so is a dangling symlink"

exported=$(nm -D --defined-only "$lib" | awk '{print $3}' | sort)
declared=$(grep -hE '^[A-Za-z_][A-Za-z0-9_ ]*[ *]seabraid_[a-z0-9_]+\(' "$prefix"/include/seabraid/*.h \
    | sed -E 's/^[^(]*[ *](seabraid_[a-z0-9_]+)\(.*/\1/' | sort -u)
[ -n "$declared" ] || fail "no function declarations found in the installed headers"
if [ "$exported" != "$declared" ]; then
    echo "exported but not declared:" >&2; echo "$exported" | grep -vxF "$declared" >&2 || true
    echo "declared but not exported:" >&2; echo "$declared" | grep -vxF "$exported" >&2 || true
    fail "exported symbols differ from the installed header's functions"
fi
echo "ok: $(echo "$declared" | wc -l) exported symbols = the header's functions; $lib is mode $(stat -c %a "$lib")"

if [ -n "$build" ]; then
    manifest="$build/install_manifest.txt"
    [ -f "$manifest" ] || fail "$manifest missing (run cmake --install from this build tree)"
    find "$prefix" \( -type f -o -type l \) | while IFS= read -r path; do
        grep -qxF "$path" "$manifest" || { echo "check_sdk: FAIL: $path is installed but not in install_manifest.txt" >&2; exit 1; }
    done
    echo "ok: every installed file is in install_manifest.txt"
fi

scratch=$(mktemp -d)
use_seabraid_cmake "$scratch"
moved="$scratch/relocated-prefix"
# WHY the trap comes BEFORE the mv: a signal between the two would leave the SDK stranded in a temp directory.
# WHY INT and TERM exit with 130/143 instead of sharing the EXIT handler: a handler that only restores returns
# control to the script, which would carry on checking (and then exit 1) after the user asked it to stop.
# WHY the scratch directory is removed only after the move back succeeded: $moved lives inside it, so removing it
# after a failed move (read-only parent, a full disk) deleted the only copy of the SDK. On failure it is kept and its
# path printed, so the SDK can be moved back by hand.
# WHY exit in the failure branch: every check may have passed (exit 0), and a script that leaves the SDK stranded and
# still exits 0 reports success to CI. The original status is kept when it was already nonzero (130/143 from a signal).
restore() {
    status=$?
    if [ -d "$moved" ]; then
        if mv "$moved" "$prefix" 2>/dev/null; then
            rm -rf "$scratch"
        else
            echo "check_sdk: COULD NOT MOVE THE SDK BACK to $prefix. The only copy is $moved (scratch directory $scratch is kept); restore it with: mv '$moved' '$prefix'" >&2
            [ "$status" -ne 0 ] || status=1
            exit "$status"
        fi
    else
        rm -rf "$scratch"
    fi
}
trap restore EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
mv "$prefix" "$moved"

series=$(sed -E 's/^([0-9]+)\.([0-9]+).*/\1.\2/' "$root/VERSION")
next="$(echo "$series" | cut -d. -f1).$(( $(echo "$series" | cut -d. -f2) + 1 ))"
# WHY the message is matched: any configure failure (no compiler, a typo) would otherwise count as "refused". The refusal
# is recognised by what it MEANS, and CMake words it two ways: 4.x says "The version found is not compatible with the
# version requested." (config found, its version file rejected the request); 3.22 says 'Could not find a configuration
# file for package "Seabraid" that is compatible with requested version "0.6"' and then lists the config it considered as
# "version: 0.5.0" (CMake wraps its messages, so the sentence is matched with whitespace flattened). That 3.22 sentence
# names only the request, not what was found, so it counts only together with the listing of THIS SDK's version: the SDK
# was found, and its version file said no.
if refuse_log=$(cmake -S "$here" -B "$scratch/refuse" -DCMAKE_PREFIX_PATH="$moved" -DSEABRAID_REQUIRE_VERSION="$next" 2>&1); then
    fail "find_package(Seabraid $next) was accepted by a $series SDK"
fi
flat_log=$(echo "$refuse_log" | tr '\n' ' ' | tr -s ' ')
if echo "$flat_log" | grep -q 'version found is not compatible with the version requested'; then
    :
elif echo "$flat_log" | grep -q 'Could not find a configuration file for package "Seabraid" that is compatible with requested version' \
    && echo "$flat_log" | grep -qE "SeabraidConfig\.cmake, version: $series\."; then
    :
else
    echo "$refuse_log" >&2
    fail "find_package(Seabraid $next) failed, but not because the SDK's version file refused it"
fi
echo "ok: find_package(Seabraid $next) is refused by the $series SDK"

cmake -S "$here" -B "$scratch/build" -DCMAKE_PREFIX_PATH="$moved" -DSEABRAID_REQUIRE_VERSION="$series" >/dev/null \
    || fail "configure against the moved prefix failed (find_package(Seabraid $series))"
cmake --build "$scratch/build" --parallel >/dev/null || fail "build against the moved prefix failed"
env -u LD_LIBRARY_PATH ctest --test-dir "$scratch/build" --output-on-failure || fail "installed consumer tests failed"
echo "ok: the prefix moved to $moved; find_package($series), epoch and consumer run all passed"
