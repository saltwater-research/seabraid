#!/bin/sh
# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
# Proves `cmake --install` puts the SDK where -DCMAKE_INSTALL_LIBDIR says, for every layout a distribution uses.
# Usage: sh native-tests/installed/check_layouts.sh        (Linux; needs cmake, cargo and a POSIX shell)
# Layouts, each one fatal on failure:
#   default (lib), lib64, lib/x86_64-linux-gnu  given untyped on the command line, as a user writes them
#   an absolute LIBDIR with DESTDIR             (the packager's form: files land under DESTDIR, the manifest does not name it)
#   an absolute LIBDIR, `install --prefix` differing from the configure prefix, then find_package, a consumer build
#                                               and run (the include directory must follow the install prefix)
#   the same absolute-LIBDIR SDK installed again: same prefix (also after `touch -r` made the installed config look up
#                                               to date), another prefix, a relative prefix, a prefix holding " $ { } and a
#                                               space; the config must name the CURRENT prefix each time; a prefix holding ';'
#                                               is refused at install time
#   an absolute LIBDIR installed under `umask 077`  the package config must still be world-readable (0644)
#   an absolute LIBDIR, `install --prefix /` with DESTDIR  the config names /include, not the working directory
#   a symlink or a read-only file left where the config goes  the install replaces it (never writes through the link)
#   --component Runtime             only the library and licenses; no symlink, headers or config
#   --component Development into a fresh prefix  headers and licenses install even without the Runtime
# WHY the absolute LIBDIRs are scratch paths: a layout check must never be able to write /usr. The DESTDIR case uses
# the made-up root /seabraid-layout-check (a DESTDIR regression then fails on a missing directory, or, as root, would
# create that one directory, which the script reports), the consumer case an absolute directory in the scratch tree.
# The script refuses to run as root unless SEABRAID_LAYOUT_CHECK_ALLOW_ROOT=1 (CI containers can run as root).
# The consumer case needs a C compiler (the same requirement as check_sdk.sh); the script says so up front when none is found.
# For every install it also checks that nothing was written outside the prefix (an untyped LIBDIR once became a
# path under the current directory) and that every install_manifest entry exists.
# OWNER: CMakeLists.txt (install rules). CHECKER: this script; tests/test_build_config.py runs it when cmake and
# cargo are present. One cargo build is shared by every layout: each case only reconfigures the same build tree.
# SEABRAID_CMAKE=/path/to/cmake runs every cmake call here with that cmake (see use_cmake.sh and the recipe in check_sdk.sh).
set -eu

here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
fail() { echo "check_layouts: FAIL: $*" >&2; exit 1; }
. "$here/use_cmake.sh"
fake_root=/seabraid-layout-check
if [ "$(id -u)" = 0 ] && [ -z "${SEABRAID_LAYOUT_CHECK_ALLOW_ROOT:-}" ]; then
    fail "refusing to run as root (a DESTDIR regression would write the host); set SEABRAID_LAYOUT_CHECK_ALLOW_ROOT=1 to override"
fi
[ ! -e "$fake_root" ] || fail "$fake_root exists on this host; remove it (it is this script's made-up root for the DESTDIR case)"
# WHY here and not at the consumer case: that case runs after the long build, and a missing compiler there surfaced as
# "find_package failed", which sent the reader to the wrong place. CMake looks for $CC, then cc, gcc, clang.
# WHY the whole value is tried first: CC may be one path that holds a space ("/opt/my tools/gcc"), which splitting at the
# first space would turn into the nonexistent "/opt/my"; only when the whole value is no command is its first word
# tried (CC="ccache gcc").
cc_word=${CC:-}
if [ -n "$cc_word" ]; then
    command -v "$cc_word" >/dev/null 2>&1 || command -v "${cc_word%% *}" >/dev/null 2>&1 \
        || fail "no C compiler: CC='$CC' is not on PATH (the absolute-consumer case builds a C consumer)"
else
    command -v cc >/dev/null 2>&1 || command -v gcc >/dev/null 2>&1 || command -v clang >/dev/null 2>&1 \
        || fail "no C compiler (cc, gcc or clang) on PATH and CC is unset (the absolute-consumer case builds a C consumer)"
fi
scratch=$(mktemp -d)
use_seabraid_cmake "$scratch"
trap 'rm -rf "$scratch"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

series=$(sed -E 's/^([0-9]+)\.([0-9]+).*/\1.\2/' "$root/VERSION")
soname="libseabraid.so.$series"
build="$scratch/build"
cwd="$scratch/cwd" # configure and install run from here; it must stay empty
mkdir "$cwd"
cmake -S "$root" -B "$build" >/dev/null || fail "initial configure failed"
cmake --build "$build" --parallel >/dev/null || fail "build failed"

n=0
# install_layout <name> <libdir: relative to the prefix, or absolute> <cmake -D/-U argument> [cmake --install args...]
# sets $prefix and $dest (the directory the library lands in, under DESTDIR when the name says destdir)
install_layout() {
    name=$1; libdir=$2; libarg=$3; shift 3
    n=$((n + 1))
    prefix="$scratch/prefix$n"
    destdir=""
    case "$name" in *destdir*) destdir="$scratch/destdir$n" ;; esac
    rm -f "$build"/install_manifest*.txt
    (cd "$cwd" && cmake -S "$root" -B "$build" "$libarg" >/dev/null) || fail "$name: configure failed"
    (cd "$cwd" && DESTDIR=$destdir cmake --install "$build" --prefix "$prefix" "$@" >"$scratch/install$n.log" 2>&1) \
        || { cat "$scratch/install$n.log" >&2; fail "$name: install failed"; }
    case "$libdir" in
        /*) dest="$destdir$libdir" ;;
        *) dest="$prefix/$libdir" ;;
    esac
    [ -z "$(ls -A "$cwd")" ] || fail "$name: install wrote outside the prefix: $(ls "$cwd")"
    for manifest in "$build"/install_manifest*.txt; do
        [ -f "$manifest" ] || fail "$name: no install manifest written"
        while IFS= read -r path; do
            [ -e "$destdir$path" ] || [ -L "$destdir$path" ] || fail "$name: manifest lists $path, which is not installed"
            case "$path" in "$scratch"/destdir*) fail "$name: manifest entry carries DESTDIR: $path" ;; esac
        done <"$manifest"
    done
}
# WHY every component, and nothing left behind: the refusal is the first install rule of each, so no file may be copied
# before it (a refusal after the runtime and headers left a half-installed prefix). It is checked in the layout that is
# currently configured, so the relative-LIBDIR layout is covered as well as the absolute one that bakes the prefix in.
refuses_semicolon_prefix() { # <label>
    label=$1
    for component in all Runtime Development; do
        bad="$scratch/refused-$component;prefix"
        if [ "$component" = all ]; then set --; else set -- --component "$component"; fi
        if (cd "$cwd" && cmake --install "$build" --prefix "$bad" "$@" >"$scratch/refused.log" 2>&1); then
            fail "$label: a ';' prefix was accepted (component: $component)"
        fi
        grep -qF "refused-$component;prefix" "$scratch/refused.log" || { cat "$scratch/refused.log" >&2; fail "$label: the refusal does not name the prefix (component: $component)"; }
        [ ! -e "$bad" ] || fail "$label: the refused ';' prefix was partly installed (component: $component): $(ls "$bad")"
        [ -z "$(ls -A "$cwd")" ] || fail "$label: a refused install wrote outside the prefix: $(ls "$cwd")"
    done
}
want_runtime() { { [ -f "$dest/$soname" ] && [ -x "$dest/$soname" ]; } || fail "$name: $dest/$soname missing or not executable"; }
want_symlink() {
    [ -L "$dest/libseabraid.so" ] || fail "$name: $dest/libseabraid.so is not a symlink"
    [ "$(readlink "$dest/libseabraid.so")" = "$soname" ] || fail "$name: libseabraid.so points to $(readlink "$dest/libseabraid.so")"
}
no_file() { { [ ! -e "$1" ] && [ ! -L "$1" ]; } || fail "$name: $1 should not be installed"; }
want_licenses() {
    for license_file in LICENSE LICENSE-MIT NOTICE; do
        installed="$prefix/share/doc/seabraid/$license_file"
        cmp -s "$root/$license_file" "$installed" || fail "$name: $license_file missing or different from source"
        [ "$(grep -xcF "$installed" "$build/$1")" -eq 1 ] || fail "$name: $license_file must appear once in the manifest"
    done
}

install_layout default lib -UCMAKE_INSTALL_LIBDIR
want_runtime; want_symlink
want_licenses install_manifest.txt
[ -f "$dest/cmake/Seabraid/SeabraidConfig.cmake" ] || fail "default: package config not under lib/cmake"
[ -f "$prefix/include/seabraid/seabraid.h" ] || fail "default: header missing"
refuses_semicolon_prefix default
echo "ok: default layout: $dest; a ';' prefix is refused"

install_layout lib64 lib64 -DCMAKE_INSTALL_LIBDIR=lib64
want_runtime; want_symlink; no_file "$prefix/lib"
[ -f "$dest/cmake/Seabraid/SeabraidConfig.cmake" ] || fail "lib64: package config not under lib64/cmake"
echo "ok: untyped -DCMAKE_INSTALL_LIBDIR=lib64: $dest"

install_layout multiarch lib/x86_64-linux-gnu -DCMAKE_INSTALL_LIBDIR=lib/x86_64-linux-gnu
want_runtime; want_symlink
echo "ok: untyped -DCMAKE_INSTALL_LIBDIR=lib/x86_64-linux-gnu: $dest"

install_layout absolute-destdir "$fake_root/lib64" "-DCMAKE_INSTALL_LIBDIR=$fake_root/lib64"
want_runtime; want_symlink; no_file "$prefix/lib"
grep -qxF "$fake_root/lib64/libseabraid.so" "$build/install_manifest.txt" || fail "absolute-destdir: manifest lacks $fake_root/lib64/libseabraid.so"
[ ! -e "$fake_root" ] || fail "absolute-destdir: DESTDIR was ignored, $fake_root was written on the host"
grep -qF "$prefix/include" "$dest/cmake/Seabraid/SeabraidConfig.cmake" || fail "absolute-destdir: the package config does not name the install prefix's include directory"
echo "ok: absolute LIBDIR $fake_root/lib64 with DESTDIR: $dest"

# WHY no DESTDIR here: the config's absolute paths must exist for find_package to follow them.
abslib="$scratch/abslib"
install_layout absolute-consumer "$abslib" "-DCMAKE_INSTALL_LIBDIR=$abslib"
want_runtime; want_symlink; no_file "$prefix/lib"
config="$abslib/cmake/Seabraid"
[ -f "$config/SeabraidConfig.cmake" ] || fail "absolute-consumer: package config missing under $config"
grep -qF "$prefix/include" "$config/SeabraidConfig.cmake" || fail "absolute-consumer: the package config does not name $prefix/include"
cmake -S "$here" -B "$scratch/consumer" -DSeabraid_DIR="$config" -DSEABRAID_REQUIRE_VERSION="$series" >"$scratch/consumer.log" 2>&1 \
    || { cat "$scratch/consumer.log" >&2; fail "absolute-consumer: find_package(Seabraid $series) failed"; }
cmake --build "$scratch/consumer" --parallel >"$scratch/consumer-build.log" 2>&1 \
    || { cat "$scratch/consumer-build.log" >&2; fail "absolute-consumer: the consumer did not build against the installed SDK"; }
env -u LD_LIBRARY_PATH ctest --test-dir "$scratch/consumer" --output-on-failure >"$scratch/consumer-test.log" 2>&1 \
    || { cat "$scratch/consumer-test.log" >&2; fail "absolute-consumer: the consumer failed to run"; }
echo "ok: absolute LIBDIR $abslib, install prefix != configure prefix: find_package, build and run passed"

# Reinstalling must work and must always name the current prefix. WHY each variant: the first rewrite of the config left
# a sentinel behind that a second install did not find (file(INSTALL) skips an up-to-date copy, which `touch -r` forces),
# so the second `cmake --install` died; a relative prefix baked a relative include directory; a prefix holding `"`, `$`
# or `${...}` broke the config's own quoting.
name=reinstall
reinstall() { # <prefix> [directory to run from]
    (cd "${2:-$cwd}" && cmake --install "$build" --prefix "$1" >"$scratch/reinstall.log" 2>&1) \
        || { cat "$scratch/reinstall.log" >&2; fail "$name: second install into $1 failed"; }
}
# Configure a throwaway project against the installed config and print the include directory its target carries.
include_of() {
    mkdir -p "$scratch/probe"
    printf 'cmake_minimum_required(VERSION 3.22)\nproject(probe NONE)\nfind_package(Seabraid CONFIG REQUIRED)\nget_target_property(inc Seabraid::runtime INTERFACE_INCLUDE_DIRECTORIES)\nmessage(STATUS "INCLUDE=[${inc}]")\n' >"$scratch/probe/CMakeLists.txt"
    rm -rf "$scratch/probe/b"
    cmake -S "$scratch/probe" -B "$scratch/probe/b" -DSeabraid_DIR="$config" >"$scratch/probe.log" 2>&1 \
        || { cat "$scratch/probe.log" >&2; fail "$name: find_package(Seabraid) failed against the config installed for $1"; }
    sed -n 's/^-- INCLUDE=\[\(.*\)\]$/\1/p' "$scratch/probe.log"
}
want_include() { # <prefix the config must name>
    got=$(include_of "$1")
    [ "$got" = "$1/include" ] || fail "$name: the config names include directory '$got', expected '$1/include'"
}
reinstall "$prefix"
want_include "$prefix"
touch -r "$build/SeabraidConfig.cmake" "$config/SeabraidConfig.cmake"
reinstall "$prefix"
want_include "$prefix"
second="$scratch/prefix-second"
reinstall "$second"
want_include "$second"
mkdir "$scratch/relcwd"
reinstall relative-prefix "$scratch/relcwd"
want_include "$scratch/relcwd/relative-prefix"
# WHY no backslash in the prefix: CMake's own file(INSTALL) of the headers fails on one ("cannot set permissions"), before
# this project's rules run (measured on CMake 4.2.3); the config escapes a backslash all the same.
weird="$scratch/we\"ird \${PACKAGE_PREFIX_DIR} \$x"
reinstall "$weird"
want_include "$weird"
semi="$scratch/semi;colon"
if (cd "$cwd" && cmake --install "$build" --prefix "$semi" >"$scratch/semi.log" 2>&1); then
    fail "$name: an install prefix holding ';' was accepted"
fi
grep -qF "semi;colon" "$scratch/semi.log" || { cat "$scratch/semi.log" >&2; fail "$name: the refusal of a ';' prefix does not name the prefix"; }
[ -z "$(ls -A "$cwd")" ] || fail "$name: a reinstall wrote outside the prefix: $(ls "$cwd")"
[ ! -e "$semi" ] || fail "$name: the refused ';' prefix was partly installed: $(ls "$semi")"
refuses_semicolon_prefix "$name"
# A symlink where the config goes must be replaced, not written through; a read-only file must be replaced too.
victim="$scratch/victim.txt"
echo untouched >"$victim"
rm -f "$config/SeabraidConfig.cmake"
ln -s "$victim" "$config/SeabraidConfig.cmake"
reinstall "$prefix"
[ ! -L "$config/SeabraidConfig.cmake" ] || fail "$name: the config is still a symlink after a reinstall"
[ "$(cat "$victim")" = untouched ] || fail "$name: the install wrote through the symlink into $victim"
want_include "$prefix"
chmod 0444 "$config/SeabraidConfig.cmake"
reinstall "$prefix"
[ "$(stat -c %a "$config/SeabraidConfig.cmake")" = 644 ] || fail "$name: a read-only config was kept read-only: $(stat -c %a "$config/SeabraidConfig.cmake")"
want_include "$prefix"
# -DCMAKE_INSTALL_MESSAGE=NEVER silences the "Installing:" line of the config as it does for CMake's own rules.
(cd "$cwd" && cmake -S "$root" -B "$build" -DCMAKE_INSTALL_MESSAGE=NEVER >/dev/null) || fail "$name: configure with CMAKE_INSTALL_MESSAGE=NEVER failed"
reinstall "$prefix"
! grep -q "Installing" "$scratch/reinstall.log" || fail "$name: CMAKE_INSTALL_MESSAGE=NEVER still printed an Installing line: $(grep Installing "$scratch/reinstall.log")"
(cd "$cwd" && cmake -S "$root" -B "$build" -UCMAKE_INSTALL_MESSAGE >/dev/null) || fail "$name: configure without CMAKE_INSTALL_MESSAGE failed"
reinstall "$prefix"
grep -q "Installing: .*SeabraidConfig.cmake" "$scratch/reinstall.log" || fail "$name: the default install no longer reports the config it writes"
echo "ok: absolute LIBDIR reinstalled: same prefix (also after touch -r), another, a relative one, one holding quote/space/dollar/braces; a ';' prefix refused"

# WHY a fresh LIBDIR and `umask 077`: file(WRITE) creates the config with 0666 minus the umask, so a packager running
# under a tight umask shipped a 0600 config that no other user could read.
umask_lib="$scratch/umasklib"
old_umask=$(umask)
umask 077
install_layout umask-077 "$umask_lib" "-DCMAKE_INSTALL_LIBDIR=$umask_lib"
umask "$old_umask"
mode=$(stat -c %a "$umask_lib/cmake/Seabraid/SeabraidConfig.cmake")
[ "$mode" = 644 ] || fail "umask-077: SeabraidConfig.cmake has mode $mode, expected 644"
echo "ok: absolute LIBDIR installed under umask 077: the package config is mode $mode"

# WHY DESTDIR: `--prefix /` is what a packager writes with DESTDIR=<staging>. The prefix reaches the install script
# empty, which used to be resolved to the working directory and baked in as the config's include directory.
root_lib="$scratch/rootlib"
rootdest="$scratch/rootdest"
(cd "$cwd" && cmake -S "$root" -B "$build" "-DCMAKE_INSTALL_LIBDIR=$root_lib" >/dev/null) || fail "prefix-root: configure failed"
(cd "$cwd" && DESTDIR=$rootdest cmake --install "$build" --prefix / >"$scratch/rootinstall.log" 2>&1) \
    || { cat "$scratch/rootinstall.log" >&2; fail "prefix-root: install failed"; }
n=$((n + 1))
rootcfg="$rootdest$root_lib/cmake/Seabraid/SeabraidConfig.cmake"
[ -f "$rootcfg" ] || fail "prefix-root: no config at $rootcfg"
grep -qF 'set(_seabraid_include "/include")' "$rootcfg" \
    || fail "prefix-root: the config does not name /include: $(grep '_seabraid_include "' "$rootcfg")"
[ -f "$rootdest/include/seabraid/seabraid.h" ] || fail "prefix-root: header not under DESTDIR/include"
[ -z "$(ls -A "$cwd")" ] || fail "prefix-root: install wrote outside DESTDIR: $(ls "$cwd")"
echo "ok: absolute LIBDIR with --prefix / under DESTDIR: the config names /include"

install_layout runtime-only lib -UCMAKE_INSTALL_LIBDIR --component Runtime
want_runtime; no_file "$dest/libseabraid.so"; no_file "$prefix/include"; no_file "$dest/cmake"
want_licenses install_manifest_Runtime.txt
echo "ok: Runtime component only: $dest"

install_layout development-only lib -UCMAKE_INSTALL_LIBDIR --component Development
want_symlink; no_file "$dest/$soname"
want_licenses install_manifest_Development.txt
[ -f "$prefix/include/seabraid/seabraid.h" ] || fail "development-only: header missing"
[ -f "$dest/cmake/Seabraid/SeabraidConfig.cmake" ] || fail "development-only: package config missing"
grep -qxF "$dest/libseabraid.so" "$build/install_manifest_Development.txt" || fail "development-only: manifest lacks the dev symlink"
echo "ok: Development component only, into a fresh prefix: $dest"
echo "check_layouts: all $n layouts passed"
