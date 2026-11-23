# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
# Builds the cdylib with cargo and proves the files CMake promised exist. Run by the root CMakeLists.txt's custom
# command: cmake -DCARGO=<cargo> -DSOURCE_DIR=<repo> -DTARGET_DIR=<cargo target dir> -DOUTPUTS=<a;b;...>
#                      [-DSONAME_LINK=<path> -DSONAME_LINK_TARGET=<file name>] -P build_cargo.cmake
# OUTPUTS    every file the build must have produced; a missing one (or a dangling symlink) fails the build.
# SONAME_LINK optional: a symlink to create (to SONAME_LINK_TARGET, a bare file name) once cargo has finished.
#
# WHY a script and not two COMMAND lines: CARGO_BUILD_TARGET (or `build.target` in a cargo config) makes cargo write
# to <target-dir>/<triple>/release instead of <target-dir>/release, so the old build "succeeded", exited 0 and left
# a dangling symlink for install to ship. The environment variable is cleared here (CMake 3.22 has no
# `cmake -E env --unset`), and the final check turns every other way of landing the file elsewhere (a cargo config
# file, for one) into a loud error instead of a silent one. The CMake build is always native: the root
# CMakeLists.txt refuses CMAKE_CROSSCOMPILING for the same reason.
# CHECKER: native-tests/check_cargo_target.cmake (CTest seabraid_cargo_build_target) builds with CARGO_BUILD_TARGET set.
foreach(var CARGO SOURCE_DIR TARGET_DIR OUTPUTS)
    if(NOT DEFINED ${var} OR "${${var}}" STREQUAL "")
        message(FATAL_ERROR "build_cargo: -D${var}=... is required")
    endif()
endforeach()

unset(ENV{CARGO_BUILD_TARGET})
execute_process(
    COMMAND "${CARGO}" build --locked --offline --release --target-dir "${TARGET_DIR}"
    WORKING_DIRECTORY "${SOURCE_DIR}"
    RESULT_VARIABLE rc)
if(NOT rc EQUAL 0)
    message(FATAL_ERROR "build_cargo: cargo build failed (${rc})")
endif()

if(SONAME_LINK)
    file(REMOVE "${SONAME_LINK}")
    file(CREATE_LINK "${SONAME_LINK_TARGET}" "${SONAME_LINK}" SYMBOLIC)
endif()

foreach(file IN LISTS OUTPUTS)
    # EXISTS follows symlinks, so a link to a file cargo never wrote counts as missing.
    if(NOT EXISTS "${file}")
        message(FATAL_ERROR "build_cargo: cargo succeeded but ${file} does not exist. If a build target is set "
            "(CARGO_BUILD_TARGET is cleared here; a cargo config file's build.target is not), cargo wrote it "
            "under a <triple> directory. Unset build.target for this build.")
    endif()
endforeach()
