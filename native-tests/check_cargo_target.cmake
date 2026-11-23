# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
# Proves the CMake build still produces the library when CARGO_BUILD_TARGET is set in the environment.
# Usage: cmake -DSOURCE=<repo> -DSCRATCH=<empty scratch dir> -DCARGO=<cargo> -DRUNTIME_NAME=<libseabraid.so>
#              -P check_cargo_target.cmake
# WHY: with a build target set cargo writes under <target-dir>/<triple>/release, and the build used to exit 0 with
# only a dangling symlink in release/ (an install then shipped a broken file). The build must now either produce
# the real library at the documented path or fail; this script accepts only the first.
# Output: FATAL_ERROR (nonzero) naming the step that failed. The scratch build is a second full cargo build of
# the crate (no dependencies), so it takes about as long as the main one.
foreach(var SOURCE SCRATCH CARGO RUNTIME_NAME)
    if(NOT DEFINED ${var} OR "${${var}}" STREQUAL "")
        message(FATAL_ERROR "check_cargo_target: -D${var}=... is required")
    endif()
endforeach()

execute_process(COMMAND "${CARGO}" -vV RESULT_VARIABLE rc OUTPUT_VARIABLE version_text)
if(NOT rc EQUAL 0 OR NOT version_text MATCHES "(^|\n)host: ([^\n]+)")
    message(FATAL_ERROR "check_cargo_target: cannot read the host triple from `cargo -vV`")
endif()
set(triple "${CMAKE_MATCH_2}")

file(REMOVE_RECURSE "${SCRATCH}")
execute_process(COMMAND "${CMAKE_COMMAND}" -S "${SOURCE}" -B "${SCRATCH}/b" RESULT_VARIABLE rc)
if(NOT rc EQUAL 0)
    message(FATAL_ERROR "check_cargo_target: configure failed (${rc})")
endif()

set(ENV{CARGO_BUILD_TARGET} "${triple}")
execute_process(COMMAND "${CMAKE_COMMAND}" --build "${SCRATCH}/b" RESULT_VARIABLE rc)
if(NOT rc EQUAL 0)
    message(FATAL_ERROR "check_cargo_target: build failed with CARGO_BUILD_TARGET=${triple} (${rc}); "
        "the build must succeed, not fail, because the target is ignored")
endif()

set(library "${SCRATCH}/b/cargo/release/${RUNTIME_NAME}")
if(NOT EXISTS "${library}" OR IS_SYMLINK "${library}")
    message(FATAL_ERROR "check_cargo_target: the build exited 0 with CARGO_BUILD_TARGET=${triple} but ${library} "
        "is not a real file (a dangling symlink is exactly the old bug)")
endif()
file(GLOB links "${SCRATCH}/b/cargo/release/${RUNTIME_NAME}.*")
foreach(link IN LISTS links)
    if(NOT EXISTS "${link}")
        message(FATAL_ERROR "check_cargo_target: dangling ${link}")
    endif()
endforeach()
message(STATUS "with CARGO_BUILD_TARGET=${triple} the build produced ${library}")
