# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
#
# The smallest downstream consumer: builds ../minimal_provider.c and ../adapter_walkthrough.c against an INSTALLED
# Seabraid SDK with find_package. To start your own project outside the tree, copy this folder as `consumer/` together
# with BOTH ../minimal_provider.c and ../adapter_walkthrough.c into one parent folder (the targets below name them as
# ../*.c), or delete the adapter_walkthrough target. Build it with:
#   cmake -S examples/consumer -B build/consumer -DCMAKE_PREFIX_PATH=/path/to/seabraid-sdk
#   cmake --build build/consumer
#   build/consumer/minimal_provider        # prints: output = [2.5, 5]
# (CMAKE_PREFIX_PATH is the --prefix you gave `cmake --install`; the targets also run through ctest, and
# `cmake --install build/consumer --prefix <dir>` installs them to <dir>/bin.)
cmake_minimum_required(VERSION 3.22)
project(SeabraidConsumer LANGUAGES C)

find_package(Seabraid CONFIG REQUIRED)

add_executable(minimal_provider ../minimal_provider.c)
target_compile_features(minimal_provider PRIVATE c_std_11)
# Seabraid::runtime carries the include directory and libseabraid.
target_link_libraries(minimal_provider PRIVATE Seabraid::runtime)

set(consumer_programs minimal_provider)
# WHY not on MSVC: the walkthrough uses C11 <stdatomic.h>, which MSVC ships only behind /experimental:c11atomics. The
# skip (not a hand-written Interlocked shim) keeps an unbuilt guess out of the Windows leg; Linux and macOS run it.
if(NOT MSVC)
    add_executable(adapter_walkthrough ../adapter_walkthrough.c)
    target_compile_features(adapter_walkthrough PRIVATE c_std_11)
    target_link_libraries(adapter_walkthrough PRIVATE Seabraid::runtime)
    list(APPEND consumer_programs adapter_walkthrough)
endif()

# WHY INSTALL_RPATH: the library's SONAME is libseabraid.so.<major.minor>, and CMake strips the build-tree rpath (the SDK's
# lib directory) when it installs a target, so an installed copy would not start ("cannot open shared object file",
# exit 127). This rpath finds lib/ beside bin/, which holds when the program is installed into the SDK prefix or ships
# with a copy of the library; for any other layout set it to wherever libseabraid lives. $ORIGIN is the ELF form
# (@loader_path on macOS); a Windows DLL is found through PATH, not an rpath.
if(APPLE)
    set(consumer_origin "@loader_path")
else()
    set(consumer_origin "$ORIGIN")
endif()
set_target_properties(${consumer_programs} PROPERTIES INSTALL_RPATH "${consumer_origin}/../lib")
install(TARGETS ${consumer_programs} RUNTIME DESTINATION bin)

# Every test runs through check_output.cmake (a local copy of the repository's native-tests/check_output.cmake, so this
# folder stays copyable on its own): ctest's PASS_REGULAR_EXPRESSION alone ignores the exit code, so a program that
# printed the right line and then exited nonzero would pass. The script requires exit 0 AND the output to match.
enable_testing()
function(consumer_test name regex)
    add_test(NAME ${name}
        COMMAND "${CMAKE_COMMAND}" "-DEXPECT_REGEX=${regex}" -P "${CMAKE_CURRENT_SOURCE_DIR}/check_output.cmake"
                -- $<TARGET_FILE:${name}>)
    # WHY: on Windows the DLL is found only through PATH, and the SDK's DLL sits in its bin/ (not verified on Windows).
    if(WIN32)
        set_tests_properties(${name} PROPERTIES
            ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:$<TARGET_FILE_DIR:Seabraid::runtime>")
    endif()
endfunction()
# check_output.cmake merges stdout and stderr in arrival order. The walkthrough's bare "ok" is its very last write and
# nothing logs after it (close() has returned), so "ok" is the last line whatever the stdio buffering; the pattern
# anchors to the end of the text, as the repository's own test does.
if(NOT MSVC)
    consumer_test(adapter_walkthrough "(^|\n)ok\n$")
endif()
consumer_test(minimal_provider "output = \\[2\\.5, 5\\]")
