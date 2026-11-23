# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
# The load simulator links only the public surface: Seabraid::runtime carries the include dir and libseabraid.
add_executable(seabraid_sim sim_main.c sim_cli.c sim_budget.c sim_providers.c sim_workload.c sim_report.c sim_placement.c)
# WHY the C_STANDARD property and not target_compile_features(c_std_11): the feature only asks for AT LEAST C11, and a
# compiler whose default is newer (gcc 15 defaults to gnu23) then got -std=c2x with extensions off, so "strict C11" was
# not what was compiled. C_STANDARD 11 + REQUIRED + EXTENSIONS OFF passes -std=c11 (the build's flags.make shows it).
set_target_properties(seabraid_sim PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
# WHY: clock_gettime and CLOCK_MONOTONIC need POSIX 2008 under strict -std=c11.
target_compile_definitions(seabraid_sim PRIVATE _POSIX_C_SOURCE=200809L)
# WHY -O2 when no build type is given: SIM_MAX_WORK and sim_estimate_work (sim_budget.c) are calibrated on an optimised
# build, and a plain `cmake -S . -B build` (the README's recipe) has CMAKE_BUILD_TYPE empty, which compiles with no -O at
# all: such a build ran up to 3.6 times over the estimate here (19 in the review that found it), so a shape the cap accepted took hours. A multi-config generator
# picks the configuration at build time, and an explicit build type is the caller's choice (Debug is allowed, and slow).
get_property(seabraid_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT seabraid_multi_config AND NOT CMAKE_BUILD_TYPE)
    target_compile_options(seabraid_sim PRIVATE -O2)
endif()
# WHY libm: the joint verifiable-shape limit in sim_cli.c takes a square root (the library is separate on glibc).
# WHY the guard: where libm is part of libc (macOS, musl) find_library finds nothing and leaves SEABRAID_SIM_LIBM-NOTFOUND,
# which target_link_libraries then treats as a library name and fails the configure; sqrt resolves from libc there.
find_library(SEABRAID_SIM_LIBM m)
target_link_libraries(seabraid_sim PRIVATE Seabraid::runtime Threads::Threads)
if(SEABRAID_SIM_LIBM)
    target_link_libraries(seabraid_sim PRIVATE ${SEABRAID_SIM_LIBM})
endif()
# WHY -Werror is switchable: a new compiler adds warnings this tree has never seen, and a developer tool must not stop a
# build for one. CI leaves it on (the default) so the tree stays clean on the compilers it tests.
option(SEABRAID_SIM_WERROR "Treat seabraid_sim warnings as errors" ON)
target_compile_options(seabraid_sim PRIVATE -Wall -Wextra)
if(SEABRAID_SIM_WERROR)
    target_compile_options(seabraid_sim PRIVATE -Werror)
endif()

# Every registration below runs through native-tests/check_output.cmake: ctest's PASS_REGULAR_EXPRESSION ignores the
# exit code (a program that printed the line and exited 3 passes), so the script checks both the exit code and the output.
# -D options must precede -P, so the helper builds the whole command line.
function(seabraid_sim_test name regex)
    add_test(NAME ${name}
        COMMAND "${CMAKE_COMMAND}" "-DEXPECT_REGEX=${regex}" -P "${PROJECT_SOURCE_DIR}/native-tests/check_output.cmake"
                -- ${ARGN})
endfunction()

# The drifting placement (4 devices, 2 layers, 24 experts, 5 rounds) commits swaps, so the smoke run also pins that
# seabraid_placement_apply moved something: swaps_applied must be at least 1. Failure injection stays on.
seabraid_sim_test(seabraid_sim_smoke "\"ok\":true,.*\"swaps_applied\":[1-9]"
    $<TARGET_FILE:seabraid_sim> --hidden 64 --experts 24 --top-k 4 --tokens 8 --devices 4 --layers 2 --steps 40
    --warmup 2 --fail-rate 0.1 --placement-rounds 5 --seed 7)

# A large expert count: the reference grows with it, so an absolute error bound failed correct runs (1024 experts at the
# default width, 4096 at any). The relative bound must pass both; sim_unit_test pins that a wrong result still fails.
seabraid_sim_test(seabraid_sim_experts_1024 "\"ok\":true," $<TARGET_FILE:seabraid_sim> --experts 1024)
seabraid_sim_test(seabraid_sim_experts_4096 "\"ok\":true," $<TARGET_FILE:seabraid_sim>
    --experts 4096 --hidden 64 --tokens 8 --steps 4 --warmup 1)

add_executable(seabraid_sim_unit sim_unit_test.c)
set_target_properties(seabraid_sim_unit PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
target_compile_definitions(seabraid_sim_unit PRIVATE _POSIX_C_SOURCE=200809L)
target_compile_options(seabraid_sim_unit PRIVATE -Wall -Wextra)
if(SEABRAID_SIM_WERROR)
    target_compile_options(seabraid_sim_unit PRIVATE -Werror)
endif()
seabraid_sim_test(seabraid_sim_unit "sim unit: ok" $<TARGET_FILE:seabraid_sim_unit>)

# Refusals (exit codes and messages of the CLI and of matrix.sh) are asserted by a shell script: they are failures on
# purpose, which the check_output.cmake helper cannot expect. It needs bash and `ulimit`, so POSIX hosts only.
if(CMAKE_HOST_UNIX)
    find_program(seabraid_bash bash)
    if(seabraid_bash)
        add_test(NAME seabraid_sim_cli
            COMMAND "${seabraid_bash}" "${CMAKE_CURRENT_SOURCE_DIR}/check_cli.sh" $<TARGET_FILE:seabraid_sim>
                    "${CMAKE_CURRENT_SOURCE_DIR}/matrix.sh")
        set_tests_properties(seabraid_sim_cli PROPERTIES TIMEOUT 300)
    else()
        message(STATUS "bash not found: seabraid_sim_cli (CLI and matrix.sh refusals) not registered")
    endif()
endif()

# Row counters: the same skip-heavy workload (4 devices, top-2 of 24 experts, so most rows have no pick on a given
# device) must ship every row without COMPACT_ROWS and strictly fewer with it. A nonzero placement epoch rides along.
set(sim_rows_args --hidden 64 --experts 24 --top-k 2 --tokens 16 --devices 4 --layers 2 --steps 20
    --warmup 1 --placement-epoch 7 --seed 3)
add_test(NAME seabraid_sim_rows_equal
    COMMAND "${CMAKE_COMMAND}" "-DSIM=$<TARGET_FILE:seabraid_sim>" "-DARGS=${sim_rows_args}"
            -DEXPECT=equal -P "${CMAKE_CURRENT_SOURCE_DIR}/check_rows.cmake")
add_test(NAME seabraid_sim_rows_compact
    COMMAND "${CMAKE_COMMAND}" "-DSIM=$<TARGET_FILE:seabraid_sim>" "-DARGS=${sim_rows_args};--compact-rows"
            -DEXPECT=less -P "${CMAKE_CURRENT_SOURCE_DIR}/check_rows.cmake")

# --help is generated from the parser's flag table: it must exit 0 and list a flag from the first and the last table row.
seabraid_sim_test(seabraid_sim_help "usage: seabraid_sim.*--hidden.*--kernel" $<TARGET_FILE:seabraid_sim> --help)
