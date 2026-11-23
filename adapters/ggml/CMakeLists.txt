# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.16)
project(seabraid_ggml LANGUAGES C CXX)

# Reuse the engine's ggml target so both sides share the same ABI and registry.
if(NOT TARGET ggml::ggml AND NOT TARGET ggml)
    find_package(ggml CONFIG REQUIRED)
endif()
if(TARGET ggml::ggml)
    set(seabraid_ggml_dependency ggml::ggml)
else()
    set(seabraid_ggml_dependency ggml)
endif()

add_library(seabraid_ggml STATIC seabraid_ggml.c _seabraid_ggml_graph.c _seabraid_ggml_routes.c _seabraid_ggml_stats.c)
# The host can link this static adapter into a shared llama library.
set_target_properties(seabraid_ggml PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(seabraid_ggml PRIVATE c_std_11)
target_include_directories(seabraid_ggml PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/../../include")
target_link_libraries(seabraid_ggml PUBLIC "${seabraid_ggml_dependency}")
# seabraid_ggml calls seabraid_log and seabraid_abi_epoch, which only the runtime defines. Measured on the Linux test machine:
# a program linking libseabraid_ggml.a without the runtime fails with "undefined reference to `seabraid_log'"
# and `seabraid_abi_epoch'. That is exactly the llama.cpp hook's link line (it links only seabraid_ggml, and
# never opens a session), so a documented build of the hook could not link.
# WHY the runtime is linked here for a normal build and not for the fixtures: the fixtures define both symbols
# themselves so they can assert on the records, and a second definition would clash. WHY a loud configure error
# rather than a link error later: the missing library is the whole fault, and CMake can say so in one line.
option(SEABRAID_GGML_TESTS "Build the packed Q4_0 parity fixture" OFF)
if(NOT SEABRAID_GGML_TESTS)
    if(NOT TARGET Seabraid::runtime)
        find_package(Seabraid CONFIG QUIET)
    endif()
    if(NOT TARGET Seabraid::runtime)
        message(FATAL_ERROR "seabraid_ggml needs the Seabraid runtime (seabraid_log, seabraid_abi_epoch). "
                "Install the SDK and pass -DSeabraid_DIR=<prefix>/lib/cmake/Seabraid (or add the Seabraid "
                "source tree before this directory so Seabraid::runtime exists).")
    endif()
    target_link_libraries(seabraid_ggml PUBLIC Seabraid::runtime)
endif()
if(SEABRAID_GGML_TESTS)
    enable_testing()
    add_executable(seabraid_ggml_q4 test_q4.c test_q4_scheduler.c)
    target_compile_features(seabraid_ggml_q4 PRIVATE c_std_11)
    target_compile_definitions(seabraid_ggml_q4 PRIVATE SEABRAID_GGML_TEST_MAIN)
    target_link_libraries(seabraid_ggml_q4 PRIVATE seabraid_ggml)
    set_target_properties(seabraid_ggml_q4 PROPERTIES LINKER_LANGUAGE CXX)
    if(NOT WIN32)
        target_link_libraries(seabraid_ggml_q4 PRIVATE m)
    endif()
    add_test(NAME seabraid_ggml_q4 COMMAND seabraid_ggml_q4)

    add_executable(seabraid_ggml_transfer_stats test_transfer_stats.c)
    target_compile_features(seabraid_ggml_transfer_stats PRIVATE c_std_11)
    target_link_libraries(seabraid_ggml_transfer_stats PRIVATE seabraid_ggml)
    set_target_properties(seabraid_ggml_transfer_stats PROPERTIES LINKER_LANGUAGE CXX)
    if(NOT WIN32)
        target_link_libraries(seabraid_ggml_transfer_stats PRIVATE m)
    endif()
    add_test(NAME seabraid_ggml_transfer_stats COMMAND seabraid_ggml_transfer_stats)

    # Edge cases: route and bounds faults, then cache lifecycle, counter races and append refusals.
    find_package(Threads REQUIRED)
    foreach(edge_test test_edges test_edges_cache)
        set(edge_target seabraid_ggml_${edge_test})
        add_executable(${edge_target} ${edge_test}.c)
        target_compile_features(${edge_target} PRIVATE c_std_11)
        target_link_libraries(${edge_target} PRIVATE seabraid_ggml Threads::Threads)
        set_target_properties(${edge_target} PROPERTIES LINKER_LANGUAGE CXX)
        if(NOT WIN32)
            target_link_libraries(${edge_target} PRIVATE m)
        endif()
        add_test(NAME ${edge_target} COMMAND ${edge_target})
    endforeach()

    # The pick rule's shared table (src/picks/golden.txt, owned by src/picks.rs): test_routes.c reads it at run time.
    # WHY a loud configure error and not a skipped row: a table that is not there must never read as "no cases".
    set(SEABRAID_PICKS_GOLDEN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../../src/picks/golden.txt")
    if(NOT EXISTS "${SEABRAID_PICKS_GOLDEN_FILE}")
        message(FATAL_ERROR "seabraid_ggml tests need the pick table ${SEABRAID_PICKS_GOLDEN_FILE}; build from a "
                "full source tree.")
    endif()
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${SEABRAID_PICKS_GOLDEN_FILE}")

    # Route-check order, the skip-weight bit test and the wide-row cost: pure C over seabraid_ggml_routes_fault.
    add_executable(seabraid_ggml_routes test_routes.c)
    target_compile_definitions(seabraid_ggml_routes PRIVATE SEABRAID_PICKS_GOLDEN="${SEABRAID_PICKS_GOLDEN_FILE}")
    target_compile_features(seabraid_ggml_routes PRIVATE c_std_11)
    target_link_libraries(seabraid_ggml_routes PRIVATE seabraid_ggml)
    set_target_properties(seabraid_ggml_routes PROPERTIES LINKER_LANGUAGE CXX)
    add_test(NAME seabraid_ggml_routes COMMAND seabraid_ggml_routes)

    # The same cases with the route code itself built -ffast-math, as a host with that flag would build it: a
    # float compare of a skip's weight may be folded there, a bit test cannot. WHY the source is compiled into
    # the test and not linked from seabraid_ggml: the flag must reach _seabraid_ggml_routes.c and nothing else.
    add_executable(seabraid_ggml_routes_fastmath test_routes.c _seabraid_ggml_routes.c)
    target_compile_features(seabraid_ggml_routes_fastmath PRIVATE c_std_11)
    target_compile_definitions(seabraid_ggml_routes_fastmath PRIVATE SEABRAID_ROUTES_FASTMATH
                               SEABRAID_PICKS_GOLDEN="${SEABRAID_PICKS_GOLDEN_FILE}")
    target_include_directories(seabraid_ggml_routes_fastmath PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}"
                               "${CMAKE_CURRENT_SOURCE_DIR}/../../include")
    if(MSVC)
        target_compile_options(seabraid_ggml_routes_fastmath PRIVATE /O2 /fp:fast)
    else()
        target_compile_options(seabraid_ggml_routes_fastmath PRIVATE -O2 -ffast-math)
    endif()
    add_test(NAME seabraid_ggml_routes_fastmath COMMAND seabraid_ggml_routes_fastmath)
endif()
