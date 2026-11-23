# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
# The ONE registration of seabraid's native host tests (no GPU, no Strata headers).  Two ways in:
#   standalone:  cmake -S compat/strata/tests -B build-compat [-DSEABRAID_STRATA_TSAN=ON]
#                cmake --build build-compat --parallel && ctest --test-dir build-compat --output-on-failure
#   in Strata:   strata-ext/CMakeLists.txt adds this directory when STRATA_BUILD_TESTS is on.
# WHY no Strata tree is needed any more: the helper wire these tests exercise is adapters/strata/wire (seabraid.cmake
# finds it in this checkout); nothing here includes a Strata header.
# WHY a project of its own and not a subdirectory of the root build: MAP.md's rule is that the core never depends on the
# add-on, and checks/check_compat_edges.py enforces it, so the core's CMake must not name compat/.
cmake_minimum_required(VERSION 3.16)
project(seabraid_host_tests LANGUAGES CXX)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "The channel is POSIX shared memory (shm_open); build on Linux")
endif()
# WHY an option: on x86 the hardware hides most reordering, so the v4 race test proves little without ThreadSanitizer
# (its header says so); it is off by default only because not every toolchain ships the TSan runtime.
option(SEABRAID_STRATA_TSAN "Build the channel race test under ThreadSanitizer" OFF)

include("${CMAKE_CURRENT_SOURCE_DIR}/../strata-ext/seabraid.cmake")   # SEABRAID_STRATA_WIRE_INCLUDE, seabraid_tier_plan, seabraid_retier
enable_testing()
set(_ext "${CMAKE_CURRENT_SOURCE_DIR}/../strata-ext")

function(seabraid_host_test name source)
    add_executable(${name} ${source} ${ARGN})
    target_compile_features(${name} PRIVATE cxx_std_17)
    target_include_directories(${name} PRIVATE "${SEABRAID_STRATA_WIRE_INCLUDE}" "${_ext}/include" "${_ext}/src")
    target_compile_options(${name} PRIVATE -Wall -Wextra -O1)
    target_link_libraries(${name} PRIVATE rt Threads::Threads)
endfunction()

# The wire contract: sizeof(Header), negotiation preamble, tier handshake sequence (one thread).
seabraid_host_test(seabraid_channel_v4_test channel_v4_test.cpp)
add_test(NAME seabraid_channel_v4 COMMAND seabraid_channel_v4_test)
# The same handshake from two threads on one mapping: visibility, not sequence.
seabraid_host_test(seabraid_channel_v4_race_test channel_v4_race_test.cpp)
if(SEABRAID_STRATA_TSAN)
    target_compile_options(seabraid_channel_v4_race_test PRIVATE -fsanitize=thread)
    target_link_options(seabraid_channel_v4_race_test PRIVATE -fsanitize=thread)
endif()
add_test(NAME seabraid_channel_v4_race COMMAND seabraid_channel_v4_race_test)
# v6/v7 layout: the golden sizes come from channel_v6_golden.txt (shared with tests/install); seabraid_layout_print
# rewrites it on a new compiler (run by hand, see its header).
seabraid_host_test(seabraid_channel_v6_test channel_v6_test.cpp)
add_test(NAME seabraid_channel_v6_test COMMAND seabraid_channel_v6_test "${CMAKE_CURRENT_SOURCE_DIR}/channel_v6_golden.txt")
seabraid_host_test(seabraid_layout_print channel_layout_print.cpp)
# v8 (pyramid): kTierSwapBlob + TierWireOp::blob_slot, pinned to channel_v8_golden.txt (seabraid_layout_print --v8 writes it).
seabraid_host_test(seabraid_channel_v8_test channel_v8_test.cpp)
add_test(NAME seabraid_channel_v8_test COMMAND seabraid_channel_v8_test "${CMAKE_CURRENT_SOURCE_DIR}/channel_v8_golden.txt")
# v9 (file moves): kTierSwapFile + FileRef + the Header's file_* fields, pinned to channel_v9_golden.txt (seabraid_layout_print --v9
# writes it; the facts list is channel_v9_facts.hpp, shared by the printer and this test).
seabraid_host_test(seabraid_channel_v9_test channel_v9_test.cpp)
add_test(NAME seabraid_channel_v9_test COMMAND seabraid_channel_v9_test "${CMAKE_CURRENT_SOURCE_DIR}/channel_v9_golden.txt")
# v10 (a lost helper closes its card): Header::helper_start, the helper's identity, pinned to channel_v10_golden.txt (seabraid_layout_print
# --v10 writes it; the facts list is channel_v10_facts.hpp, shared by the printer and this test).
seabraid_host_test(seabraid_channel_v10_test channel_v10_test.cpp)
add_test(NAME seabraid_channel_v10_test COMMAND seabraid_channel_v10_test "${CMAKE_CURRENT_SOURCE_DIR}/channel_v10_golden.txt")
# The helper's tier round with the blob swap, over a fake cache (the sequence, every refusal, the read-back), and the same
# on a real shared-memory channel with main and the helper on two threads.
seabraid_host_test(seabraid_blob_swap_test blob_swap_test.cpp)
add_test(NAME seabraid_blob_swap_test COMMAND seabraid_blob_swap_test)
seabraid_host_test(seabraid_channel_blob_swap_test channel_blob_swap_test.cpp)
add_test(NAME seabraid_channel_blob_swap_test COMMAND seabraid_channel_blob_swap_test)
seabraid_host_test(seabraid_channel_private_test channel_private_test.cpp)
add_test(NAME seabraid_channel_private_test COMMAND seabraid_channel_private_test)
seabraid_host_test(seabraid_tier_residency_test tier_residency_test.cpp)
add_test(NAME seabraid_tier_residency_test COMMAND seabraid_tier_residency_test)
# D4 helper-side prefill service: a request main published is always answered (shm).
seabraid_host_test(seabraid_pf_service_test pf_service_test.cpp)
add_test(NAME seabraid_pf_service_test COMMAND seabraid_pf_service_test)
seabraid_host_test(seabraid_card_bench_test card_bench_test.cpp "${_ext}/src/card_bench.cpp")
add_test(NAME seabraid_card_bench_test COMMAND seabraid_card_bench_test)
seabraid_host_test(seabraid_log_text_test log_text_test.cpp)   # path scrubbing for helper logs (header only)
add_test(NAME seabraid_log_text_test COMMAND seabraid_log_text_test)
seabraid_host_test(seabraid_tier_plan_test tier_plan_test.cpp)
target_link_libraries(seabraid_tier_plan_test PRIVATE seabraid_tier_plan)
add_test(NAME seabraid_tier_plan_test COMMAND seabraid_tier_plan_test)
seabraid_host_test(seabraid_tier_ladder_test tier_ladder_test.cpp)   # M3b: the N-rung cascade
target_link_libraries(seabraid_tier_ladder_test PRIVATE seabraid_tier_plan)
add_test(NAME seabraid_tier_ladder_test COMMAND seabraid_tier_ladder_test)
seabraid_host_test(seabraid_retier_loop_test retier_loop_test.cpp)
target_link_libraries(seabraid_retier_loop_test PRIVATE seabraid_retier)
add_test(NAME seabraid_retier_loop_test COMMAND seabraid_retier_loop_test)
# The engine's --retier* startup rules over tests/data/retier_cases.tsv, the file the installer's mirror (install/_retier_rules.py) is
# tested on too (tests/install/test_retier_rules.py): the two sides cannot drift apart without one of them failing.
seabraid_host_test(seabraid_retier_rules_case_test retier_rules_case_test.cpp)
target_link_libraries(seabraid_retier_rules_case_test PRIVATE seabraid_retier)
add_test(NAME seabraid_retier_rules_case_test COMMAND seabraid_retier_rules_case_test "${CMAKE_CURRENT_SOURCE_DIR}/data/retier_cases.tsv")

# The direct executable must reject contradictory switches before model or GPU setup.
if(TARGET strata)
  add_test(NAME seabraid_retier_argv COMMAND ${CMAKE_COMMAND}
    "-DENGINE=$<TARGET_FILE:strata>" -P "${CMAKE_CURRENT_SOURCE_DIR}/retier_argv.cmake")
endif()
# File moves (wire v9): the helper's reader of a FileRef, the same on a real channel, the page-cache proof, and the sums tool's logic.
# They hash with Strata's vendored xxhash (seabraid.cmake), so they exist only when it was found; without it ONE test fails and says why,
# so a missing hash is never a green run.  The synthetic packs go under SEABRAID_STRATA_FR_TEST_DIR: it must be a DISK (tmpfs has no direct I/O
# alignment and keeps every page resident, so the direct and page-cache arms SKIP there; SEABRAID_STRATA_FR_REQUIRE_DIRECT=1 makes a skip fail).
set(SEABRAID_STRATA_FR_TEST_DIR "${CMAKE_CURRENT_BINARY_DIR}" CACHE PATH "directory for the file-move tests' synthetic packs (a disk, not tmpfs)")
if(TARGET seabraid_file_fetch)
    foreach(_t file_fetch_test channel_file_swap_test file_cache_test sums_test)
        seabraid_host_test(seabraid_${_t} ${_t}.cpp)
        target_link_libraries(seabraid_${_t} PRIVATE seabraid_file_fetch)
        add_test(NAME seabraid_${_t} COMMAND seabraid_${_t})
        set_tests_properties(seabraid_${_t} PROPERTIES ENVIRONMENT "SEABRAID_STRATA_FR_TEST_DIR=${SEABRAID_STRATA_FR_TEST_DIR}")
    endforeach()
else()
    add_test(NAME seabraid_file_tests_need_xxhash COMMAND "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_SOURCE_DIR}/need_xxhash.cmake")
endif()

# A latency benchmark for the doorbell round trip (DESIGN-LOG gate 1: p50 <= 15 us), not a pass/fail test: built, never registered.
seabraid_host_test(seabraid_channel_pingpong channel_pingpong.cpp)

# L13 (R8 review): the C++ ladder still reproduces the Rust parity golden byte for byte.  The golden belongs to the
# core (src/placement/tests/ladder_golden.txt, read by parity.rs); this test only reads it.  A drift on either side
# fails here or in parity.rs, so the two ladders cannot part silently.
seabraid_host_test(seabraid_ladder_golden_gen ladder_golden_gen.cpp)
target_link_libraries(seabraid_ladder_golden_gen PRIVATE seabraid_retier)
add_test(NAME seabraid_ladder_golden
         COMMAND "${CMAKE_COMMAND}" "-DGEN=$<TARGET_FILE:seabraid_ladder_golden_gen>"
                 "-DGOLDEN=${CMAKE_CURRENT_SOURCE_DIR}/../../../src/placement/tests/ladder_golden.txt"
                 "-DOUT=${CMAKE_CURRENT_BINARY_DIR}/ladder_golden.regenerated.txt"
                 -P "${CMAKE_CURRENT_SOURCE_DIR}/ladder_golden_check.cmake")
