# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
# OWNER: the host-only half of seabraid's native build, shared by the two places it is built: strata-ext/CMakeLists.txt
# (inside a patched Strata tree, next to seabraid-helper) and tests/CMakeLists.txt (standalone, no Strata at all).
# It owns three things and nothing else:
#   SEABRAID_STRATA_WIRE_INCLUDE  the helper wire contract's directory.  adapters/strata/wire owns the wire (channel.hpp,
#                         channel_layout.hpp, expert_spec.hpp, helper_options.hpp); compat includes it from there and
#                         never carries a copy (compat -> adapters, never back; see channel_layout.hpp for why).
#   seabraid_harden()     the hardening flags, from adapters/strata/cmake/seabraid_harden.cmake (one owner for both ends
#                         of the channel).
#   seabraid_tier_plan / seabraid_retier   the tier policy and the re-tier loop: host C++, no GPU, no Strata headers.
# WHY include_guard: both consumers may include it in one build (strata-ext, then its tests); the targets exist once.
include_guard(GLOBAL)

get_filename_component(SEABRAID_STRATA_WIRE_INCLUDE "${CMAKE_CURRENT_LIST_DIR}/../../../adapters/strata/wire" ABSOLUTE)
if(NOT EXISTS "${SEABRAID_STRATA_WIRE_INCLUDE}/strata/seabraid/channel.hpp")
  message(FATAL_ERROR "seabraid builds from a full Seabraid checkout: the helper wire is ${SEABRAID_STRATA_WIRE_INCLUDE} (missing)")
endif()
include("${CMAKE_CURRENT_LIST_DIR}/../../../adapters/strata/cmake/seabraid_harden.cmake")
find_package(Threads REQUIRED)

set(_seabraid_ext "${CMAKE_CURRENT_LIST_DIR}")
add_library(seabraid_tier_plan STATIC ${_seabraid_ext}/src/tier_plan.cpp ${_seabraid_ext}/src/tier_ladder.cpp)   # two-set round + the N-rung ladder (M3b)
target_compile_features(seabraid_tier_plan PUBLIC cxx_std_17)
target_include_directories(seabraid_tier_plan PUBLIC ${_seabraid_ext}/include ${SEABRAID_STRATA_WIRE_INCLUDE})
add_library(seabraid_retier STATIC ${_seabraid_ext}/src/retier_loop.cpp ${_seabraid_ext}/src/retier_help.cpp ${_seabraid_ext}/src/_retier_ladder.cpp)
target_link_libraries(seabraid_retier PUBLIC seabraid_tier_plan Threads::Threads)
# WHY the two libraries are NOT hardened here: they link into Strata's own `strata` binary, whose flags are Strata's
# policy; only the process this add-on owns outright (seabraid-helper) takes seabraid_harden.

# ---- file moves (wire v9): seabraid_file_fetch, the helper's reader of a FileRef (src/file_fetch.cpp).
# Host-only like the two libraries above, but it hashes with Strata's vendored xxhash (XXH3-64 is the wire's sum, and one vendored copy
# is the one definition), so it needs that header's directory: found by itself inside a Strata build (CMAKE_SOURCE_DIR), else from
# -DSEABRAID_STRATA_XXHASH_DIR=<dir> or a Strata checkout named by -DSTRATA_SOURCE / $STRATA_SOURCE.  Without it the target does not exist, and
# tests/CMakeLists.txt registers one FAILING test that says so, so a missing hash can never look like a green run.
set(SEABRAID_STRATA_XXHASH_DIR "" CACHE PATH "directory holding Strata's vendored xxhash.h (third_party/strata-llama/vendor/hash/xxhash)")
if(NOT SEABRAID_STRATA_XXHASH_DIR)
  foreach(_root IN ITEMS "${CMAKE_SOURCE_DIR}" "${STRATA_SOURCE}" "$ENV{STRATA_SOURCE}")
    if(_root AND EXISTS "${_root}/third_party/strata-llama/vendor/hash/xxhash/xxhash.h")
      set(SEABRAID_STRATA_XXHASH_DIR "${_root}/third_party/strata-llama/vendor/hash/xxhash" CACHE PATH "" FORCE)
      break()
    endif()
  endforeach()
endif()
if(SEABRAID_STRATA_XXHASH_DIR AND EXISTS "${SEABRAID_STRATA_XXHASH_DIR}/xxhash.h")
  add_library(seabraid_file_fetch STATIC ${_seabraid_ext}/src/file_fetch.cpp)
  target_compile_features(seabraid_file_fetch PUBLIC cxx_std_17)
  target_include_directories(seabraid_file_fetch PUBLIC ${_seabraid_ext}/src ${SEABRAID_STRATA_WIRE_INCLUDE} PRIVATE ${SEABRAID_STRATA_XXHASH_DIR})
endif()
