# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
# seabraid's helper and tools, built inside a patched Strata tree (Strata CMake: -DSEABRAID_STRATA_SOURCE_DIR=<this checkout>).
# Inherits Strata's GPU backend: a CUDA Strata build gives a CUDA helper, a HIP build a HIP helper.
# seabraid.cmake owns the wire include path, the hardening flags and the host libraries (shared with tests/).
include("${CMAKE_CURRENT_SOURCE_DIR}/seabraid.cmake")
set(_bl ${CMAKE_CURRENT_SOURCE_DIR})
add_executable(seabraid-helper ${_bl}/src/helper.cpp ${_bl}/src/helper_serve.cpp ${_bl}/src/card_bench.cpp ${_bl}/src/card_bench_run.cpp)
# The wire (channel + helper_options.hpp, the controller<->helper CLI) is the Strata adapter's (compat -> adapters, never back).
target_include_directories(seabraid-helper PRIVATE ${_bl}/include ${_bl}/src ${SEABRAID_STRATA_WIRE_INCLUDE} ${_strata_gpu_include_directories})
if(NOT TARGET seabraid_file_fetch)
  message(FATAL_ERROR "seabraid_file_fetch needs Strata's vendored xxhash.h (third_party/strata-llama/vendor/hash/xxhash); set -DSEABRAID_STRATA_XXHASH_DIR=<dir>")
endif()
target_link_libraries(seabraid-helper PRIVATE seabraid_file_fetch strata_engine strata_prefill strata_spec ${_strata_gpu_runtime_target})
# The one process this add-on owns outright gets the same hardening as the adapter that spawns it (L17, R8 review).
seabraid_harden(seabraid-helper EXE)
# File moves (wire v9): writes / verifies <pack>/expert-sums.v1, the sums every file move is checked against.  Run by the installer.
add_executable(seabraid-sums ${_bl}/src/seabraid_sums.cpp)
target_include_directories(seabraid-sums PRIVATE ${_bl}/include ${_bl}/src ${SEABRAID_STRATA_WIRE_INCLUDE} ${_strata_gpu_include_directories})
target_link_libraries(seabraid-sums PRIVATE seabraid_file_fetch strata_engine ${_strata_gpu_runtime_target})
seabraid_harden(seabraid-sums EXE)
# ---- dsv41 helper: REMOVED for Strata >= 0.2.14.  Strata deleted the DeepSeek-V4.1 engine (src/dsv41, ds4_*) in 0.2.14/0.2.15, and these
# targets compiled sources from it (ds4_gguf.cu, ds4_iq.cu, ds4_tq.cu, ds4_gguf_cpu.cpp, ds4_helper_link.cpp ...); with them the CUDA configure fails
# ("Cannot find source file: src/dsv41/ds4_gguf_cpu.cpp").  Patches 0008/0009 (the dsv41 helper link) are dropped for the same reason.

# The add-on's host tests have ONE registration, tests/CMakeLists.txt; a Strata build with tests on runs the same list.
if(STRATA_BUILD_TESTS)
  add_subdirectory("${_bl}/../tests" "${CMAKE_BINARY_DIR}/seabraid-tests")
endif()
