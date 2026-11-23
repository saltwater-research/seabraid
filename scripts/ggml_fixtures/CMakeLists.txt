# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
# The adapters/ggml fixtures (SEABRAID_GGML_TESTS=ON) built against the engine's ggml source, CPU-only, and run by CTest.
# OWNER: this directory; driven by run.sh and by the ci.yml `sdk` job.  WHY here and not in adapters/ggml: the adapter's own
# CMake wants an installed ggml package; CI has the ggml SOURCE (the same pinned checkout the sdk link check uses), so this
# tiny wrapper adds it as a subdirectory, exactly as adapters/ggml/sdk_check does for the tests-off consumer.
cmake_minimum_required(VERSION 3.22)
project(seabraid_ggml_fixtures LANGUAGES C CXX)

set(GGML_SOURCE "" CACHE PATH "A ggml source tree (the engine's ggml/ directory), built here CPU-only")
if(NOT EXISTS "${GGML_SOURCE}/CMakeLists.txt")
    message(FATAL_ERROR "pass -DGGML_SOURCE=<ggml source directory>")
endif()
set(GGML_CUDA OFF)
set(GGML_HIP OFF)
set(GGML_VULKAN OFF)
set(BUILD_SHARED_LIBS OFF)
add_subdirectory("${GGML_SOURCE}" ggml-build)

# WHY FORCE: the fixtures define seabraid_log and seabraid_abi_epoch themselves, so the tests-on path (no SDK needed) is the point.
set(SEABRAID_GGML_TESTS ON CACHE BOOL "the ggml fixtures" FORCE)
enable_testing()
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../adapters/ggml" adapter-build)
