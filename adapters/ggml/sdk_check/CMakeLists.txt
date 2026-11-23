# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
# A downstream consumer of adapters/ggml with SEABRAID_GGML_TESTS=OFF, exactly the llama.cpp hook's shape:
# the engine's ggml target, then the adapter, then one program linked against the adapter alone. The adapter must
# find the INSTALLED SDK through find_package(Seabraid) and link Seabraid::runtime (seabraid_log and
# seabraid_abi_epoch come only from there). Driven by run.sh, which installs the SDK first.
# OWNER: this directory. The fixtures (SEABRAID_GGML_TESTS=ON) define both symbols themselves and never reach this
# path, so nothing else exercises it.
cmake_minimum_required(VERSION 3.16)
project(seabraid_ggml_sdk_check LANGUAGES C CXX)

set(GGML_SOURCE "" CACHE PATH "A ggml source tree (the engine's ggml/ directory), built here CPU-only")
if(NOT EXISTS "${GGML_SOURCE}/CMakeLists.txt")
    message(FATAL_ERROR "pass -DGGML_SOURCE=<ggml source directory>")
endif()
set(GGML_CUDA OFF)
set(GGML_HIP OFF)
set(GGML_VULKAN OFF)
set(BUILD_SHARED_LIBS OFF)
add_subdirectory("${GGML_SOURCE}" ggml-build)

# The default is the tests-off adapter: the configure path under test. Do not set SEABRAID_GGML_TESTS here.
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/.." adapter-build)

enable_testing()
add_executable(seabraid_ggml_sdk_link sdk_link_main.c)
target_link_libraries(seabraid_ggml_sdk_link PRIVATE seabraid_ggml)
set_target_properties(seabraid_ggml_sdk_link PROPERTIES LINKER_LANGUAGE CXX)
add_test(NAME seabraid_ggml_sdk_link COMMAND seabraid_ggml_sdk_link)
