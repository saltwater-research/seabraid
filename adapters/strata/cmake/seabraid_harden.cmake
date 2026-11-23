# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
# OWNER: the one definition of the hardened compile and link flags for the native code that speaks the helper wire:
# the Strata adapter (adapters/strata/CMakeLists.txt) and the Strata integration's seabraid-helper (its strata-ext build)
# include this file, so the two processes on either side of the channel cannot drift apart.  The edge is one-way: the
# add-on includes this file, and nothing here names the add-on (checks/check_compat_edges.py).
# WHY each flag (GCC/Clang only; other compilers get none rather than a guess):
#   -fstack-protector-strong       canaries on every frame with an array or address-taken local.
#   -D_FORTIFY_SOURCE=3            bounds-checked libc calls; needs optimization, so only for optimized configs, and
#                                  -U first because some distributions predefine a level.
#   _GLIBCXX_ASSERTIONS            bounds-checked operator[] and friends in libstdc++.
#   -Wformat=2 -Wformat-security   printf-family misuse is a warning everywhere.
#   PIE + -z relro -z now          executables only (libraries are PIC already and link into the host's own policy).
# Every flag is limited to C++ translation units: a target that also compiles CUDA or HIP keeps the GPU compiler's own
# flags (nvcc does not accept -fstack-protector-strong without -Xcompiler).
# CHECKER: adapters/strata/helper_hardening_test.cpp (the adapter's fixtures are built with these flags) and
# the add-on's tests/test_wire_owner.py (both consumers include this file and no other CMake file defines a flag).
include_guard(GLOBAL)

function(seabraid_harden target kind)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    return()
  endif()
  set(_cxx "$<COMPILE_LANGUAGE:CXX>")
  set(_optimized "$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>,$<CONFIG:MinSizeRel>>")
  target_compile_options(${target} PRIVATE
    "$<${_cxx}:-Wall;-Wextra;-Wformat=2;-Wformat-security;-fstack-protector-strong>"
    "$<$<AND:${_cxx},${_optimized}>:-U_FORTIFY_SOURCE;-D_FORTIFY_SOURCE=3>")
  target_compile_definitions(${target} PRIVATE "$<${_cxx}:_GLIBCXX_ASSERTIONS>")
  if(kind STREQUAL "EXE")
    set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_options(${target} PRIVATE -Wl,-z,relro,-z,now)
  elseif(NOT kind STREQUAL "LIB")
    message(FATAL_ERROR "seabraid_harden(${target} ${kind}): kind must be EXE or LIB")
  endif()
endfunction()
