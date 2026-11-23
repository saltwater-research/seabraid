# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
# ctest -P script: regenerate the ladder golden with the C++ ladder (GEN OUT) and require it byte-identical to GOLDEN.
# FAILURE: fail-fast; a mismatch prints both paths so `diff` shows the drift.  WHY cmake -E compare_files and not a
# Python diff: the test then needs nothing beyond the CMake that already built it.
foreach(_v GEN GOLDEN OUT)
  if(NOT DEFINED ${_v})
    message(FATAL_ERROR "ladder_golden_check.cmake needs -D${_v}=...")
  endif()
endforeach()
if(NOT EXISTS "${GOLDEN}")
  message(FATAL_ERROR "ladder golden not found: ${GOLDEN}")
endif()
execute_process(COMMAND "${GEN}" "${OUT}" RESULT_VARIABLE _gen)
if(NOT _gen EQUAL 0)
  message(FATAL_ERROR "ladder_golden_gen exited ${_gen}")
endif()
execute_process(COMMAND "${CMAKE_COMMAND}" -E compare_files "${OUT}" "${GOLDEN}" RESULT_VARIABLE _same)
if(NOT _same EQUAL 0)
  message(FATAL_ERROR "the C++ ladder no longer reproduces the golden: diff ${OUT} ${GOLDEN}")
endif()
message(STATUS "ladder golden: identical (${GOLDEN})")
