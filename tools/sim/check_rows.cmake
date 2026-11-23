# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
# CTest checker for the row counters in seabraid_sim's JSON line. WHY a script: ctest's regex cannot compare two
# numbers, and the claim under test is an inequality (rows_shipped < rows_submitted when COMPACT_ROWS drops rows).
# Usage: cmake -DSIM=<exe> "-DARGS=<a;b;...>" -DEXPECT=less|equal -P check_rows.cmake
if(NOT EXPECT MATCHES "^(less|equal)$")
    message(FATAL_ERROR "EXPECT must be less or equal")
endif()
execute_process(COMMAND "${SIM}" ${ARGS} OUTPUT_VARIABLE out RESULT_VARIABLE rc)
message(STATUS "${out}")
if(NOT rc EQUAL 0)
    message(FATAL_ERROR "seabraid_sim exited ${rc}")
endif()
string(JSON ok GET "${out}" ok)
string(JSON submitted GET "${out}" rows_submitted)
string(JSON shipped GET "${out}" rows_shipped)
if(NOT ok)
    message(FATAL_ERROR "ok is ${ok}")
endif()
if(submitted EQUAL 0)
    message(FATAL_ERROR "rows_submitted is 0: the run submitted nothing")
endif()
if(EXPECT STREQUAL "less" AND NOT shipped LESS submitted)
    message(FATAL_ERROR "expected rows_shipped < rows_submitted, got ${shipped} vs ${submitted}")
elseif(EXPECT STREQUAL "equal" AND NOT shipped EQUAL submitted)
    message(FATAL_ERROR "expected rows_shipped == rows_submitted, got ${shipped} vs ${submitted}")
endif()
message(STATUS "rows_submitted ${submitted} rows_shipped ${shipped} (${EXPECT}): ok")
