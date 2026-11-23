# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
# Checks SEABRAID_DEPRECATED is opt-in: a deprecated v1 call compiles silently by default and warns (an error under
# -Werror=deprecated-declarations) once SEABRAID_WARN_DEPRECATED is defined, in strict C89 both times.
# Run as: cmake -DCC=<c compiler> -DINCLUDE=<include dir> -DSOURCE=<deprecated_use.c> -P check_deprecated.cmake
foreach(var CC INCLUDE SOURCE)
    if(NOT ${var})
        message(FATAL_ERROR "check_deprecated.cmake: -D${var}=... is required")
    endif()
endforeach()

set(common -std=c89 -pedantic-errors -fsyntax-only -I "${INCLUDE}" "${SOURCE}")

execute_process(COMMAND "${CC}" -Wall -Wextra -Werror ${common}
    RESULT_VARIABLE quiet_result ERROR_VARIABLE quiet_text)
if(NOT quiet_result EQUAL 0)
    message(FATAL_ERROR "the default build must not warn about deprecated entry points:\n${quiet_text}")
endif()

execute_process(COMMAND "${CC}" -DSEABRAID_WARN_DEPRECATED -Werror=deprecated-declarations ${common}
    RESULT_VARIABLE loud_result ERROR_VARIABLE loud_text)
if(loud_result EQUAL 0 OR NOT loud_text MATCHES "seabraid_open_reference")
    message(FATAL_ERROR "SEABRAID_WARN_DEPRECATED did not flag seabraid_open_reference (exit ${loud_result}):\n${loud_text}")
endif()
message(STATUS "deprecation is opt-in and fires: ${loud_text}")
