# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
# Exercise the engine's actual argv loop. No request, model or helper is needed for a flag-only refusal.
function(expect_refusal diagnostic)
  execute_process(COMMAND "${ENGINE}" ${ARGN} RESULT_VARIABLE status ERROR_VARIABLE error TIMEOUT 15)
  string(FIND "${error}" "${diagnostic}" match)
  if(NOT "${status}" STREQUAL "2" OR match EQUAL -1)
    message(FATAL_ERROR "${ARGN}: expected exit 2 and '${diagnostic}', got ${status}: ${error}")
  endif()
endfunction()

expect_refusal("--retier-promote legacy|seabraid needs --retier on" --serve --retier-promote seabraid)
expect_refusal("--retier-file-moves 1 needs --retier-promote seabraid" --serve --retier-file-moves 1)
expect_refusal("--retier-fallback 1 and --retier-hops > 1 need --retier-promote seabraid" --retier off --retier-hops 2)
# A later --retier on supplies the prerequisite. The remaining error is that this invocation omitted --serve.
expect_refusal("--retier requires --serve" --retier-promote seabraid --retier on)
