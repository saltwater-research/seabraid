# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
# Runs one command and passes only when BOTH its exit code and its output are right.
#
# CONTRACT
#   cmake -DEXPECT_REGEX=<regex> [-DEXPECT_RC=<n>] [-DFORBID_REGEX=<regex>] [-DTIMEOUT=<seconds>]
#         -P check_output.cmake -- <command> [args...]
#   EXPECT_REGEX  required. A CMake regex (CMake has no multiline mode: ^ and $ anchor the whole text, use
#                 (^|\n) and \n to anchor a line). Matched against stdout and stderr merged in one string.
#   EXPECT_RC     optional, default 0. The exit code the command must return (compared as a string, so a signal
#                 name such as "Segmentation fault" can be expected too).
#   FORBID_REGEX  optional. The check fails if this matches the output.
#   TIMEOUT       optional, a whole number of seconds (digits only; anything else stops the script with a message
#                 naming TIMEOUT). A command that overruns is killed and the check fails. -DTIMEOUT=0 is NOT "no limit":
#                 this script fails at once with "timed out after 0 seconds" and never starts the command, on every CMake.
#                 (WHY it does not pass 0 on: execute_process(TIMEOUT 0) is "no limit" on CMake 3.22.6 and "time out at
#                 once" on 4.2.3, both measured; tests/test_native_checks.py pins each.) Leave TIMEOUT unset (or empty) for no limit.
#   Everything after the first "--" following the script name is the command. It reaches the OS one element per
#   argument, unchanged: empty arguments, semicolons, brackets and a command named false/off/no all survive.
# LIMITS (each is refused loudly, never run wrongly):
#   - an argument equal to an execute_process keyword (COMMAND, TIMEOUT, OUTPUT_VARIABLE, ...) would be read by CMake
#     as that keyword, so the script stops with an error naming it; wrap the command in `sh -c` to pass such text;
#   - an argument holding a carriage return is refused (CMake's script reader rewrites it).
#   - on TIMEOUT only the command itself is killed and the script returns promptly (measured 1.03 s for a 1 s timeout
#     with a `sleep 30 &` grandchild); the grandchild is left running, orphaned, and a test that leaves such children
#     should exec or clean up after itself. On a NORMAL exit the opposite holds: a grandchild that keeps the output
#     pipe open (a shell that forked and did not exec) delays the return until it closes the pipe or exits.
# EXIT: 0 when the exit code equals EXPECT_RC and the output matches; otherwise FATAL_ERROR (nonzero) with the
#   reason and the command's whole output. A command that overruns TIMEOUT always fails with "timed out", even when
#   EXPECT_RC names the timeout's own result string.
#
# WHY this exists: CTest ignores the exit code of a test that sets PASS_REGULAR_EXPRESSION, so
# `echo ok; exit 3` passes such a test. This script is the one place that checks both, and every test that wants
# "right output AND clean exit" runs through it (root CMakeLists.txt, tools/sim, examples/consumer).
# CHECKER: tests/test_native_checks.py plants a failing exit code with matching output, a missing line, a forbidden
# line and a timeout, and requires this script to fail each time (it skips when cmake is not installed). It also
# plants a TIMEOUT that carries a second command and requires it to be refused, and fails when `cmake --help-command
# execute_process` documents a keyword that _keywords below lacks.

# WHY a floor: -P mode otherwise runs under the oldest CMake policies, and this script relies on IN_LIST and
# cmake_language(EVAL), which the root CMakeLists.txt's own minimum (3.22) already guarantees.
cmake_minimum_required(VERSION 3.22)

if(NOT DEFINED EXPECT_REGEX OR EXPECT_REGEX STREQUAL "")
    message(FATAL_ERROR "check_output: -DEXPECT_REGEX=<regex> is required")
endif()
if(NOT DEFINED EXPECT_RC)
    set(EXPECT_RC 0)
endif()
# WHY validated: the timeout is spliced into the text that cmake_language(EVAL) runs, so an unchecked value is code
# (a newline followed by file(WRITE ...) ran), and a non-number is a parse error with no mention of TIMEOUT.
if(DEFINED TIMEOUT AND NOT TIMEOUT STREQUAL "" AND NOT TIMEOUT MATCHES "^[0-9]+$")
    message(FATAL_ERROR "check_output: -DTIMEOUT must be a whole number of seconds (digits only), not '${TIMEOUT}'")
endif()

# Locate the command: everything after "--" that follows "-P <script>".
# WHY not a list: a CMake list drops empty elements and splits on ";", and execute_process(COMMAND ${list}) would
# also lose arguments. Each argument is instead written as a bracket argument (always literal, may be empty) into
# the text of one execute_process call that cmake_language(EVAL) then runs. A newline right after the opening
# bracket is skipped by CMake, so one is always emitted and an argument's own leading newline survives.
# WHY a hard-coded keyword list: execute_process itself reads an argument equal to one of its keywords as that keyword,
# and neither bracket arguments nor a temporary file change that (the keyword is matched on the argument's value, after
# parsing), so there is no way to pass such an argument; the only safe move is to refuse it. The list is the synopsis
# of `cmake --help-command execute_process` (CMake 4.2); tests/test_native_checks.py compares it with the installed
# CMake's synopsis, so a release that adds a keyword fails a test instead of being silently misread.
set(_keywords COMMAND WORKING_DIRECTORY TIMEOUT RESULT_VARIABLE RESULTS_VARIABLE OUTPUT_VARIABLE ERROR_VARIABLE
    INPUT_FILE OUTPUT_FILE ERROR_FILE OUTPUT_QUIET ERROR_QUIET COMMAND_ECHO OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE ENCODING ECHO_OUTPUT_VARIABLE ECHO_ERROR_VARIABLE COMMAND_ERROR_IS_FATAL)
set(_code "")
set(_shown "")
set(_count 0)
set(_seen_script FALSE)
set(_collect FALSE)
math(EXPR _last "${CMAKE_ARGC} - 1")
foreach(_i RANGE 0 ${_last})
    set(_arg "${CMAKE_ARGV${_i}}")
    if(_collect)
        if(_arg IN_LIST _keywords)
            message(FATAL_ERROR "check_output: the argument '${_arg}' is an execute_process keyword and cannot be passed through; wrap the command in sh -c")
        endif()
        if(_arg MATCHES "\r")
            message(FATAL_ERROR "check_output: an argument holds a carriage return, which CMake cannot pass through")
        endif()
        set(_eq "=")
        while(_arg MATCHES "\\]${_eq}")
            string(APPEND _eq "=")
        endwhile()
        string(APPEND _code " [${_eq}[\n${_arg}]${_eq}]")
        string(APPEND _shown " [${_arg}]")
        math(EXPR _count "${_count} + 1")
    elseif(_seen_script AND _arg STREQUAL "--")
        set(_collect TRUE)
    elseif(_arg STREQUAL "-P")
        set(_seen_script TRUE)
    endif()
endforeach()
if(_count EQUAL 0)
    message(FATAL_ERROR "check_output: no command given (expected: -P check_output.cmake -- <command> [args...])")
endif()

# WHY before running: see TIMEOUT in the header; 0 means different things to different CMake versions, so it is decided here.
if(DEFINED TIMEOUT AND TIMEOUT MATCHES "^[0-9]+$" AND TIMEOUT EQUAL 0)
    message(FATAL_ERROR "check_output: timed out after ${TIMEOUT} seconds (a timeout of 0 gives the command no time; it was not started): ${_shown}")
endif()

set(_timeout_args "")
if(DEFINED TIMEOUT AND NOT TIMEOUT STREQUAL "")
    # WHY a reference and not the value: the EVAL text then holds no caller-supplied characters at all.
    set(_timeout_args " TIMEOUT \${TIMEOUT}")
endif()
cmake_language(EVAL CODE "execute_process(COMMAND${_code}${_timeout_args} RESULT_VARIABLE _rc OUTPUT_VARIABLE _out ERROR_VARIABLE _out)")
message("${_out}")

if("${_rc}" MATCHES "timeout")
    message(FATAL_ERROR "check_output: timed out after ${TIMEOUT} seconds ('${_rc}'): ${_shown}")
endif()
if(NOT "${_rc}" STREQUAL "${EXPECT_RC}")
    message(FATAL_ERROR "check_output: exit code was '${_rc}', expected '${EXPECT_RC}': ${_shown}")
endif()
if(NOT "${_out}" MATCHES "${EXPECT_REGEX}")
    message(FATAL_ERROR "check_output: output does not match '${EXPECT_REGEX}': ${_shown}")
endif()
if(DEFINED FORBID_REGEX AND NOT FORBID_REGEX STREQUAL "" AND "${_out}" MATCHES "${FORBID_REGEX}")
    message(FATAL_ERROR "check_output: output matches the forbidden '${FORBID_REGEX}': ${_shown}")
endif()
