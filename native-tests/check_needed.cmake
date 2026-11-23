# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
# Asserts the SONAME contract of the Linux library: the library declares SONAME, and a consumer records exactly
# that bare name in DT_NEEDED. Usage: cmake -DEXE=<elf> -DLIB=<libseabraid.so> -DSONAME=<libseabraid.so.0.5>
#                                           -P check_needed.cmake
# WHY a script: a path in DT_NEEDED only breaks when the engine starts from another directory, so the build
# must check the recorded entry directly, not rely on the cwd the test happens to run in. WHY SONAME is passed in:
# the expected name is derived from VERSION by CMake, so this test fails when build.rs and CMake disagree.
if(NOT EXE OR NOT EXISTS "${EXE}")
    message(FATAL_ERROR "check_needed: EXE must name an existing ELF file")
endif()
if(NOT LIB OR NOT EXISTS "${LIB}")
    message(FATAL_ERROR "check_needed: LIB must name the built libseabraid.so")
endif()
if(NOT SONAME MATCHES "^libseabraid\\.so\\.[0-9]+\\.[0-9]+$")
    message(FATAL_ERROR "check_needed: SONAME must look like libseabraid.so.MAJOR.MINOR, got '${SONAME}'")
endif()
find_program(_tool NAMES readelf objdump)
if(NOT _tool)
    message(FATAL_ERROR "check_needed: neither readelf nor objdump is installed")
endif()
get_filename_component(_tool_name "${_tool}" NAME)
set(_flag -d)
if(_tool_name STREQUAL "objdump")
    set(_flag -p)
endif()
string(REPLACE "." "\\." _re "${SONAME}")

# The library's own SONAME. readelf: "(SONAME) Library soname: [x]"; objdump: "SONAME               x".
execute_process(COMMAND "${_tool}" ${_flag} "${LIB}" RESULT_VARIABLE _rc OUTPUT_VARIABLE _lib_out)
if(NOT _rc EQUAL 0)
    message(FATAL_ERROR "check_needed: ${_tool_name} failed on the library (exit ${_rc})")
endif()
string(REGEX MATCHALL "[^\n]*SONAME[^\n]*" _sonames "${_lib_out}")
if(NOT _sonames)
    message(FATAL_ERROR "check_needed: the library has no SONAME")
endif()
foreach(_s IN LISTS _sonames)
    if(NOT _s MATCHES "(\\[${_re}\\]|[ \t]${_re})$")
        message(FATAL_ERROR "check_needed: library SONAME is not ${SONAME}: ${_s}")
    endif()
endforeach()

execute_process(COMMAND "${_tool}" ${_flag} "${EXE}" RESULT_VARIABLE _rc OUTPUT_VARIABLE _out)
if(NOT _rc EQUAL 0)
    message(FATAL_ERROR "check_needed: ${_tool_name} failed (exit ${_rc})")
endif()
# Only NEEDED lines count: a RUNPATH line may legitimately contain "seabraid" in a directory name.
# readelf: "(NEEDED) Shared library: [libseabraid.so.0.5]"; objdump: "NEEDED               libseabraid.so.0.5".
string(REGEX MATCHALL "[^\n]*NEEDED[^\n]*seabraid[^\n]*" _entries "${_out}")
if(NOT _entries)
    message(FATAL_ERROR "check_needed: no libseabraid entry in DT_NEEDED")
endif()
foreach(_e IN LISTS _entries)
    if(NOT _e MATCHES "(\\[${_re}\\]|[ \t]${_re})$")
        message(FATAL_ERROR "check_needed: DT_NEEDED is not the bare soname ${SONAME}: ${_e}")
    endif()
endforeach()
