# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.22)
project(SeabraidInstalledSmoke LANGUAGES C)

# SEABRAID_REQUIRE_VERSION is what a consumer would write after the package name (e.g. 0.5); native-tests/installed/
# check_sdk.sh sets it to prove the same-minor rule. Empty requests no version.
set(SEABRAID_REQUIRE_VERSION "" CACHE STRING "Version passed to find_package(Seabraid ...)")
# Exercise the same placement ABI fixture through a downstream package import.
find_package(Seabraid ${SEABRAID_REQUIRE_VERSION} CONFIG REQUIRED)
add_executable(seabraid_installed_epoch epoch_check.c)
target_compile_features(seabraid_installed_epoch PRIVATE c_std_11)
target_compile_definitions(seabraid_installed_epoch PRIVATE CONFIG_EPOCH=${Seabraid_ABI_EPOCH})
target_link_libraries(seabraid_installed_epoch PRIVATE Seabraid::runtime)
add_executable(seabraid_installed_smoke ../placement_smoke.c)
target_compile_features(seabraid_installed_smoke PRIVATE c_std_11)
target_link_libraries(seabraid_installed_smoke PRIVATE Seabraid::runtime)

enable_testing()
add_test(NAME seabraid_installed_smoke COMMAND seabraid_installed_smoke)
add_test(NAME seabraid_installed_epoch COMMAND seabraid_installed_epoch)
if(WIN32)
    # The Windows loader needs the installed DLL directory when the fixture runs.
    set_tests_properties(seabraid_installed_smoke seabraid_installed_epoch PROPERTIES
        ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:$<TARGET_FILE_DIR:Seabraid::runtime>")
endif()
