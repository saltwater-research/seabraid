# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.22)

# VERSION is the single source for the release number. CMake versions are numeric, so the
# pre-release suffix ("-alpha.1") is kept only for the seabraid_version() check and Cargo.
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" seabraid_version_text)
string(STRIP "${seabraid_version_text}" seabraid_version_full)
string(REGEX MATCH "^[0-9]+\\.[0-9]+\\.[0-9]+" seabraid_version_numeric "${seabraid_version_full}")
if(NOT seabraid_version_numeric)
    message(FATAL_ERROR "VERSION must start with MAJOR.MINOR.PATCH")
endif()
# The header owns SEABRAID_ABI_EPOCH (read below into the package config), so editing it must re-run configure.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS VERSION Cargo.toml include/seabraid/seabraid.h)
# Cargo reports this number through seabraid_version(), so a mismatch must stop the build.
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml" seabraid_cargo_version REGEX "^version = \"" LIMIT_COUNT 1)
if(NOT seabraid_cargo_version STREQUAL "version = \"${seabraid_version_full}\"")
    message(FATAL_ERROR "Cargo.toml (${seabraid_cargo_version}) disagrees with VERSION (${seabraid_version_full})")
endif()

project(Seabraid VERSION ${seabraid_version_numeric} LANGUAGES NONE)

# While the major version is 0 any minor bump may break the ABI, so the ELF SONAME carries MAJOR.MINOR. build.rs
# derives the same string from Cargo's version, and the SONAME CTest asserts that the two agree. Windows keeps
# seabraid.dll (no ABI in the name), macOS keeps libseabraid.dylib with a compatibility version.
# WHY Linux only (not "UNIX AND NOT APPLE"): build.rs sets the SONAME on target_os = linux and nowhere else, so
# FreeBSD and the other ELF systems get a plain libseabraid.so and must not be told otherwise.
set(seabraid_soname "libseabraid.so.${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}")
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(seabraid_has_soname TRUE)
else()
    set(seabraid_has_soname FALSE)
endif()
# VERSION says 0.5.0-alpha.8 but CMake's version, and so SeabraidConfigVersion.cmake, is 0.5.0: a pre-release is
# not visible to find_package. The project is LANGUAGES NONE, so the package version file also skips its 32/64-bit
# check; the header's 64-bit-pointer typedef rejects 32-bit targets instead. See docs/ABI.md, "Versioning".
# The layout epoch has one owner, the public header; the package config republishes it as Seabraid_ABI_EPOCH.
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/include/seabraid/seabraid.h" seabraid_epoch_line
     REGEX "^#define SEABRAID_ABI_EPOCH [0-9]+u" LIMIT_COUNT 1)
if(NOT seabraid_epoch_line MATCHES "SEABRAID_ABI_EPOCH ([0-9]+)u")
    message(FATAL_ERROR "include/seabraid/seabraid.h must define SEABRAID_ABI_EPOCH")
endif()
set(seabraid_abi_epoch "${CMAKE_MATCH_1}")

# Cargo owns the native artifact. Consumers link the same binary and header
# that the ABI tests exercise.
if(CMAKE_CROSSCOMPILING)
    message(FATAL_ERROR "The Seabraid Cargo target must match the native CMake host")
endif()

find_program(SEABRAID_CARGO_EXECUTABLE NAMES cargo REQUIRED)
set(seabraid_target_dir "${CMAKE_CURRENT_BINARY_DIR}/cargo")
set(seabraid_artifact_dir "${seabraid_target_dir}/release")

set(seabraid_soname_link "")
if(WIN32)
    # rustc names the import library by ABI: seabraid.dll.lib for MSVC, libseabraid.dll.a for windows-gnu.
    execute_process(COMMAND "${SEABRAID_CARGO_EXECUTABLE}" -vV OUTPUT_VARIABLE seabraid_cargo_info)
    set(seabraid_import_name seabraid.dll.lib)
    if(seabraid_cargo_info MATCHES "(^|\n)host: [^\n]*-windows-gnu")
        set(seabraid_import_name libseabraid.dll.a)
    endif()
    set(seabraid_runtime "${seabraid_artifact_dir}/seabraid.dll")
    set(seabraid_import "${seabraid_artifact_dir}/${seabraid_import_name}")
    set(seabraid_outputs "${seabraid_runtime}" "${seabraid_import}")
elseif(APPLE)
    set(seabraid_runtime "${seabraid_artifact_dir}/libseabraid.dylib")
    set(seabraid_outputs "${seabraid_runtime}")
else()
    set(seabraid_runtime "${seabraid_artifact_dir}/libseabraid.so")
    set(seabraid_outputs "${seabraid_runtime}")
    if(seabraid_has_soname)
        # Cargo writes only libseabraid.so. The loader looks for the SONAME, so the build tree gets a symlink of
        # that name next to it (the install tree gets a real file, see below).
        set(seabraid_soname_link "${seabraid_artifact_dir}/${seabraid_soname}")
        list(APPEND seabraid_outputs "${seabraid_soname_link}")
    endif()
endif()

# cmake/build_cargo.cmake runs cargo, makes the SONAME symlink and fails when any output is missing.
# src/layout/abi_layout.txt is include_bytes!'d by src/layout.rs, so it is a dependency like the sources.
file(GLOB_RECURSE seabraid_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.rs")
add_custom_command(
    OUTPUT ${seabraid_outputs}
    COMMAND "${CMAKE_COMMAND}" "-DCARGO=${SEABRAID_CARGO_EXECUTABLE}" "-DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
            "-DTARGET_DIR=${seabraid_target_dir}" "-DOUTPUTS=${seabraid_outputs}"
            "-DSONAME_LINK=${seabraid_soname_link}" "-DSONAME_LINK_TARGET=libseabraid.so"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/build_cargo.cmake"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    DEPENDS ${seabraid_sources} Cargo.toml Cargo.lock build.rs cmake/build_cargo.cmake
            "${CMAKE_CURRENT_SOURCE_DIR}/src/layout/abi_layout.txt"
    VERBATIM
)
add_custom_target(seabraid_runtime_build ALL DEPENDS ${seabraid_outputs})

add_library(Seabraid::runtime SHARED IMPORTED GLOBAL)
set_target_properties(Seabraid::runtime PROPERTIES
    IMPORTED_LOCATION "${seabraid_runtime}"
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
if(WIN32)
    set_property(TARGET Seabraid::runtime PROPERTY IMPORTED_IMPLIB "${seabraid_import}")
endif()
if(seabraid_has_soname)
    # build.rs gives the cdylib this SONAME. Declaring it makes consumers record the bare name in
    # DT_NEEDED (not the build path) and lets CMake add the library directory to the build RPATH.
    set_property(TARGET Seabraid::runtime PROPERTY IMPORTED_SONAME "${seabraid_soname}")
endif()
add_dependencies(Seabraid::runtime seabraid_runtime_build)

# Keep standalone install rules out of hosts that embed Seabraid with add_subdirectory.
if(PROJECT_IS_TOP_LEVEL)
    include(CMakePackageConfigHelpers)
    # WHY the LIBDIR default: this project enables no language, and GNUInstallDirs warns (and cannot pick lib64
    # or a multiarch directory) without one. "lib" is what the SDK has always installed; -DCMAKE_INSTALL_LIBDIR
    # still overrides it. WHY STRING and not PATH: an untyped -DCMAKE_INSTALL_LIBDIR=lib64 on the command line is
    # turned into a path relative to the current directory when the cache entry is typed PATH, which installed the
    # library outside the prefix. LIBDIR is relative to the prefix unless the caller gives an absolute path.
    set(CMAKE_INSTALL_LIBDIR "lib" CACHE STRING "Object code libraries (lib, lib64, lib/<multiarch> or an absolute path)")
    include(GNUInstallDirs)
    # WHY refuse these: the LIBDIR is pasted into install(CODE "...") text below, which CMake parses a second time at
    # install time. A quote ends the string, `$` starts a variable reference and `\` an escape, so a value like lib$x
    # would install somewhere the configure never named; `;` splits a list, so lib;x names two directories; a leading
    # `~` is IS_ABSOLUTE to CMake (`~lib` reads as a home-directory path) yet installs a directory literally named
    # `~lib` under the current directory. Refusing at configure is the one place that fails before anything is
    # built; no distribution layout uses these characters.
    set(seabraid_dir_vars CMAKE_INSTALL_LIBDIR CMAKE_INSTALL_INCLUDEDIR)
    if(WIN32)
        # The Windows runtime path (BINDIR) is pasted into install(CODE) too, in the absolute-LIBDIR layout.
        list(APPEND seabraid_dir_vars CMAKE_INSTALL_BINDIR)
    endif()
    foreach(seabraid_dir_var IN LISTS seabraid_dir_vars)
        string(FIND "${${seabraid_dir_var}}" "\n" seabraid_newline_at)
        if("${${seabraid_dir_var}}" MATCHES [=[["$\;]]=] OR "${${seabraid_dir_var}}" MATCHES "^~" OR NOT seabraid_newline_at EQUAL -1)
            message(FATAL_ERROR "${seabraid_dir_var}='${${seabraid_dir_var}}' contains a quote, '$', a backslash, ';', a newline or a leading '~'; "
                    "the install rules cannot carry it. Use forward slashes and plain directory names.")
        endif()
    endforeach()
    # A plain `cmake --install` installs both the Runtime and Development components.
    # WHY this is the FIRST install rule, and in both components: a ';' in the prefix is refused in every layout (the
    # absolute-LIBDIR config cannot carry it, and a list-splitting prefix is a trap for every later rule), and refusing
    # after the runtime and headers were copied left a half-installed prefix. install(CODE) takes one component, so the
    # same text is installed once per component; a plain install runs it twice, which is harmless. It tests the prefix
    # as given: a relative prefix is resolved later, and the resolved path is checked again where it is baked in.
    set(seabraid_prefix_guard [=[
        if("${CMAKE_INSTALL_PREFIX}" MATCHES ";|\n")
            message(FATAL_ERROR "install prefix '${CMAKE_INSTALL_PREFIX}' contains ';' or a newline; the install rules cannot carry it")
        endif()
    ]=])
    install(CODE "${seabraid_prefix_guard}" COMPONENT Runtime)
    install(CODE "${seabraid_prefix_guard}" COMPONENT Development)
    set(seabraid_sdk_libdir "${CMAKE_INSTALL_LIBDIR}")
    set(seabraid_sdk_include "${CMAKE_INSTALL_INCLUDEDIR}")
    set(seabraid_sdk_config_dir "${seabraid_sdk_libdir}/cmake/Seabraid")
    set(seabraid_sdk_runtime_dir "${seabraid_sdk_libdir}")
    # WHY seabraid_sdk_include_cfg: with an absolute LIBDIR the package config is installed outside the prefix, and
    # configure_package_config_file derives its prefix from the CONFIGURE-time CMAKE_INSTALL_PREFIX, so a relative
    # include directory would resolve against the wrong prefix whenever `cmake --install --prefix` differs from it.
    # In that one case the config carries a sentinel that an install(CODE) rule below replaces with the install-time
    # prefix; everywhere else the config is relative to its own location and relocates.
    # The Windows runtime path is the same case: it lives under BINDIR, which is relative to the prefix, while the config
    # sits under an absolute LIBDIR. (Written without a Windows host to run it: the Linux legs never take that branch.)
    set(seabraid_include_sentinel "SEABRAID_INCLUDE_DIR_AT_INSTALL")
    set(seabraid_runtime_sentinel "SEABRAID_RUNTIME_PATH_AT_INSTALL")
    set(seabraid_include_at_install FALSE)
    set(seabraid_runtime_at_install FALSE)
    set(seabraid_sdk_include_cfg "${seabraid_sdk_include}")
    if(IS_ABSOLUTE "${seabraid_sdk_libdir}" AND NOT IS_ABSOLUTE "${seabraid_sdk_include}")
        set(seabraid_include_at_install TRUE)
        set(seabraid_sdk_include_cfg "${seabraid_include_sentinel}")
    endif()
    set(seabraid_sdk_paths seabraid_sdk_include_cfg seabraid_sdk_runtime_cfg)
    if(WIN32)
        # The DLL is loaded at runtime and the import library is used by the linker.
        set(seabraid_sdk_runtime_dir "${CMAKE_INSTALL_BINDIR}")
        get_filename_component(seabraid_import_name "${seabraid_import}" NAME)
        set(seabraid_sdk_import "${seabraid_sdk_libdir}/${seabraid_import_name}")
        list(APPEND seabraid_sdk_paths seabraid_sdk_import)
        install(FILES "${seabraid_import}" DESTINATION "${seabraid_sdk_libdir}" COMPONENT Development)
    endif()
    get_filename_component(seabraid_runtime_name "${seabraid_runtime}" NAME)
    if(seabraid_has_soname)
        # The installed file carries the SONAME as its name; libseabraid.so is the development symlink for -lseabraid.
        set(seabraid_runtime_name "${seabraid_soname}")
    endif()
    set(seabraid_sdk_runtime "${seabraid_sdk_runtime_dir}/${seabraid_runtime_name}")
    set(seabraid_sdk_runtime_cfg "${seabraid_sdk_runtime}")
    if(WIN32 AND IS_ABSOLUTE "${seabraid_sdk_libdir}" AND NOT IS_ABSOLUTE "${seabraid_sdk_runtime_dir}")
        set(seabraid_runtime_at_install TRUE)
        set(seabraid_sdk_runtime_cfg "${seabraid_runtime_sentinel}")
    endif()
    # WHY install(PROGRAMS): install(FILES) gave the library mode 0644, which the loader tolerates but packaging
    # tools and `ldd` warn about; PROGRAMS installs it 0755 like a built shared library. Strip is decided in
    # Cargo.toml ([profile.release] strip), because `cmake --install --strip` does not strip files installed this way.
    install(PROGRAMS "${seabraid_runtime}" DESTINATION "${seabraid_sdk_runtime_dir}"
            RENAME "${seabraid_runtime_name}" COMPONENT Runtime)
    if(seabraid_has_soname)
        # WHY install(CODE) and not install(FILES) of a build-tree symlink: the link must be relative and created
        # in the destination. WHY the manifest line: CMake records only what its own install() rules copy, so
        # without it `xargs rm < install_manifest.txt` would leave the dev symlink behind.
        install(CODE "
            set(seabraid_libdir \"${seabraid_sdk_runtime_dir}\")
            if(IS_ABSOLUTE \"\${seabraid_libdir}\")
                set(seabraid_dev_dir \"\${seabraid_libdir}\")
            else()
                set(seabraid_dev_dir \"\${CMAKE_INSTALL_PREFIX}/\${seabraid_libdir}\")
            endif()
            set(seabraid_dev_link \"\${seabraid_dev_dir}/libseabraid.so\")
            # WHY MAKE_DIRECTORY: a Development-only install into a fresh prefix has no lib directory yet (the
            # Runtime component creates it). WHY DESTDIR only on the file operations: CMake's own manifest entries
            # do not carry DESTDIR, so neither does ours.
            file(MAKE_DIRECTORY \"\$ENV{DESTDIR}\${seabraid_dev_dir}\")
            file(REMOVE \"\$ENV{DESTDIR}\${seabraid_dev_link}\")
            file(CREATE_LINK \"${seabraid_soname}\" \"\$ENV{DESTDIR}\${seabraid_dev_link}\" SYMBOLIC)
            list(APPEND CMAKE_INSTALL_MANIFEST_FILES \"\${seabraid_dev_link}\")
        " COMPONENT Development)
    endif()
    install(DIRECTORY include/seabraid DESTINATION "${seabraid_sdk_include}" COMPONENT Development
            FILES_MATCHING PATTERN "*.h")
    # Both components ship license texts and NOTICE. Exclude the second rule from full installs to record each file once.
    install(FILES LICENSE LICENSE-MIT NOTICE DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/seabraid" COMPONENT Runtime)
    install(FILES LICENSE LICENSE-MIT NOTICE DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/seabraid" COMPONENT Development EXCLUDE_FROM_ALL)

    # Imported targets cannot be exported with install(EXPORT). Generate the
    # installed target with paths relative to the package's current prefix.
    configure_package_config_file(
        cmake/SeabraidConfig.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/SeabraidConfig.cmake"
        INSTALL_DESTINATION "${seabraid_sdk_config_dir}"
        PATH_VARS ${seabraid_sdk_paths}
    )
    # While the major version is 0 any minor bump may break the ABI, so only the same minor matches.
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/SeabraidConfigVersion.cmake"
        VERSION "${PROJECT_VERSION}"
        COMPATIBILITY SameMinorVersion
    )
    if(NOT seabraid_include_at_install AND NOT seabraid_runtime_at_install)
        install(FILES "${CMAKE_CURRENT_BINARY_DIR}/SeabraidConfig.cmake"
                      "${CMAKE_CURRENT_BINARY_DIR}/SeabraidConfigVersion.cmake"
                DESTINATION "${seabraid_sdk_config_dir}" COMPONENT Development)
    else()
        install(FILES "${CMAKE_CURRENT_BINARY_DIR}/SeabraidConfigVersion.cmake"
                DESTINATION "${seabraid_sdk_config_dir}" COMPONENT Development)
        # WHY the config is generated here and not patched after install(FILES): the first design copied the pristine
        # file, then replaced a sentinel inside the COPY. A second `cmake --install` found the copy up to date
        # (file(INSTALL) skips an unchanged file, and an identical timestamp counts), so the sentinel was already
        # gone and the patch step died; `touch -r` or a second --prefix did the same. Here the final file is always
        # produced from the pristine build-tree file and written whole, so an install is idempotent by construction.
        # WHY two install(CODE) calls: the first carries the configure-time values (validated above; the build-tree
        # path is escaped for the quoted string it lands in), the second is a bracket argument that CMake does not
        # touch, so no further escaping layers. WHY the manifest line: CMake records only what its own install()
        # rules copy. WHY fatal when nothing changed: the sentinel's form comes from configure_package_config_file; if a
        # CMake release spells it differently the config would silently keep the configure-time prefix, which is the
        # bug this rule exists to prevent.
        string(REPLACE "\\" "\\\\" seabraid_config_src_q "${CMAKE_CURRENT_BINARY_DIR}/SeabraidConfig.cmake")
        string(REPLACE "\"" "\\\"" seabraid_config_src_q "${seabraid_config_src_q}")
        string(REPLACE "$" "\\$" seabraid_config_src_q "${seabraid_config_src_q}")
        # An empty sentinel switches its replacement off (the runtime one is used on Windows only).
        set(seabraid_include_sentinel_at "")
        set(seabraid_runtime_sentinel_at "")
        if(seabraid_include_at_install)
            set(seabraid_include_sentinel_at "${seabraid_include_sentinel}")
        endif()
        if(seabraid_runtime_at_install)
            set(seabraid_runtime_sentinel_at "${seabraid_runtime_sentinel}")
        endif()
        # WHY a boolean: CMAKE_INSTALL_MESSAGE is read when the install script is generated (it is not visible to an
        # install(CODE) block), and only NEVER has an effect we must copy: this rule always writes, so ALWAYS and LAZY both
        # print "Installing".
        set(seabraid_install_quiet FALSE)
        if(CMAKE_INSTALL_MESSAGE STREQUAL "NEVER")
            set(seabraid_install_quiet TRUE)
        endif()
        install(CODE "
            set(seabraid_install_quiet ${seabraid_install_quiet})
            set(seabraid_config_src \"${seabraid_config_src_q}\")
            set(seabraid_config_dir \"${seabraid_sdk_config_dir}\")
            set(seabraid_include_rel \"${seabraid_sdk_include}\")
            set(seabraid_include_sentinel \"${seabraid_include_sentinel_at}\")
            set(seabraid_runtime_rel \"${seabraid_sdk_runtime}\")
            set(seabraid_runtime_sentinel \"${seabraid_runtime_sentinel_at}\")
        " COMPONENT Development)
        install(CODE [=[
            # WHY absolute: file(INSTALL) resolves a relative prefix against the working directory; so must the
            # config, or it would carry a relative include directory that means something else to a consumer.
            # WHY the empty case is its own branch: `--prefix /` reaches this script as an empty CMAKE_INSTALL_PREFIX (the
            # trailing slash is dropped), which get_filename_component turns into the working directory. WHY the
            # trailing slashes are stripped: "<prefix>/<dir>" below must not read "//dir" for a prefix of "/" either way.
            if("${CMAKE_INSTALL_PREFIX}" STREQUAL "")
                set(seabraid_prefix "")
            else()
                get_filename_component(seabraid_prefix "${CMAKE_INSTALL_PREFIX}" ABSOLUTE)
                string(REGEX REPLACE "/+$" "" seabraid_prefix "${seabraid_prefix}")
            endif()
            # WHY refuse ';' and a newline: the prefix becomes INTERFACE_INCLUDE_DIRECTORIES, a list, where a ';' splits it in two.
            # The first install rule already refused them in the prefix as given; this is the resolved path (a relative
            # prefix picks up the working directory).
            if(seabraid_prefix MATCHES ";|\n")
                message(FATAL_ERROR "install prefix '${seabraid_prefix}' contains ';' or a newline; the package config cannot carry it")
            endif()
            # WHY escape: the prefix is written inside a quoted argument of the installed config, where a backslash,
            # a quote and a '$' (so also ${PACKAGE_PREFIX_DIR}) would be read as syntax.
            string(REPLACE "\\" "\\\\" seabraid_prefix_q "${seabraid_prefix}")
            string(REPLACE "\"" "\\\"" seabraid_prefix_q "${seabraid_prefix_q}")
            string(REPLACE "$" "\\$" seabraid_prefix_q "${seabraid_prefix_q}")
            file(READ "${seabraid_config_src}" seabraid_text)
            set(seabraid_patched "${seabraid_text}")
            foreach(seabraid_kind include runtime)
                if(NOT "${seabraid_${seabraid_kind}_sentinel}" STREQUAL "")
                    set(seabraid_before "${seabraid_patched}")
                    string(REPLACE "\${PACKAGE_PREFIX_DIR}/${seabraid_${seabraid_kind}_sentinel}"
                           "${seabraid_prefix_q}/${seabraid_${seabraid_kind}_rel}" seabraid_patched "${seabraid_patched}")
                    if(seabraid_patched STREQUAL seabraid_before)
                        message(FATAL_ERROR "${seabraid_config_src} has no ${seabraid_kind} placeholder to resolve against the install prefix")
                    endif()
                endif()
            endforeach()
            set(seabraid_config "${seabraid_config_dir}/SeabraidConfig.cmake")
            if(NOT seabraid_install_quiet AND NOT CMAKE_INSTALL_MESSAGE STREQUAL "NEVER")
                message(STATUS "Installing: $ENV{DESTDIR}${seabraid_config}")
            endif()
            # WHY remove first: file(WRITE) writes THROUGH a symlink (an old install that left a link would overwrite its
            # target) and keeps the mode of an existing file (a read-only copy stayed read-only and unwritable on the next
            # install). WHY CHMOD: file(WRITE) creates the file with 0666 minus the umask, so under umask 077 the config was
            # 0600 and no other user could find_package the SDK; the headers and library CMake installs itself are not umask-bound.
            file(REMOVE "$ENV{DESTDIR}${seabraid_config}")
            file(WRITE "$ENV{DESTDIR}${seabraid_config}" "${seabraid_patched}")
            file(CHMOD "$ENV{DESTDIR}${seabraid_config}" PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ)
            list(APPEND CMAKE_INSTALL_MANIFEST_FILES "${seabraid_config}")
        ]=] COMPONENT Development)
    endif()
endif()

option(SEABRAID_BUILD_NATIVE_TESTS "Build the native Seabraid provider fixtures" OFF)
if(SEABRAID_BUILD_NATIVE_TESTS)
    enable_language(C CXX)
    enable_testing()
    find_package(Threads REQUIRED)
    # Warnings are errors in native tests: the headers must stay clean for strict C and C++ consumers.
    # Optional trailing argument: a CMake regex the output must match. With it the test runs through
    # native-tests/check_output.cmake, which checks the exit code AND the output. WHY not CTest's own
    # PASS_REGULAR_EXPRESSION: it makes CTest ignore the exit code (`echo ok; exit 3` passes).
    function(seabraid_native_test name source standard_feature)
        add_executable(${name} ${source})
        target_compile_features(${name} PRIVATE ${standard_feature})
        target_link_libraries(${name} PRIVATE Seabraid::runtime Threads::Threads)
        if(MSVC)
            target_compile_options(${name} PRIVATE /W4 /WX)
        else()
            target_compile_options(${name} PRIVATE -Wall -Wextra -Werror)
        endif()
        if(ARGC GREATER 3)
            add_test(NAME ${name}
                COMMAND "${CMAKE_COMMAND}" "-DEXPECT_REGEX=${ARGV3}"
                        -P "${CMAKE_CURRENT_SOURCE_DIR}/native-tests/check_output.cmake" -- $<TARGET_FILE:${name}>)
        else()
            add_test(NAME ${name} COMMAND ${name})
        endif()
        if(WIN32)
            set_tests_properties(${name} PROPERTIES
                ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${seabraid_artifact_dir}")
        endif()
    endfunction()

    seabraid_native_test(seabraid_provider_smoke native-tests/provider_smoke.cpp cxx_std_17)
    seabraid_native_test(seabraid_abi_contract native-tests/abi_contract_smoke.cpp cxx_std_17)
    seabraid_native_test(seabraid_placement_smoke native-tests/placement_smoke.c c_std_11)
    seabraid_native_test(seabraid_placement_smoke_cxx native-tests/placement_smoke_cxx.cpp cxx_std_17)
    seabraid_native_test(seabraid_diag_smoke native-tests/diag_smoke.c c_std_11)
    target_compile_definitions(seabraid_diag_smoke PRIVATE SEABRAID_EXPECTED_VERSION="${seabraid_version_full}")
    # Both examples exit nonzero on any unexpected status and print one documented line; check_output.cmake
    # requires a zero exit AND that line (the second regex pins that the walkthrough's last line is the bare "ok").
    seabraid_native_test(seabraid_minimal_provider examples/minimal_provider.c c_std_11 "output = \\[2\\.5, 5\\]")
    # WHY not on MSVC: the walkthrough uses C11 <stdatomic.h>, which MSVC ships only behind /experimental:c11atomics,
    # and it builds here with /W4 /WX. Skipping (not a hand-written Interlocked shim) keeps an unbuilt guess out of
    # the Windows leg; Linux and macOS run it. This is unverified on Windows: no MSVC was available.
    if(MSVC)
        message(STATUS "seabraid_adapter_walkthrough skipped on MSVC (needs C11 <stdatomic.h>)")
    else()
        seabraid_native_test(seabraid_adapter_walkthrough examples/adapter_walkthrough.c c_std_11 "(^|\n)ok\n$")
    endif()

    # Layout parity: src/layout/abi_layout.txt owns every sizeof/offsetof; the asserts are generated from it so no
    # number is written twice here. file(GENERATE) rewrites the header only when its text changes.
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS src/layout/abi_layout.txt)
    file(STRINGS src/layout/abi_layout.txt seabraid_layout_rows REGEX "^seabraid_")
    set(seabraid_layout_asserts "/* Generated from src/layout/abi_layout.txt by CMake. Do not edit. */\n#include <stddef.h>\n")
    foreach(row IN LISTS seabraid_layout_rows)
        if(row MATCHES "^([a-z0-9_]+)\\.([a-z0-9_]+) ([0-9]+)")
            string(APPEND seabraid_layout_asserts
                "_Static_assert(offsetof(${CMAKE_MATCH_1}, ${CMAKE_MATCH_2}) == ${CMAKE_MATCH_3}, \"${CMAKE_MATCH_1}.${CMAKE_MATCH_2}\");\n")
        elseif(row MATCHES "^([a-z0-9_]+) ([0-9]+)")
            string(APPEND seabraid_layout_asserts
                "_Static_assert(sizeof(${CMAKE_MATCH_1}) == ${CMAKE_MATCH_2}, \"sizeof ${CMAKE_MATCH_1}\");\n")
        else()
            message(FATAL_ERROR "src/layout/abi_layout.txt: unreadable row: ${row}")
        endif()
    endforeach()
    file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/abi_layout_asserts.h" CONTENT "${seabraid_layout_asserts}")
    seabraid_native_test(seabraid_abi_layout native-tests/abi_layout.c c_std_11)
    target_include_directories(seabraid_abi_layout PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")

    # The public headers must stay strict ANSI C. GCC/Clang only: MSVC has no -std=c89 -pedantic equivalent, and
    # its C front end does not accept C89 as a mode, so the check is skipped there.
    if(NOT MSVC)
        add_executable(seabraid_c89_headers native-tests/abi_c89.c)
        set_target_properties(seabraid_c89_headers PROPERTIES C_STANDARD 90 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
        target_include_directories(seabraid_c89_headers PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
        target_compile_options(seabraid_c89_headers PRIVATE -pedantic-errors -Wall -Wextra -Werror)
        add_test(NAME seabraid_c89_headers COMMAND seabraid_c89_headers)
        # The documented C++ floor is C++11, but the C++ fixtures need C++17; this one proves the headers alone.
        add_executable(seabraid_cxx11_headers native-tests/abi_cxx11.cpp)
        set_target_properties(seabraid_cxx11_headers PROPERTIES CXX_STANDARD 11 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
        target_include_directories(seabraid_cxx11_headers PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
        target_compile_options(seabraid_cxx11_headers PRIVATE -pedantic-errors -Wall -Wextra -Werror)
        add_test(NAME seabraid_cxx11_headers COMMAND seabraid_cxx11_headers)
        # SEABRAID_DEPRECATED stays silent unless SEABRAID_WARN_DEPRECATED is defined, and fires when it is.
        add_test(NAME seabraid_deprecated_opt_in
            COMMAND "${CMAKE_COMMAND}" "-DCC=${CMAKE_C_COMPILER}" "-DINCLUDE=${CMAKE_CURRENT_SOURCE_DIR}/include"
                    "-DSOURCE=${CMAKE_CURRENT_SOURCE_DIR}/native-tests/deprecated_use.c"
                    -P "${CMAKE_CURRENT_SOURCE_DIR}/native-tests/check_deprecated.cmake")
    endif()

    if(seabraid_has_soname)
        # The engine records DT_NEEDED from this link. A path there only works from one cwd, so check the
        # entry itself (and the library's own SONAME) and also run a test from an unrelated directory.
        add_test(NAME seabraid_needed_is_soname
            COMMAND "${CMAKE_COMMAND}" "-DEXE=$<TARGET_FILE:seabraid_diag_smoke>" "-DLIB=${seabraid_runtime}"
                    "-DSONAME=${seabraid_soname}"
                    -P "${CMAKE_CURRENT_SOURCE_DIR}/native-tests/check_needed.cmake")
        add_test(NAME seabraid_runs_from_other_cwd COMMAND seabraid_diag_smoke)
        set_tests_properties(seabraid_runs_from_other_cwd PROPERTIES WORKING_DIRECTORY "/")
        # CARGO_BUILD_TARGET must not break the build (it used to leave a dangling symlink and exit 0). The check
        # makes a second, scratch build, so it is Linux only like the SONAME tests above.
        add_test(NAME seabraid_cargo_build_target
            COMMAND "${CMAKE_COMMAND}" "-DSOURCE=${CMAKE_CURRENT_SOURCE_DIR}"
                    "-DSCRATCH=${CMAKE_CURRENT_BINARY_DIR}/cargo_target_check"
                    "-DCARGO=${SEABRAID_CARGO_EXECUTABLE}" "-DRUNTIME_NAME=libseabraid.so"
                    -P "${CMAKE_CURRENT_SOURCE_DIR}/native-tests/check_cargo_target.cmake")
    endif()
    # WHY --release: the CTest entry shares the target dir the library was just built in, so it costs no second
    # build. It is not CI's stricter run: CI's own `cargo test` is a debug build, with debug assertions on.
    add_test(NAME seabraid_rust_core
        COMMAND "${SEABRAID_CARGO_EXECUTABLE}" test --locked --offline --release
                --target-dir "${seabraid_target_dir}"
    )
    set_tests_properties(seabraid_rust_core PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(NAME seabraid_python
            COMMAND "${CMAKE_COMMAND}" -E env "SEABRAID_LIBRARY=${seabraid_runtime}"
                    "${Python3_EXECUTABLE}" -B -m unittest discover -s bindings/python
        )
        set_tests_properties(seabraid_python PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
    else()
        message(STATUS "No Python 3 interpreter found: the seabraid_python test is not registered")
    endif()
endif()

# The load simulator is a developer tool, never part of the runtime or its install set. POSIX and GCC/Clang flags
# only; it needs C11 <stdatomic.h>, so MSVC is skipped with a warning rather than failing the configure.
option(SEABRAID_BUILD_TOOLS "Build the Seabraid load simulator (tools/sim)" OFF)
if(SEABRAID_BUILD_TOOLS)
    if(MSVC)
        message(WARNING "SEABRAID_BUILD_TOOLS needs C11 atomics and a POSIX clock; skipped on MSVC")
    else()
        enable_language(C)
        enable_testing()
        find_package(Threads REQUIRED)
        add_subdirectory(tools/sim)
    endif()
endif()

option(SEABRAID_BUILD_STRATA_ADAPTER "Build the private Linux Strata integration" OFF)
if(SEABRAID_BUILD_STRATA_ADAPTER)
    # WHY unconditional: BUILD_TESTING exists only after include(CTest), which this file never does, so the old
    # `if(BUILD_TESTING)` dropped the adapter's tests unless the caller happened to pass -DBUILD_TESTING=ON.
    enable_testing()
    add_subdirectory(adapters/strata)
endif()
