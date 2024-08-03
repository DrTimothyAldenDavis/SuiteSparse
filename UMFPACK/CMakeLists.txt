#-------------------------------------------------------------------------------
# SuiteSparse/UMFPACK/CMakeLists.txt:  cmake for UMFPACK
#-------------------------------------------------------------------------------

# Copyright (c) 1995-2023, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: GPL-2.0+

#-------------------------------------------------------------------------------
# get the version
#-------------------------------------------------------------------------------

# cmake 3.22 is required to find the BLAS in SuiteSparse_config
cmake_minimum_required ( VERSION 3.22 )

set ( UMFPACK_DATE "June 20, 2024" )
set ( UMFPACK_VERSION_MAJOR 6 CACHE STRING "" FORCE )
set ( UMFPACK_VERSION_MINOR 3 CACHE STRING "" FORCE )
set ( UMFPACK_VERSION_SUB   4 CACHE STRING "" FORCE )

message ( STATUS "Building UMFPACK version: v"
    ${UMFPACK_VERSION_MAJOR}.
    ${UMFPACK_VERSION_MINOR}.
    ${UMFPACK_VERSION_SUB} " (" ${UMFPACK_DATE} ")" )

#-------------------------------------------------------------------------------
# define the project
#-------------------------------------------------------------------------------

project ( UMFPACK
    VERSION "${UMFPACK_VERSION_MAJOR}.${UMFPACK_VERSION_MINOR}.${UMFPACK_VERSION_SUB}"
    LANGUAGES C )

#-------------------------------------------------------------------------------
# SuiteSparse policies
#-------------------------------------------------------------------------------

set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    ${PROJECT_SOURCE_DIR}/../CHOLMOD/cmake_modules
    ${PROJECT_SOURCE_DIR}/../SuiteSparse_config/cmake_modules )

include ( SuiteSparsePolicy )

#-------------------------------------------------------------------------------
# find library dependencies
#-------------------------------------------------------------------------------

if ( NOT SUITESPARSE_ROOT_CMAKELISTS )
    find_package ( SuiteSparse_config 7.8.0
        PATHS ${CMAKE_SOURCE_DIR}/../SuiteSparse_config/build NO_DEFAULT_PATH )
    if ( NOT TARGET SuiteSparse::SuiteSparseConfig )
        find_package ( SuiteSparse_config 7.8.0 REQUIRED )
    endif ( )

    find_package ( AMD 3.3.2
        PATHS ${CMAKE_SOURCE_DIR}/../AMD/build NO_DEFAULT_PATH )
    if ( NOT TARGET SuiteSparse::AMD )
        find_package ( AMD 3.3.2 REQUIRED )
    endif ( )
endif ( )

include ( SuiteSparseBLAS )     # requires cmake 3.22

#-------------------------------------------------------------------------------
# find CHOLMOD
#-------------------------------------------------------------------------------

option ( UMFPACK_USE_CHOLMOD "ON (default): use CHOLMOD in UMFPACK.  OFF: do not use CHOLMOD in UMFPACK" ON )

if ( SUITESPARSE_ROOT_CMAKELISTS )
    # if UMFPACK_USE_CHOLMOD is true, then CHOLMOD has been added to the
    # list of packages to compile in the root CMakeLists.txt.
    set ( UMFPACK_HAS_CHOLMOD ${UMFPACK_USE_CHOLMOD} )
else ( )
    if ( UMFPACK_USE_CHOLMOD )
        # look for CHOLMOD (optional fill-reducing orderings)
        find_package ( CHOLMOD 5.2.1
            PATHS ${CMAKE_SOURCE_DIR}/../CHOLMOD/build NO_DEFAULT_PATH )
        if ( NOT TARGET SuiteSparse::CHOLMOD )
            find_package ( CHOLMOD 5.2.1 )
        endif ( )
        if ( NOT CHOLMOD_FOUND )
            # CHOLMOD not found so disable it
            set ( UMFPACK_HAS_CHOLMOD OFF )
        else ( )
            set ( UMFPACK_HAS_CHOLMOD ON )
        endif ( )
    else ( )
        set ( UMFPACK_HAS_CHOLMOD OFF )
    endif ( )
endif ( )

if ( UMFPACK_HAS_CHOLMOD )
    message ( STATUS "Using CHOLMOD for addtional pre-ordering options" )
else ( )
    message ( STATUS "CHOLMOD not found or not requested" )
endif ( )

# check for strict usage
if ( SUITESPARSE_USE_STRICT AND UMFPACK_USE_CHOLMOD AND NOT UMFPACK_HAS_CHOLMOD )
    message ( FATAL_ERROR "CHOLMOD required for UMFPACK but not found" )
endif ( )

#-------------------------------------------------------------------------------
# configure files
#-------------------------------------------------------------------------------

configure_file ( "Config/umfpack.h.in"
    "${PROJECT_SOURCE_DIR}/Include/umfpack.h"
    NEWLINE_STYLE LF )
configure_file ( "Config/umfpack_version.tex.in"
    "${PROJECT_SOURCE_DIR}/Doc/umfpack_version.tex"
    NEWLINE_STYLE LF )

#-------------------------------------------------------------------------------
# include directories
#-------------------------------------------------------------------------------

include_directories ( Source Include )

#-------------------------------------------------------------------------------
# dynamic umfpack library properties
#-------------------------------------------------------------------------------

file ( GLOB UMFPACK_SOURCES "Source2/*.c" )

if ( BUILD_SHARED_LIBS )
    add_library ( UMFPACK SHARED ${UMFPACK_SOURCES} )

    set_target_properties ( UMFPACK PROPERTIES
        VERSION ${UMFPACK_VERSION_MAJOR}.${UMFPACK_VERSION_MINOR}.${UMFPACK_VERSION_SUB}
        C_STANDARD 11
        C_STANDARD_REQUIRED ON
        OUTPUT_NAME umfpack
        SOVERSION ${UMFPACK_VERSION_MAJOR}
        PUBLIC_HEADER "Include/umfpack.h"
        WINDOWS_EXPORT_ALL_SYMBOLS ON )

    if ( ${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.25" )
        set_target_properties ( UMFPACK PROPERTIES EXPORT_NO_SYSTEM ON )
    endif ( )

    target_include_directories ( UMFPACK 
        INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>
                  $<INSTALL_INTERFACE:${SUITESPARSE_INCLUDEDIR}> )
endif ( )

#-------------------------------------------------------------------------------
# static umfpack library properties
#-------------------------------------------------------------------------------

if ( BUILD_STATIC_LIBS )
    add_library ( UMFPACK_static STATIC ${UMFPACK_SOURCES} )

    set_target_properties ( UMFPACK_static PROPERTIES
        C_STANDARD 11
        C_STANDARD_REQUIRED ON
        OUTPUT_NAME umfpack
        PUBLIC_HEADER "Include/umfpack.h" )

    if ( MSVC OR ("${CMAKE_C_SIMULATE_ID}" STREQUAL "MSVC") )
        set_target_properties ( UMFPACK_static PROPERTIES
            OUTPUT_NAME umfpack_static )
    endif ( )

    if ( ${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.25" )
        set_target_properties ( UMFPACK_static PROPERTIES EXPORT_NO_SYSTEM ON )
    endif ( )

    target_include_directories ( UMFPACK_static 
        INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>
                  $<INSTALL_INTERFACE:${SUITESPARSE_INCLUDEDIR}> )
endif ( )

#-------------------------------------------------------------------------------
# add the library dependencies
#-------------------------------------------------------------------------------

# SuiteSparseConfig:
if ( BUILD_SHARED_LIBS )
    target_link_libraries ( UMFPACK PRIVATE SuiteSparse::SuiteSparseConfig )
    target_include_directories ( UMFPACK PUBLIC
        "$<TARGET_PROPERTY:SuiteSparse::SuiteSparseConfig,INTERFACE_INCLUDE_DIRECTORIES>" )
endif ( )
if ( BUILD_STATIC_LIBS )
    if ( TARGET SuiteSparse::SuiteSparseConfig_static )
        target_link_libraries ( UMFPACK_static PUBLIC SuiteSparse::SuiteSparseConfig_static )
    else ( )
        target_link_libraries ( UMFPACK_static PUBLIC SuiteSparse::SuiteSparseConfig )
    endif ( )
endif ( )

# AMD:
if ( BUILD_SHARED_LIBS )
    target_link_libraries ( UMFPACK PRIVATE SuiteSparse::AMD )
    target_include_directories ( UMFPACK PUBLIC
        "$<TARGET_PROPERTY:SuiteSparse::AMD,INTERFACE_INCLUDE_DIRECTORIES>" )
endif ( )
if ( BUILD_STATIC_LIBS )
    if ( TARGET SuiteSparse::AMD_static )
        target_link_libraries ( UMFPACK_static PUBLIC SuiteSparse::AMD_static )
    else ( )
        target_link_libraries ( UMFPACK_static PUBLIC SuiteSparse::AMD )
    endif ( )
endif ( )

# CHOLMOD:
if ( UMFPACK_HAS_CHOLMOD )
    # link with CHOLMOD and its dependencies, both required and optional
    if ( BUILD_SHARED_LIBS )
        target_link_libraries ( UMFPACK PRIVATE SuiteSparse::CHOLMOD )
    endif ( )
    if ( BUILD_STATIC_LIBS )
        set ( UMFPACK_STATIC_MODULES "${UMFPACK_STATIC_MODULES} CHOLMOD" )
        if ( TARGET SuiteSparse::CHOLMOD_static )
            target_link_libraries ( UMFPACK_static PRIVATE SuiteSparse::CHOLMOD_static )
        else ( )
            target_link_libraries ( UMFPACK_static PRIVATE SuiteSparse::CHOLMOD )
        endif ( )
    endif ( )
else ( )
    # tell UMFPACK that CHOLMOD is not available
    if ( BUILD_SHARED_LIBS )
        target_compile_definitions ( UMFPACK PRIVATE NCHOLMOD )
    endif ( )
    if ( BUILD_STATIC_LIBS )
        target_compile_definitions ( UMFPACK_static PRIVATE NCHOLMOD )
    endif ( )
endif ( )

# libm:
include ( CheckSymbolExists )
check_symbol_exists ( fmax "math.h" NO_LIBM )
if ( NOT NO_LIBM )
    if ( BUILD_SHARED_LIBS )
        target_link_libraries ( UMFPACK PRIVATE m )
    endif ( )
    if ( BUILD_STATIC_LIBS )
        list ( APPEND UMFPACK_STATIC_LIBS "m" )
        target_link_libraries ( UMFPACK_static PUBLIC m )
    endif ( )
endif ( )

# BLAS:
message ( STATUS "BLAS libraries:      ${BLAS_LIBRARIES}" )
message ( STATUS "BLAS include:        ${BLAS_INCLUDE_DIRS}" )
message ( STATUS "BLAS linker flags:   ${BLAS_LINKER_FLAGS}" )
if ( BUILD_SHARED_LIBS )
    target_link_libraries ( UMFPACK PRIVATE ${BLAS_LIBRARIES} )
    target_include_directories ( UMFPACK PRIVATE ${BLAS_INCLUDE_DIRS} )
endif ( )
if ( BUILD_STATIC_LIBS )
    list ( APPEND UMFPACK_STATIC_LIBS ${BLAS_LIBRARIES} )
    target_link_libraries ( UMFPACK_static PUBLIC ${BLAS_LIBRARIES} )
    target_include_directories ( UMFPACK_static PRIVATE ${BLAS_INCLUDE_DIRS} )
endif ( )

#-------------------------------------------------------------------------------
# check compiler features
#-------------------------------------------------------------------------------

include ( CheckSourceCompiles )

set ( _orig_CMAKE_C_FLAGS ${CMAKE_C_FLAGS} )
include ( CheckCCompilerFlag )
check_c_compiler_flag ( "-Wunknown-pragmas" HAVE_UNKNOWN_PRAGMAS )
if ( HAVE_UNKNOWN_PRAGMAS )
    set ( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wunknown-pragmas" )
endif ( )

# check syntax for ivdep pragma
set ( _pragma_found OFF )
# GCC compatible
check_source_compiles ( C
    "int main(void) {
    #pragma GCC ivdep
    for (int j = 0; j < 1; j++) { }
    return 0;
    }"
    HAVE_PRAGMA_GCC_IVDEP
    FAIL_REGEX "[wW]arning" )

if ( HAVE_PRAGMA_GCC_IVDEP )
    if ( BUILD_SHARED_LIBS )
        target_compile_definitions ( UMFPACK PRIVATE "HAVE_PRAGMA_GCC_IVDEP" )
    endif ( )
    if ( BUILD_STATIC_LIBS )
        target_compile_definitions ( UMFPACK_static PRIVATE "HAVE_PRAGMA_GCC_IVDEP" )
    endif ( )
    set ( _pragma_found ON )
endif ( )

# Clang compatible
if ( NOT _pragma_found )
    check_source_compiles ( C
        "int main(void) {
        #pragma clang loop vectorize(enable)
        for (int j = 0; j < 1; j++) { }
        return 0;
        }"
        HAVE_PRAGMA_CLANG_LOOP_VECTORIZE
        FAIL_REGEX "[wW]arning" )

    if ( HAVE_PRAGMA_CLANG_LOOP_VECTORIZE )
        # Clang is very verbose if a hint for vectorization cannot be performed.
        # Disable the respective warning.
        check_c_compiler_flag ( "-Wno-pass-failed" HAVE_NO_PASS_FAILED )
        if ( BUILD_SHARED_LIBS )
            target_compile_definitions ( UMFPACK PRIVATE "HAVE_PRAGMA_CLANG_LOOP_VECTORIZE" )
            if ( HAVE_NO_PASS_FAILED )
                target_compile_options ( UMFPACK PRIVATE "-Wno-pass-failed" )
            endif ( )
        endif ( )
        if ( BUILD_STATIC_LIBS )
            target_compile_definitions ( UMFPACK_static PRIVATE "HAVE_PRAGMA_CLANG_LOOP_VECTORIZE" )
            if ( HAVE_NO_PASS_FAILED )
                target_compile_options ( UMFPACK_static PRIVATE "-Wno-pass-failed" )
            endif ( )
        endif ( )
        set ( _pragma_found ON )
    endif ( )
endif ( )

# Intel compatible
if ( NOT _pragma_found )
    check_source_compiles ( C
        "int main(void) {
        #pragma ivdep
        for (int j = 0; j < 1; j++) { }
        return 0;
        }"
        HAVE_PRAGMA_IVDEP
        FAIL_REGEX "[wW]arning" )

    if ( HAVE_PRAGMA_IVDEP )
        if ( BUILD_SHARED_LIBS )
            target_compile_definitions ( UMFPACK PRIVATE "HAVE_PRAGMA_IVDEP" )
        endif ( )
        if ( BUILD_STATIC_LIBS )
            target_compile_definitions ( UMFPACK_static PRIVATE "HAVE_PRAGMA_IVDEP" )
        endif ( )
        set ( _pragma_found ON )
    endif ( )
endif ( )

# MSVC compatible
if ( NOT _pragma_found )
    check_source_compiles ( C
        "int main(void) {
        #pragma loop( ivdep )
        for (int j = 0; j < 1; j++) { }
        return 0;
        }"
        HAVE_PRAGMA_LOOP_IVDEP
        FAIL_REGEX "[wW]arning" )

    if ( HAVE_PRAGMA_LOOP_IVDEP )
        if ( BUILD_SHARED_LIBS )
            target_compile_definitions ( UMFPACK PRIVATE "HAVE_PRAGMA_LOOP_IVDEP" )
        endif ( )
        if ( BUILD_STATIC_LIBS )
            target_compile_definitions ( UMFPACK_static PRIVATE "HAVE_PRAGMA_LOOP_IVDEP" )
        endif ( )
        set ( _pragma_found ON )
    endif ( )
endif ( )

# check syntax for novector pragma
set ( _pragma_found ${HAVE_PRAGMA_CLANG_LOOP_VECTORIZE} )
# GCC compatible
if ( NOT _pragma_found )
    check_source_compiles ( C
        "int main(void) {
        #pragma GCC novector
        for (int j = 0; j < 1; j++) { }
        return 0;
        }"
        HAVE_PRAGMA_GCC_NOVECTOR
        FAIL_REGEX "[wW]arning" )

    if ( HAVE_PRAGMA_GCC_NOVECTOR )
        if ( BUILD_SHARED_LIBS )
            target_compile_definitions ( UMFPACK PRIVATE "HAVE_PRAGMA_GCC_NOVECTOR" )
        endif ( )
        if ( BUILD_STATIC_LIBS )
            target_compile_definitions ( UMFPACK_static PRIVATE "HAVE_PRAGMA_GCC_NOVECTOR" )
        endif ( )
        set ( _pragma_found ON )
    endif ( )
endif ( )

# Intel compatible
if ( NOT _pragma_found )
    check_source_compiles ( C
        "int main(void) {
        #pragma novector
        for (int j = 0; j < 1; j++) { }
        return 0;
        }"
        HAVE_PRAGMA_NOVECTOR
        FAIL_REGEX "[wW]arning" )

    if ( HAVE_PRAGMA_NOVECTOR )
        if ( BUILD_SHARED_LIBS )
            target_compile_definitions ( UMFPACK PRIVATE "HAVE_PRAGMA_NOVECTOR" )
        endif ( )
        if ( BUILD_STATIC_LIBS )
            target_compile_definitions ( UMFPACK_static PRIVATE "HAVE_PRAGMA_NOVECTOR" )
        endif ( )
        set ( _pragma_found ON )
    endif ( )
endif ( )

# MSVC compatible
if ( NOT _pragma_found )
    check_source_compiles ( C
        "int main(void) {
        #pragma loop( no_vector )
        for (int j = 0; j < 1; j++) { }
        return 0;
        }"
        HAVE_PRAGMA_LOOP_NO_VECTOR
        FAIL_REGEX "[wW]arning" )

    if ( HAVE_PRAGMA_LOOP_NO_VECTOR )
        if ( BUILD_SHARED_LIBS )
            target_compile_definitions ( UMFPACK PRIVATE "HAVE_PRAGMA_LOOP_NO_VECTOR" )
        endif ( )
        if ( BUILD_STATIC_LIBS )
            target_compile_definitions ( UMFPACK_static PRIVATE "HAVE_PRAGMA_LOOP_NO_VECTOR" )
        endif ( )
        set ( _pragma_found ON )
    endif ( )
endif ( )

set ( CMAKE_C_FLAGS ${_orig_CMAKE_C_FLAGS} )

#-------------------------------------------------------------------------------
# UMFPACK installation location
#-------------------------------------------------------------------------------

include ( CMakePackageConfigHelpers )

if ( BUILD_SHARED_LIBS )
    install ( TARGETS UMFPACK
        EXPORT UMFPACKTargets
        LIBRARY DESTINATION ${SUITESPARSE_LIBDIR}
        ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR}
        RUNTIME DESTINATION ${SUITESPARSE_BINDIR}
        PUBLIC_HEADER DESTINATION ${SUITESPARSE_INCLUDEDIR} )
endif ( )
if ( BUILD_STATIC_LIBS )
    install ( TARGETS UMFPACK_static
        EXPORT UMFPACKTargets
        ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR}
        PUBLIC_HEADER DESTINATION ${SUITESPARSE_INCLUDEDIR} )
endif ( )

# create (temporary) export target file during build
export ( EXPORT UMFPACKTargets
    NAMESPACE SuiteSparse::
    FILE ${CMAKE_CURRENT_BINARY_DIR}/UMFPACKTargets.cmake )

# install export target, config and version files for find_package
install ( EXPORT UMFPACKTargets
    NAMESPACE SuiteSparse::
    DESTINATION ${SUITESPARSE_PKGFILEDIR}/cmake/UMFPACK )

# generate config file to be used in common build tree
set ( SUITESPARSE_IN_BUILD_TREE ON )
configure_package_config_file (
    Config/UMFPACKConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/UMFPACKConfig.cmake
    INSTALL_DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/UMFPACKConfig.cmake )

# generate config file to be installed
set ( SUITESPARSE_IN_BUILD_TREE OFF )
configure_package_config_file (
    Config/UMFPACKConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/target/UMFPACKConfig.cmake
    INSTALL_DESTINATION ${SUITESPARSE_PKGFILEDIR}/cmake/UMFPACK )

write_basic_package_version_file (
    ${CMAKE_CURRENT_BINARY_DIR}/UMFPACKConfigVersion.cmake
    COMPATIBILITY SameMajorVersion )

install ( FILES
    ${CMAKE_CURRENT_BINARY_DIR}/target/UMFPACKConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/UMFPACKConfigVersion.cmake
    DESTINATION ${SUITESPARSE_PKGFILEDIR}/cmake/UMFPACK )

#-------------------------------------------------------------------------------
# create pkg-config file
#-------------------------------------------------------------------------------

if ( NOT MSVC )
    # This might be something like:
    #   /usr/lib/libgomp.so;/usr/lib/libpthread.a;m
    # convert to -l flags for pkg-config, i.e.: "-lgomp -lpthread -lm"
    set ( UMFPACK_STATIC_LIBS_LIST ${UMFPACK_STATIC_LIBS} )
    set ( UMFPACK_STATIC_LIBS "" )
    foreach ( _lib ${UMFPACK_STATIC_LIBS_LIST} )
        string ( FIND ${_lib} "." _pos REVERSE )
        if ( ${_pos} EQUAL "-1" )
            set ( UMFPACK_STATIC_LIBS "${UMFPACK_STATIC_LIBS} -l${_lib}" )
            continue ()
        endif ( )
        set ( _kinds "SHARED" "STATIC" )
        if ( WIN32 )
            list ( PREPEND _kinds "IMPORT" )
        endif ( )
        foreach ( _kind IN LISTS _kinds )
            set ( _regex ".*\\/(lib)?([^\\.]*)(${CMAKE_${_kind}_LIBRARY_SUFFIX})" )
            if ( ${_lib} MATCHES ${_regex} )
                string ( REGEX REPLACE ${_regex} "\\2" _libname ${_lib} )
                if ( NOT "${_libname}" STREQUAL "" )
                    set ( UMFPACK_STATIC_LIBS "${UMFPACK_STATIC_LIBS} -l${_libname}" )
                    break ()
                endif ( )
            endif ( )
        endforeach ( )
    endforeach ( )

    set ( prefix "${CMAKE_INSTALL_PREFIX}" )
    set ( exec_prefix "\${prefix}" )
    cmake_path ( IS_ABSOLUTE SUITESPARSE_LIBDIR SUITESPARSE_LIBDIR_IS_ABSOLUTE )
    if (SUITESPARSE_LIBDIR_IS_ABSOLUTE)
        set ( libdir "${SUITESPARSE_LIBDIR}")
    else ( )
        set ( libdir "\${exec_prefix}/${SUITESPARSE_LIBDIR}")
    endif ( )
    cmake_path ( IS_ABSOLUTE SUITESPARSE_INCLUDEDIR SUITESPARSE_INCLUDEDIR_IS_ABSOLUTE )
    if (SUITESPARSE_INCLUDEDIR_IS_ABSOLUTE)
        set ( includedir "${SUITESPARSE_INCLUDEDIR}")
    else ( )
        set ( includedir "\${prefix}/${SUITESPARSE_INCLUDEDIR}")
    endif ( )
    if ( BUILD_SHARED_LIBS )
        set ( SUITESPARSE_LIB_BASE_NAME $<TARGET_FILE_BASE_NAME:UMFPACK> )
    else ( )
        set ( SUITESPARSE_LIB_BASE_NAME $<TARGET_FILE_BASE_NAME:UMFPACK_static> )
    endif ( )
    configure_file (
        Config/UMFPACK.pc.in
        UMFPACK.pc.out
        @ONLY
        NEWLINE_STYLE LF )
    file ( GENERATE
        OUTPUT UMFPACK.pc
        INPUT ${CMAKE_CURRENT_BINARY_DIR}/UMFPACK.pc.out
        NEWLINE_STYLE LF )
    install ( FILES
        ${CMAKE_CURRENT_BINARY_DIR}/UMFPACK.pc
        DESTINATION ${SUITESPARSE_PKGFILEDIR}/pkgconfig )
endif ( )

#-------------------------------------------------------------------------------
# Demo library and programs
#-------------------------------------------------------------------------------

if ( SUITESPARSE_DEMOS )

    #---------------------------------------------------------------------------
    # demo library
    #---------------------------------------------------------------------------

    message ( STATUS "Also compiling the demos in UMFPACK/Demo" )

    #---------------------------------------------------------------------------
    # Demo programs
    #---------------------------------------------------------------------------

    add_executable ( umfpack_simple  "Demo/umfpack_simple.c" )
    add_executable ( umfpack_di_demo "Demo/umfpack_di_demo.c" )
    add_executable ( umfpack_dl_demo "Demo/umfpack_dl_demo.c" )
    add_executable ( umfpack_zi_demo "Demo/umfpack_zi_demo.c" )
    add_executable ( umfpack_zl_demo "Demo/umfpack_zl_demo.c" )

    if ( SUITESPARSE_HAS_FORTRAN )
        # Fortran demos
        add_executable ( umf4            "Demo/umf4.c" )
        add_executable ( umfreadhb       "Demo/readhb.f" )
        set_target_properties ( umfreadhb PROPERTIES OUTPUT_NAME readhb )
        add_executable ( umfreadhb_nozeros "Demo/readhb_nozeros.f" )
        set_target_properties ( umfreadhb_nozeros PROPERTIES OUTPUT_NAME readhb_nozeros )
        add_executable ( umfreadhb_size  "Demo/readhb_size.f" )
        set_target_properties ( umfreadhb_size PROPERTIES OUTPUT_NAME readhb_size )
        add_executable ( umf4hb          "Demo/umf4hb.f"
                                         "Demo/umf4_f77wrapper.c" )
        add_executable ( umf4zhb         "Demo/umf4zhb.f"
                                         "Demo/umf4_f77zwrapper.c" )
        add_executable ( umf4hb64        "Demo/umf4hb64.f"
                                         "Demo/umf4_f77wrapper64.c" )
        add_executable ( umf4zhb64       "Demo/umf4zhb64.f"
                                         "Demo/umf4_f77zwrapper64.c" )
    endif ( )

    # Libraries required for Demo programs
    if ( BUILD_SHARED_LIBS )
        target_link_libraries ( umfpack_simple PUBLIC UMFPACK )
        target_link_libraries ( umfpack_di_demo PUBLIC UMFPACK )
        target_link_libraries ( umfpack_dl_demo PUBLIC UMFPACK )
        target_link_libraries ( umfpack_zi_demo PUBLIC UMFPACK )
        target_link_libraries ( umfpack_zl_demo PUBLIC UMFPACK )
    else ( )
        target_link_libraries ( umfpack_simple PUBLIC UMFPACK_static )
        target_link_libraries ( umfpack_di_demo PUBLIC UMFPACK_static )
        target_link_libraries ( umfpack_dl_demo PUBLIC UMFPACK_static )
        target_link_libraries ( umfpack_zi_demo PUBLIC UMFPACK_static )
        target_link_libraries ( umfpack_zl_demo PUBLIC UMFPACK_static )
    endif ( )
    target_link_libraries ( umfpack_simple PUBLIC SuiteSparse::SuiteSparseConfig )

    if ( SUITESPARSE_HAS_FORTRAN )
        # Fortran demos
        if ( BUILD_SHARED_LIBS )
            target_link_libraries ( umf4 PUBLIC UMFPACK )
            target_link_libraries ( umf4hb PUBLIC UMFPACK )
            target_link_libraries ( umf4zhb PUBLIC UMFPACK )
            target_link_libraries ( umf4hb64 PUBLIC UMFPACK )
            target_link_libraries ( umf4zhb64 PUBLIC UMFPACK )
        else ( )
            target_link_libraries ( umf4 PUBLIC UMFPACK_static )
            target_link_libraries ( umf4hb PUBLIC UMFPACK_static )
            target_link_libraries ( umf4zhb PUBLIC UMFPACK_static )
            target_link_libraries ( umf4hb64 PUBLIC UMFPACK_static )
            target_link_libraries ( umf4zhb64 PUBLIC UMFPACK_static )
        endif ( )
        target_link_libraries ( umf4 PUBLIC SuiteSparse::AMD )
    endif ( )

else ( )

    message ( STATUS "Skipping the demos in UMFPACK/Demo" )

endif ( )

#-------------------------------------------------------------------------------
# report status
#-------------------------------------------------------------------------------

include ( SuiteSparseReport )
