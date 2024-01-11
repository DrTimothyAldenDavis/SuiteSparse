#-------------------------------------------------------------------------------
# SuiteSparse/ParU/CMakeLists.txt:  cmake for ParU
#-------------------------------------------------------------------------------

# ParU, Copyright (c) 2022-2023, Mohsen Aznaveh and Timothy A. Davis,
# All Rights Reserved.
# SPDX-License-Identifier: GNU GPL 3.0

#-------------------------------------------------------------------------------
# get the version
#-------------------------------------------------------------------------------

# cmake 3.22 is required to find the BLAS in SuiteSparse_config
cmake_minimum_required ( VERSION 3.22 )

set ( PARU_DATE "Jan 10, 2024" )
set ( PARU_VERSION_MAJOR  0 CACHE STRING "" FORCE )
set ( PARU_VERSION_MINOR  1 CACHE STRING "" FORCE )
set ( PARU_VERSION_UPDATE 1 CACHE STRING "" FORCE )

message ( STATUS "Building PARU version: v"
    ${PARU_VERSION_MAJOR}.
    ${PARU_VERSION_MINOR}.
    ${PARU_VERSION_UPDATE} " (" ${PARU_DATE} ")" )

#-------------------------------------------------------------------------------
# define the project
#-------------------------------------------------------------------------------

project ( ParU
    VERSION "${PARU_VERSION_MAJOR}.${PARU_VERSION_MINOR}.${PARU_VERSION_UPDATE}" )

#-------------------------------------------------------------------------------
# SuiteSparse policies
#-------------------------------------------------------------------------------

set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    ${PROJECT_SOURCE_DIR}/../SuiteSparse_config/cmake_modules
    )

include ( SuiteSparsePolicy )

#-------------------------------------------------------------------------------
# find languages
#-------------------------------------------------------------------------------

enable_language ( CXX )

if ( SUITESPARSE_DEMOS )
    enable_language ( C )
endif ( )

#-------------------------------------------------------------------------------
# find OpenMP
#-------------------------------------------------------------------------------

option ( PARU_USE_OPENMP "ON: Use OpenMP in ParU if available.  OFF: Do not use OpenMP.  (Default: SUITESPARSE_USE_OPENMP)" ${SUITESPARSE_USE_OPENMP} )
if ( NOT SUITESPARSE_USE_OPENMP )
    set ( PARU_USE_OPENMP "OFF" CACHE STRING "" FORCE )
endif ( )

if ( PARU_USE_OPENMP )
    # OpenMP 4.5 or later is required
    if ( CMAKE_VERSION VERSION_LESS 3.24 )
        find_package ( OpenMP COMPONENTS C CXX )
    else ( )
        find_package ( OpenMP COMPONENTS C CXX GLOBAL )
    endif ( )
    if ( OpenMP_CXX_FOUND AND OpenMP_C_FOUND )
        set ( PARU_HAS_OPENMP ON )
        if (( OpenMP_CXX_VERSION VERSION_LESS 4.5 ) OR
            ( OpenMP_C_VERSION   VERSION_LESS 4.5 ) )
            message ( STATUS "ParU requires OpenMP 4.5 or later" )
            set ( PARU_HAS_OPENMP OFF )
            set ( OpenMP_CXX_FOUND OFF )
            set ( OpenMP_C_FOUND OFF )
        endif ( )
    else ( )
        # OpenMP has not been found
        message ( STATUS "OpenMP not found" )
        set ( PARU_HAS_OPENMP OFF )
        set ( OpenMP_CXX_FOUND OFF )
        set ( OpenMP_C_FOUND OFF )
    endif ( )
else ( )
    # OpenMP has not been requested
    set ( PARU_HAS_OPENMP OFF )
    set ( OpenMP_CXX_FOUND OFF )
    set ( OpenMP_C_FOUND OFF )
endif ( )
message ( STATUS "ParU has OpenMP: ${PARU_HAS_OPENMP}" )

# check for strict usage
if ( SUITESPARSE_USE_STRICT AND PARU_USE_OPENMP AND NOT PARU_HAS_OPENMP )
    message ( FATAL_ERROR "OpenMP 4.5 required for ParU but not found" )
endif ( )

#-------------------------------------------------------------------------------
# find other dependencies
#-------------------------------------------------------------------------------

if ( NOT SUITESPARSE_ROOT_CMAKELISTS )
    find_package ( SuiteSparse_config 7.5.0
        PATHS ${CMAKE_SOURCE_DIR}/../SuiteSparse_config/build NO_DEFAULT_PATH )
    if ( NOT TARGET SuiteSparse::SuiteSparseConfig )
        find_package ( SuiteSparse_config 7.5.0 REQUIRED )
    endif ( )

    find_package ( CHOLMOD 5.1.1
        PATHS ${CMAKE_SOURCE_DIR}/../CHOLMOD/build NO_DEFAULT_PATH )
    if ( NOT CHOLMOD_FOUND )
        find_package ( CHOLMOD 5.1.1 REQUIRED )
    endif ( )

    find_package ( UMFPACK 6.3.0
        PATHS ${CMAKE_SOURCE_DIR}/../UMFPACK/build NO_DEFAULT_PATH )
    if ( NOT CAMD_FOUND )
        find_package ( UMFPACK 6.3.0 REQUIRED )
    endif ( )
endif ( )

include ( SuiteSparseBLAS )     # requires cmake 3.22

#-------------------------------------------------------------------------------
# configure files
#-------------------------------------------------------------------------------

configure_file ( "Config/ParU_definitions.h.in"
    "${PROJECT_SOURCE_DIR}/Include/ParU_definitions.h"
    NEWLINE_STYLE LF )
configure_file ( "Config/paru_version.tex.in"
    "${PROJECT_SOURCE_DIR}/Doc/paru_version.tex"
    NEWLINE_STYLE LF )

#-------------------------------------------------------------------------------
# include directories
#-------------------------------------------------------------------------------

# FIXME: "paru_internal.hpp" includes "umf_internal.h" which is not installed.
#        That means the two libraries are interdependent and ParU cannot be
#        built as a stand-alone project outside a common build tree.
include_directories ( Source Include
    ${PROJECT_SOURCE_DIR}/../UMFPACK/Source
    )

#-------------------------------------------------------------------------------
# dynamic paru library properties
#-------------------------------------------------------------------------------

file ( GLOB PARU_SOURCES "Source/*.cpp" )

if ( BUILD_SHARED_LIBS )
    add_library ( ParU SHARED ${PARU_SOURCES} )

    set_target_properties ( ParU PROPERTIES
        VERSION ${PARU_VERSION_MAJOR}.${PARU_VERSION_MINOR}.${PARU_VERSION_UPDATE}
        CXX_STANDARD 11
        CXX_STANDARD_REQUIRED ON
        OUTPUT_NAME paru
        SOVERSION ${PARU_VERSION_MAJOR}
        PUBLIC_HEADER "Include/ParU.hpp;Include/ParU_C.h;Include/ParU_definitions.h"
        WINDOWS_EXPORT_ALL_SYMBOLS ON )

    if ( ${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.25" )
        set_target_properties ( ParU PROPERTIES EXPORT_NO_SYSTEM ON )
    endif ( )

    target_include_directories ( ParU
        INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>
                  $<INSTALL_INTERFACE:${SUITESPARSE_INCLUDEDIR}> )
endif ( )

#-------------------------------------------------------------------------------
# static paru library properties
#-------------------------------------------------------------------------------

if ( BUILD_STATIC_LIBS )
    add_library ( ParU_static STATIC ${PARU_SOURCES} )

    set_target_properties ( ParU_static PROPERTIES
        CXX_STANDARD 11
        CXX_STANDARD_REQUIRED ON
        OUTPUT_NAME paru
        PUBLIC_HEADER "Include/ParU.hpp;Include/ParU_C.h;Include/ParU_definitions.h"
        WINDOWS_EXPORT_ALL_SYMBOLS ON )

    if ( MSVC )
        set_target_properties ( ParU_static PROPERTIES
            OUTPUT_NAME ParU_static )
    endif ( )

    if ( ${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.25" )
        set_target_properties ( ParU_static PROPERTIES EXPORT_NO_SYSTEM ON )
    endif ( )

    target_include_directories ( ParU_static
        INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>
                  $<INSTALL_INTERFACE:${SUITESPARSE_INCLUDEDIR}> )
endif ( )

#-------------------------------------------------------------------------------
# additional definitions
#-------------------------------------------------------------------------------

if ( (APPLE OR WIN32) AND CMAKE_C_COMPILER_ID STREQUAL "GNU" )
    # Some parallelization levels don't work correctly with GCC on Windows or
    # Mac (because of emuTLS?). Omit them.
    if ( BUILD_SHARED_LIBS )
        target_compile_definitions (ParU PRIVATE PARU_1TASK )
    endif ( )
    if ( BUILD_STATIC_LIBS )
        target_compile_definitions (ParU_static PRIVATE PARU_1TASK )
    endif ( )
endif ( )

#-------------------------------------------------------------------------------
# add the library dependencies
#-------------------------------------------------------------------------------

# suitesparseconfig:
if ( BUILD_SHARED_LIBS )
    target_link_libraries ( ParU PRIVATE SuiteSparse::SuiteSparseConfig )
    target_include_directories ( ParU PUBLIC
        "$<TARGET_PROPERTY:SuiteSparse::SuiteSparseConfig,INTERFACE_INCLUDE_DIRECTORIES>" )
endif ( )
if ( BUILD_STATIC_LIBS )
    if ( TARGET SuiteSparse::SuiteSparseConfig_static )
        target_link_libraries ( ParU_static PUBLIC SuiteSparse::SuiteSparseConfig_static )
    else ( )
        target_link_libraries ( ParU_static PUBLIC SuiteSparse::SuiteSparseConfig )
    endif ( )
endif ( )

# umfpack:
if ( BUILD_SHARED_LIBS )
    target_link_libraries ( ParU PRIVATE SuiteSparse::UMFPACK )
    target_include_directories ( ParU PUBLIC
        "$<TARGET_PROPERTY:SuiteSparse::UMFPACK,INTERFACE_INCLUDE_DIRECTORIES>" )
endif ( )
if ( BUILD_STATIC_LIBS )
    if ( TARGET SuiteSparse::UMFPACK_static )
        target_link_libraries ( ParU_static PUBLIC SuiteSparse::UMFPACK_static )
    else ( )
        target_link_libraries ( ParU_static PUBLIC SuiteSparse::UMFPACK )
    endif ( )
endif ( )

# cholmod:
if ( BUILD_SHARED_LIBS )
    target_include_directories ( ParU PUBLIC
        "$<TARGET_PROPERTY:SuiteSparse::CHOLMOD,INTERFACE_INCLUDE_DIRECTORIES>" )
endif ( )
if ( BUILD_STATIC_LIBS )
    target_include_directories ( ParU_static PUBLIC
        "$<TARGET_PROPERTY:SuiteSparse::CHOLMOD,INTERFACE_INCLUDE_DIRECTORIES>" )
endif ( )

# OpenMP:
if ( PARU_HAS_OPENMP )
    message ( STATUS "OpenMP C++ libraries:    ${OpenMP_CXX_LIBRARIES} ")
    message ( STATUS "OpenMP C++ include:      ${OpenMP_CXX_INCLUDE_DIRS} ")
    message ( STATUS "OpenMP C++ flags:        ${OpenMP_CXX_FLAGS} ")
    if ( BUILD_SHARED_LIBS )
        target_link_libraries ( ParU PRIVATE OpenMP::OpenMP_CXX )
    endif ( )
    if ( BUILD_STATIC_LIBS )
        list ( APPEND PARU_STATIC_LIBS ${OpenMP_CXX_LIBRARIES} )
        target_link_libraries ( ParU_static PRIVATE OpenMP::OpenMP_CXX )
    endif ( )
endif ( )

# libm:
if ( NOT WIN32 )
    if ( BUILD_SHARED_LIBS )
        target_link_libraries ( ParU PRIVATE m )
    endif ( )
    if ( BUILD_STATIC_LIBS )
        list ( APPEND PARU_STATIC_LIBS m )
        target_link_libraries ( ParU_static PUBLIC m )
    endif ( )
endif ( )

# BLAS:
message ( STATUS "BLAS libraries:      ${BLAS_LIBRARIES} ")
message ( STATUS "BLAS linker flags:   ${BLAS_LINKER_FLAGS} ")
if ( BUILD_SHARED_LIBS )
    target_link_libraries ( ParU PRIVATE ${BLAS_LIBRARIES} )
    target_include_directories ( ParU PRIVATE ${BLAS_INCLUDE_DIRS} )
endif ( )
if ( BUILD_STATIC_LIBS )
    list ( APPEND PARU_STATIC_LIBS ${BLAS_LIBRARIES} )
    target_link_libraries ( ParU_static PUBLIC ${BLAS_LIBRARIES} )
    target_include_directories ( ParU_static PRIVATE ${BLAS_INCLUDE_DIRS} )
endif ( )

#-------------------------------------------------------------------------------
# PARU installation location
#-------------------------------------------------------------------------------

include ( CMakePackageConfigHelpers )

if ( BUILD_SHARED_LIBS )
    install ( TARGETS ParU
        EXPORT ParUTargets
        LIBRARY DESTINATION ${SUITESPARSE_LIBDIR}
        ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR}
        RUNTIME DESTINATION ${SUITESPARSE_BINDIR}
        PUBLIC_HEADER DESTINATION ${SUITESPARSE_INCLUDEDIR} )
endif ( )

if ( BUILD_STATIC_LIBS )
    install ( TARGETS ParU_static
        EXPORT ParUTargets
        ARCHIVE       DESTINATION ${SUITESPARSE_LIBDIR}
        PUBLIC_HEADER DESTINATION ${SUITESPARSE_INCLUDEDIR} )
endif ( )

# create (temporary) export target file during build
export ( EXPORT ParUTargets
    NAMESPACE SuiteSparse::
    FILE ${CMAKE_CURRENT_BINARY_DIR}/ParUTargets.cmake )

# install export target, config and version files for find_package
install ( EXPORT ParUTargets
    NAMESPACE SuiteSparse::
    DESTINATION ${SUITESPARSE_PKGFILEDIR}/cmake/ParU )

# generate config file to be used in common build tree
set ( SUITESPARSE_IN_BUILD_TREE ON )
configure_package_config_file (
    Config/ParUConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/ParUConfig.cmake
    INSTALL_DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/ParUConfig.cmake )

# generate config file to be installed
set ( SUITESPARSE_IN_BUILD_TREE OFF )
configure_package_config_file (
    Config/ParUConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/target/ParUConfig.cmake
    INSTALL_DESTINATION ${SUITESPARSE_PKGFILEDIR}/cmake/ParU )

write_basic_package_version_file (
    ${CMAKE_CURRENT_BINARY_DIR}/ParUConfigVersion.cmake
    COMPATIBILITY SameMajorVersion )

install ( FILES
    ${CMAKE_CURRENT_BINARY_DIR}/target/ParUConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/ParUConfigVersion.cmake
    DESTINATION ${SUITESPARSE_PKGFILEDIR}/cmake/ParU )

#-------------------------------------------------------------------------------
# create pkg-config file
#-------------------------------------------------------------------------------

if ( NOT MSVC )
    # This might be something like:
    #   /usr/lib/libgomp.so;/usr/lib/libpthread.a;m
    # convert to -l flags for pkg-config, i.e.: "-lgomp -lpthread -lm"
    set ( PARU_STATIC_LIBS_LIST ${PARU_STATIC_LIBS} )
    set ( PARU_STATIC_LIBS "" )
    foreach ( _lib ${PARU_STATIC_LIBS_LIST} )
        string ( FIND ${_lib} "." _pos REVERSE )
        if ( ${_pos} EQUAL "-1" )
            set ( PARU_STATIC_LIBS "${PARU_STATIC_LIBS} -l${_lib}" )
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
                    set ( PARU_STATIC_LIBS "${PARU_STATIC_LIBS} -l${_libname}" )
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
        set ( SUITESPARSE_LIB_BASE_NAME $<TARGET_FILE_BASE_NAME:ParU> )
    else ( )
        set ( SUITESPARSE_LIB_BASE_NAME $<TARGET_FILE_BASE_NAME:ParU_static> )
    endif ( )
    configure_file (
        Config/ParU.pc.in
        ParU.pc.out
        @ONLY
        NEWLINE_STYLE LF )
    file ( GENERATE
        OUTPUT ParU.pc
        INPUT ${CMAKE_CURRENT_BINARY_DIR}/ParU.pc.out
        NEWLINE_STYLE LF )
    install ( FILES
        ${CMAKE_CURRENT_BINARY_DIR}/ParU.pc
        DESTINATION ${SUITESPARSE_PKGFILEDIR}/pkgconfig )
endif ( )

#-------------------------------------------------------------------------------
# Demo library and programs
#-------------------------------------------------------------------------------

if ( SUITESPARSE_DEMOS )

    #---------------------------------------------------------------------------
    # demo library
    #---------------------------------------------------------------------------

    message ( STATUS "Also compiling the demos in ParU/Demo" )

    #---------------------------------------------------------------------------
    # Demo programs
    #---------------------------------------------------------------------------

    if ( PARU_HAS_OPENMP )
        add_executable ( paru_demo "Demo/paru_demo.cpp" )
        if ( BUILD_SHARED_LIBS )
            target_link_libraries ( paru_demo PUBLIC ParU )
        else ( )
            target_link_libraries ( paru_demo PUBLIC ParU_static )
        endif ( )
        target_link_libraries ( paru_demo
            PUBLIC SuiteSparse::CHOLMOD SuiteSparse::UMFPACK OpenMP::OpenMP_CXX )

        add_executable ( paru_democ "Demo/paru_democ.c" )
        if ( BUILD_SHARED_LIBS )
            target_link_libraries ( paru_democ PUBLIC ParU )
        else ( )
            target_link_libraries ( paru_democ PUBLIC ParU_static )
        endif ( )
        target_link_libraries ( paru_democ
            PUBLIC SuiteSparse::CHOLMOD SuiteSparse::UMFPACK OpenMP::OpenMP_C )
    endif ( )

    add_executable ( paru_simple  "Demo/paru_simple.cpp" )
    add_executable ( paru_simplec "Demo/paru_simplec.c"  )
    if ( BUILD_SHARED_LIBS )
        target_link_libraries ( paru_simple PUBLIC ParU )
        target_link_libraries ( paru_simplec PUBLIC ParU )
    else ( )
        target_link_libraries ( paru_simple PUBLIC ParU_static )
        target_link_libraries ( paru_simplec PUBLIC ParU_static )
    endif ( )
    target_link_libraries ( paru_simple PUBLIC SuiteSparse::CHOLMOD )
    target_link_libraries ( paru_simplec PUBLIC SuiteSparse::CHOLMOD )

else ( )

    message ( STATUS "Skipping the demos in ParU/Demo" )

endif ( )

#-------------------------------------------------------------------------------
# report status
#-------------------------------------------------------------------------------

include ( SuiteSparseReport )
