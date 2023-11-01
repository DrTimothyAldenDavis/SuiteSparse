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

set ( PARU_DATE "Dec 20, 2022" )    # FIXME: pick a date
set ( PARU_VERSION_MAJOR 1 )
set ( PARU_VERSION_MINOR 0 )
set ( PARU_VERSION_UPDATE 0 )

message ( STATUS "Building PARU version: v"
    ${PARU_VERSION_MAJOR}.
    ${PARU_VERSION_MINOR}.
    ${PARU_VERSION_UPDATE} " (" ${PARU_DATE} ")" )

#-------------------------------------------------------------------------------
# SuiteSparse policies
#-------------------------------------------------------------------------------

set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    ${CMAKE_SOURCE_DIR}/../SuiteSparse_config/cmake_modules
    )

include ( SuiteSparsePolicy )

#-------------------------------------------------------------------------------
# define the project
#-------------------------------------------------------------------------------

project ( paru
    VERSION "${PARU_VERSION_MAJOR}.${PARU_VERSION_MINOR}.${PARU_VERSION_UPDATE}" )

#-------------------------------------------------------------------------------
# find library dependencies
#-------------------------------------------------------------------------------

enable_language ( CXX )

if ( DEMO )
    enable_language ( C )
endif ( )

option ( NOPENMP "ON: do not use OpenMP.  OFF (default): use OpenMP" OFF )
if ( NOPENMP )
    # OpenMP has been disabled
    set ( OpenMP_CXX_FOUND OFF )
    set ( OpenMP_C_FOUND OFF )
else ( )
    find_package ( OpenMP )
    if ( OpenMP_CXX_FOUND AND OpenMP_CXX_VERSION VERSION_LESS 4.5 )
        set ( NOPENMP ON )
    endif ( )
    if ( DEMO AND OpenMP_C_FOUND AND OpenMP_C_VERSION VERSION_LESS 4.5 )
        set ( NOPENMP ON )
    endif ( )
    if ( NOPENMP )
        message ( STATUS "Version of OpenMP is too old." )
        set ( OpenMP_CXX_FOUND OFF )
        set ( OpenMP_C_FOUND OFF )
    endif ( )
endif ( )

find_package ( SuiteSparse_config 7.3.0
    PATHS ${CMAKE_SOURCE_DIR}/../SuiteSparse_config/build NO_DEFAULT_PATH )
if ( NOT TARGET SuiteSparse::SuiteSparseConfig )
    find_package ( SuiteSparse_config 7.3.0 REQUIRED )
endif ( )

find_package ( CHOLMOD 5.0.0
    PATHS ${CMAKE_SOURCE_DIR}/../CHOLMOD/build NO_DEFAULT_PATH )
if ( NOT CHOLMOD_FOUND )
    find_package ( CHOLMOD 5.0.0 REQUIRED )
endif ( )

find_package ( UMFPACK 6.2.2
    PATHS ${CMAKE_SOURCE_DIR}/../UMFPACK/build NO_DEFAULT_PATH )
if ( NOT CAMD_FOUND )
    find_package ( UMFPACK 6.2.2 REQUIRED )
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
    ${CMAKE_SOURCE_DIR}/../UMFPACK/Source
    )

#-------------------------------------------------------------------------------
# dynamic paru library properties
#-------------------------------------------------------------------------------

file ( GLOB PARU_SOURCES "Source/*.cpp" )

add_library ( ParU SHARED ${PARU_SOURCES} )

set_target_properties ( ParU PROPERTIES
    VERSION ${PARU_VERSION_MAJOR}.${PARU_VERSION_MINOR}.${PARU_VERSION_UPDATE}
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
    OUTPUT_NAME paru
    SOVERSION ${PARU_VERSION_MAJOR}
    PUBLIC_HEADER "Include/ParU.hpp;Include/ParU_C.h;Include/ParU_definitions.h"
    WINDOWS_EXPORT_ALL_SYMBOLS ON )

target_include_directories ( ParU
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>
              $<INSTALL_INTERFACE:${SUITESPARSE_INCLUDEDIR}> )

#-------------------------------------------------------------------------------
# static paru library properties
#-------------------------------------------------------------------------------

if ( NOT NSTATIC )
    add_library ( ParU_static STATIC ${PARU_SOURCES} )

    set_target_properties ( ParU_static PROPERTIES
        VERSION ${PARU_VERSION_MAJOR}.${PARU_VERSION_MINOR}.${PARU_VERSION_UPDATE}
        CXX_STANDARD 11
        CXX_STANDARD_REQUIRED ON
        OUTPUT_NAME paru
        SOVERSION ${PARU_VERSION_MAJOR}
        PUBLIC_HEADER "Include/ParU.hpp;Include/ParU_C.h;Include/ParU_definitions.h"
        WINDOWS_EXPORT_ALL_SYMBOLS ON )

    if ( MSVC )
        set_target_properties ( ParU_static PROPERTIES
            OUTPUT_NAME ParU_static )
    endif ( )

    target_include_directories ( ParU_static
        INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>
                  $<INSTALL_INTERFACE:${SUITESPARSE_INCLUDEDIR}> )
endif ( )

#-------------------------------------------------------------------------------
# add the library dependencies
#-------------------------------------------------------------------------------

# suitesparseconfig:
target_link_libraries ( ParU PRIVATE SuiteSparse::SuiteSparseConfig )
target_include_directories ( ParU PUBLIC
    "$<TARGET_PROPERTY:SuiteSparse::SuiteSparseConfig,INTERFACE_INCLUDE_DIRECTORIES>" )
if ( NOT NSTATIC )
    if ( TARGET SuiteSparse::SuiteSparseConfig_static )
        target_link_libraries ( ParU_static PUBLIC SuiteSparse::SuiteSparseConfig_static )
    else ( )
        target_link_libraries ( ParU_static PUBLIC SuiteSparse::SuiteSparseConfig )
    endif ( )
endif ( )

# OpenMP:
if ( OpenMP_CXX_FOUND )
    message ( STATUS "OpenMP C++ libraries:    ${OpenMP_CXX_LIBRARIES} ")
    message ( STATUS "OpenMP C++ include:      ${OpenMP_CXX_INCLUDE_DIRS} ")
    message ( STATUS "OpenMP C++ flags:        ${OpenMP_CXX_FLAGS} ")
    target_link_libraries ( ParU PRIVATE OpenMP::OpenMP_CXX )
    if ( NOT NSTATIC )
        list ( APPEND PARU_STATIC_LIBS ${OpenMP_CXX_LIBRARIES} )
        target_link_libraries ( ParU_static PUBLIC OpenMP::OpenMP_CXX )
    endif ( )
endif ( )

# libm:
if ( NOT MSVC )
    target_link_libraries ( ParU PRIVATE m )
    if ( NOT NSTATIC )
        list ( APPEND PARU_STATIC_LIBS m )
        target_link_libraries ( ParU_static PUBLIC m )
    endif ( )
endif ( )

# umfpack:
target_link_libraries ( ParU PRIVATE SuiteSparse::UMFPACK )
target_include_directories ( ParU PUBLIC
    "$<TARGET_PROPERTY:SuiteSparse::UMFPACK,INTERFACE_INCLUDE_DIRECTORIES>" )
if ( NOT NSTATIC )
    if ( TARGET SuiteSparse::UMFPACK_static )
        target_link_libraries ( ParU_static PUBLIC SuiteSparse::UMFPACK_static )
    else ( )
        target_link_libraries ( ParU_static PUBLIC SuiteSparse::UMFPACK )
    endif ( )
endif ( )

# BLAS:
message ( STATUS "BLAS libraries:      ${BLAS_LIBRARIES} ")
message ( STATUS "BLAS linker flags:   ${BLAS_LINKER_FLAGS} ")
target_link_libraries ( ParU PRIVATE ${BLAS_LIBRARIES} )
if ( NOT NSTATIC )
    list ( APPEND PARU_STATIC_LIBS ${BLAS_LIBRARIES} )
    target_link_libraries ( ParU_static PUBLIC ${BLAS_LIBRARIES} )
    include_directories ( ${BLAS_INCLUDE_DIRS} )
endif ( )

# cholmod:
target_include_directories ( ParU PUBLIC
    "$<TARGET_PROPERTY:SuiteSparse::CHOLMOD,INTERFACE_INCLUDE_DIRECTORIES>" )
if ( NOT NSTATIC )
    target_include_directories ( ParU_static PUBLIC
        "$<TARGET_PROPERTY:SuiteSparse::CHOLMOD,INTERFACE_INCLUDE_DIRECTORIES>" )
endif ( )

#-------------------------------------------------------------------------------
# PARU installation location
#-------------------------------------------------------------------------------

include ( CMakePackageConfigHelpers )

install ( TARGETS ParU
    EXPORT ParUTargets
    LIBRARY DESTINATION ${SUITESPARSE_LIBDIR}
    ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR}
    RUNTIME DESTINATION ${SUITESPARSE_BINDIR}
    PUBLIC_HEADER DESTINATION ${SUITESPARSE_INCLUDEDIR} )

if ( NOT NSTATIC )
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
    configure_file (
        Config/ParU.pc.in
        ParU.pc
        @ONLY
        NEWLINE_STYLE LF )
    install ( FILES
        ${CMAKE_CURRENT_BINARY_DIR}/ParU.pc
        DESTINATION ${SUITESPARSE_PKGFILEDIR}/pkgconfig )
endif ( )

#-------------------------------------------------------------------------------
# Demo library and programs
#-------------------------------------------------------------------------------

if ( DEMO )

    #---------------------------------------------------------------------------
    # demo library
    #---------------------------------------------------------------------------

    message ( STATUS "Also compiling the demos in ParU/Demo" )

    #---------------------------------------------------------------------------
    # Demo programs
    #---------------------------------------------------------------------------

    if ( OpenMP_CXX_FOUND )
        add_executable ( paru_demo    "Demo/paru_demo.cpp"   )
        target_link_libraries ( paru_demo
            PUBLIC ParU SuiteSparse::CHOLMOD SuiteSparse::UMFPACK OpenMP::OpenMP_CXX )
    endif ( )

    if ( OpenMP_C_FOUND )
        add_executable ( paru_democ   "Demo/paru_democ.c"    )
        target_link_libraries ( paru_democ
            PUBLIC ParU SuiteSparse::CHOLMOD SuiteSparse::UMFPACK OpenMP::OpenMP_C )
    endif ( )

    add_executable ( paru_simple  "Demo/paru_simple.cpp" )
    target_link_libraries ( paru_simple
        PUBLIC ParU SuiteSparse::CHOLMOD )

    add_executable ( paru_simplec "Demo/paru_simplec.c"  )
    target_link_libraries ( paru_simplec
        PUBLIC ParU SuiteSparse::CHOLMOD )

else ( )

    message ( STATUS "Skipping the demos in ParU/Demo" )

endif ( )

#-------------------------------------------------------------------------------
# report status
#-------------------------------------------------------------------------------

include ( SuiteSparseReport )

