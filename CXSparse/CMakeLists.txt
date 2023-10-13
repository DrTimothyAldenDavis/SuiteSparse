#-------------------------------------------------------------------------------
# CXSparse/CMakeLists.txt:  cmake for CXSparse
#-------------------------------------------------------------------------------

# Copyright (c) 2006-2023, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: LGPL-2.1+

#-------------------------------------------------------------------------------
# get the version
#-------------------------------------------------------------------------------

cmake_minimum_required ( VERSION 3.20 )

set ( CXSPARSE_DATE "Sept 18, 2023" )
set ( CXSPARSE_VERSION_MAJOR 4 )
set ( CXSPARSE_VERSION_MINOR 2 )
set ( CXSPARSE_VERSION_SUB   1 )

message ( STATUS "Building CXSparse version: v"
    ${CXSPARSE_VERSION_MAJOR}.
    ${CXSPARSE_VERSION_MINOR}.
    ${CXSPARSE_VERSION_SUB} " (" ${CXSPARSE_DATE} ")" )

#-------------------------------------------------------------------------------
# SuiteSparse policies
#-------------------------------------------------------------------------------

set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    ${CMAKE_SOURCE_DIR}/../SuiteSparse_config/cmake_modules )

include ( SuiteSparsePolicy )

#-------------------------------------------------------------------------------
# complex support
#-------------------------------------------------------------------------------

# MS Visual Studio does not support the complex type for ANSI C11.

if ( MSVC )
    option ( NCOMPLEX "ON (default): complex data type disabled.  OFF: complex data type enabled." on )
else ( )
    option ( NCOMPLEX "ON: complex data type disabled.  OFF (default): complex data type enabled." off )
endif ( )

if ( NCOMPLEX )
    message ( STATUS "complex data type: disabled" )
    add_compile_definitions ( NCOMPLEX )
    set ( CXSPARSE_USE_COMPLEX 0 )
else ( )
    message ( STATUS "complex data type: enabled" )
    set ( CXSPARSE_USE_COMPLEX 1 )
endif ( )

#-------------------------------------------------------------------------------
# define the project
#-------------------------------------------------------------------------------

project ( cxsparse
    VERSION "${CXSPARSE_VERSION_MAJOR}.${CXSPARSE_VERSION_MINOR}.${CXSPARSE_VERSION_SUB}"
    LANGUAGES C )

#-------------------------------------------------------------------------------
# find library dependencies
#-------------------------------------------------------------------------------

find_package ( SuiteSparse_config 7.2.0
    PATHS ${CMAKE_SOURCE_DIR}/../SuiteSparse_config/build NO_DEFAULT_PATH )
if ( NOT TARGET SuiteSparse::SuiteSparseConfig )
    find_package ( SuiteSparse_config 7.2.0 REQUIRED )
endif ( )

#-------------------------------------------------------------------------------
# Configure cs.h with version number
#-------------------------------------------------------------------------------

configure_file ( "Config/cs.h.in"
    "${PROJECT_SOURCE_DIR}/Include/cs.h"
    NEWLINE_STYLE LF )

#-------------------------------------------------------------------------------
# include directories
#-------------------------------------------------------------------------------

include_directories ( Source Include )

#-------------------------------------------------------------------------------
# dynamic cxsparse library properties
#-------------------------------------------------------------------------------

if ( NCOMPLEX )
    # exclude functions that work on complex data types
    file ( GLOB CXSPARSE_SOURCES
        "Source/cs_a*.c"
        "Source/cs_chol*.c"
        "Source/cs_compress.c"
        "Source/cs_counts.c"
        "Source/cs_cumsum.c"
        "Source/cs_[d-z]*.c"
        )
else ( )
    file ( GLOB CXSPARSE_SOURCES "Source/*.c" )
endif ( )

add_library ( CXSparse SHARED ${CXSPARSE_SOURCES} )

set_target_properties ( CXSparse PROPERTIES
    VERSION ${CXSPARSE_VERSION_MAJOR}.${CXSPARSE_VERSION_MINOR}.${CXSPARSE_VERSION_SUB}
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
    OUTPUT_NAME cxsparse
    SOVERSION ${CXSPARSE_VERSION_MAJOR}
    PUBLIC_HEADER "Include/cs.h"
    WINDOWS_EXPORT_ALL_SYMBOLS ON )

target_include_directories ( CXSparse
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>
              $<INSTALL_INTERFACE:${SUITESPARSE_INCLUDEDIR}> )

#-------------------------------------------------------------------------------
# static cxsparse library properties
#-------------------------------------------------------------------------------

if ( NOT NSTATIC )
    add_library ( CXSparse_static STATIC ${CXSPARSE_SOURCES} )

    set_target_properties ( CXSparse_static PROPERTIES
        VERSION ${CXSPARSE_VERSION_MAJOR}.${CXSPARSE_VERSION_MINOR}.${CXSPARSE_VERSION_SUB}
        OUTPUT_NAME cxsparse
        C_STANDARD 11
        C_STANDARD_REQUIRED ON
        SOVERSION ${CXSPARSE_VERSION_MAJOR} )

    if ( MSVC )
        set_target_properties ( CXSparse_static PROPERTIES
            OUTPUT_NAME cxsparse_static )
    endif ( )

    target_include_directories ( CXSparse_static
        INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>
                  $<INSTALL_INTERFACE:${SUITESPARSE_INCLUDEDIR}> )

endif ( )

#-------------------------------------------------------------------------------
# add the library dependencies
#-------------------------------------------------------------------------------

target_link_libraries ( CXSparse PRIVATE SuiteSparse::SuiteSparseConfig )
target_include_directories ( CXSparse PUBLIC
    "$<TARGET_PROPERTY:SuiteSparse::SuiteSparseConfig,INTERFACE_INCLUDE_DIRECTORIES>" )
if ( NOT NSTATIC )
    if ( TARGET SuiteSparse::SuiteSparseConfig_static )
        target_link_libraries ( CXSparse_static PUBLIC SuiteSparse::SuiteSparseConfig_static )
    else ( )
        target_link_libraries ( CXSparse_static PUBLIC SuiteSparse::SuiteSparseConfig )
    endif ( )
endif ( )

# libm:
if ( NOT WIN32 )
    target_link_libraries ( CXSparse PRIVATE m )
    if ( NOT NSTATIC )
        set ( CXSPARSE_STATIC_LIBS "${CXSPARSE_STATIC_LIBS} -lm" )
        target_link_libraries ( CXSparse_static PUBLIC m )
    endif ( )
endif ( )

#-------------------------------------------------------------------------------
# installation location
#-------------------------------------------------------------------------------

include ( CMakePackageConfigHelpers )

install ( TARGETS CXSparse
    EXPORT CXSparseTargets
    LIBRARY DESTINATION ${SUITESPARSE_LIBDIR}
    ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR}
    RUNTIME DESTINATION ${SUITESPARSE_BINDIR}
    PUBLIC_HEADER DESTINATION ${SUITESPARSE_INCLUDEDIR} )
if ( NOT NSTATIC )
    install ( TARGETS CXSparse_static
        EXPORT CXSparseTargets
        ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR} )
endif ( )

# create (temporary) export target file during build
export ( EXPORT CXSparseTargets
    NAMESPACE SuiteSparse::
    FILE ${CMAKE_CURRENT_BINARY_DIR}/CXSparseTargets.cmake )

# install export target, config and version files for find_package
install ( EXPORT CXSparseTargets
    NAMESPACE SuiteSparse::
    DESTINATION ${SUITESPARSE_LIBDIR}/cmake/CXSparse )

configure_package_config_file (
    Config/CXSparseConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/CXSparseConfig.cmake
    INSTALL_DESTINATION ${SUITESPARSE_LIBDIR}/cmake/CXSparse )

write_basic_package_version_file (
    ${CMAKE_CURRENT_BINARY_DIR}/CXSparseConfigVersion.cmake
    COMPATIBILITY SameMajorVersion )

install ( FILES
    ${CMAKE_CURRENT_BINARY_DIR}/CXSparseConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/CXSparseConfigVersion.cmake
    DESTINATION ${SUITESPARSE_LIBDIR}/cmake/CXSparse )

#-------------------------------------------------------------------------------
# create pkg-config file
#-------------------------------------------------------------------------------

if ( NOT MSVC )
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
        Config/CXSparse.pc.in
        CXSparse.pc
        @ONLY
        NEWLINE_STYLE LF )
    install ( FILES
        ${CMAKE_CURRENT_BINARY_DIR}/CXSparse.pc
        DESTINATION ${SUITESPARSE_LIBDIR}/pkgconfig )
endif ( )

#-------------------------------------------------------------------------------
# Demo library and programs
#-------------------------------------------------------------------------------

option ( DEMO "ON: Build the demo programs.  OFF (default): do not build the demo programs." off )
if ( DEMO )

    #---------------------------------------------------------------------------
    # demo library
    #---------------------------------------------------------------------------

    message ( STATUS "Also compiling the demos in CXSparse/Demo" )

    #---------------------------------------------------------------------------
    # Demo programs
    #---------------------------------------------------------------------------

    add_executable ( cs_demo1 "Demo/cs_demo1.c" "Demo/cs_demo.c" )
    add_executable ( cs_demo2 "Demo/cs_demo2.c" "Demo/cs_demo.c" )
    add_executable ( cs_demo3 "Demo/cs_demo3.c" "Demo/cs_demo.c" )

    add_executable ( cs_di_demo1 "Demo/cs_di_demo1.c" "Demo/cs_di_demo.c" )
    add_executable ( cs_di_demo2 "Demo/cs_di_demo2.c" "Demo/cs_di_demo.c" )
    add_executable ( cs_di_demo3 "Demo/cs_di_demo3.c" "Demo/cs_di_demo.c" )

    add_executable ( cs_dl_demo1 "Demo/cs_dl_demo1.c" "Demo/cs_dl_demo.c" )
    add_executable ( cs_dl_demo2 "Demo/cs_dl_demo2.c" "Demo/cs_dl_demo.c" )
    add_executable ( cs_dl_demo3 "Demo/cs_dl_demo3.c" "Demo/cs_dl_demo.c" )

    if ( NOT NCOMPLEX )
        add_executable ( cs_ci_demo1 "Demo/cs_ci_demo1.c" "Demo/cs_ci_demo.c" )
        add_executable ( cs_ci_demo2 "Demo/cs_ci_demo2.c" "Demo/cs_ci_demo.c" )
        add_executable ( cs_ci_demo3 "Demo/cs_ci_demo3.c" "Demo/cs_ci_demo.c" )

        add_executable ( cs_cl_demo1 "Demo/cs_cl_demo1.c" "Demo/cs_cl_demo.c" )
        add_executable ( cs_cl_demo2 "Demo/cs_cl_demo2.c" "Demo/cs_cl_demo.c" )
        add_executable ( cs_cl_demo3 "Demo/cs_cl_demo3.c" "Demo/cs_cl_demo.c" )

        add_executable ( cs_idemo "Demo/cs_idemo.c" )
        add_executable ( cs_ldemo "Demo/cs_ldemo.c" )
    endif ( )

    # Libraries required for Demo programs
    target_link_libraries ( cs_demo1 PUBLIC CXSparse )
    target_link_libraries ( cs_demo2 PUBLIC CXSparse )
    target_link_libraries ( cs_demo3 PUBLIC CXSparse )

    target_link_libraries ( cs_di_demo1 PUBLIC CXSparse )
    target_link_libraries ( cs_di_demo2 PUBLIC CXSparse )
    target_link_libraries ( cs_di_demo3 PUBLIC CXSparse )

    target_link_libraries ( cs_dl_demo1 PUBLIC CXSparse )
    target_link_libraries ( cs_dl_demo2 PUBLIC CXSparse )
    target_link_libraries ( cs_dl_demo3 PUBLIC CXSparse )

    if ( NOT NCOMPLEX )
        target_link_libraries ( cs_ci_demo1 PUBLIC CXSparse )
        target_link_libraries ( cs_ci_demo2 PUBLIC CXSparse )
        target_link_libraries ( cs_ci_demo3 PUBLIC CXSparse )

        target_link_libraries ( cs_cl_demo1 PUBLIC CXSparse )
        target_link_libraries ( cs_cl_demo2 PUBLIC CXSparse )
        target_link_libraries ( cs_cl_demo3 PUBLIC CXSparse )

        if ( NOT WIN32 )
            target_link_libraries ( cs_ci_demo1 PUBLIC m )
            target_link_libraries ( cs_ci_demo2 PUBLIC m )
            target_link_libraries ( cs_ci_demo3 PUBLIC m )

            target_link_libraries ( cs_cl_demo1 PUBLIC m )
            target_link_libraries ( cs_cl_demo2 PUBLIC m )
            target_link_libraries ( cs_cl_demo3 PUBLIC m )
        endif ( )

        target_link_libraries ( cs_idemo PUBLIC CXSparse )
        target_link_libraries ( cs_ldemo PUBLIC CXSparse )
    endif ( )

else ( )

    message ( STATUS "Skipping the demos in CXSparse/Demo" )

endif ( )

#-------------------------------------------------------------------------------
# report status
#-------------------------------------------------------------------------------

include ( SuiteSparseReport )

