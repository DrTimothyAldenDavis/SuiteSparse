#-------------------------------------------------------------------------------
# CXSparse/CMakeLists.txt:  cmake for CXSparse
#-------------------------------------------------------------------------------

# Copyright (c) 2006-2023, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: LGPL-2.1+

#-------------------------------------------------------------------------------
# get the version
#-------------------------------------------------------------------------------

cmake_minimum_required ( VERSION 3.19 )

set ( CXSPARSE_DATE "Jan 9, 2023" )
set ( CXSPARSE_VERSION_MAJOR 4 )
set ( CXSPARSE_VERSION_MINOR 0 )
set ( CXSPARSE_VERSION_SUB   3 )

message ( STATUS "Building CXSparse version: v"
    ${CXSPARSE_VERSION_MAJOR}.
    ${CXSPARSE_VERSION_MINOR}.
    ${CXSPARSE_VERSION_SUB} " (" ${CXSPARSE_DATE} ")" )

#-------------------------------------------------------------------------------
# SuiteSparse policies
#-------------------------------------------------------------------------------

set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    ${CMAKE_SOURCE_DIR}/cmake_modules
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
else ( )
    message ( STATUS "complex data type: enabled" )
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

find_package ( SuiteSparse_config 7.0.0 REQUIRED )

#-------------------------------------------------------------------------------
# Configure cs.h with version number
#-------------------------------------------------------------------------------

configure_file ( "Config/cs.h.in"
    "${PROJECT_SOURCE_DIR}/Include/cs.h"
    NEWLINE_STYLE LF )

#-------------------------------------------------------------------------------
# include directories
#-------------------------------------------------------------------------------

include_directories ( Source Include ${SUITESPARSE_CONFIG_INCLUDE_DIR} )

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

add_library ( cxsparse SHARED ${CXSPARSE_SOURCES} )

set_target_properties ( cxsparse PROPERTIES
    VERSION ${CXSPARSE_VERSION_MAJOR}.${CXSPARSE_VERSION_MINOR}.${CXSPARSE_VERSION_SUB}
    C_STANDARD_REQUIRED 11
    SOVERSION ${CXSPARSE_VERSION_MAJOR}
    PUBLIC_HEADER "Include/cs.h"
    WINDOWS_EXPORT_ALL_SYMBOLS ON )

#-------------------------------------------------------------------------------
# static cxsparse library properties
#-------------------------------------------------------------------------------

if ( NOT NSTATIC )
    add_library ( cxsparse_static STATIC ${CXSPARSE_SOURCES} )

    set_target_properties ( cxsparse_static PROPERTIES
        VERSION ${CXSPARSE_VERSION_MAJOR}.${CXSPARSE_VERSION_MINOR}.${CXSPARSE_VERSION_SUB}
        OUTPUT_NAME cxsparse
        C_STANDARD_REQUIRED 11
        SOVERSION ${CXSPARSE_VERSION_MAJOR} )

    if ( MSVC )
        set_target_properties ( cxsparse_static PROPERTIES
            OUTPUT_NAME cxsparse_static )
    endif ( )
endif ( )

#-------------------------------------------------------------------------------
# add the library dependencies
#-------------------------------------------------------------------------------

target_link_libraries ( cxsparse PUBLIC ${SUITESPARSE_CONFIG_LIBRARY} )
if ( NOT NSTATIC )
    target_link_libraries ( cxsparse_static PUBLIC ${SUITESPARSE_CONFIG_LIBRARY} )
endif ( )

# libm:
if ( NOT WIN32 )
    target_link_libraries ( cxsparse PUBLIC m )
    if ( NOT NSTATIC )
        target_link_libraries ( cxsparse_static PUBLIC m )
    endif ( )
endif ( )

#-------------------------------------------------------------------------------
# installation location
#-------------------------------------------------------------------------------

if ( GLOBAL_INSTALL )
    # install in /usr/local/lib and /usr/local/include.
    # requires "sudo make install"
    message ( STATUS "Installation will be system-wide (requires 'sudo make install')" )
    install ( TARGETS cxsparse
        LIBRARY       DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} )
    install ( FILES ${CMAKE_SOURCE_DIR}/cmake_modules/FindCXSparse.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SuiteSparse 
        COMPONENT Development )
    if ( NOT NSTATIC )
    install ( TARGETS cxsparse_static
        ARCHIVE       DESTINATION ${CMAKE_INSTALL_LIBDIR} )
    endif ( )
endif ( )

if ( INSIDE_SUITESPARSE )
    # also install in SuiteSparse/lib and SuiteSparse/include;
    # does not require "sudo make install", just "make install"
    message ( STATUS "Installation in ../lib and ../include," )
    message ( STATUS "  with 'make local ; make install'. No 'sudo' required." )
    install ( TARGETS cxsparse
        LIBRARY       DESTINATION ${SUITESPARSE_LIBDIR}
        PUBLIC_HEADER DESTINATION ${SUITESPARSE_INCLUDEDIR} )
    install ( FILES ${CMAKE_SOURCE_DIR}/cmake_modules/FindCXSparse.cmake
        DESTINATION ${SUITESPARSE_LIBDIR}/cmake/SuiteSparse 
        COMPONENT Development )
    if ( NOT NSTATIC )
    install ( TARGETS cxsparse_static
        ARCHIVE       DESTINATION ${SUITESPARSE_LIBDIR} )
    endif ( )
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
    target_link_libraries ( cs_demo1 PUBLIC cxsparse )
    target_link_libraries ( cs_demo2 PUBLIC cxsparse )
    target_link_libraries ( cs_demo3 PUBLIC cxsparse )

    target_link_libraries ( cs_di_demo1 PUBLIC cxsparse )
    target_link_libraries ( cs_di_demo2 PUBLIC cxsparse )
    target_link_libraries ( cs_di_demo3 PUBLIC cxsparse )

    target_link_libraries ( cs_dl_demo1 PUBLIC cxsparse )
    target_link_libraries ( cs_dl_demo2 PUBLIC cxsparse )
    target_link_libraries ( cs_dl_demo3 PUBLIC cxsparse )

    if ( NOT NCOMPLEX )
        target_link_libraries ( cs_ci_demo1 PUBLIC cxsparse )
        target_link_libraries ( cs_ci_demo2 PUBLIC cxsparse )
        target_link_libraries ( cs_ci_demo3 PUBLIC cxsparse )

        target_link_libraries ( cs_cl_demo1 PUBLIC cxsparse )
        target_link_libraries ( cs_cl_demo2 PUBLIC cxsparse )
        target_link_libraries ( cs_cl_demo3 PUBLIC cxsparse )

        target_link_libraries ( cs_idemo PUBLIC cxsparse )
        target_link_libraries ( cs_ldemo PUBLIC cxsparse )
    endif ( )

else ( )

    message ( STATUS "Skipping the demos in CXSparse/Demo" )

endif ( )

#-------------------------------------------------------------------------------
# report status
#-------------------------------------------------------------------------------

include ( SuiteSparseReport )

