#-------------------------------------------------------------------------------
# CSparse/CMakeLists.txt:  cmake for CSparse
#-------------------------------------------------------------------------------

# Copyright (c) 2006-2023, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: LGPL-2.1+

# CSparse/CMakeLists.txt is a standalone script, which can be used outside of
# SuiteSparse.  There is no "make install" target for CSparse.  The library
# (libcsparse.so) and include file (cs.h) are NOT installed in ../lib and
# ../include.  CSparse is meant for classwork and stand-alone development,
# while SuiteSparse/CXSparse should be used in production.

#-------------------------------------------------------------------------------
# get the version
#-------------------------------------------------------------------------------

cmake_minimum_required ( VERSION 3.13 )

set ( CSPARSE_DATE "Dec 30, 2023" )
set ( CSPARSE_VERSION_MAJOR 4 CACHE STRING "" FORCE )
set ( CSPARSE_VERSION_MINOR 3 CACHE STRING "" FORCE )
set ( CSPARSE_VERSION_SUB   0 CACHE STRING "" FORCE )

message ( STATUS "Building CSparse version: v"
    ${CSPARSE_VERSION_MAJOR}.
    ${CSPARSE_VERSION_MINOR}.
    ${CSPARSE_VERSION_SUB} " (" ${CSPARSE_DATE} ")" )

#-------------------------------------------------------------------------------
# define the project
#-------------------------------------------------------------------------------

project ( csparse
    VERSION "${CSPARSE_VERSION_MAJOR}.${CSPARSE_VERSION_MINOR}.${CSPARSE_VERSION_SUB}"
    LANGUAGES C )

#-------------------------------------------------------------------------------
# SuiteSparse policies (a subset)
#-------------------------------------------------------------------------------

message ( STATUS "Source:        " ${PROJECT_SOURCE_DIR} )
message ( STATUS "Build:         " ${CMAKE_BINARY_DIR} )

cmake_policy ( SET CMP0042 NEW )    # enable MACOSX_RPATH by default
cmake_policy ( SET CMP0048 NEW )    # VERSION variable policy
cmake_policy ( SET CMP0054 NEW )    # if ( expression ) handling policy

if ( WIN32 )
    set ( CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS true )
    add_compile_definitions ( _CRT_SECURE_NO_WARNINGS )
endif ( )

set ( CMAKE_MACOSX_RPATH TRUE )
include ( GNUInstallDirs )

if ( NOT CMAKE_BUILD_TYPE )
    set ( CMAKE_BUILD_TYPE Release )
endif ( )

message ( STATUS "Build type:    " ${CMAKE_BUILD_TYPE} )

# BUILD_SHARED_LIBS and BUILD_STATIC_LIBS options
option ( BUILD_SHARED_LIBS "OFF: do not build shared libraries.  ON (default): build shared libraries" ON )
option ( BUILD_STATIC_LIBS "OFF: do not build static libraries.  ON (default): build static libraries" ON )

#-------------------------------------------------------------------------------
# Configure cs.h with version number
#-------------------------------------------------------------------------------

configure_file ( "Config/cs.h.in"
    "${PROJECT_SOURCE_DIR}/Include/cs.h"
    NEWLINE_STYLE LF )

#-------------------------------------------------------------------------------
# include directories
#-------------------------------------------------------------------------------

set ( CMAKE_INCLUDE_CURRENT_DIR ON )
include_directories ( Source Include )

#-------------------------------------------------------------------------------
# dynamic csparse library properties
#-------------------------------------------------------------------------------

file ( GLOB CSPARSE_SOURCES "Source/*.c" )

if ( BUILD_SHARED_LIBS )
    add_library ( csparse SHARED ${CSPARSE_SOURCES} )

    set_target_properties ( csparse PROPERTIES
        VERSION ${CSPARSE_VERSION_MAJOR}.${CSPARSE_VERSION_MINOR}.${CSPARSE_VERSION_SUB}
        C_STANDARD 11
        C_STANDARD_REQUIRED ON
        SOVERSION ${CSPARSE_VERSION_MAJOR}
        PUBLIC_HEADER "Include/cs.h"
        WINDOWS_EXPORT_ALL_SYMBOLS ON )
endif ( )

#-------------------------------------------------------------------------------
# static csparse library properties
#-------------------------------------------------------------------------------

if ( BUILD_STATIC_LIBS )
    add_library ( csparse_static STATIC ${CSPARSE_SOURCES} )

    set_target_properties ( csparse_static PROPERTIES
        OUTPUT_NAME csparse
        C_STANDARD 11
        C_STANDARD_REQUIRED ON
        PUBLIC_HEADER "Include/cs.h" )

    if ( MSVC )
        set_target_properties ( csparse_static PROPERTIES
            OUTPUT_NAME csparse_static )
    endif ( )
endif ( )

#-------------------------------------------------------------------------------
# add the library dependencies
#-------------------------------------------------------------------------------

# libm:
if ( NOT WIN32 )
    if ( BUILD_SHARED_LIBS )
        target_link_libraries ( csparse PRIVATE m )
    endif ( )
    if ( BUILD_STATIC_LIBS )
        set ( CSPARSE_STATIC_LIBS "${CSPARSE_STATIC_LIBS} -lm" )
        target_link_libraries ( csparse_static PUBLIC m )
    endif ( )
endif ( )


#-------------------------------------------------------------------------------
# create pkg-config file
#-------------------------------------------------------------------------------

if ( NOT MSVC )
    set ( prefix "${CMAKE_INSTALL_PREFIX}" )
    set ( exec_prefix "\${prefix}" )
    cmake_path ( IS_ABSOLUTE CMAKE_INSTALL_LIBDIR SUITESPARSE_LIBDIR_IS_ABSOLUTE )
    if (SUITESPARSE_LIBDIR_IS_ABSOLUTE)
        set ( libdir "${CMAKE_INSTALL_LIBDIR}")
    else ( )
        set ( libdir "\${exec_prefix}/${CMAKE_INSTALL_LIBDIR}")
    endif ( )
    cmake_path ( IS_ABSOLUTE CMAKE_INSTALL_INCLUDEDIR SUITESPARSE_INCLUDEDIR_IS_ABSOLUTE )
    if (SUITESPARSE_INCLUDEDIR_IS_ABSOLUTE)
        set ( includedir "${CMAKE_INSTALL_INCLUDEDIR}")
    else ( )
        set ( includedir "\${prefix}/${CMAKE_INSTALL_INCLUDEDIR}")
    endif ( )
    configure_file (
        Config/CSparse.pc.in
        CSparse.pc
        @ONLY
        NEWLINE_STYLE LF )
    # install ( FILES
    #     ${CMAKE_CURRENT_BINARY_DIR}/CSparse.pc
    #     DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig )
endif ( )

#-------------------------------------------------------------------------------
# Demo library and programs
#-------------------------------------------------------------------------------

option ( DEMO "ON: Build the demo programs.  OFF (default): do not build the demo programs." off )
if ( DEMO )

    #---------------------------------------------------------------------------
    # demo library
    #---------------------------------------------------------------------------

    message ( STATUS "Also compiling the demos in CSparse/Demo" )

    #---------------------------------------------------------------------------
    # Demo programs
    #---------------------------------------------------------------------------

    add_executable ( cs_demo1 "Demo/cs_demo1.c" "Demo/cs_demo.c" )
    add_executable ( cs_demo2 "Demo/cs_demo2.c" "Demo/cs_demo.c" )
    add_executable ( cs_demo3 "Demo/cs_demo3.c" "Demo/cs_demo.c" )

    # Libraries required for Demo programs
    if ( BUILD_SHARED_LIBS )
        target_link_libraries ( cs_demo1 PUBLIC csparse )
        target_link_libraries ( cs_demo2 PUBLIC csparse )
        target_link_libraries ( cs_demo3 PUBLIC csparse )
    else ( )
        target_link_libraries ( cs_demo1 PUBLIC csparse_static )
        target_link_libraries ( cs_demo2 PUBLIC csparse_static )
        target_link_libraries ( cs_demo3 PUBLIC csparse_static )
    endif ( )

else ( )

    message ( STATUS "Skipping the demos in CSparse/Demo" )

endif ( )

