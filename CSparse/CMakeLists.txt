#-------------------------------------------------------------------------------
# CSparse/CMakeLists.txt:  cmake for CSparse
#-------------------------------------------------------------------------------

# Copyright (c) 2022, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: LGPL-2.1+

# CSparse/CMakeLists.txt is a standalone script, which can be used outside of
# SuiteSparse.  There is no "make install" target for CSparse.  The library
# (libcsparse.so) and include file (cs.h) are NOT installed in ../lib and
# ../include.  CSparse is meant for classwork and stand-alone development,
# while SuiteSpare/CXSparse should be used in production.

#-------------------------------------------------------------------------------
# get the version
#-------------------------------------------------------------------------------

cmake_minimum_required ( VERSION 3.13 )

set ( CSPARSE_DATE "Oct 1, 2022" )  # FIXME
set ( CSPARSE_VERSION_MAJOR 4 )
set ( CSPARSE_VERSION_MINOR 0 )
set ( CSPARSE_VERSION_SUB   0 )

message ( STATUS "Building CSparse version: v"
    ${CSPARSE_VERSION_MAJOR}.
    ${CSPARSE_VERSION_MINOR}.
    ${CSPARSE_VERSION_SUB} " (" ${CSPARSE_DATE} ")" )

#-------------------------------------------------------------------------------
# SuiteSparse policies (a subset)
#-------------------------------------------------------------------------------

message ( STATUS "Source:        " ${CMAKE_SOURCE_DIR} )
message ( STATUS "Build:         " ${CMAKE_BINARY_DIR} )

cmake_policy ( SET CMP0042 NEW )    # enable MACOSX_RPATH by default
cmake_policy ( SET CMP0048 NEW )    # VERSION variable policy
cmake_policy ( SET CMP0054 NEW )    # if ( expression ) handling policy

set ( CMAKE_MACOSX_RPATH TRUE )
enable_language ( C )
include ( GNUInstallDirs )

if ( NOT CMAKE_BUILD_TYPE )
    set ( CMAKE_BUILD_TYPE Release )
endif ( )

message ( STATUS "Build type:    " ${CMAKE_BUILD_TYPE} )

#-------------------------------------------------------------------------------
# define the project
#-------------------------------------------------------------------------------

project ( csparse
    VERSION "${CSPARSE_VERSION_MAJOR}.${CSPARSE_VERSION_MINOR}.${CSPARSE_VERSION_SUB}"
    LANGUAGES C ) # <-- add CXX here for packages that use C++

#-------------------------------------------------------------------------------
# Configure cs.h with version number
#-------------------------------------------------------------------------------

configure_file ( "Config/cs.h.in" "${PROJECT_SOURCE_DIR}/Include/cs.h" )

#-------------------------------------------------------------------------------
# report status and set compile flags
#-------------------------------------------------------------------------------

message ( STATUS "CMAKE build type:        " ${CMAKE_BUILD_TYPE} )

if ( ${CMAKE_BUILD_TYPE} STREQUAL "Debug" )
    message ( STATUS "CMAKE C Flags debug:     " ${CMAKE_C_FLAGS_DEBUG} )
#   message ( STATUS "CMAKE C++ Flags debug:   " ${CMAKE_CXX_FLAGS_DEBUG} )
else ( )
    message ( STATUS "CMAKE C Flags release:   " ${CMAKE_C_FLAGS_RELEASE} )
#   message ( STATUS "CMAKE C++ Flags release: " ${CMAKE_CXX_FLAGS_RELEASE} )
endif ( )

message ( STATUS "CMAKE C compiler:        " ${CMAKE_C_COMPILER_ID} )

# message ( STATUS "CMAKE C++ compiler:      " ${CMAKE_CXX_COMPILER_ID} )

#-------------------------------------------------------------------------------
# include directories
#-------------------------------------------------------------------------------

set ( CMAKE_INCLUDE_CURRENT_DIR ON )
include_directories ( Source Include )

#-------------------------------------------------------------------------------
# dynamic csparse library properties
#-------------------------------------------------------------------------------

file ( GLOB CSPARSE_SOURCES "Source/*.c" )

add_library ( csparse SHARED ${CSPARSE_SOURCES} )

SET_TARGET_PROPERTIES ( csparse PROPERTIES
    VERSION ${CSPARSE_VERSION_MAJOR}.${CSPARSE_VERSION_MINOR}.${CSPARSE_VERSION_SUB}
    C_STANDARD_REQUIRED 11
    SOVERSION ${CSPARSE_VERSION_MAJOR}
    PUBLIC_HEADER "Include/cs.h" )

#-------------------------------------------------------------------------------
# static csparse library properties
#-------------------------------------------------------------------------------

add_library ( csparse_static STATIC ${CSPARSE_SOURCES} )

SET_TARGET_PROPERTIES ( csparse_static PROPERTIES
    VERSION ${CSPARSE_VERSION_MAJOR}.${CSPARSE_VERSION_MINOR}.${CSPARSE_VERSION_SUB}
    OUTPUT_NAME csparse
    C_STANDARD_REQUIRED 11
    SOVERSION ${CSPARSE_VERSION_MAJOR} )

#-------------------------------------------------------------------------------
# add the library dependencies
#-------------------------------------------------------------------------------

# libm:
if ( NOT "${CMAKE_C_COMPILER_ID}" STREQUAL "MSVC" )
    target_link_libraries ( csparse PUBLIC m )
    target_link_libraries ( csparse_static PUBLIC m )
endif ( )

#-------------------------------------------------------------------------------
# Demo library and programs
#-------------------------------------------------------------------------------

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
    target_link_libraries ( cs_demo1 PUBLIC csparse )
    target_link_libraries ( cs_demo2 PUBLIC csparse )
    target_link_libraries ( cs_demo3 PUBLIC csparse )

else ( )

    message ( STATUS "Skipping the demos in CSparse/Demo" )

endif ( )

