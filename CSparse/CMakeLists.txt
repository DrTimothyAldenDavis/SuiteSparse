#-------------------------------------------------------------------------------
# CSparse/CMakeLists.txt:  cmake for CSparse
#-------------------------------------------------------------------------------

# Copyright (c) 2022, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: LGPL-2.1+

#-------------------------------------------------------------------------------
# get the version
#-------------------------------------------------------------------------------

# cmake 3.17 is required
cmake_minimum_required ( VERSION 3.17 )

cmake_policy ( SET CMP0042 NEW )
cmake_policy ( SET CMP0048 NEW )
set ( CMAKE_MACOSX_RPATH TRUE )

set ( CSPARSE_DATE "Sept FIXME, 2022" )
set ( CSPARSE_VERSION_MAJOR 4 )
set ( CSPARSE_VERSION_MINOR 0 )
set ( CSPARSE_VERSION_SUB   0 )

message ( STATUS "Building CSparse version: v"
    ${CSPARSE_VERSION_MAJOR}.
    ${CSPARSE_VERSION_MINOR}.
    ${CSPARSE_VERSION_SUB} " (" ${CSPARSE_DATE} ")" )

project ( csparse
    VERSION "${CSPARSE_VERSION_MAJOR}.${CSPARSE_VERSION_MINOR}.${CSPARSE_VERSION_SUB}"
    LANGUAGES C ) # <-- add CXX here for packages that use C++

message ( STATUS "CSparse project: " ${PROJECT_SOURCE_DIR} )
message ( STATUS "CSparse build:   " ${CMAKE_BINARY_DIR} )

#-------------------------------------------------------------------------------
# determine build type
#-------------------------------------------------------------------------------

if ( NOT CMAKE_BUILD_TYPE )
    set ( CMAKE_BUILD_TYPE Release )
endif ( )

#-------------------------------------------------------------------------------
# Configure cs.h with version number
#-------------------------------------------------------------------------------

configure_file (
    "Config/cs.h.in"
    "${PROJECT_SOURCE_DIR}/Include/cs.h"
)

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
include_directories ( Source Include ${SUITESPARSE_CONFIG_INCLUDE_DIR} )

#-------------------------------------------------------------------------------
# dynamic csparse library properties
#-------------------------------------------------------------------------------

file ( GLOB CSPARSE_SOURCES "Source/*.c" )

add_library ( csparse SHARED ${CSPARSE_SOURCES} )

SET_TARGET_PROPERTIES ( csparse PROPERTIES
    VERSION ${CSPARSE_VERSION_MAJOR}.${CSPARSE_VERSION_MINOR}.${CSPARSE_VERSION_SUB}
    SOVERSION ${CSPARSE_VERSION_MAJOR}
    PUBLIC_HEADER "Include/cs.h" )

#-------------------------------------------------------------------------------
# static csparse library properties
#-------------------------------------------------------------------------------

add_library ( csparse_static STATIC ${CSPARSE_SOURCES} )

SET_TARGET_PROPERTIES ( csparse_static PROPERTIES
    VERSION ${CSPARSE_VERSION_MAJOR}.${CSPARSE_VERSION_MINOR}.${CSPARSE_VERSION_SUB}
    OUTPUT_NAME csparse
    SOVERSION ${CSPARSE_VERSION_MAJOR} )

#-------------------------------------------------------------------------------
# add the library dependencies
#-------------------------------------------------------------------------------

target_link_libraries ( csparse PUBLIC ${SUITESPARSE_CONFIG_LIBRARY} )
target_link_libraries ( csparse_static PUBLIC ${SUITESPARSE_CONFIG_LIBRARY} )

# libm:
if ( NOT "${CMAKE_C_COMPILER_ID}" STREQUAL "MSVC" )
    target_link_libraries ( csparse PUBLIC m )
    target_link_libraries ( csparse_static PUBLIC m )
endif ( )

#-------------------------------------------------------------------------------
# CSparse installation location
#-------------------------------------------------------------------------------

# use "cmake -DLOCAL=1" to install only in SuiteSparse/lib and
# SuiteSparse/include.  

if ( NOT LOCAL )

    # install in /usr/local/lib and /usr/local/include.
    # requires "sudo make install"
    message ( STATUS "Installation will be system-wide (requires 'sudo make install')" )
    include ( GNUInstallDirs )

    install ( TARGETS csparse
        LIBRARY       DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} )

    install ( TARGETS csparse_static
        ARCHIVE       DESTINATION ${CMAKE_INSTALL_LIBDIR} )

else ( )

    message ( STATUS "Installation in ../lib and ../include only," )
    message ( STATUS "  with 'make install'. No 'sudo' required." )

endif ( )

if ( ( EXISTS ${CMAKE_SOURCE_DIR}/../lib ) AND
     ( EXISTS ${CMAKE_SOURCE_DIR}/../include ))

    # also install in SuiteSparse/lib and SuiteSparse/include;
    # does not require "sudo make install", just "make install"
    install ( TARGETS csparse
        LIBRARY       DESTINATION ${CMAKE_SOURCE_DIR}/../lib
        PUBLIC_HEADER DESTINATION ${CMAKE_SOURCE_DIR}/../include )

    install ( TARGETS csparse_static
        ARCHIVE       DESTINATION ${CMAKE_SOURCE_DIR}/../lib )

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
