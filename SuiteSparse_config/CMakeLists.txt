#-------------------------------------------------------------------------------
# SuiteSparse/SuiteSparse_config/CMakeLists.txt:  cmake for SuiteSparse_config
#-------------------------------------------------------------------------------

# SuiteSparse_config, Copyright (c) 2012-2022, Timothy A. Davis.
# All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

#-------------------------------------------------------------------------------
# get the version
#-------------------------------------------------------------------------------

cmake_minimum_required ( VERSION 3.19 )

# version of both SuiteSparse and SuiteSparse_config
set ( SUITESPARSE_DATE "Oct 1, 2022" )  # FIXME
set ( SUITESPARSE_VERSION_MAJOR 6 )
set ( SUITESPARSE_VERSION_MINOR 0 )
set ( SUITESPARSE_VERSION_SUB   0 )

message ( STATUS "Building SuiteSparse_config version: v"
    ${SUITESPARSE_VERSION_MAJOR}.
    ${SUITESPARSE_VERSION_MINOR}.
    ${SUITESPARSE_VERSION_SUB} " (" ${SUITESPARSE_DATE} ")" )

#-------------------------------------------------------------------------------
# SuiteSparse policies
#-------------------------------------------------------------------------------

set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    ${CMAKE_SOURCE_DIR}/cmake_modules
    ${CMAKE_SOURCE_DIR}/../cmake_modules
    ${CMAKE_SOURCE_DIR}/../SuiteSparse/cmake_modules
    )

include ( SuiteSparsePolicy )

#-------------------------------------------------------------------------------
# define the project
#-------------------------------------------------------------------------------

project ( suitesparseconfig
    VERSION "${SUITESPARSE_VERSION_MAJOR}.${SUITESPARSE_VERSION_MINOR}.${SUITESPARSE_VERSION_SUB}"
    LANGUAGES C ) # <-- add CXX here for packages that use C++

#-------------------------------------------------------------------------------
# report status and set compile flags
#-------------------------------------------------------------------------------

# FIXME: move this to cmake_modules/SuiteSparseReport.cmake
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
# dynamic suitesparseconfig library properties
#-------------------------------------------------------------------------------

file ( GLOB SUITESPARSECONFIG_SOURCES "*.c" )

add_library ( suitesparseconfig SHARED ${SUITESPARSECONFIG_SOURCES} )

SET_TARGET_PROPERTIES ( suitesparseconfig PROPERTIES
    VERSION ${SUITESPARSE_VERSION_MAJOR}.${SUITESPARSE_VERSION_MINOR}.${SUITESPARSE_VERSION_SUB}
    C_STANDARD_REQUIRED 11
    SOVERSION ${SUITESPARSE_VERSION_MAJOR}
    PUBLIC_HEADER "SuiteSparse_config.h" )

#-------------------------------------------------------------------------------
# static suitesparseconfig library properties
#-------------------------------------------------------------------------------

add_library ( suitesparseconfig_static STATIC ${SUITESPARSECONFIG_SOURCES} )

SET_TARGET_PROPERTIES ( suitesparseconfig_static PROPERTIES
    VERSION ${SUITESPARSE_VERSION_MAJOR}.${SUITESPARSE_VERSION_MINOR}.${SUITESPARSE_VERSION_SUB}
    C_STANDARD_REQUIRED 11
    OUTPUT_NAME suitesparseconfig
    SOVERSION ${SUITESPARSE_VERSION_MAJOR} )

#-------------------------------------------------------------------------------
# configure files
#-------------------------------------------------------------------------------

configure_file ( "Config/SuiteSparse_config.h.in" "${PROJECT_SOURCE_DIR}/SuiteSparse_config.h" )

#-------------------------------------------------------------------------------
# add the library dependencies
#-------------------------------------------------------------------------------

# libm:
if ( NOT "${CMAKE_C_COMPILER_ID}" STREQUAL "MSVC" )
    target_link_libraries ( suitesparseconfig PUBLIC m )
    target_link_libraries ( suitesparseconfig_static PUBLIC m )
endif ( )

#-------------------------------------------------------------------------------
# suitesparseconfig installation location
#-------------------------------------------------------------------------------

# use "cmake -DLOCAL=1" to install only in SuiteSparse/lib and
# SuiteSparse/include.  

if ( NOT LOCAL )
    # install in /usr/local/lib and /usr/local/include.
    # requires "sudo make install"
    message ( STATUS "Installation will be system-wide (requires 'sudo make install')" )
    install ( TARGETS suitesparseconfig
        LIBRARY       DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} )
    install ( TARGETS suitesparseconfig_static
        ARCHIVE       DESTINATION ${CMAKE_INSTALL_LIBDIR} )
endif ( )

if ( INSIDE_SUITESPARSE )
    # also install in SuiteSparse/lib and SuiteSparse/include;
    # does not require "sudo make install", just "make install"
    message ( STATUS "Installation in ../lib and ../include," )
    message ( STATUS "  with 'make local ; make install'. No 'sudo' required." )
    install ( TARGETS suitesparseconfig
        LIBRARY       DESTINATION ${LOCAL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${LOCAL_INCLUDEDIR} )
    install ( TARGETS suitesparseconfig_static
        ARCHIVE       DESTINATION ${LOCAL_LIBDIR} )
endif ( )

