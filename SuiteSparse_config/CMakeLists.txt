#-------------------------------------------------------------------------------
# SuiteSparse/SuiteSparse_config/CMakeLists.txt:  cmake for SuiteSparse_config
#-------------------------------------------------------------------------------

# Copyright (c) 2022, Timothy A. Davis.  All Rights Reserved.
# SPEX-License-Identifier: BSD-3-clause

#-------------------------------------------------------------------------------
# get the version
#-------------------------------------------------------------------------------

# cmake 3.17 is required
cmake_minimum_required ( VERSION 3.17 )

set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    ${CMAKE_SOURCE_DIR}/cmake_modules
    ${CMAKE_SOURCE_DIR}/../cmake_modules
    ${CMAKE_SOURCE_DIR}/../SuiteSparse/cmake_modules
    )

# version of both SuiteSparse and SuiteSparse_config
set ( SUITESPARSE_DATE "Sept FIXME, 2022" )
set ( SUITESPARSE_VERSION_MAJOR 6 )
set ( SUITESPARSE_VERSION_MINOR 0 )
set ( SUITESPARSE_VERSION_SUB   0 )

message ( STATUS "Building SuiteSparse_config version: v"
    ${SUITESPARSE_VERSION_MAJOR}.
    ${SUITESPARSE_VERSION_MINOR}.
    ${SUITESPARSE_VERSION_SUB} " (" ${SUITESPARSE_DATE} ")" )

project ( suitesparseconfig
    VERSION "${SUITESPARSE_VERSION_MAJOR}.${SUITESPARSE_VERSION_MINOR}.${SUITESPARSE_VERSION_SUB}"
    LANGUAGES C ) # <-- add CXX here for packages that use C++

message ( STATUS "SuiteSparse_config project: " ${PROJECT_SOURCE_DIR} )
message ( STATUS "SuiteSparse_config build:   " ${CMAKE_BINARY_DIR} )

#-------------------------------------------------------------------------------
# find library dependencies
#-------------------------------------------------------------------------------

find_package ( OpenMP )

#-------------------------------------------------------------------------------
# determine build type
#-------------------------------------------------------------------------------

if ( NOT CMAKE_BUILD_TYPE )
    set ( CMAKE_BUILD_TYPE Release )
endif ( )

#-------------------------------------------------------------------------------
# Configure SuiteSparse_config.h with version number
#-------------------------------------------------------------------------------

configure_file (
    "Config/SuiteSparse_config.h.in"
    "${PROJECT_SOURCE_DIR}/SuiteSparse_config.h"
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
message ( STATUS "CMAKE have OpenMP:       " ${OPENMP_FOUND} )

# message ( STATUS "CMAKE C++ compiler:      " ${CMAKE_CXX_COMPILER_ID} )

#-------------------------------------------------------------------------------
# include directories
#-------------------------------------------------------------------------------

set ( CMAKE_INCLUDE_CURRENT_DIR ON )

#-------------------------------------------------------------------------------
# dynamic suitesparseconfig library properties
#-------------------------------------------------------------------------------

file ( GLOB SUITESPARSECONFIG_SOURCES "*.c" )

add_library ( suitesparseconfig SHARED ${SUITESPARSECONFIG_SOURCES} )

SET_TARGET_PROPERTIES ( suitesparseconfig PROPERTIES
    VERSION ${SUITESPARSE_VERSION_MAJOR}.${SUITESPARSE_VERSION_MINOR}.${SUITESPARSE_VERSION_SUB}
    SOVERSION ${SUITESPARSE_VERSION_MAJOR}
    PUBLIC_HEADER "SuiteSparse_config.h" )

#-------------------------------------------------------------------------------
# static suitesparseconfig library properties
#-------------------------------------------------------------------------------

add_library ( suitesparseconfig_static STATIC ${SUITESPARSECONFIG_SOURCES} )

SET_TARGET_PROPERTIES ( suitesparseconfig_static PROPERTIES
    VERSION ${SUITESPARSE_VERSION_MAJOR}.${SUITESPARSE_VERSION_MINOR}.${SUITESPARSE_VERSION_SUB}
    OUTPUT_NAME suitesparseconfig
    SOVERSION ${SUITESPARSE_VERSION_MAJOR} )

#-------------------------------------------------------------------------------
# add the library dependencies
#-------------------------------------------------------------------------------

# OpenMP:
if ( OPENMP_FOUND )
    message ( STATUS "OpenMP C libraries:      " ${OpenMP_C_LIBRARIES} )
    message ( STATUS "OpenMP C include:        " ${OpenMP_C_INCLUDE_DIRS} )
    message ( STATUS "OpenMP C flags:          " ${OpenMP_C_FLAGS} )
    target_link_libraries ( suitesparseconfig PUBLIC ${OpenMP_C_LIBRARIES} )
    target_link_libraries ( suitesparseconfig_static PUBLIC ${OpenMP_C_LIBRARIES} )
    set ( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} " )
    include_directories ( ${OpenMP_C_INCLUDE_DIRS} )
#   for packages that use OpenMP in C++:
#   message ( STATUS "OpenMP C++ libraries:    " ${OpenMP_CXX_LIBRARIES} )
#   message ( STATUS "OpenMP C++ include:      " ${OpenMP_CXX_INCLUDE_DIRS} )
#   message ( STATUS "OpenMP C++ flags:        " ${OpenMP_CXX_FLAGS} )
#   target_link_libraries ( suitesparseconfig PUBLIC ${OpenMP_CXX_LIBRARIES} )
#   target_link_libraries ( suitesparseconfig_static PUBLIC ${OpenMP_CXX_LIBRARIES} )
#   set ( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} " )
#   include_directories ( ${OpenMP_CXX_INCLUDE_DIRS} )
endif ( )

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
    message ( STATUS "Installation will be system-wide (requires 'sudo make install')." )
    message ( STATUS "    If you do not have 'sudo' privilege, do 'make local' instead." )
    include ( GNUInstallDirs )

    install ( TARGETS suitesparseconfig
        LIBRARY       DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} )

    install ( TARGETS suitesparseconfig_static
        ARCHIVE       DESTINATION ${CMAKE_INSTALL_LIBDIR} )

else ( )

    message ( STATUS "Installation in ../lib and ../include only," )
    message ( STATUS "    with 'make install'. No 'sudo' required." )

endif ( )

# also install in SuiteSparse/lib and SuiteSparse/include;
# does not require "sudo make install", just "make install"
install ( TARGETS suitesparseconfig
    LIBRARY       DESTINATION ${CMAKE_SOURCE_DIR}/../lib
    PUBLIC_HEADER DESTINATION ${CMAKE_SOURCE_DIR}/../include )

install ( TARGETS suitesparseconfig_static
    ARCHIVE       DESTINATION ${CMAKE_SOURCE_DIR}/../lib )

