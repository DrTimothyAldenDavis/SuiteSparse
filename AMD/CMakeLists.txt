#-------------------------------------------------------------------------------
# SuiteSparse/AMD/CMakeLists.txt:  cmake for AMD
#-------------------------------------------------------------------------------

# Copyright (c) 2022, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

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

set ( AMD_DATE "Sept FIXME, 2022" )
set ( AMD_VERSION_MAJOR 3 )
set ( AMD_VERSION_MINOR 0 )
set ( AMD_VERSION_SUB   0 )

message ( STATUS "Building AMD version: v"
    ${AMD_VERSION_MAJOR}.
    ${AMD_VERSION_MINOR}.
    ${AMD_VERSION_SUB} " (" ${AMD_DATE} ")" )

project ( amd
    VERSION "${AMD_VERSION_MAJOR}.${AMD_VERSION_MINOR}.${AMD_VERSION_SUB}"
    LANGUAGES C ) # <-- add CXX here for packages that use C++

message ( STATUS "AMD project: " ${PROJECT_SOURCE_DIR} )
message ( STATUS "AMD build:   " ${CMAKE_BINARY_DIR} )

#-------------------------------------------------------------------------------
# find library dependencies
#-------------------------------------------------------------------------------

find_package ( OpenMP )
find_package ( SuiteSparse_config REQUIRED )

#-------------------------------------------------------------------------------
# determine build type
#-------------------------------------------------------------------------------

if ( NOT CMAKE_BUILD_TYPE )
    set ( CMAKE_BUILD_TYPE Release )
endif ( )

#-------------------------------------------------------------------------------
# Configure amd.h with version number
#-------------------------------------------------------------------------------

configure_file (
    "Config/amd.h.in"
    "${PROJECT_SOURCE_DIR}/Include/amd.h"
)

configure_file (
    "Config/amd_version.tex.in"
    "${PROJECT_SOURCE_DIR}/Doc/amd_version.tex"
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
include_directories ( Source Include ${SUITESPARSE_CONFIG_INCLUDE_DIR} )

#-------------------------------------------------------------------------------
# dynamic amd library properties
#-------------------------------------------------------------------------------

file ( GLOB AMD_SOURCES "Source/*.c" )

add_library ( amd SHARED ${AMD_SOURCES} )

SET_TARGET_PROPERTIES ( amd PROPERTIES
    VERSION ${AMD_VERSION_MAJOR}.${AMD_VERSION_MINOR}.${AMD_VERSION_SUB}
    SOVERSION ${AMD_VERSION_MAJOR}
    PUBLIC_HEADER "Include/amd.h" )

#-------------------------------------------------------------------------------
# static amd library properties
#-------------------------------------------------------------------------------

add_library ( amd_static STATIC ${AMD_SOURCES} )

SET_TARGET_PROPERTIES ( amd_static PROPERTIES
    VERSION ${AMD_VERSION_MAJOR}.${AMD_VERSION_MINOR}.${AMD_VERSION_SUB}
    OUTPUT_NAME amd
    SOVERSION ${AMD_VERSION_MAJOR} )

#-------------------------------------------------------------------------------
# add the library dependencies
#-------------------------------------------------------------------------------

target_link_libraries ( amd PUBLIC ${SUITESPARSE_CONFIG_LIBRARY} )
target_link_libraries ( amd_static PUBLIC ${SUITESPARSE_CONFIG_LIBRARY} )

# OpenMP:
if ( OPENMP_FOUND )
    message ( STATUS "OpenMP C libraries:      " ${OpenMP_C_LIBRARIES} )
    message ( STATUS "OpenMP C include:        " ${OpenMP_C_INCLUDE_DIRS} )
    message ( STATUS "OpenMP C flags:          " ${OpenMP_C_FLAGS} )
    target_link_libraries ( amd PUBLIC ${OpenMP_C_LIBRARIES} )
    target_link_libraries ( amd_static PUBLIC ${OpenMP_C_LIBRARIES} )
    set ( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} " )
    include_directories ( ${OpenMP_C_INCLUDE_DIRS} )
endif ( )

# libm:
if ( NOT "${CMAKE_C_COMPILER_ID}" STREQUAL "MSVC" )
    target_link_libraries ( amd PUBLIC m )
    target_link_libraries ( amd_static PUBLIC m )
endif ( )

#-------------------------------------------------------------------------------
# amd installation location
#-------------------------------------------------------------------------------

# install in /usr/local/lib and /usr/local/include
include ( GNUInstallDirs )

install ( TARGETS amd
    LIBRARY       DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} )

install ( TARGETS amd_static
    ARCHIVE       DESTINATION ${CMAKE_INSTALL_LIBDIR} )

# also install in ../lib and ../include
install ( TARGETS amd
    LIBRARY       DESTINATION ${CMAKE_SOURCE_DIR}/../lib
    PUBLIC_HEADER DESTINATION ${CMAKE_SOURCE_DIR}/../include )

install ( TARGETS amd_static
    ARCHIVE       DESTINATION ${CMAKE_SOURCE_DIR}/../lib)

#-------------------------------------------------------------------------------
# Demo library and programs
#-------------------------------------------------------------------------------

if ( DEMO )

    #---------------------------------------------------------------------------
    # demo library
    #---------------------------------------------------------------------------

    message ( STATUS "Also compiling the demos in AMD/Demo" )

    #---------------------------------------------------------------------------
    # Demo programs
    #---------------------------------------------------------------------------

    add_executable ( amd_demo   "Demo/amd_demo.c" )
    add_executable ( amd_l_demo "Demo/amd_l_demo.c" )
    add_executable ( amd_demo2  "Demo/amd_demo2.c" )
    add_executable ( amd_simple "Demo/amd_simple.c" )

    # Libraries required for Demo programs
    target_link_libraries ( amd_demo   PUBLIC amd )
    target_link_libraries ( amd_l_demo PUBLIC amd )
    target_link_libraries ( amd_demo2  PUBLIC amd )
    target_link_libraries ( amd_simple PUBLIC amd )

else ( )

    message ( STATUS "Skipping the demos in GraphBLAS/Demo" )

endif ( )
