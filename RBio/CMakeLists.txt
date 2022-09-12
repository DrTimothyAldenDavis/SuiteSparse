#-------------------------------------------------------------------------------
# SuiteSparse/RBio/CMakeLists.txt:  cmake for RBio
#-------------------------------------------------------------------------------

# Copyright (c) 2009-2022, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: GPL-2.0+

#-------------------------------------------------------------------------------
# get the version
#-------------------------------------------------------------------------------

cmake_minimum_required ( VERSION 3.19 )

set ( RBIO_DATE "Sept FIXME, 2022" )
set ( RBIO_VERSION_MAJOR 3 )
set ( RBIO_VERSION_MINOR 0 )
set ( RBIO_VERSION_SUB   0 )

message ( STATUS "Building RBIO version: v"
    ${RBIO_VERSION_MAJOR}.
    ${RBIO_VERSION_MINOR}.
    ${RBIO_VERSION_SUB} " (" ${RBIO_DATE} ")" )

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

project ( rbio
    VERSION "${RBIO_VERSION_MAJOR}.${RBIO_VERSION_MINOR}.${RBIO_VERSION_SUB}"
    LANGUAGES C ) # <-- add CXX here for packages that use C++

#-------------------------------------------------------------------------------
# find library dependencies
#-------------------------------------------------------------------------------

find_package ( OpenMP )
find_package ( SuiteSparse_config REQUIRED )

#-------------------------------------------------------------------------------
# configure files
#-------------------------------------------------------------------------------

configure_file ( "Config/RBio.h.in" "${PROJECT_SOURCE_DIR}/Include/RBio.h")

#-------------------------------------------------------------------------------
# report status and set compile flags
#-------------------------------------------------------------------------------

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

include_directories ( Source Include ${SUITESPARSE_CONFIG_INCLUDE_DIR} )

#-------------------------------------------------------------------------------
# dynamic rbio library properties
#-------------------------------------------------------------------------------

file ( GLOB RBIO_SOURCES "Source/*.c" )

add_library ( rbio SHARED ${RBIO_SOURCES} )

SET_TARGET_PROPERTIES ( rbio PROPERTIES
    VERSION ${RBIO_VERSION_MAJOR}.${RBIO_VERSION_MINOR}.${RBIO_VERSION_SUB}
    C_STANDARD_REQUIRED 11
    SOVERSION ${RBIO_VERSION_MAJOR}
    PUBLIC_HEADER "Include/RBio.h" )

#-------------------------------------------------------------------------------
# static rbio library properties
#-------------------------------------------------------------------------------

add_library ( rbio_static STATIC ${RBIO_SOURCES} )

SET_TARGET_PROPERTIES ( rbio_static PROPERTIES
    VERSION ${RBIO_VERSION_MAJOR}.${RBIO_VERSION_MINOR}.${RBIO_VERSION_SUB}
    C_STANDARD_REQUIRED 11
    OUTPUT_NAME rbio
    SOVERSION ${RBIO_VERSION_MAJOR} )

#-------------------------------------------------------------------------------
# add the library dependencies
#-------------------------------------------------------------------------------

# suitesparseconfig:
target_link_libraries ( rbio PUBLIC ${SUITESPARSE_CONFIG_LIBRARIES} )
target_link_libraries ( rbio_static PUBLIC ${SUITESPARSE_CONFIG_LIBRARIES} )

# OpenMP:
if ( OPENMP_FOUND )
    message ( STATUS "OpenMP C libraries:      " ${OpenMP_C_LIBRARIES} )
    message ( STATUS "OpenMP C include:        " ${OpenMP_C_INCLUDE_DIRS} )
    message ( STATUS "OpenMP C flags:          " ${OpenMP_C_FLAGS} )
    target_link_libraries ( rbio PUBLIC ${OpenMP_C_LIBRARIES} )
    target_link_libraries ( rbio_static PUBLIC ${OpenMP_C_LIBRARIES} )
    set ( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} " )
    include_directories ( ${OpenMP_C_INCLUDE_DIRS} )
endif ( )

# libm:
if ( NOT "${CMAKE_C_COMPILER_ID}" STREQUAL "MSVC" )
    target_link_libraries ( rbio PUBLIC m )
    target_link_libraries ( rbio_static PUBLIC m )
endif ( )

#-------------------------------------------------------------------------------
# RBIO installation location
#-------------------------------------------------------------------------------

# use "cmake -DLOCAL=1" to install only in SuiteSparse/lib and
# SuiteSparse/include.  

if ( NOT LOCAL )
    # install in /usr/local/lib and /usr/local/include.
    # requires "sudo make install"
    message ( STATUS "Installation will be system-wide (requires 'sudo make install')" )
    install ( TARGETS rbio
        LIBRARY       DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} )
    install ( TARGETS rbio_static
        ARCHIVE       DESTINATION ${CMAKE_INSTALL_LIBDIR} )
endif ( )

if ( INSIDE_SUITESPARSE )
    # also install in SuiteSparse/lib and SuiteSparse/include;
    # does not require "sudo make install", just "make install"
    message ( STATUS "Installation in ../lib and ../include only," )
    message ( STATUS "  with 'make install'. No 'sudo' required." )
    install ( TARGETS rbio
        LIBRARY       DESTINATION ${LOCAL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${LOCAL_INCLUDEDIR} )
    install ( TARGETS rbio_static
        ARCHIVE       DESTINATION ${LOCAL_LIBDIR} )
endif ( )

#-------------------------------------------------------------------------------
# Demo library and programs
#-------------------------------------------------------------------------------

if ( DEMO )

    #---------------------------------------------------------------------------
    # demo library
    #---------------------------------------------------------------------------

    message ( STATUS "Also compiling the demos in RBio/Demo" )

    #---------------------------------------------------------------------------
    # Demo programs
    #---------------------------------------------------------------------------

    add_executable ( RBdemo "Demo/RBdemo.c" )

    # Libraries required for Demo programs
    target_link_libraries ( RBdemo PUBLIC rbio )

else ( )

    message ( STATUS "Skipping the demos in RBio/Demo" )

endif ( )
