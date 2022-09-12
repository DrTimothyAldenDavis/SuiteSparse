#-------------------------------------------------------------------------------
# SuiteSparse/CCOLAMD/CMakeLists.txt:  cmake for CCOLAMD
#-------------------------------------------------------------------------------

# Copyright (c) 2022, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

#-------------------------------------------------------------------------------
# get the version
#-------------------------------------------------------------------------------

cmake_minimum_required ( VERSION 3.19 )

set ( CCOLAMD_DATE "Sept FIXME, 2022" )
set ( CCOLAMD_VERSION_MAJOR 3 )
set ( CCOLAMD_VERSION_MINOR 0 )
set ( CCOLAMD_VERSION_SUB   0 )

message ( STATUS "Building CCOLAMD version: v"
    ${CCOLAMD_VERSION_MAJOR}.
    ${CCOLAMD_VERSION_MINOR}.
    ${CCOLAMD_VERSION_SUB} " (" ${CCOLAMD_DATE} ")" )

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

project ( ccolamd
    VERSION "${CCOLAMD_VERSION_MAJOR}.${CCOLAMD_VERSION_MINOR}.${CCOLAMD_VERSION_SUB}"
    LANGUAGES C ) # <-- add CXX here for packages that use C++

#-------------------------------------------------------------------------------
# find library dependencies
#-------------------------------------------------------------------------------

find_package ( OpenMP )
find_package ( SuiteSparse_config REQUIRED )

#-------------------------------------------------------------------------------
# configure files
#-------------------------------------------------------------------------------

configure_file ( "Config/ccolamd.h.in" "${PROJECT_SOURCE_DIR}/Include/ccolamd.h" )

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

include_directories ( Source Include ${SUITESPARSE_CONFIG_INCLUDE_DIR} )

#-------------------------------------------------------------------------------
# dynamic ccolamd library properties
#-------------------------------------------------------------------------------

file ( GLOB CCOLAMD_SOURCES "Source/*.c" )

add_library ( ccolamd SHARED ${CCOLAMD_SOURCES} )

SET_TARGET_PROPERTIES ( ccolamd PROPERTIES
    VERSION ${CCOLAMD_VERSION_MAJOR}.${CCOLAMD_VERSION_MINOR}.${CCOLAMD_VERSION_SUB}
    C_STANDARD_REQUIRED 11
    SOVERSION ${CCOLAMD_VERSION_MAJOR}
    PUBLIC_HEADER "Include/ccolamd.h" )

#-------------------------------------------------------------------------------
# static ccolamd library properties
#-------------------------------------------------------------------------------

add_library ( ccolamd_static STATIC ${CCOLAMD_SOURCES} )

SET_TARGET_PROPERTIES ( ccolamd_static PROPERTIES
    VERSION ${CCOLAMD_VERSION_MAJOR}.${CCOLAMD_VERSION_MINOR}.${CCOLAMD_VERSION_SUB}
    OUTPUT_NAME ccolamd
    C_STANDARD_REQUIRED 11
    SOVERSION ${CCOLAMD_VERSION_MAJOR} )

#-------------------------------------------------------------------------------
# add the library dependencies
#-------------------------------------------------------------------------------

target_link_libraries ( ccolamd PUBLIC ${SUITESPARSE_CONFIG_LIBRARY} )
target_link_libraries ( ccolamd_static PUBLIC ${SUITESPARSE_CONFIG_LIBRARY} )

# OpenMP:
if ( OPENMP_FOUND )
    message ( STATUS "OpenMP C libraries:      " ${OpenMP_C_LIBRARIES} )
    message ( STATUS "OpenMP C include:        " ${OpenMP_C_INCLUDE_DIRS} )
    message ( STATUS "OpenMP C flags:          " ${OpenMP_C_FLAGS} )
    target_link_libraries ( ccolamd PUBLIC ${OpenMP_C_LIBRARIES} )
    target_link_libraries ( ccolamd_static PUBLIC ${OpenMP_C_LIBRARIES} )
    set ( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} " )
    include_directories ( ${OpenMP_C_INCLUDE_DIRS} )
endif ( )

# libm:
if ( NOT "${CMAKE_C_COMPILER_ID}" STREQUAL "MSVC" )
    target_link_libraries ( ccolamd PUBLIC m )
    target_link_libraries ( ccolamd_static PUBLIC m )
endif ( )

#-------------------------------------------------------------------------------
# COLAMD installation location
#-------------------------------------------------------------------------------

# use "cmake -DLOCAL=1" to install only in SuiteSparse/lib and
# SuiteSparse/include.  

if ( NOT LOCAL )
    # install in /usr/local/lib and /usr/local/include.
    # requires "sudo make install"
    message ( STATUS "Installation will be system-wide (requires 'sudo make install')" )
    install ( TARGETS ccolamd
        LIBRARY       DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} )
    install ( TARGETS ccolamd_static
        ARCHIVE       DESTINATION ${CMAKE_INSTALL_LIBDIR} )
endif ( )

if ( INSIDE_SUITESPARSE )
    # also install in SuiteSparse/lib and SuiteSparse/include;
    # does not require "sudo make install", just "make install"
    message ( STATUS "Installation in ../lib and ../include only," )
    message ( STATUS "  with 'make install'. No 'sudo' required." )
    install ( TARGETS ccolamd
        LIBRARY       DESTINATION ${LOCAL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${LOCAL_INCLUDEDIR} )
    install ( TARGETS ccolamd_static
        ARCHIVE       DESTINATION ${LOCAL_LIBDIR} )
endif ( )

#-------------------------------------------------------------------------------
# Demo library and programs
#-------------------------------------------------------------------------------

if ( DEMO )

    #---------------------------------------------------------------------------
    # demo library
    #---------------------------------------------------------------------------

    message ( STATUS "Also compiling the demos in CCOLAMD/Demo" )

    #---------------------------------------------------------------------------
    # Demo programs
    #---------------------------------------------------------------------------

    add_executable ( ccolamd_example   "Demo/ccolamd_example.c" )
    add_executable ( ccolamd_l_example "Demo/ccolamd_l_example.c" )

    # Libraries required for Demo programs
    target_link_libraries ( ccolamd_example   PUBLIC ccolamd )
    target_link_libraries ( ccolamd_l_example PUBLIC ccolamd )

else ( )

    message ( STATUS "Skipping the demos in CCCOLAMD/Demo" )

endif ( )
