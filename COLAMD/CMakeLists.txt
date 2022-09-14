#-------------------------------------------------------------------------------
# SuiteSparse/COLAMD/CMakeLists.txt:  cmake for COLAMD
#-------------------------------------------------------------------------------

# Copyright (c) 2022, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

#-------------------------------------------------------------------------------
# get the version
#-------------------------------------------------------------------------------

cmake_minimum_required ( VERSION 3.19 )

set ( COLAMD_DATE "Sept FIXME, 2022" )
set ( COLAMD_VERSION_MAJOR 3 )
set ( COLAMD_VERSION_MINOR 0 )
set ( COLAMD_VERSION_SUB   0 )

message ( STATUS "Building COLAMD version: v"
    ${COLAMD_VERSION_MAJOR}.
    ${COLAMD_VERSION_MINOR}.
    ${COLAMD_VERSION_SUB} " (" ${COLAMD_DATE} ")" )

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

project ( colamd
    VERSION "${COLAMD_VERSION_MAJOR}.${COLAMD_VERSION_MINOR}.${COLAMD_VERSION_SUB}"
    LANGUAGES C ) # <-- add CXX here for packages that use C++

#-------------------------------------------------------------------------------
# find library dependencies
#-------------------------------------------------------------------------------

find_package ( SuiteSparse_config 6.0.0 REQUIRED )

#-------------------------------------------------------------------------------
# configure files
#-------------------------------------------------------------------------------

configure_file ( "Config/colamd.h.in" "${PROJECT_SOURCE_DIR}/Include/colamd.h" )

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

# message ( STATUS "CMAKE C++ compiler:      " ${CMAKE_CXX_COMPILER_ID} )

#-------------------------------------------------------------------------------
# include directories
#-------------------------------------------------------------------------------

include_directories ( Source Include ${SUITESPARSE_CONFIG_INCLUDE_DIR} )

#-------------------------------------------------------------------------------
# dynamic colamd library properties
#-------------------------------------------------------------------------------

file ( GLOB COLAMD_SOURCES "Source/*.c" )

add_library ( colamd SHARED ${COLAMD_SOURCES} )

SET_TARGET_PROPERTIES ( colamd PROPERTIES
    VERSION ${COLAMD_VERSION_MAJOR}.${COLAMD_VERSION_MINOR}.${COLAMD_VERSION_SUB}
    C_STANDARD_REQUIRED 11
    SOVERSION ${COLAMD_VERSION_MAJOR}
    PUBLIC_HEADER "Include/colamd.h" )

#-------------------------------------------------------------------------------
# static colamd library properties
#-------------------------------------------------------------------------------

add_library ( colamd_static STATIC ${COLAMD_SOURCES} )

SET_TARGET_PROPERTIES ( colamd_static PROPERTIES
    VERSION ${COLAMD_VERSION_MAJOR}.${COLAMD_VERSION_MINOR}.${COLAMD_VERSION_SUB}
    OUTPUT_NAME colamd
    C_STANDARD_REQUIRED 11
    SOVERSION ${COLAMD_VERSION_MAJOR} )

#-------------------------------------------------------------------------------
# add the library dependencies
#-------------------------------------------------------------------------------

target_link_libraries ( colamd PUBLIC ${SUITESPARSE_CONFIG_LIBRARY} )
target_link_libraries ( colamd_static PUBLIC ${SUITESPARSE_CONFIG_LIBRARY} )

# libm:
if ( NOT "${CMAKE_C_COMPILER_ID}" STREQUAL "MSVC" )
    target_link_libraries ( colamd PUBLIC m )
    target_link_libraries ( colamd_static PUBLIC m )
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
    install ( TARGETS colamd
        LIBRARY       DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} )
    install ( TARGETS colamd_static
        ARCHIVE       DESTINATION ${CMAKE_INSTALL_LIBDIR} )
endif ( )

if ( INSIDE_SUITESPARSE )
    # also install in SuiteSparse/lib and SuiteSparse/include;
    # does not require "sudo make install", just "make install"
    message ( STATUS "Installation in ../lib and ../include," )
    message ( STATUS "  with 'make local ; make install'. No 'sudo' required." )
    install ( TARGETS colamd
        LIBRARY       DESTINATION ${LOCAL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${LOCAL_INCLUDEDIR} )
    install ( TARGETS colamd_static
        ARCHIVE       DESTINATION ${LOCAL_LIBDIR} )
endif ( )

#-------------------------------------------------------------------------------
# Demo library and programs
#-------------------------------------------------------------------------------

if ( DEMO )

    #---------------------------------------------------------------------------
    # demo library
    #---------------------------------------------------------------------------

    message ( STATUS "Also compiling the demos in COLAMD/Demo" )

    #---------------------------------------------------------------------------
    # Demo programs
    #---------------------------------------------------------------------------

    add_executable ( colamd_example   "Demo/colamd_example.c" )
    add_executable ( colamd_l_example "Demo/colamd_l_example.c" )

    # Libraries required for Demo programs
    target_link_libraries ( colamd_example   PUBLIC colamd )
    target_link_libraries ( colamd_l_example PUBLIC colamd )

else ( )

    message ( STATUS "Skipping the demos in COLAMD/Demo" )

endif ( )
