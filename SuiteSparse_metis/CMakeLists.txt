#-------------------------------------------------------------------------------
# SuiteSparse/SuiteSparse_metis/CMakeLists.txt:  cmake for METIS 5.1.0
#-------------------------------------------------------------------------------

# METIS, Copyright 1995-2013, Regents of the University of Minnesota.
# Author: George Karypis
# SPDX-License-identifier: Apache-2.0

# This CMakeLists.txt script was added by Tim Davis, Oct 2022,
# to build SuiteSparse_metis for use in SuiteSparse.  See
# include/suitesparse_metis.h for a description of the changes made in this
# version, as compared to the original METIS v5.1.0.  This cmake script builds
# libsuitesparse_metis.so.

# The original METIS v5.1.0 CMakeLists.txt has been moved to
# original/CMakeLists.txt.

#-------------------------------------------------------------------------------
# get the version
#-------------------------------------------------------------------------------

cmake_minimum_required ( VERSION 3.19 )

set ( SUITESPARSE_METIS_DATE "Oct 31, 2022" )  # FIXME
set ( SUITESPARSE_METIS_VERSION_MAJOR 5 )
set ( SUITESPARSE_METIS_VERSION_MINOR 1 )
set ( SUITESPARSE_METIS_VERSION_SUB   0 )

message ( STATUS "Building SuiteSparse_metis version: v"
    ${SUITESPARSE_METIS_VERSION_MAJOR}.
    ${SUITESPARSE_METIS_VERSION_MINOR}.
    ${SUITESPARSE_METIS_VERSION_SUB} " (" ${SUITESPARSE_METIS_DATE} ")" )

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

project ( suitesparse_metis
    VERSION "${SUITESPARSE_METIS_VERSION_MAJOR}.${SUITESPARSE_METIS_VERSION_MINOR}.${SUITESPARSE_METIS_VERSION_SUB}"
    LANGUAGES C ) # <-- add CXX here for packages that use C++

#-------------------------------------------------------------------------------
# find library dependencies
#-------------------------------------------------------------------------------

find_package ( SuiteSparse_config 6.0.0 REQUIRED )

#-------------------------------------------------------------------------------
# compiler options
#-------------------------------------------------------------------------------

file ( REAL_PATH ${CMAKE_SOURCE_DIR}/GKlib GKLIB_PATH )
include ( ${GKLIB_PATH}/GKlibSystem.cmake )

#-------------------------------------------------------------------------------
# report status and set compile flags
#-------------------------------------------------------------------------------

message ( STATUS "CMAKE C flags:   ${CMAKE_C_FLAGS}" )
# message ( STATUS "CMAKE C++ flags: ${CMAKE_CXX_FLAGS}" )
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

include_directories ( include libmetis ${GKLIB_PATH} ${SUITESPARSE_CONFIG_INCLUDE_DIR} )

#-------------------------------------------------------------------------------
# dynamic suitesparse_metis library properties
#-------------------------------------------------------------------------------

file ( GLOB SUITESPARSE_METIS_SOURCES "GKlib/*.c" "libmetis/*.c" )

add_library ( suitesparse_metis SHARED ${SUITESPARSE_METIS_SOURCES} )

SET_TARGET_PROPERTIES ( suitesparse_metis PROPERTIES
    VERSION ${SUITESPARSE_METIS_VERSION_MAJOR}.${SUITESPARSE_METIS_VERSION_MINOR}.${SUITESPARSE_METIS_VERSION_SUB}
    C_STANDARD_REQUIRED 11
    SOVERSION ${SUITESPARSE_METIS_VERSION_MAJOR}
    PUBLIC_HEADER "include/suitesparse_metis.h" )

#-------------------------------------------------------------------------------
# static suitesparse_metis library properties
#-------------------------------------------------------------------------------

add_library ( suitesparse_metis_static STATIC ${SUITESPARSE_METIS_SOURCES} )

SET_TARGET_PROPERTIES ( suitesparse_metis_static PROPERTIES
    VERSION ${SUITESPARSE_METIS_VERSION_MAJOR}.${SUITESPARSE_METIS_VERSION_MINOR}.${SUITESPARSE_METIS_VERSION_SUB}
    C_STANDARD_REQUIRED 11
    OUTPUT_NAME suitesparse_metis
    SOVERSION ${SUITESPARSE_METIS_VERSION_MAJOR} )

#-------------------------------------------------------------------------------
# add the library dependencies
#-------------------------------------------------------------------------------

# suitesparseconfig:
target_link_libraries ( suitesparse_metis PUBLIC ${SUITESPARSE_CONFIG_LIBRARIES} )
target_link_libraries ( suitesparse_metis_static PUBLIC ${SUITESPARSE_CONFIG_LIBRARIES} )

# libm:
if ( NOT "${CMAKE_C_COMPILER_ID}" STREQUAL "MSVC" )
    target_link_libraries ( suitesparse_metis PUBLIC m )
    target_link_libraries ( suitesparse_metis_static PUBLIC m )
endif ( )

#-------------------------------------------------------------------------------
# SuiteSparse_metis installation location
#-------------------------------------------------------------------------------

# use "cmake -DSUITESPARSE_LOCAL=1" to install only in SuiteSparse/lib and
# SuiteSparse/include.  

if ( NOT SUITESPARSE_LOCAL )
    # install in /usr/local/lib and /usr/local/include.
    # requires "sudo make install"
    message ( STATUS "Installation will be system-wide (requires 'sudo make install')" )
    install ( TARGETS suitesparse_metis
        LIBRARY       DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} )
    install ( TARGETS suitesparse_metis_static
        ARCHIVE       DESTINATION ${CMAKE_INSTALL_LIBDIR} )
endif ( )

if ( INSIDE_SUITESPARSE )
    # also install in SuiteSparse/lib and SuiteSparse/include;
    # does not require "sudo make install", just "make install"
    message ( STATUS "Installation in ../lib and ../include," )
    message ( STATUS "  with 'make local ; make install'. No 'sudo' required." )
    install ( TARGETS suitesparse_metis
        LIBRARY       DESTINATION ${SUITESPARSE_LIBDIR}
        PUBLIC_HEADER DESTINATION ${SUITESPARSE_INCLUDEDIR} )
    install ( TARGETS suitesparse_metis_static
        ARCHIVE       DESTINATION ${SUITESPARSE_LIBDIR} )
endif ( )

