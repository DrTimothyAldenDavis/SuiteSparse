#-------------------------------------------------------------------------------
# SuiteSparse/SuiteSparse_metis/CMakeLists.txt:  cmake for METIS 5.1.0
#-------------------------------------------------------------------------------

# METIS, Copyright 1995-2013, Regents of the University of Minnesota.
# Author: George Karypis
# SPDX-License-identifier: Apache-2.0

# This CMakeLists.txt script was added by Tim Davis, Nov 2022,
# to build SuiteSparse_metis for use in SuiteSparse.  See
# include/SuiteSparse_metis.h for a description of the changes made in this
# version, as compared to the original METIS v5.1.0.  This cmake script builds
# libsuitesparse_metis.so.

# The original METIS v5.1.0 CMakeLists.txt has been moved to
# original/CMakeLists.txt.

#-------------------------------------------------------------------------------
# get the version
#-------------------------------------------------------------------------------

cmake_minimum_required ( VERSION 3.19 )

set ( SUITESPARSE_METIS_DATE "Nov 4, 2022" )
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
    ${CMAKE_SOURCE_DIR}/../SuiteSparse_config/cmake_modules
    ${CMAKE_SOURCE_DIR}/../SuiteSparse/SuiteSparse_config/cmake_modules
    )

include ( SuiteSparsePolicy )

#-------------------------------------------------------------------------------
# define the project
#-------------------------------------------------------------------------------

project ( suitesparse_metis
    VERSION "${SUITESPARSE_METIS_VERSION_MAJOR}.${SUITESPARSE_METIS_VERSION_MINOR}.${SUITESPARSE_METIS_VERSION_SUB}"
    LANGUAGES C )

#-------------------------------------------------------------------------------
# find library dependencies
#-------------------------------------------------------------------------------

find_package ( SuiteSparse_config 6.0.0 REQUIRED )

#-------------------------------------------------------------------------------
# compiler options
#-------------------------------------------------------------------------------

file ( REAL_PATH ${CMAKE_SOURCE_DIR}/GKlib GKLIB_PATH )
include ( ${GKLIB_PATH}/GKlibSystem.cmake )

if ( "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU" )
    set ( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -Wunused-result" )
endif ( )

#-------------------------------------------------------------------------------
# include directories
#-------------------------------------------------------------------------------

include_directories ( include libmetis ${GKLIB_PATH} ${SUITESPARSE_CONFIG_INCLUDE_DIR} )

#-------------------------------------------------------------------------------
# dynamic suitesparse_metis library properties
#-------------------------------------------------------------------------------

file ( GLOB SUITESPARSE_METIS_SOURCES "GKlib/*.c" "libmetis/*.c" )

add_library ( suitesparse_metis SHARED ${SUITESPARSE_METIS_SOURCES} )

set_target_properties ( suitesparse_metis PROPERTIES
    VERSION ${SUITESPARSE_METIS_VERSION_MAJOR}.${SUITESPARSE_METIS_VERSION_MINOR}.${SUITESPARSE_METIS_VERSION_SUB}
    C_STANDARD_REQUIRED 11
    SOVERSION ${SUITESPARSE_METIS_VERSION_MAJOR}
    PUBLIC_HEADER "include/SuiteSparse_metis.h" )

#-------------------------------------------------------------------------------
# static suitesparse_metis library properties
#-------------------------------------------------------------------------------

if ( NOT NSTATIC )
add_library ( suitesparse_metis_static STATIC ${SUITESPARSE_METIS_SOURCES} )

set_target_properties ( suitesparse_metis_static PROPERTIES
    VERSION ${SUITESPARSE_METIS_VERSION_MAJOR}.${SUITESPARSE_METIS_VERSION_MINOR}.${SUITESPARSE_METIS_VERSION_SUB}
    C_STANDARD_REQUIRED 11
    OUTPUT_NAME suitesparse_metis
    SOVERSION ${SUITESPARSE_METIS_VERSION_MAJOR} )
endif ( )

#-------------------------------------------------------------------------------
# add the library dependencies
#-------------------------------------------------------------------------------

# suitesparseconfig:
target_link_libraries ( suitesparse_metis PUBLIC ${SUITESPARSE_CONFIG_LIBRARIES} )
if ( NOT NSTATIC )
target_link_libraries ( suitesparse_metis_static PUBLIC ${SUITESPARSE_CONFIG_LIBRARIES} )
endif ( )

# libm:
if ( NOT "${CMAKE_C_COMPILER_ID}" STREQUAL "MSVC" )
    target_link_libraries ( suitesparse_metis PUBLIC m )
    if ( NOT NSTATIC )
    target_link_libraries ( suitesparse_metis_static PUBLIC m )
    endif ( )
endif ( )

#-------------------------------------------------------------------------------
# SuiteSparse_metis installation location
#-------------------------------------------------------------------------------

if ( GLOBAL_INSTALL )
    # install in /usr/local/lib and /usr/local/include.
    # requires "sudo make install"
    message ( STATUS "Installation will be system-wide (requires 'sudo make install')" )
    install ( TARGETS suitesparse_metis
        LIBRARY       DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} )
    if ( NOT NSTATIC )
    install ( TARGETS suitesparse_metis_static
        ARCHIVE       DESTINATION ${CMAKE_INSTALL_LIBDIR} )
    endif ( )
endif ( )

if ( INSIDE_SUITESPARSE )
    # also install in SuiteSparse/lib and SuiteSparse/include;
    # does not require "sudo make install", just "make install"
    message ( STATUS "Installation in ../lib and ../include," )
    message ( STATUS "  with 'make local ; make install'. No 'sudo' required." )
    install ( TARGETS suitesparse_metis
        LIBRARY       DESTINATION ${SUITESPARSE_LIBDIR}
        PUBLIC_HEADER DESTINATION ${SUITESPARSE_INCLUDEDIR} )
    if ( NOT NSTATIC )
    install ( TARGETS suitesparse_metis_static
        ARCHIVE       DESTINATION ${SUITESPARSE_LIBDIR} )
    endif ( )
endif ( )

#-------------------------------------------------------------------------------
# report status
#-------------------------------------------------------------------------------

include ( SuiteSparseReport )

