#-------------------------------------------------------------------------------
# SuiteSparse/RBio/CMakeLists.txt:  cmake for RBio
#-------------------------------------------------------------------------------

# RBio, Copyright (c) 2009-2022, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: GPL-2.0+

#-------------------------------------------------------------------------------
# get the version
#-------------------------------------------------------------------------------

cmake_minimum_required ( VERSION 3.19 )

set ( RBIO_DATE "Jan 1, 2023" )
set ( RBIO_VERSION_MAJOR 3 )
set ( RBIO_VERSION_MINOR 0 )
set ( RBIO_VERSION_SUB   3 )

message ( STATUS "Building RBIO version: v"
    ${RBIO_VERSION_MAJOR}.
    ${RBIO_VERSION_MINOR}.
    ${RBIO_VERSION_SUB} " (" ${RBIO_DATE} ")" )

#-------------------------------------------------------------------------------
# SuiteSparse policies
#-------------------------------------------------------------------------------

set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    ${CMAKE_SOURCE_DIR}/cmake_modules
    ${CMAKE_SOURCE_DIR}/../SuiteSparse_config/cmake_modules )

include ( SuiteSparsePolicy )

#-------------------------------------------------------------------------------
# define the project
#-------------------------------------------------------------------------------

project ( rbio
    VERSION "${RBIO_VERSION_MAJOR}.${RBIO_VERSION_MINOR}.${RBIO_VERSION_SUB}"
    LANGUAGES C )

#-------------------------------------------------------------------------------
# find library dependencies
#-------------------------------------------------------------------------------

find_package ( SuiteSparse_config 7.0.0 REQUIRED )

#-------------------------------------------------------------------------------
# configure files
#-------------------------------------------------------------------------------

configure_file ( "Config/RBio.h.in"
    "${PROJECT_SOURCE_DIR}/Include/RBio.h"
    NEWLINE_STYLE LF )

#-------------------------------------------------------------------------------
# include directories
#-------------------------------------------------------------------------------

include_directories ( Source Include ${SUITESPARSE_CONFIG_INCLUDE_DIR} )

#-------------------------------------------------------------------------------
# dynamic rbio library properties
#-------------------------------------------------------------------------------

file ( GLOB RBIO_SOURCES "Source/*.c" )

add_library ( rbio SHARED ${RBIO_SOURCES} )

set_target_properties ( rbio PROPERTIES
    VERSION ${RBIO_VERSION_MAJOR}.${RBIO_VERSION_MINOR}.${RBIO_VERSION_SUB}
    C_STANDARD_REQUIRED 11
    SOVERSION ${RBIO_VERSION_MAJOR}
    PUBLIC_HEADER "Include/RBio.h" )

#-------------------------------------------------------------------------------
# static rbio library properties
#-------------------------------------------------------------------------------

if ( NOT NSTATIC )
add_library ( rbio_static STATIC ${RBIO_SOURCES} )

set_target_properties ( rbio_static PROPERTIES
    VERSION ${RBIO_VERSION_MAJOR}.${RBIO_VERSION_MINOR}.${RBIO_VERSION_SUB}
    C_STANDARD_REQUIRED 11
    OUTPUT_NAME rbio
    SOVERSION ${RBIO_VERSION_MAJOR} )
endif ( )

#-------------------------------------------------------------------------------
# add the library dependencies
#-------------------------------------------------------------------------------

# suitesparseconfig:
target_link_libraries ( rbio PUBLIC ${SUITESPARSE_CONFIG_LIBRARIES} )
if ( NOT NSTATIC )
target_link_libraries ( rbio_static PUBLIC ${SUITESPARSE_CONFIG_LIBRARIES} )
endif ( )

# libm:
if ( NOT WIN32 )
    target_link_libraries ( rbio PUBLIC m )
    if ( NOT NSTATIC )
    target_link_libraries ( rbio_static PUBLIC m )
    endif ( )
endif ( )

#-------------------------------------------------------------------------------
# RBIO installation location
#-------------------------------------------------------------------------------

if ( GLOBAL_INSTALL )
    # install in /usr/local/lib and /usr/local/include.
    # requires "sudo make install"
    message ( STATUS "Installation will be system-wide (requires 'sudo make install')" )
    install ( TARGETS rbio
        LIBRARY       DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} )
    install ( FILES ${CMAKE_SOURCE_DIR}/cmake_modules/FindRBio.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SuiteSparse 
        COMPONENT Development )
    if ( NOT NSTATIC )
    install ( TARGETS rbio_static
        ARCHIVE       DESTINATION ${CMAKE_INSTALL_LIBDIR} )
    endif ( )
endif ( )

if ( INSIDE_SUITESPARSE )
    # also install in SuiteSparse/lib and SuiteSparse/include;
    # does not require "sudo make install", just "make install"
    message ( STATUS "Installation in ../lib and ../include," )
    message ( STATUS "  with 'make local ; make install'. No 'sudo' required." )
    install ( TARGETS rbio
        LIBRARY       DESTINATION ${SUITESPARSE_LIBDIR}
        PUBLIC_HEADER DESTINATION ${SUITESPARSE_INCLUDEDIR} )
    install ( FILES ${CMAKE_SOURCE_DIR}/cmake_modules/FindRBio.cmake
        DESTINATION ${SUITESPARSE_LIBDIR}/cmake/SuiteSparse 
        COMPONENT Development )
    if ( NOT NSTATIC )
    install ( TARGETS rbio_static
        ARCHIVE       DESTINATION ${SUITESPARSE_LIBDIR} )
    endif ( )
endif ( )

#-------------------------------------------------------------------------------
# Demo library and programs
#-------------------------------------------------------------------------------

option ( DEMO "ON: Build the demo programs.  OFF (default): do not build the demo programs." off )
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

#-------------------------------------------------------------------------------
# report status
#-------------------------------------------------------------------------------

include ( SuiteSparseReport )

