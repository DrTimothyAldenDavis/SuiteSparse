#-------------------------------------------------------------------------------
# SuiteSparse/COLAMD/CMakeLists.txt:  cmake for COLAMD
#-------------------------------------------------------------------------------

# Copyright (c) 1998-2023, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

#-------------------------------------------------------------------------------
# get the version
#-------------------------------------------------------------------------------

cmake_minimum_required ( VERSION 3.19 )

set ( COLAMD_DATE "Jan 17, 2023" )
set ( COLAMD_VERSION_MAJOR 3 )
set ( COLAMD_VERSION_MINOR 0 )
set ( COLAMD_VERSION_SUB   3 )

message ( STATUS "Building COLAMD version: v"
    ${COLAMD_VERSION_MAJOR}.
    ${COLAMD_VERSION_MINOR}.
    ${COLAMD_VERSION_SUB} " (" ${COLAMD_DATE} ")" )

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

project ( colamd
    VERSION "${COLAMD_VERSION_MAJOR}.${COLAMD_VERSION_MINOR}.${COLAMD_VERSION_SUB}"
    LANGUAGES C )

#-------------------------------------------------------------------------------
# find library dependencies
#-------------------------------------------------------------------------------

find_package ( SuiteSparse_config 7.0.0 REQUIRED )

#-------------------------------------------------------------------------------
# configure files
#-------------------------------------------------------------------------------

configure_file ( "Config/colamd.h.in"
    "${PROJECT_SOURCE_DIR}/Include/colamd.h"
    NEWLINE_STYLE LF )

#-------------------------------------------------------------------------------
# include directories
#-------------------------------------------------------------------------------

include_directories ( Source Include ${SUITESPARSE_CONFIG_INCLUDE_DIR} )

#-------------------------------------------------------------------------------
# dynamic colamd library properties
#-------------------------------------------------------------------------------

file ( GLOB COLAMD_SOURCES "Source/*.c" )

add_library ( colamd SHARED ${COLAMD_SOURCES} )

set_target_properties ( colamd PROPERTIES
    VERSION ${COLAMD_VERSION_MAJOR}.${COLAMD_VERSION_MINOR}.${COLAMD_VERSION_SUB}
    C_STANDARD_REQUIRED 11
    SOVERSION ${COLAMD_VERSION_MAJOR}
    PUBLIC_HEADER "Include/colamd.h"
    WINDOWS_EXPORT_ALL_SYMBOLS ON )

#-------------------------------------------------------------------------------
# static colamd library properties
#-------------------------------------------------------------------------------

if ( NOT NSTATIC )
    add_library ( colamd_static STATIC ${COLAMD_SOURCES} )

    set_target_properties ( colamd_static PROPERTIES
        VERSION ${COLAMD_VERSION_MAJOR}.${COLAMD_VERSION_MINOR}.${COLAMD_VERSION_SUB}
        OUTPUT_NAME colamd
        C_STANDARD_REQUIRED 11
        SOVERSION ${COLAMD_VERSION_MAJOR} )

    if ( MSVC )
        set_target_properties ( colamd_static PROPERTIES
            OUTPUT_NAME colamd_static )
    endif ( )
endif ( )

#-------------------------------------------------------------------------------
# add the library dependencies
#-------------------------------------------------------------------------------

target_link_libraries ( colamd PUBLIC ${SUITESPARSE_CONFIG_LIBRARY} )
if ( NOT NSTATIC )
    target_link_libraries ( colamd_static PUBLIC ${SUITESPARSE_CONFIG_STATIC} )
endif ( )

# libm:
if ( NOT WIN32 )
    target_link_libraries ( colamd PUBLIC m )
    if ( NOT NSTATIC )
        target_link_libraries ( colamd_static PUBLIC m )
    endif ( )
endif ( )

#-------------------------------------------------------------------------------
# COLAMD installation location
#-------------------------------------------------------------------------------

install ( TARGETS colamd
    LIBRARY DESTINATION ${SUITESPARSE_LIBDIR}
    ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR}
    RUNTIME DESTINATION ${SUITESPARSE_BINDIR}
    PUBLIC_HEADER DESTINATION ${SUITESPARSE_INCLUDEDIR} )
install ( FILES ${CMAKE_SOURCE_DIR}/cmake_modules/FindCOLAMD.cmake
    DESTINATION ${SUITESPARSE_LIBDIR}/cmake/SuiteSparse 
    COMPONENT Development )
if ( NOT NSTATIC )
    install ( TARGETS colamd_static
        ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR} )
endif ( )

#-------------------------------------------------------------------------------
# Demo library and programs
#-------------------------------------------------------------------------------

option ( DEMO "ON: Build the demo programs.  OFF (default): do not build the demo programs." off )
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

#-------------------------------------------------------------------------------
# report status
#-------------------------------------------------------------------------------

include ( SuiteSparseReport )

