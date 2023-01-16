#-------------------------------------------------------------------------------
# SuiteSparse/AMD/CMakeLists.txt:  cmake for AMD
#-------------------------------------------------------------------------------

# Copyright (c) 1996-2022, Timothy A. Davis, Patrick Amestoy, Iain Duff.
# All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

#-------------------------------------------------------------------------------
# get the version
#-------------------------------------------------------------------------------

cmake_minimum_required ( VERSION 3.19 )

set ( AMD_DATE "Jan 17, 2023" )
set ( AMD_VERSION_MAJOR 3 )
set ( AMD_VERSION_MINOR 0 )
set ( AMD_VERSION_SUB   3 )

message ( STATUS "Building AMD version: v"
    ${AMD_VERSION_MAJOR}.
    ${AMD_VERSION_MINOR}.
    ${AMD_VERSION_SUB} " (" ${AMD_DATE} ")" )

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

if ( WIN32 )
    # disable Fortran in AMD when compiling on Windows
    set ( NFORTRAN true )
endif ( )

if ( NOT NFORTRAN )
    # Fortan is available and enabled
    project ( amd
        VERSION "${AMD_VERSION_MAJOR}.${AMD_VERSION_MINOR}.${AMD_VERSION_SUB}"
        LANGUAGES C Fortran )
else ( )
    # no Fortran compiler available; do not compile Source/*.f or Demo/*.f
    project ( amd
        VERSION "${AMD_VERSION_MAJOR}.${AMD_VERSION_MINOR}.${AMD_VERSION_SUB}"
        LANGUAGES C )
endif ( )

#-------------------------------------------------------------------------------
# find library dependencies
#-------------------------------------------------------------------------------

find_package ( SuiteSparse_config 7.0.0 REQUIRED )

#-------------------------------------------------------------------------------
# configure files
#-------------------------------------------------------------------------------

configure_file ( "Config/amd.h.in" "${PROJECT_SOURCE_DIR}/Include/amd.h"
    NEWLINE_STYLE LF )
configure_file ( "Config/amd_version.tex.in" "${PROJECT_SOURCE_DIR}/Doc/amd_version.tex"
    NEWLINE_STYLE LF )

#-------------------------------------------------------------------------------
# include directories
#-------------------------------------------------------------------------------

include_directories ( Source Include ${SUITESPARSE_CONFIG_INCLUDE_DIR} )

#-------------------------------------------------------------------------------
# dynamic amd library properties
#-------------------------------------------------------------------------------

if ( NOT NFORTRAN )
    file ( GLOB AMD_SOURCES "Source/*.c" "Source/*.f" )
else ( )
    file ( GLOB AMD_SOURCES "Source/*.c" )
endif ( )

add_library ( amd SHARED ${AMD_SOURCES} )
set_target_properties ( amd PROPERTIES
    VERSION ${AMD_VERSION_MAJOR}.${AMD_VERSION_MINOR}.${AMD_VERSION_SUB}
    C_STANDARD_REQUIRED 11
    SOVERSION ${AMD_VERSION_MAJOR}
    PUBLIC_HEADER "Include/amd.h"
    WINDOWS_EXPORT_ALL_SYMBOLS ON )

#-------------------------------------------------------------------------------
# static amd library properties
#-------------------------------------------------------------------------------

if ( NOT NSTATIC )
    add_library ( amd_static STATIC ${AMD_SOURCES} )
    set_target_properties ( amd_static PROPERTIES
        VERSION ${AMD_VERSION_MAJOR}.${AMD_VERSION_MINOR}.${AMD_VERSION_SUB}
        C_STANDARD_REQUIRED 11
        OUTPUT_NAME amd
        SOVERSION ${AMD_VERSION_MAJOR} )

    if ( MSVC )
        set_target_properties ( amd_static PROPERTIES
            OUTPUT_NAME amd_static )
    endif ( )
endif ( )

#-------------------------------------------------------------------------------
# add the library dependencies
#-------------------------------------------------------------------------------

# suitesparseconfig:
target_link_libraries ( amd PUBLIC ${SUITESPARSE_CONFIG_LIBRARIES} )
if ( NOT NSTATIC )
    target_link_libraries ( amd_static PUBLIC ${SUITESPARSE_CONFIG_STATIC} )
endif ( )

# libm:
if ( NOT WIN32 )
    target_link_libraries ( amd PUBLIC m )
    if ( NOT NSTATIC )
        target_link_libraries ( amd_static PUBLIC m )
    endif ( )
endif ( )

#-------------------------------------------------------------------------------
# AMD installation location
#-------------------------------------------------------------------------------

install ( TARGETS amd
    LIBRARY DESTINATION ${SUITESPARSE_LIBDIR}
    ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR}
    RUNTIME DESTINATION ${SUITESPARSE_BINDIR}
    PUBLIC_HEADER DESTINATION ${SUITESPARSE_INCLUDEDIR} )
install ( FILES ${CMAKE_SOURCE_DIR}/cmake_modules/FindAMD.cmake
    DESTINATION ${SUITESPARSE_LIBDIR}/cmake/SuiteSparse 
    COMPONENT Development )
if ( NOT NSTATIC )
    install ( TARGETS amd_static
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

    message ( STATUS "Also compiling the demos in AMD/Demo" )

    #---------------------------------------------------------------------------
    # Demo programs
    #---------------------------------------------------------------------------

    add_executable ( amd_demo      "Demo/amd_demo.c" )
    add_executable ( amd_l_demo    "Demo/amd_l_demo.c" )
    add_executable ( amd_demo2     "Demo/amd_demo2.c" )
    add_executable ( amd_simple    "Demo/amd_simple.c" )
    if ( NOT NFORTRAN )
        add_executable ( amd_f77demo   "Demo/amd_f77demo.f" )
        add_executable ( amd_f77simple "Demo/amd_f77simple.f" )
    endif ( )

    # Libraries required for Demo programs
    target_link_libraries ( amd_demo      PUBLIC amd )
    target_link_libraries ( amd_l_demo    PUBLIC amd )
    target_link_libraries ( amd_demo2     PUBLIC amd )
    target_link_libraries ( amd_simple    PUBLIC amd )
    if ( NOT NFORTRAN )
        target_link_libraries ( amd_f77demo   PUBLIC amd )
        target_link_libraries ( amd_f77simple PUBLIC amd )
    endif ( )

else ( )

    message ( STATUS "Skipping the demos in AMD/Demo" )

endif ( )

#-------------------------------------------------------------------------------
# report status
#-------------------------------------------------------------------------------

include ( SuiteSparseReport )

