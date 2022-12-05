#-------------------------------------------------------------------------------
# SuiteSparse/SPEX/CMakeLists.txt:  cmake for SPEX
#-------------------------------------------------------------------------------

# Copyright (c) 1996-2022, Timothy A. Davis, Patrick Amestoy, Iain Duff.
# All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

#-------------------------------------------------------------------------------
# get the version
#-------------------------------------------------------------------------------

cmake_minimum_required ( VERSION 3.19 )

set ( SPEX_DATE "Dec 9, 2022" )
set ( SPEX_VERSION_MAJOR 2 )
set ( SPEX_VERSION_MINOR 0 )
set ( SPEX_VERSION_SUB   2 )

message ( STATUS "Building SPEX version: v"
    ${SPEX_VERSION_MAJOR}.
    ${SPEX_VERSION_MINOR}.
    ${SPEX_VERSION_SUB} " (" ${SPEX_DATE} ")" )

#-------------------------------------------------------------------------------
# SuiteSparse policies
#-------------------------------------------------------------------------------

set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    ${CMAKE_SOURCE_DIR}/cmake_modules
    ${CMAKE_SOURCE_DIR}/../COLAMD/cmake_modules
    ${CMAKE_SOURCE_DIR}/../AMD/cmake_modules
    ${CMAKE_SOURCE_DIR}/../SuiteSparse_config/cmake_modules )

include ( SuiteSparsePolicy )

#-------------------------------------------------------------------------------
# define the project
#-------------------------------------------------------------------------------

project ( spex
    VERSION "${SPEX_VERSION_MAJOR}.${SPEX_VERSION_MINOR}.${SPEX_VERSION_SUB}"
    LANGUAGES C )

#-------------------------------------------------------------------------------
# find library dependencies
#-------------------------------------------------------------------------------

find_package ( SuiteSparse_config 6.0.2 REQUIRED )
find_package ( COLAMD 3.0.2 REQUIRED )
find_package ( AMD 3.0.2 REQUIRED )
find_package ( GMP 6.1.2 REQUIRED )     # from SPEX/cmake_modules
find_package ( MPFR 4.0.2 REQUIRED )    # from SPEX/cmake_modules

#-------------------------------------------------------------------------------
# configure files
#-------------------------------------------------------------------------------

configure_file ( "Config/SPEX.h.in" "${PROJECT_SOURCE_DIR}/Include/SPEX.h")
configure_file ( "Config/SPEX_version.tex.in" "${PROJECT_SOURCE_DIR}/Doc/SPEX_version.tex")

#-------------------------------------------------------------------------------
# include directories
#-------------------------------------------------------------------------------

include_directories ( SPEX_Left_LU/Source SPEX_Util/Source Include 
    SPEX_Left_LU/Demo
    ${SUITESPARSE_CONFIG_INCLUDE_DIR} 
    ${GMP_INCLUDE_DIR} ${MPFR_INCLUDE_DIR}
    ${AMD_INCLUDE_DIR} ${COLAMD_INCLUDE_DIR} )

#-------------------------------------------------------------------------------
# compiler flags
#-------------------------------------------------------------------------------

# gcc -O3 complains about SPEX_mpq_init

if ( "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU" )
    set ( CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O2" )
endif ( )

#-------------------------------------------------------------------------------
# dynamic spex library properties
#-------------------------------------------------------------------------------

file ( GLOB SPEX_SOURCES "SPEX*/Source/*.c" )

add_library ( spex SHARED ${SPEX_SOURCES} )

set_target_properties ( spex PROPERTIES
    VERSION ${SPEX_VERSION_MAJOR}.${SPEX_VERSION_MINOR}.${SPEX_VERSION_SUB}
    C_STANDARD_REQUIRED 11
    SOVERSION ${SPEX_VERSION_MAJOR}
    PUBLIC_HEADER "Include/SPEX.h" )

#-------------------------------------------------------------------------------
# static spex library properties
#-------------------------------------------------------------------------------

if ( NOT NSTATIC )
add_library ( spex_static STATIC ${SPEX_SOURCES} )

set_target_properties ( spex_static PROPERTIES
    VERSION ${SPEX_VERSION_MAJOR}.${SPEX_VERSION_MINOR}.${SPEX_VERSION_SUB}
    C_STANDARD_REQUIRED 11
    OUTPUT_NAME spex
    SOVERSION ${SPEX_VERSION_MAJOR} )
endif ( )

#-------------------------------------------------------------------------------
# add the library dependencies
#-------------------------------------------------------------------------------

# suitesparseconfig:
target_link_libraries ( spex PUBLIC ${SUITESPARSE_CONFIG_LIBRARIES} )
if ( NOT NSTATIC )
target_link_libraries ( spex_static PUBLIC ${SUITESPARSE_CONFIG_LIBRARIES} )
endif ( )

# AMD:
target_link_libraries ( spex PUBLIC ${AMD_LIBRARIES} )
if ( NOT NSTATIC )
    target_link_libraries ( spex_static PUBLIC ${AMD_STATIC} )
endif ( )

# COLAMD:
target_link_libraries ( spex PUBLIC ${COLAMD_LIBRARIES} )
if ( NOT NSTATIC )
    target_link_libraries ( spex_static PUBLIC ${COLAMD_STATIC} )
endif ( )

# GMP:
target_link_libraries ( spex PUBLIC ${GMP_LIBRARIES} )
if ( NOT NSTATIC )
    target_link_libraries ( spex_static PUBLIC ${GMP_STATIC} )
endif ( )

# MPFR:
target_link_libraries ( spex PUBLIC ${MPFR_LIBRARIES} )
if ( NOT NSTATIC )
    target_link_libraries ( spex_static PUBLIC ${MPFR_STATIC} )
endif ( )

# libm:
if ( NOT MSVC )
    target_link_libraries ( spex PUBLIC m )
    if ( NOT NSTATIC )
        target_link_libraries ( spex_static PUBLIC m )
    endif ( )
endif ( )

#-------------------------------------------------------------------------------
# SPEX installation location
#-------------------------------------------------------------------------------

if ( GLOBAL_INSTALL )
    # install in /usr/local/lib and /usr/local/include.
    # requires "sudo make install"
    message ( STATUS "Installation will be system-wide (requires 'sudo make install')" )
    install ( TARGETS spex
        LIBRARY       DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} )
    install ( FILES ${CMAKE_SOURCE_DIR}/cmake_modules/FindSPEX.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SuiteSparse 
        COMPONENT Development )
    if ( NOT NSTATIC )
    install ( TARGETS spex_static
        ARCHIVE       DESTINATION ${CMAKE_INSTALL_LIBDIR} )
    endif ( )
endif ( )

if ( INSIDE_SUITESPARSE )
    # also install in SuiteSparse/lib and SuiteSparse/include;
    # does not require "sudo make install", just "make install"
    message ( STATUS "Installation in ../lib and ../include," )
    message ( STATUS "  with 'make local ; make install'. No 'sudo' required." )
    install ( TARGETS spex
        LIBRARY       DESTINATION ${SUITESPARSE_LIBDIR}
        PUBLIC_HEADER DESTINATION ${SUITESPARSE_INCLUDEDIR} )
    install ( FILES ${CMAKE_SOURCE_DIR}/cmake_modules/FindSPEX.cmake
        DESTINATION ${SUITESPARSE_LIBDIR}/cmake/SuiteSparse 
        COMPONENT Development )
    if ( NOT NSTATIC )
    install ( TARGETS spex_static
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

    message ( STATUS "Also compiling the demos in SPEX/Demo" )

    #---------------------------------------------------------------------------
    # Demo programs
    #---------------------------------------------------------------------------

    add_executable ( spexlu_demo "SPEX_Left_LU/Demo/spexlu_demo.c"
                                 "SPEX_Left_LU/Demo/demos.c" )
    add_executable ( example     "SPEX_Left_LU/Demo/example.c" )
    add_executable ( example2    "SPEX_Left_LU/Demo/example2.c"
                                 "SPEX_Left_LU/Demo/demos.c" )

    # Libraries required for Demo programs
    target_link_libraries ( spexlu_demo PUBLIC spex )
    target_link_libraries ( example  PUBLIC spex )
    target_link_libraries ( example2 PUBLIC spex )

else ( )

    message ( STATUS "Skipping the demos in SPEX/Demo" )

endif ( )

#-------------------------------------------------------------------------------
# report status
#-------------------------------------------------------------------------------

include ( SuiteSparseReport )

