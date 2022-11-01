#-------------------------------------------------------------------------------
# SuiteSparse/SLIP_LU/CMakeLists.txt:  cmake for SLIP_LU
#-------------------------------------------------------------------------------

# Copyright (c) 1996-2022, Timothy A. Davis, Patrick Amestoy, Iain Duff.
# All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

#-------------------------------------------------------------------------------
# get the version
#-------------------------------------------------------------------------------

cmake_minimum_required ( VERSION 3.19 )

set ( SLIP_LU_DATE "Nov 4, 2022" )
set ( SLIP_LU_VERSION_MAJOR 2 )
set ( SLIP_LU_VERSION_MINOR 0 )
set ( SLIP_LU_VERSION_SUB   0 )

message ( STATUS "Building SLIP_LU version: v"
    ${SLIP_LU_VERSION_MAJOR}.
    ${SLIP_LU_VERSION_MINOR}.
    ${SLIP_LU_VERSION_SUB} " (" ${SLIP_LU_DATE} ")" )

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

project ( sliplu
    VERSION "${SLIP_LU_VERSION_MAJOR}.${SLIP_LU_VERSION_MINOR}.${SLIP_LU_VERSION_SUB}"
    LANGUAGES C )

#-------------------------------------------------------------------------------
# find library dependencies
#-------------------------------------------------------------------------------

find_package ( SuiteSparse_config 6.0.0 REQUIRED )
find_package ( COLAMD 3.0.0 REQUIRED )
find_package ( AMD 3.0.0 REQUIRED )
find_package ( GMP 6.1.2 REQUIRED )
find_package ( MPFR 4.0.2 REQUIRED )

#-------------------------------------------------------------------------------
# configure files
#-------------------------------------------------------------------------------

configure_file ( "Config/SLIP_LU.h.in" "${PROJECT_SOURCE_DIR}/Include/SLIP_LU.h")
configure_file ( "Config/SLIP_LU_version.tex.in" "${PROJECT_SOURCE_DIR}/Doc/SLIP_LU_version.tex")

#-------------------------------------------------------------------------------
# include directories
#-------------------------------------------------------------------------------

include_directories ( Source Include Demo ${SUITESPARSE_CONFIG_INCLUDE_DIR} 
    ${GMP_INCLUDE_DIR} ${MPFR_INCLUDE_DIR}
    ${AMD_INCLUDE_DIR} ${COLAMD_INCLUDE_DIR} )

#-------------------------------------------------------------------------------
# dynamic sliplu library properties
#-------------------------------------------------------------------------------

file ( GLOB SLIP_LU_SOURCES "Source/*.c" )

add_library ( sliplu SHARED ${SLIP_LU_SOURCES} )

set_target_properties ( sliplu PROPERTIES
    VERSION ${SLIP_LU_VERSION_MAJOR}.${SLIP_LU_VERSION_MINOR}.${SLIP_LU_VERSION_SUB}
    C_STANDARD_REQUIRED 11
    SOVERSION ${SLIP_LU_VERSION_MAJOR}
    PUBLIC_HEADER "Include/SLIP_LU.h" )

#-------------------------------------------------------------------------------
# static sliplu library properties
#-------------------------------------------------------------------------------

add_library ( sliplu_static STATIC ${SLIP_LU_SOURCES} )

set_target_properties ( sliplu_static PROPERTIES
    VERSION ${SLIP_LU_VERSION_MAJOR}.${SLIP_LU_VERSION_MINOR}.${SLIP_LU_VERSION_SUB}
    C_STANDARD_REQUIRED 11
    OUTPUT_NAME sliplu
    SOVERSION ${SLIP_LU_VERSION_MAJOR} )

#-------------------------------------------------------------------------------
# add the library dependencies
#-------------------------------------------------------------------------------

# suitesparseconfig:
target_link_libraries ( sliplu PUBLIC ${SUITESPARSE_CONFIG_LIBRARIES} )
target_link_libraries ( sliplu_static PUBLIC ${SUITESPARSE_CONFIG_LIBRARIES} )

# AMD:
target_link_libraries ( sliplu PUBLIC ${AMD_LIBRARIES} )
target_link_libraries ( sliplu_static PUBLIC ${AMD_LIBRARIES} )

# COLAMD:
target_link_libraries ( sliplu PUBLIC ${COLAMD_LIBRARIES} )
target_link_libraries ( sliplu_static PUBLIC ${COLAMD_LIBRARIES} )

# GMP:
target_link_libraries ( sliplu PUBLIC ${GMP_LIBRARIES} )
target_link_libraries ( sliplu_static PUBLIC ${GMP_LIBRARIES} )

# MPFR:
target_link_libraries ( sliplu PUBLIC ${MPFR_LIBRARIES} )
target_link_libraries ( sliplu_static PUBLIC ${MPFR_LIBRARIES} )

# libm:
if ( NOT "${CMAKE_C_COMPILER_ID}" STREQUAL "MSVC" )
    target_link_libraries ( sliplu PUBLIC m )
    target_link_libraries ( sliplu_static PUBLIC m )
endif ( )

#-------------------------------------------------------------------------------
# SLIP_LU installation location
#-------------------------------------------------------------------------------

# use "cmake -DSUITESPARSE_LOCAL=1" to install only in SuiteSparse/lib and
# SuiteSparse/include.  

if ( NOT SUITESPARSE_LOCAL )
    # install in /usr/local/lib and /usr/local/include.
    # requires "sudo make install"
    message ( STATUS "Installation will be system-wide (requires 'sudo make install')" )
    install ( TARGETS sliplu
        LIBRARY       DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} )
    install ( TARGETS sliplu_static
        ARCHIVE       DESTINATION ${CMAKE_INSTALL_LIBDIR} )
endif ( )

if ( INSIDE_SUITESPARSE )
    # also install in SuiteSparse/lib and SuiteSparse/include;
    # does not require "sudo make install", just "make install"
    message ( STATUS "Installation in ../lib and ../include," )
    message ( STATUS "  with 'make local ; make install'. No 'sudo' required." )
    install ( TARGETS sliplu
        LIBRARY       DESTINATION ${SUITESPARSE_LIBDIR}
        PUBLIC_HEADER DESTINATION ${SUITESPARSE_INCLUDEDIR} )
    install ( TARGETS sliplu_static
        ARCHIVE       DESTINATION ${SUITESPARSE_LIBDIR} )
endif ( )

#-------------------------------------------------------------------------------
# Demo library and programs
#-------------------------------------------------------------------------------

if ( DEMO )

    #---------------------------------------------------------------------------
    # demo library
    #---------------------------------------------------------------------------

    message ( STATUS "Also compiling the demos in SLIP_LU/Demo" )

    #---------------------------------------------------------------------------
    # Demo programs
    #---------------------------------------------------------------------------

    add_executable ( SLIPLU   "Demo/SLIPLU.c" "Demo/demos.c" )
    add_executable ( example  "Demo/example.c" )
    add_executable ( example2 "Demo/example2.c" "Demo/demos.c" )

    # Libraries required for Demo programs
    target_link_libraries ( SLIPLU   PUBLIC sliplu )
    target_link_libraries ( example  PUBLIC sliplu )
    target_link_libraries ( example2 PUBLIC sliplu )

else ( )

    message ( STATUS "Skipping the demos in SLIP_LU/Demo" )

endif ( )

#-------------------------------------------------------------------------------
# report status
#-------------------------------------------------------------------------------

include ( SuiteSparseReport )

