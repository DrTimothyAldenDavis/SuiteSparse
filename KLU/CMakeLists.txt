#-------------------------------------------------------------------------------
# SuiteSparse/KLU/CMakeLists.txt:  cmake for KLU
#-------------------------------------------------------------------------------

# Copyright (c) 1995-2022, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: GPL-2.0+

#-------------------------------------------------------------------------------
# get the version
#-------------------------------------------------------------------------------

cmake_minimum_required ( VERSION 3.19 )

set ( KLU_DATE "Sept FIXME, 2022" )
set ( KLU_VERSION_MAJOR 2 )
set ( KLU_VERSION_MINOR 0 )
set ( KLU_VERSION_SUB   0 )

message ( STATUS "Building KLU version: v"
    ${KLU_VERSION_MAJOR}.
    ${KLU_VERSION_MINOR}.
    ${KLU_VERSION_SUB} " (" ${KLU_DATE} ")" )

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

project ( klu
    VERSION "${KLU_VERSION_MAJOR}.${KLU_VERSION_MINOR}.${KLU_VERSION_SUB}"
    LANGUAGES C ) # <-- add CXX here for packages that use C++

#-------------------------------------------------------------------------------
# find library dependencies
#-------------------------------------------------------------------------------

enable_language ( C Fortran )
find_package ( SuiteSparse_config 6.0.0 REQUIRED )
find_package ( BTF 2.0.0 REQUIRED )
find_package ( AMD 3.0.0 REQUIRED )
find_package ( COLAMD 3.0.0 REQUIRED )

#-------------------------------------------------------------------------------
# configure files
#-------------------------------------------------------------------------------

configure_file ( "Config/klu.h.in" "${PROJECT_SOURCE_DIR}/Include/klu.h")
configure_file ( "Config/klu_version.tex.in" "${PROJECT_SOURCE_DIR}/Doc/klu_version.tex")

#-------------------------------------------------------------------------------
# report status and set compile flags
#-------------------------------------------------------------------------------

if ( ${CMAKE_BUILD_TYPE} STREQUAL "Debug" )
    message ( STATUS "CMAKE C Flags debug:     ${CMAKE_C_FLAGS_DEBUG} ")
#   message ( STATUS "CMAKE C++ Flags debug:   ${CMAKE_CXX_FLAGS_DEBUG} ")
else ( )
    message ( STATUS "CMAKE C Flags release:   ${CMAKE_C_FLAGS_RELEASE} ")
#   message ( STATUS "CMAKE C++ Flags release: ${CMAKE_CXX_FLAGS_RELEASE} ")
endif ( )

message ( STATUS "CMAKE C compiler:        ${CMAKE_C_COMPILER_ID} ")

# message ( STATUS "CMAKE C++ compiler:      " ${CMAKE_CXX_COMPILER_ID} )

#-------------------------------------------------------------------------------
# include directories
#-------------------------------------------------------------------------------

include_directories ( Source Include ${SUITESPARSE_CONFIG_INCLUDE_DIR}
    ${AMD_INCLUDE_DIR} ${COLAMD_INCLUDE_DIR} ${BTF_INCLUDE_DIR} )

#-------------------------------------------------------------------------------
# dynamic klu library properties
#-------------------------------------------------------------------------------

file ( GLOB KLU_SOURCES "Source/*.c" )

add_library ( klu SHARED ${KLU_SOURCES} )

SET_TARGET_PROPERTIES ( klu PROPERTIES
    VERSION ${KLU_VERSION_MAJOR}.${KLU_VERSION_MINOR}.${KLU_VERSION_SUB}
    C_STANDARD_REQUIRED 11
    SOVERSION ${KLU_VERSION_MAJOR}
    PUBLIC_HEADER "Include/klu.h" )

#-------------------------------------------------------------------------------
# static klu library properties
#-------------------------------------------------------------------------------

add_library ( klu_static STATIC ${KLU_SOURCES} )

SET_TARGET_PROPERTIES ( klu_static PROPERTIES
    VERSION ${KLU_VERSION_MAJOR}.${KLU_VERSION_MINOR}.${KLU_VERSION_SUB}
    C_STANDARD_REQUIRED 11
    OUTPUT_NAME klu
    SOVERSION ${KLU_VERSION_MAJOR} )

#-------------------------------------------------------------------------------
# add the library dependencies
#-------------------------------------------------------------------------------

# suitesparseconfig:
target_link_libraries ( klu PUBLIC ${SUITESPARSE_CONFIG_LIBRARIES} )
target_link_libraries ( klu_static PUBLIC ${SUITESPARSE_CONFIG_LIBRARIES} )

# libm:
if ( NOT "${CMAKE_C_COMPILER_ID}" STREQUAL "MSVC" )
    target_link_libraries ( klu PUBLIC m )
    target_link_libraries ( klu_static PUBLIC m )
endif ( )

# amd:
target_link_libraries ( klu PUBLIC ${AMD_LIBRARIES} )
target_link_libraries ( klu_static PUBLIC ${AMD_LIBRARIES} )

# colamd:
target_link_libraries ( klu PUBLIC ${COLAMD_LIBRARIES} )
target_link_libraries ( klu_static PUBLIC ${COLAMD_LIBRARIES} )

# btf:
target_link_libraries ( klu PUBLIC ${BTF_LIBRARIES} )
target_link_libraries ( klu_static PUBLIC ${BTF_LIBRARIES} )

#-------------------------------------------------------------------------------
# KLU installation location
#---------------------------------------------------------------Z--------------

# use "cmake -DLOCAL=1" to install only in SuiteSparse/lib and
# SuiteSparse/include.  

if ( NOT LOCAL )
    # install in /usr/local/lib and /usr/local/include.
    # requires "sudo make install"
    message ( STATUS "Installation will be system-wide (requires 'sudo make install')" )
    install ( TARGETS klu
        LIBRARY       DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} )
    install ( TARGETS klu_static
        ARCHIVE       DESTINATION ${CMAKE_INSTALL_LIBDIR} )
endif ( )

if ( INSIDE_SUITESPARSE )
    # also install in SuiteSparse/lib and SuiteSparse/include;
    # does not require "sudo make install", just "make install"
    message ( STATUS "Installation in ../lib and ../include only," )
    message ( STATUS "  with 'make install'. No 'sudo' required." )
    install ( TARGETS klu
        LIBRARY       DESTINATION ${LOCAL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${LOCAL_INCLUDEDIR} )
    install ( TARGETS klu_static
        ARCHIVE       DESTINATION ${LOCAL_LIBDIR} )
endif ( )

#-------------------------------------------------------------------------------
# Demo library and programs
#-------------------------------------------------------------------------------

if ( DEMO )

    #---------------------------------------------------------------------------
    # demo library
    #---------------------------------------------------------------------------

    message ( STATUS "Also compiling the demos in KLU/Demo" )

    #---------------------------------------------------------------------------
    # Demo programs
    #---------------------------------------------------------------------------

    add_executable ( klu_simple  "Demo/klu_simple.c" )
#   FIXME requires CHOLMOD:
#   add_executable ( kludemo     "Demo/kludemo.c" )
#   add_executable ( kluldemo    "Demo/kluldemo.c" )

    # Libraries required for Demo programs
    target_link_libraries ( klu_simple  PUBLIC klu )
#   target_link_libraries ( kludemo     PUBLIC klu )
#   target_link_libraries ( kluldemo    PUBLIC klu )

else ( )

    message ( STATUS "Skipping the demos in KLU/Demo" )

endif ( )

