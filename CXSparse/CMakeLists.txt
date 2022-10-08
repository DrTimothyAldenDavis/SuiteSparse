#-------------------------------------------------------------------------------
# CXSparse/CMakeLists.txt:  cmake for CXSparse
#-------------------------------------------------------------------------------

# Copyright (c) 2022, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: LGPL-2.1+

#-------------------------------------------------------------------------------
# get the version
#-------------------------------------------------------------------------------

cmake_minimum_required ( VERSION 3.22 )

set ( CXSPARSE_DATE "Oct 31, 2022" ) # FIXME
set ( CXSPARSE_VERSION_MAJOR 4 )
set ( CXSPARSE_VERSION_MINOR 0 )
set ( CXSPARSE_VERSION_SUB   0 )

message ( STATUS "Building CXSparse version: v"
    ${CXSPARSE_VERSION_MAJOR}.
    ${CXSPARSE_VERSION_MINOR}.
    ${CXSPARSE_VERSION_SUB} " (" ${CXSPARSE_DATE} ")" )

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

project ( cxsparse
    VERSION "${CXSPARSE_VERSION_MAJOR}.${CXSPARSE_VERSION_MINOR}.${CXSPARSE_VERSION_SUB}"
    LANGUAGES C )

#-------------------------------------------------------------------------------
# find library dependencies
#-------------------------------------------------------------------------------

find_package ( SuiteSparse_config 6.0.0 REQUIRED )

#-------------------------------------------------------------------------------
# Configure cs.h with version number
#-------------------------------------------------------------------------------

configure_file ( "Config/cs.h.in" "${PROJECT_SOURCE_DIR}/Include/cs.h" )

#-------------------------------------------------------------------------------
# include directories
#-------------------------------------------------------------------------------

include_directories ( Source Include ${SUITESPARSE_CONFIG_INCLUDE_DIR} )

#-------------------------------------------------------------------------------
# dynamic cxsparse library properties
#-------------------------------------------------------------------------------

file ( GLOB CXSPARSE_SOURCES "Source/*.c" )

add_library ( cxsparse SHARED ${CXSPARSE_SOURCES} )

SET_TARGET_PROPERTIES ( cxsparse PROPERTIES
    VERSION ${CXSPARSE_VERSION_MAJOR}.${CXSPARSE_VERSION_MINOR}.${CXSPARSE_VERSION_SUB}
    C_STANDARD_REQUIRED 11
    SOVERSION ${CXSPARSE_VERSION_MAJOR}
    PUBLIC_HEADER "Include/cs.h" )

#-------------------------------------------------------------------------------
# static cxsparse library properties
#-------------------------------------------------------------------------------

add_library ( cxsparse_static STATIC ${CXSPARSE_SOURCES} )

SET_TARGET_PROPERTIES ( cxsparse_static PROPERTIES
    VERSION ${CXSPARSE_VERSION_MAJOR}.${CXSPARSE_VERSION_MINOR}.${CXSPARSE_VERSION_SUB}
    OUTPUT_NAME cxsparse
    C_STANDARD_REQUIRED 11
    SOVERSION ${CXSPARSE_VERSION_MAJOR} )

#-------------------------------------------------------------------------------
# add the library dependencies
#-------------------------------------------------------------------------------

target_link_libraries ( cxsparse PUBLIC ${SUITESPARSE_CONFIG_LIBRARY} )
target_link_libraries ( cxsparse_static PUBLIC ${SUITESPARSE_CONFIG_LIBRARY} )

# libm:
if ( NOT "${CMAKE_C_COMPILER_ID}" STREQUAL "MSVC" )
    target_link_libraries ( cxsparse PUBLIC m )
    target_link_libraries ( cxsparse_static PUBLIC m )
endif ( )

#-------------------------------------------------------------------------------
# installation location
#-------------------------------------------------------------------------------

# use "cmake -DSUITESPARSE_LOCAL=1" to install only in SuiteSparse/lib and
# SuiteSparse/include.  

if ( NOT SUITESPARSE_LOCAL )
    # install in /usr/local/lib and /usr/local/include.
    # requires "sudo make install"
    message ( STATUS "Installation will be system-wide (requires 'sudo make install')" )
    install ( TARGETS cxsparse
        LIBRARY       DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} )
    install ( TARGETS cxsparse_static
        ARCHIVE       DESTINATION ${CMAKE_INSTALL_LIBDIR} )
endif ( )

if ( INSIDE_SUITESPARSE )
    # also install in SuiteSparse/lib and SuiteSparse/include;
    # does not require "sudo make install", just "make install"
    message ( STATUS "Installation in ../lib and ../include," )
    message ( STATUS "  with 'make local ; make install'. No 'sudo' required." )
    install ( TARGETS cxsparse
        LIBRARY       DESTINATION ${SUITESPARSE_LIBDIR}
        PUBLIC_HEADER DESTINATION ${SUITESPARSE_INCLUDEDIR} )
    install ( TARGETS cxsparse_static
        ARCHIVE       DESTINATION ${SUITESPARSE_LIBDIR} )
endif ( )

#-------------------------------------------------------------------------------
# Demo library and programs
#-------------------------------------------------------------------------------

if ( DEMO )

    #---------------------------------------------------------------------------
    # demo library
    #---------------------------------------------------------------------------

    message ( STATUS "Also compiling the demos in CXSparse/Demo" )

    #---------------------------------------------------------------------------
    # Demo programs
    #---------------------------------------------------------------------------

    add_executable ( cs_demo1 "Demo/cs_demo1.c" "Demo/cs_demo.c" )
    add_executable ( cs_demo2 "Demo/cs_demo2.c" "Demo/cs_demo.c" )
    add_executable ( cs_demo3 "Demo/cs_demo3.c" "Demo/cs_demo.c" )

    add_executable ( cs_di_demo1 "Demo/cs_di_demo1.c" "Demo/cs_di_demo.c" )
    add_executable ( cs_di_demo2 "Demo/cs_di_demo2.c" "Demo/cs_di_demo.c" )
    add_executable ( cs_di_demo3 "Demo/cs_di_demo3.c" "Demo/cs_di_demo.c" )

    add_executable ( cs_dl_demo1 "Demo/cs_dl_demo1.c" "Demo/cs_dl_demo.c" )
    add_executable ( cs_dl_demo2 "Demo/cs_dl_demo2.c" "Demo/cs_dl_demo.c" )
    add_executable ( cs_dl_demo3 "Demo/cs_dl_demo3.c" "Demo/cs_dl_demo.c" )

    add_executable ( cs_ci_demo1 "Demo/cs_ci_demo1.c" "Demo/cs_ci_demo.c" )
    add_executable ( cs_ci_demo2 "Demo/cs_ci_demo2.c" "Demo/cs_ci_demo.c" )
    add_executable ( cs_ci_demo3 "Demo/cs_ci_demo3.c" "Demo/cs_ci_demo.c" )

    add_executable ( cs_cl_demo1 "Demo/cs_cl_demo1.c" "Demo/cs_cl_demo.c" )
    add_executable ( cs_cl_demo2 "Demo/cs_cl_demo2.c" "Demo/cs_cl_demo.c" )
    add_executable ( cs_cl_demo3 "Demo/cs_cl_demo3.c" "Demo/cs_cl_demo.c" )

    add_executable ( cs_idemo "Demo/cs_idemo.c" )
    add_executable ( cs_ldemo "Demo/cs_ldemo.c" )

    # Libraries required for Demo programs
    target_link_libraries ( cs_demo1 PUBLIC cxsparse )
    target_link_libraries ( cs_demo2 PUBLIC cxsparse )
    target_link_libraries ( cs_demo3 PUBLIC cxsparse )

    target_link_libraries ( cs_di_demo1 PUBLIC cxsparse )
    target_link_libraries ( cs_di_demo2 PUBLIC cxsparse )
    target_link_libraries ( cs_di_demo3 PUBLIC cxsparse )

    target_link_libraries ( cs_dl_demo1 PUBLIC cxsparse )
    target_link_libraries ( cs_dl_demo2 PUBLIC cxsparse )
    target_link_libraries ( cs_dl_demo3 PUBLIC cxsparse )

    target_link_libraries ( cs_ci_demo1 PUBLIC cxsparse )
    target_link_libraries ( cs_ci_demo2 PUBLIC cxsparse )
    target_link_libraries ( cs_ci_demo3 PUBLIC cxsparse )

    target_link_libraries ( cs_cl_demo1 PUBLIC cxsparse )
    target_link_libraries ( cs_cl_demo2 PUBLIC cxsparse )
    target_link_libraries ( cs_cl_demo3 PUBLIC cxsparse )

    target_link_libraries ( cs_idemo PUBLIC cxsparse )
    target_link_libraries ( cs_ldemo PUBLIC cxsparse )

else ( )

    message ( STATUS "Skipping the demos in CXSparse/Demo" )

endif ( )

#-------------------------------------------------------------------------------
# report status
#-------------------------------------------------------------------------------

include ( SuiteSparseReport )

