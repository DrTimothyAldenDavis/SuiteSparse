#-------------------------------------------------------------------------------
# SuiteSparse/TestConfig/CHOLMOD/CMakeLists.txt:  cmake for project linking to CHOLMOD
#-------------------------------------------------------------------------------

# Copyright (c) 2024, Timothy A. Davis, All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

cmake_minimum_required ( VERSION 3.22 )

#-------------------------------------------------------------------------------
# configure options
#-------------------------------------------------------------------------------

option ( BUILD_SHARED_LIBS "OFF: do not build shared libraries.  ON (default): build shared libraries" ON )
option ( BUILD_STATIC_LIBS "OFF: do not build static libraries.  ON (default): build static libraries" ON )

#-------------------------------------------------------------------------------
# variables
#-------------------------------------------------------------------------------

set ( TEST_PROJECT_NAME cholmod_demo )
set ( TEST_PACKAGE_NAME CHOLMOD )
set ( TEST_SHARED_BIN cholmod_demo )
set ( TEST_STATIC_BIN cholmod_demo_static )
set ( TEST_IMPORT_TARGET SuiteSparse::CHOLMOD )
set ( TEST_IMPORT_TARGET_STATIC SuiteSparse::CHOLMOD_static )

#-------------------------------------------------------------------------------
# define project
#-------------------------------------------------------------------------------

project ( ${TEST_PROJECT_NAME} LANGUAGES C CXX )

#-------------------------------------------------------------------------------
# find library dependencies
#-------------------------------------------------------------------------------

find_package ( ${TEST_PACKAGE_NAME} REQUIRED )

#-------------------------------------------------------------------------------
# dynamically linked executable properties
#-------------------------------------------------------------------------------

file ( GLOB MY_SOURCES "*.cc" )
if ( BUILD_SHARED_LIBS )
    add_executable ( ${TEST_SHARED_BIN} ${MY_SOURCES} )
    set_target_properties ( ${TEST_SHARED_BIN} PROPERTIES
        C_STANDARD 11
        C_STANDARD_REQUIRED ON )
endif ( )

#-------------------------------------------------------------------------------
# staticcally linked executable properties
#-------------------------------------------------------------------------------

if ( BUILD_STATIC_LIBS )
    add_executable ( ${TEST_STATIC_BIN} ${MY_SOURCES} )
    set_target_properties ( ${TEST_STATIC_BIN} PROPERTIES
        C_STANDARD 11
        C_STANDARD_REQUIRED ON )
endif ( )

#-------------------------------------------------------------------------------
# add library dependency
#-------------------------------------------------------------------------------

if ( BUILD_SHARED_LIBS )
    target_link_libraries ( ${TEST_SHARED_BIN} PRIVATE ${TEST_IMPORT_TARGET} )
endif ( )
if ( BUILD_STATIC_LIBS )
    if ( TARGET ${TEST_IMPORT_TARGET_STATIC} )
        target_link_libraries ( ${TEST_STATIC_BIN} PUBLIC ${TEST_IMPORT_TARGET_STATIC} )
    else ( )
        target_link_libraries ( ${TEST_STATIC_BIN} PUBLIC ${TEST_IMPORT_TARGET} )
    endif ( )
endif ( )

#-------------------------------------------------------------------------------
# installation location
#-------------------------------------------------------------------------------

include ( GNUInstallDirs )

if ( BUILD_SHARED_LIBS )
    install ( TARGETS ${TEST_SHARED_BIN}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} )
endif ( )
if ( BUILD_STATIC_LIBS )
    install ( TARGETS ${TEST_STATIC_BIN}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} )
endif ( )
