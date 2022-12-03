#-------------------------------------------------------------------------------
# SuiteSparse/SuiteSparse_config/cmake_modules/FindCAMD.cmake
#-------------------------------------------------------------------------------

# Copyright (c) 2022, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

# Finds the CAMD include file and compiled library and sets:

# CAMD_INCLUDE_DIR - where to find camd.h
# CAMD_LIBRARY     - dynamic CAMD library
# CAMD_STATIC      - static CAMD library
# CAMD_LIBRARIES   - libraries when using CAMD
# CAMD_FOUND       - true if CAMD found

# set ``CAMD_ROOT`` to a CAMD installation root to
# tell this module where to look.

# To use this file in your application, copy this file into MyApp/cmake_modules
# where MyApp is your application and add the following to your
# MyApp/CMakeLists.txt file:
#
#   set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake_modules")
#
# or, assuming MyApp and SuiteSparse sit side-by-side in a common folder, you
# can leave this file in place and use this command (revise as needed):
#
#   set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
#       "${CMAKE_SOURCE_DIR}/../SuiteSparse/SuiteSparse_config/cmake_modules")

#-------------------------------------------------------------------------------

# include files for CAMD
find_path ( CAMD_INCLUDE_DIR
    NAMES camd.h
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/CAMD
    HINTS ${CMAKE_SOURCE_DIR}/../CAMD
    PATH_SUFFIXES include Include
)

# dynamic CAMD library
find_library ( CAMD_LIBRARY
    NAMES camd
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/CAMD
    HINTS ${CMAKE_SOURCE_DIR}/../CAMD
    PATH_SUFFIXES lib build
)

# static CAMD library
set ( save ${CMAKE_FIND_LIBRARY_SUFFIXES} )
set ( CMAKE_FIND_LIBRARY_SUFFIXES ${STATIC_SUFFIX} ${CMAKE_FIND_LIBRARY_SUFFIXES} )
find_library ( CAMD_LIBRARY
    NAMES camd
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/CAMD
    HINTS ${CMAKE_SOURCE_DIR}/../CAMD
    PATH_SUFFIXES lib build
)
set ( ${CMAKE_FIND_LIBRARY_SUFFIXES} save )

# get version of the library from the dynamic library filename
get_filename_component ( CAMD_LIBRARY  ${CAMD_LIBRARY} REALPATH )
get_filename_component ( CAMD_FILENAME ${CAMD_LIBRARY} NAME )
string (
    REGEX MATCH "[0-9]+.[0-9]+.[0-9]+"
    CAMD_VERSION
    ${CAMD_FILENAME}
)

if ( NOT CAMD_VERSION )
    # if the version does not appear in the filename, read the include file
    foreach ( _VERSION MAIN_VERSION SUB_VERSION SUBSUB_VERSION )
        file ( STRINGS ${CAMD_INCLUDE_DIR}/camd.h _VERSION_LINE REGEX "define[ ]+CAMD_${_VERSION}" )
        if ( _VERSION_LINE )
            string ( REGEX REPLACE ".*define[ ]+CAMD_${_VERSION}[ ]+([0-9]*).*" "\\1" _CAMD_${_VERSION} "${_VERSION_LINE}" )
        endif ( )
        unset ( _VERSION_LINE )
    endforeach ( )
    set ( CAMD_VERSION "${_CAMD_MAIN_VERSION}.${_CAMD_SUB_VERSION}.${_CAMD_SUBSUB_VERSION}" )
endif ( )

set ( CAMD_LIBRARIES ${CAMD_LIBRARY} )

include (FindPackageHandleStandardArgs)

find_package_handle_standard_args ( CAMD
    REQUIRED_VARS CAMD_LIBRARIES CAMD_INCLUDE_DIR
    VERSION_VAR CAMD_VERSION
)

mark_as_advanced (
    CAMD_INCLUDE_DIR
    CAMD_LIBRARY
    CAMD_STATIC
    CAMD_LIBRARIES
)

if ( CAMD_FOUND )
    message ( STATUS "CAMD version:     ${CAMD_VERSION}" )
    message ( STATUS "CAMD include dir: ${CAMD_INCLUDE_DIR}" )
    message ( STATUS "CAMD dynamic:     ${CAMD_LIBRARY}" )
    message ( STATUS "CAMD static:      ${CAMD_STATIC}" )
else ( )
    message ( STATUS "CAMD not found" )
endif ( )

