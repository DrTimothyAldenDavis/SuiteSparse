#-------------------------------------------------------------------------------
# SuiteSparse/SuiteSparse_config/cmake_modules/FindCOLAMD.cmake
#-------------------------------------------------------------------------------

# Copyright (c) 2022, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

# Finds the COLAMD include file and compiled library and sets:

# COLAMD_INCLUDE_DIR - where to find colamd.h
# COLAMD_LIBRARY     - dynamic COLAMD library
# COLAMD_STATIC      - static COLAMD library
# COLAMD_LIBRARIES   - libraries when using COLAMD
# COLAMD_FOUND       - true if COLAMD found

# set ``COLAMD_ROOT`` to a COLAMD installation root to
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

# include files for COLAMD
find_path ( COLAMD_INCLUDE_DIR
    NAMES colamd.h
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/COLAMD
    HINTS ${CMAKE_SOURCE_DIR}/../COLAMD
    PATH_SUFFIXES include Include
)

# dynamic COLAMD library
find_library ( COLAMD_LIBRARY
    NAMES colamd
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/COLAMD
    HINTS ${CMAKE_SOURCE_DIR}/../COLAMD
    PATH_SUFFIXES lib build
)

# static COLAMD library
set ( save ${CMAKE_FIND_LIBRARY_SUFFIXES} )
set ( CMAKE_FIND_LIBRARY_SUFFIXES ${STATIC_SUFFIX} ${CMAKE_FIND_LIBRARY_SUFFIXES} )
find_library ( COLAMD_LIBRARY
    NAMES colamd
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/COLAMD
    HINTS ${CMAKE_SOURCE_DIR}/../COLAMD
    PATH_SUFFIXES lib build
)
set ( ${CMAKE_FIND_LIBRARY_SUFFIXES} save )

# get version of the library from the dynamic library name
get_filename_component ( COLAMD_LIBRARY  ${COLAMD_LIBRARY} REALPATH )
get_filename_component ( COLAMD_FILENAME ${COLAMD_LIBRARY} NAME )
string (
    REGEX MATCH "[0-9]+.[0-9]+.[0-9]+"
    COLAMD_VERSION
    ${COLAMD_FILENAME}
)

if ( NOT COLAMD_VERSION )
    # if the version does not appear in the filename, read the include file
    foreach (_VERSION MAIN_VERSION SUB_VERSION SUBSUB_VERSION)
        file (STRINGS ${COLAMD_INCLUDE_DIR}/colamd.h _VERSION_LINE REGEX "define[ ]+COLAMD_${_VERSION}")
        if (_VERSION_LINE)
            string (REGEX REPLACE ".*define[ ]+COLAMD_${_VERSION}[ ]+([0-9]*).*" "\\1" _COLAMD_${_VERSION} "${_VERSION_LINE}")
        endif ()
        unset (_VERSION_LINE)
    endforeach ()
    set (COLAMD_VERSION "${_COLAMD_MAIN_VERSION}.${_COLAMD_SUB_VERSION}.${_COLAMD_SUBSUB_VERSION}")
endif ( )

set (COLAMD_LIBRARIES ${COLAMD_LIBRARY})

include (FindPackageHandleStandardArgs)

find_package_handle_standard_args ( COLAMD
    REQUIRED_VARS COLAMD_LIBRARIES COLAMD_INCLUDE_DIR
    VERSION_VAR COLAMD_VERSION
)

mark_as_advanced (
    COLAMD_INCLUDE_DIR
    COLAMD_LIBRARY
    COLAMD_STATIC
    COLAMD_LIBRARIES
)

if ( COLAMD_FOUND )
    message ( STATUS "COLAMD version:     ${COLAMD_VERSION}" )
    message ( STATUS "COLAMD include dir: ${COLAMD_INCLUDE_DIR}" )
    message ( STATUS "COLAMD dynamic:     ${COLAMD_LIBRARY}" )
    message ( STATUS "COLAMD static:      ${COLAMD_STATIC}" )
else ( )
    message ( STATUS "COLAMD not found" )
endif ( )

