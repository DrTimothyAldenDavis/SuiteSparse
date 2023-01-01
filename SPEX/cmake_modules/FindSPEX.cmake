#-------------------------------------------------------------------------------
# SuiteSparse/SPEX/cmake_modules/FindSPEX.cmake
#-------------------------------------------------------------------------------

# The following copyright and license applies to just this file only, not to
# the library itself:
# FindSPEX.cmake, Copyright (c) 2022-2023, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

#-------------------------------------------------------------------------------

# Finds the SPEX include file and compiled library and sets:

# SPEX_INCLUDE_DIR - where to find SPEX.h
# SPEX_LIBRARY     - dynamic SPEX library
# SPEX_STATIC      - static SPEX library
# SPEX_LIBRARIES   - libraries when using SPEX
# SPEX_FOUND       - true if SPEX found

# set ``SPEX_ROOT`` to a SPEX installation root to
# tell this module where to look.

# All the Find*.cmake files in SuiteSparse are installed by 'make install' into
# /usr/local/lib/cmake/SuiteSparse (where '/usr/local' is the
# ${CMAKE_INSTALL_PREFIX}).  To access this file, place the following commands
# in your CMakeLists.txt file.  See also SuiteSparse/Example/CMakeLists.txt:
#
#   set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
#       ${CMAKE_INSTALL_PREFIX}/lib/cmake/SuiteSparse )

#-------------------------------------------------------------------------------

# include files for SPEX
find_path ( SPEX_INCLUDE_DIR
    NAMES SPEX.h
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/SPEX
    HINTS ${CMAKE_SOURCE_DIR}/../SPEX
    PATH_SUFFIXES include Include
)

# dynamic SPEX library
find_library ( SPEX_LIBRARY
    NAMES spex
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/SPEX
    HINTS ${CMAKE_SOURCE_DIR}/../SPEX
    PATH_SUFFIXES lib build
)

if ( MSVC )
    set ( STATIC_SUFFIX .lib )
else ( )
    set ( STATIC_SUFFIX .a )
endif ( )

# static SPEX library
set ( save ${CMAKE_FIND_LIBRARY_SUFFIXES} )
set ( CMAKE_FIND_LIBRARY_SUFFIXES ${STATIC_SUFFIX} ${CMAKE_FIND_LIBRARY_SUFFIXES} )
find_library ( SPEX_STATIC
    NAMES spex
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/SPEX
    HINTS ${CMAKE_SOURCE_DIR}/../SPEX
    PATH_SUFFIXES lib build
)
set ( CMAKE_FIND_LIBRARY_SUFFIXES ${save} )

# get version of the library from the dynamic library name
get_filename_component ( SPEX_LIBRARY  ${SPEX_LIBRARY} REALPATH )
get_filename_component ( SPEX_FILENAME ${SPEX_LIBRARY} NAME )
string (
    REGEX MATCH "[0-9]+.[0-9]+.[0-9]+"
    SPEX_VERSION
    ${SPEX_FILENAME}
)

# set ( SPEX_VERSION "" )
if ( EXISTS "${SPEX_INCLUDE_DIR}" AND NOT SPEX_VERSION )
    # if the version does not appear in the filename, read the include file
    file ( STRINGS ${SPEX_INCLUDE_DIR}/SPEX.h SPEX_MAJOR_STR
        REGEX "define SPEX_VERSION_MAJOR " )
    file ( STRINGS ${SPEX_INCLUDE_DIR}/SPEX.h SPEX_MINOR_STR
        REGEX "define SPEX_VERSION_MINOR " )
    file ( STRINGS ${SPEX_INCLUDE_DIR}/SPEX.h SPEX_PATCH_STR
        REGEX "define SPEX_VERSION_SUB " )
    message ( STATUS "major: ${SPEX_MAJOR_STR}" )
    message ( STATUS "minor: ${SPEX_MINOR_STR}" )
    message ( STATUS "patch: ${SPEX_PATCH_STR}" )
    string ( REGEX MATCH "[0-9]+" SPEX_MAJOR ${SPEX_MAJOR_STR} )
    string ( REGEX MATCH "[0-9]+" SPEX_MINOR ${SPEX_MINOR_STR} )
    string ( REGEX MATCH "[0-9]+" SPEX_PATCH ${SPEX_PATCH_STR} )
    set (SPEX_VERSION "${SPEX_MAJOR}.${SPEX_MINOR}.${SPEX_PATCH}")
endif ( )

set ( SPEX_LIBRARIES ${SPEX_LIBRARY} )

include (FindPackageHandleStandardArgs)

find_package_handle_standard_args ( SPEX
    REQUIRED_VARS SPEX_LIBRARIES SPEX_INCLUDE_DIR
    VERSION_VAR SPEX_VERSION
)

mark_as_advanced (
    SPEX_INCLUDE_DIR
    SPEX_LIBRARY
    SPEX_STATIC
    SPEX_LIBRARIES
)

if ( SPEX_FOUND )
    message ( STATUS "SPEX version: ${SPEX_VERSION}" )
    message ( STATUS "SPEX include: ${SPEX_INCLUDE_DIR}" )
    message ( STATUS "SPEX library: ${SPEX_LIBRARY}" )
    message ( STATUS "SPEX static:  ${SPEX_STATIC}" )
else ( )
    message ( STATUS "SPEX not found" )
    set ( SPEX_INCLUDE_DIR "" )
    set ( SPEX_LIBRARIES "" )
    set ( SPEX_LIBRARY "" )
    set ( SPEX_STATIC "" )
endif ( )

