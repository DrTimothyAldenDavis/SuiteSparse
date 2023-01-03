#-------------------------------------------------------------------------------
# SuiteSparse/SPQR/cmake_modules/FindSPQR.cmake
#-------------------------------------------------------------------------------

# The following copyright and license applies to just this file only, not to
# the library itself:
# FindSPQR.cmake, Copyright (c) 2022-2023, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

#-------------------------------------------------------------------------------

# Finds the SPQR include file and compiled library and sets:

# SPQR_INCLUDE_DIR - where to find SuiteSparseQR.hpp and other headers
# SPQR_LIBRARY     - dynamic SPQR library
# SPQR_STATIC      - static SPQR library
# SPQR_LIBRARIES   - libraries when using SPQR
# SPQR_FOUND       - true if SPQR found

# set ``SPQR_ROOT`` to a SPQR installation root to
# tell this module where to look.

# All the Find*.cmake files in SuiteSparse are installed by 'make install' into
# /usr/local/lib/cmake/SuiteSparse (where '/usr/local' is the
# ${CMAKE_INSTALL_PREFIX}).  To access this file, place the following commands
# in your CMakeLists.txt file.  See also SuiteSparse/Example/CMakeLists.txt:
#
#   set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
#       ${CMAKE_INSTALL_PREFIX}/lib/cmake/SuiteSparse )

#-------------------------------------------------------------------------------

# include files for SPQR
find_path ( SPQR_INCLUDE_DIR
    NAMES SuiteSparseQR.hpp
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/SPQR
    HINTS ${CMAKE_SOURCE_DIR}/../SPQR
    PATH_SUFFIXES include Include
)

# dynamic SPQR library
find_library ( SPQR_LIBRARY
    NAMES spqr
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/SPQR
    HINTS ${CMAKE_SOURCE_DIR}/../SPQR
    PATH_SUFFIXES lib build
)

if ( MSVC )
    set ( STATIC_SUFFIX .lib )
else ( )
    set ( STATIC_SUFFIX .a )
endif ( )

# static SPQR library
set ( save ${CMAKE_FIND_LIBRARY_SUFFIXES} )
set ( CMAKE_FIND_LIBRARY_SUFFIXES ${STATIC_SUFFIX} ${CMAKE_FIND_LIBRARY_SUFFIXES} )
find_library ( SPQR_STATIC
    NAMES spqr_static spqr
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/SPQR
    HINTS ${CMAKE_SOURCE_DIR}/../SPQR
    PATH_SUFFIXES lib build
)
set ( CMAKE_FIND_LIBRARY_SUFFIXES ${save} )

# get version of the library from the dynamic library name
get_filename_component ( SPQR_LIBRARY  ${SPQR_LIBRARY} REALPATH )
get_filename_component ( SPQR_FILENAME ${SPQR_LIBRARY} NAME )
string (
    REGEX MATCH "[0-9]+.[0-9]+.[0-9]+"
    SPQR_VERSION
    ${SPQR_FILENAME}
)

# set ( SPQR_VERSION "" )
if ( EXISTS "${SPQR_INCLUDE_DIR}" AND NOT SPQR_VERSION )
    # if the version does not appear in the filename, read the include file
    file ( STRINGS ${SPQR_INCLUDE_DIR}/SuiteSparseQR_definitions.h SPQR_MAJOR_STR
        REGEX "define SPQR_MAIN_VERSION" )
    file ( STRINGS ${SPQR_INCLUDE_DIR}/SuiteSparseQR_definitions.h SPQR_MINOR_STR
        REGEX "define SPQR_SUB_VERSION" )
    file ( STRINGS ${SPQR_INCLUDE_DIR}/SuiteSparseQR_definitions.h SPQR_PATCH_STR
        REGEX "define SPQR_SUBSUB_VERSION" )
    message ( STATUS "major: ${SPQR_MAJOR_STR}" )
    message ( STATUS "minor: ${SPQR_MINOR_STR}" )
    message ( STATUS "patch: ${SPQR_PATCH_STR}" )
    string ( REGEX MATCH "[0-9]+" SPQR_MAJOR ${SPQR_MAJOR_STR} )
    string ( REGEX MATCH "[0-9]+" SPQR_MINOR ${SPQR_MINOR_STR} )
    string ( REGEX MATCH "[0-9]+" SPQR_PATCH ${SPQR_PATCH_STR} )
    set (SPQR_VERSION "${SPQR_MAJOR}.${SPQR_MINOR}.${SPQR_PATCH}")
endif ( )

set ( SPQR_LIBRARIES ${SPQR_LIBRARY} )

include (FindPackageHandleStandardArgs)

find_package_handle_standard_args ( SPQR
    REQUIRED_VARS SPQR_LIBRARY SPQR_INCLUDE_DIR
    VERSION_VAR SPQR_VERSION
)

mark_as_advanced (
    SPQR_INCLUDE_DIR
    SPQR_LIBRARY
    SPQR_STATIC
    SPQR_LIBRARIES
)

if ( SPQR_FOUND )
    message ( STATUS "SPQR version: ${SPQR_VERSION}" )
    message ( STATUS "SPQR include: ${SPQR_INCLUDE_DIR}" )
    message ( STATUS "SPQR library: ${SPQR_LIBRARY}" )
    message ( STATUS "SPQR static:  ${SPQR_STATIC}" )
else ( )
    message ( STATUS "SPQR not found" )
    set ( SPQR_INCLUDE_DIR "" )
    set ( SPQR_LIBRARIES "" )
    set ( SPQR_LIBRARY "" )
    set ( SPQR_STATIC "" )
endif ( )

