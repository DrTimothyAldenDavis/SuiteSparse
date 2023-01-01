#-------------------------------------------------------------------------------
# SuiteSparse/CXSparse/cmake_modules/FindCXSparse.cmake
#-------------------------------------------------------------------------------

# The following copyright and license applies to just this file only, not to
# the library itself:
# FindCXSparse.cmake, Copyright (c) 2022-2023, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

#-------------------------------------------------------------------------------

# Finds the CXSparse include file and compiled library and sets:

# CXSPARSE_INCLUDE_DIR - where to find cs.h
# CXSPARSE_LIBRARY     - dynamic CXSPARSE library
# CXSPARSE_STATIC      - static CXSPARSE library
# CXSPARSE_LIBRARIES   - libraries when using CXSPARSE
# CXSPARSE_FOUND       - true if CXSPARSE found

# set ``CXSparse_ROOT`` or ``CXSPARSE_ROOT`` to a CXSPARSE installation root to
# tell this module where to look.

# All the Find*.cmake files in SuiteSparse are installed by 'make install' into
# /usr/local/lib/cmake/SuiteSparse (where '/usr/local' is the
# ${CMAKE_INSTALL_PREFIX}).  To access this file, place the following commands
# in your CMakeLists.txt file.  See also SuiteSparse/Example/CMakeLists.txt:
#
#   set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
#       ${CMAKE_INSTALL_PREFIX}/lib/cmake/SuiteSparse )

#-------------------------------------------------------------------------------

# include files for CXSPARSE
find_path ( CXSPARSE_INCLUDE_DIR
    NAMES cs.h
    HINTS ${CXSPARSE_ROOT}
    HINTS ENV CXSPARSE_ROOT
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/CXSparse
    HINTS ${CMAKE_SOURCE_DIR}/../CXSparse
    PATH_SUFFIXES include Include
)

# dynamic CXSPARSE library
find_library ( CXSPARSE_LIBRARY
    NAMES cxsparse
    HINTS ${CXSPARSE_ROOT}
    HINTS ENV CXSPARSE_ROOT
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/CXSparse
    HINTS ${CMAKE_SOURCE_DIR}/../CXSparse
    PATH_SUFFIXES lib build
)

if ( MSVC )
    set ( STATIC_SUFFIX .lib )
else ( )
    set ( STATIC_SUFFIX .a )
endif ( )

# static CXSPARSE library
set ( save ${CMAKE_FIND_LIBRARY_SUFFIXES} )
set ( CMAKE_FIND_LIBRARY_SUFFIXES ${STATIC_SUFFIX} ${CMAKE_FIND_LIBRARY_SUFFIXES} )
find_library ( CXSPARSE_STATIC
    NAMES cxsparse
    HINTS ${CXSPARSE_ROOT}
    HINTS ENV CXSPARSE_ROOT
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/CXSparse
    HINTS ${CMAKE_SOURCE_DIR}/../CXSparse
    PATH_SUFFIXES lib build
)
set ( CMAKE_FIND_LIBRARY_SUFFIXES ${save} )

# get version of the library from the dynamic library name
get_filename_component ( CXSPARSE_LIBRARY  ${CXSPARSE_LIBRARY} REALPATH )
get_filename_component ( CXSPARSE_FILENAME ${CXSPARSE_LIBRARY} NAME )
string (
    REGEX MATCH "[0-9]+.[0-9]+.[0-9]+"
    CXSPARSE_VERSION
    ${CXSPARSE_FILENAME}
)

# set ( CXSPARSE_VERSION "" )
if ( EXISTS "${CXSPARSE_INCLUDE_DIR}" AND NOT CXSPARSE_VERSION )
    # if the version does not appear in the filename, read the include file
    file ( STRINGS ${CXSPARSE_INCLUDE_DIR}/cs.h CXSPARSE_MAJOR_STR
        REGEX "define CS_VER" )
    file ( STRINGS ${CXSPARSE_INCLUDE_DIR}/cs.h CXSPARSE_MINOR_STR
        REGEX "define CS_SUBVER" )
    file ( STRINGS ${CXSPARSE_INCLUDE_DIR}/cs.h CXSPARSE_PATCH_STR
        REGEX "define CS_SUBSUB" )
    message ( STATUS "major: ${CXSPARSE_MAJOR_STR}" )
    message ( STATUS "minor: ${CXSPARSE_MINOR_STR}" )
    message ( STATUS "patch: ${CXSPARSE_PATCH_STR}" )
    string ( REGEX MATCH "[0-9]+" CXSPARSE_MAJOR ${CXSPARSE_MAJOR_STR} )
    string ( REGEX MATCH "[0-9]+" CXSPARSE_MINOR ${CXSPARSE_MINOR_STR} )
    string ( REGEX MATCH "[0-9]+" CXSPARSE_PATCH ${CXSPARSE_PATCH_STR} )
    set (CXSPARSE_VERSION "${CXSPARSE_MAJOR}.${CXSPARSE_MINOR}.${CXSPARSE_PATCH}")
endif ( )

set ( CXSPARSE_LIBRARIES ${CXSPARSE_LIBRARY} )

include (FindPackageHandleStandardArgs)

find_package_handle_standard_args ( CXSparse
    REQUIRED_VARS CXSPARSE_LIBRARIES CXSPARSE_INCLUDE_DIR
    VERSION_VAR CXSPARSE_VERSION
)

mark_as_advanced (
    CXSPARSE_INCLUDE_DIR
    CXSPARSE_LIBRARY
    CXSPARSE_STATIC
    CXSPARSE_LIBRARIES
)

if ( CXSPARSE_FOUND )
    message ( STATUS "CXSparse version: ${CXSPARSE_VERSION}" )
    message ( STATUS "CXSparse include: ${CXSPARSE_INCLUDE_DIR}" )
    message ( STATUS "CXSparse library: ${CXSPARSE_LIBRARY}" )
    message ( STATUS "CXSparse static:  ${CXSPARSE_STATIC}" )
else ( )
    message ( STATUS "CXSparse not found" )
    set ( CXSPARSE_INCLUDE_DIR "" )
    set ( CXSPARSE_LIBRARIES "" )
    set ( CXSPARSE_LIBRARY "" )
    set ( CXSPARSE_STATIC "" )
endif ( )

