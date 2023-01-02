#-------------------------------------------------------------------------------
# SuiteSparse/RBio/cmake_modules/FindRBio.cmake
#-------------------------------------------------------------------------------

# The following copyright and license applies to just this file only, not to
# the library itself:
# FindRBio.cmake, Copyright (c) 2022-2023, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

#-------------------------------------------------------------------------------

# Finds the RBio include file and compiled library and sets:

# RBIO_INCLUDE_DIR - where to find RBio.h
# RBIO_LIBRARY     - dynamic RBio library
# RBIO_STATIC      - static RBio library
# RBIO_LIBRARIES   - libraries when using RBio
# RBIO_FOUND       - true if RBio found

# set ``RBIO_ROOT`` or ``RBio_ROOT`` to a RBio installation root to
# tell this module where to look.

# All the Find*.cmake files in SuiteSparse are installed by 'make install' into
# /usr/local/lib/cmake/SuiteSparse (where '/usr/local' is the
# ${CMAKE_INSTALL_PREFIX}).  To access this file, place the following commands
# in your CMakeLists.txt file.  See also SuiteSparse/Example/CMakeLists.txt:
#
#   set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
#       ${CMAKE_INSTALL_PREFIX}/lib/cmake/SuiteSparse )

#-------------------------------------------------------------------------------

# include files for RBio
find_path ( RBIO_INCLUDE_DIR
    NAMES RBio.h
    HINTS ${RBIO_ROOT}
    HINTS ENV RBIO_ROOT
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/RBio
    HINTS ${CMAKE_SOURCE_DIR}/../RBio
    PATH_SUFFIXES include Include
)

# dynamic RBio library
find_library ( RBIO_LIBRARY
    NAMES rbio
    HINTS ${RBIO_ROOT}
    HINTS ENV RBIO_ROOT
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/RBio
    HINTS ${CMAKE_SOURCE_DIR}/../RBio
    PATH_SUFFIXES lib build alternative
)

if ( MSVC )
    set ( STATIC_SUFFIX .lib )
    set ( STATIC_NAME rbio_static )
else ( )
    set ( STATIC_SUFFIX .a )
    set ( STATIC_NAME rbio )
endif ( )

# static RBio library
set ( save ${CMAKE_FIND_LIBRARY_SUFFIXES} )
set ( CMAKE_FIND_LIBRARY_SUFFIXES ${STATIC_SUFFIX} ${CMAKE_FIND_LIBRARY_SUFFIXES} )
find_library ( RBIO_STATIC
    NAMES ${STATIC_NAME}
    HINTS ${RBIO_ROOT}
    HINTS ENV RBIO_ROOT
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/RBio
    HINTS ${CMAKE_SOURCE_DIR}/../RBio
    PATH_SUFFIXES lib build alternative
)
set ( CMAKE_FIND_LIBRARY_SUFFIXES ${save} )

# get version of the library from the dynamic library name
get_filename_component ( RBIO_LIBRARY  ${RBIO_LIBRARY} REALPATH )
get_filename_component ( RBIO_FILENAME ${RBIO_LIBRARY} NAME )
string (
    REGEX MATCH "[0-9]+.[0-9]+.[0-9]+"
    RBIO_VERSION
    ${RBIO_FILENAME}
)

# set ( RBIO_VERSION "" )
if ( EXISTS "${RBIO_INCLUDE_DIR}" AND NOT RBIO_VERSION )
    # if the version does not appear in the filename, read the include file
    file ( STRINGS ${RBIO_INCLUDE_DIR}/RBio.h RBIO_MAJOR_STR
        REGEX "define RBIO_MAIN_VERSION" )
    file ( STRINGS ${RBIO_INCLUDE_DIR}/RBio.h RBIO_MINOR_STR
        REGEX "define RBIO_SUB_VERSION" )
    file ( STRINGS ${RBIO_INCLUDE_DIR}/RBio.h RBIO_PATCH_STR
        REGEX "define RBIO_SUBSUB_VERSION" )
    message ( STATUS "major: ${RBIO_MAJOR_STR}" )
    message ( STATUS "minor: ${RBIO_MINOR_STR}" )
    message ( STATUS "patch: ${RBIO_PATCH_STR}" )
    string ( REGEX MATCH "[0-9]+" RBIO_MAJOR ${RBIO_MAJOR_STR} )
    string ( REGEX MATCH "[0-9]+" RBIO_MINOR ${RBIO_MINOR_STR} )
    string ( REGEX MATCH "[0-9]+" RBIO_PATCH ${RBIO_PATCH_STR} )
    set (RBIO_VERSION "${RBIO_MAJOR}.${RBIO_MINOR}.${RBIO_PATCH}")
endif ( )
set ( RBIO_LIBRARIES ${RBIO_LIBRARY} )

include (FindPackageHandleStandardArgs)

find_package_handle_standard_args ( RBio
    REQUIRED_VARS RBIO_LIBRARIES RBIO_INCLUDE_DIR
    VERSION_VAR RBIO_VERSION
)

mark_as_advanced (
    RBIO_INCLUDE_DIR
    RBIO_LIBRARY
    RBIO_STATIC
    RBIO_LIBRARIES
)

if ( RBIO_FOUND )
    message ( STATUS "RBio version: ${RBIO_VERSION}" )
    message ( STATUS "RBio include: ${RBIO_INCLUDE_DIR}")
    message ( STATUS "RBio library: ${RBIO_LIBRARY}" )
    message ( STATUS "RBio static:  ${RBIO_STATIC}" )
else ( )
    message ( STATUS "RBio not found" )
    set ( RBIO_INCLUDE_DIR "" )
    set ( RBIO_LIBRARIES "" )
    set ( RBIO_LIBRARY "" )
    set ( RBIO_STATIC "" )
endif ( )

