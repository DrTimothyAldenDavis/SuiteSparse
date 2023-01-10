#-------------------------------------------------------------------------------
# SuiteSparse/CCOLAMD/cmake_modules/FindCCOLAMD.cmake
#-------------------------------------------------------------------------------

# The following copyright and license applies to just this file only, not to
# the library itself:
# FindCCOLAMD.cmake, Copyright (c) 2022-2023, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

#-------------------------------------------------------------------------------

# Finds the CCOLAMD include file and compiled library and sets:

# CCOLAMD_INCLUDE_DIR - where to find ccolamd.h
# CCOLAMD_LIBRARY     - dynamic CCOLAMD library
# CCOLAMD_STATIC      - static CCOLAMD library
# CCOLAMD_LIBRARIES   - libraries when using CCOLAMD
# CCOLAMD_FOUND       - true if CCOLAMD found

# set ``CCOLAMD_ROOT`` to a CCOLAMD installation root to
# tell this module where to look.

# All the Find*.cmake files in SuiteSparse are installed by 'make install' into
# /usr/local/lib/cmake/SuiteSparse (where '/usr/local' is the
# ${CMAKE_INSTALL_PREFIX}).  To access this file, place the following commands
# in your CMakeLists.txt file.  See also SuiteSparse/Example/CMakeLists.txt:
#
#   set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
#       ${CMAKE_INSTALL_PREFIX}/lib/cmake/SuiteSparse )

#-------------------------------------------------------------------------------

# include files for CCOLAMD
find_path ( CCOLAMD_INCLUDE_DIR
    NAMES ccolamd.h
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/CCOLAMD
    HINTS ${CMAKE_SOURCE_DIR}/../CCOLAMD
    PATH_SUFFIXES include Include
)

# dynamic CCOLAMD library (or static if no dynamic library was built)
find_library ( CCOLAMD_LIBRARY
    NAMES ccolamd ccolamd_static
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/CCOLAMD
    HINTS ${CMAKE_SOURCE_DIR}/../CCOLAMD
    PATH_SUFFIXES lib build build/Release build/Debug
)

if ( MSVC )
    set ( STATIC_NAME ccolamd_static )
else ( )
    set ( STATIC_NAME ccolamd )
    set ( save ${CMAKE_FIND_LIBRARY_SUFFIXES} )
    set ( CMAKE_FIND_LIBRARY_SUFFIXES
        ${CMAKE_STATIC_LIBRARY_SUFFIX} ${CMAKE_FIND_LIBRARY_SUFFIXES} )
endif ( )

# static CCOLAMD library
find_library ( CCOLAMD_STATIC
    NAMES ${STATIC_NAME}
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/CCOLAMD
    HINTS ${CMAKE_SOURCE_DIR}/../CCOLAMD
    PATH_SUFFIXES lib build build/Release build/Debug
)

if ( NOT MSVC )
    # restore the CMAKE_FIND_LIBRARY_SUFFIXES variable
    set ( CMAKE_FIND_LIBRARY_SUFFIXES ${save} )
endif ( )

# get version of the library from the dynamic library name
get_filename_component ( CCOLAMD_LIBRARY  ${CCOLAMD_LIBRARY} REALPATH )
get_filename_component ( CCOLAMD_FILENAME ${CCOLAMD_LIBRARY} NAME )
string (
    REGEX MATCH "[0-9]+.[0-9]+.[0-9]+"
    CCOLAMD_VERSION
    ${CCOLAMD_FILENAME}
)

# set ( CCOLAMD_VERSION "" )
if ( EXISTS "${CCOLAMD_INCLUDE_DIR}" AND NOT CCOLAMD_VERSION )
    # if the version does not appear in the filename, read the include file
    file ( STRINGS ${CCOLAMD_INCLUDE_DIR}/ccolamd.h CCOLAMD_MAJOR_STR
        REGEX "define CCOLAMD_MAIN_VERSION" )
    file ( STRINGS ${CCOLAMD_INCLUDE_DIR}/ccolamd.h CCOLAMD_MINOR_STR
        REGEX "define CCOLAMD_SUB_VERSION" )
    file ( STRINGS ${CCOLAMD_INCLUDE_DIR}/ccolamd.h CCOLAMD_PATCH_STR
        REGEX "define CCOLAMD_SUBSUB_VERSION" )
    message ( STATUS "major: ${CCOLAMD_MAJOR_STR}" )
    message ( STATUS "minor: ${CCOLAMD_MINOR_STR}" )
    message ( STATUS "patch: ${CCOLAMD_PATCH_STR}" )
    string ( REGEX MATCH "[0-9]+" CCOLAMD_MAJOR ${CCOLAMD_MAJOR_STR} )
    string ( REGEX MATCH "[0-9]+" CCOLAMD_MINOR ${CCOLAMD_MINOR_STR} )
    string ( REGEX MATCH "[0-9]+" CCOLAMD_PATCH ${CCOLAMD_PATCH_STR} )
    set (CCOLAMD_VERSION "${CCOLAMD_MAJOR}.${CCOLAMD_MINOR}.${CCOLAMD_PATCH}")
endif ( )

set ( CCOLAMD_LIBRARIES ${CCOLAMD_LIBRARY} )

include (FindPackageHandleStandardArgs)

find_package_handle_standard_args ( CCOLAMD
    REQUIRED_VARS CCOLAMD_LIBRARY CCOLAMD_INCLUDE_DIR
    VERSION_VAR CCOLAMD_VERSION
)

mark_as_advanced (
    CCOLAMD_INCLUDE_DIR
    CCOLAMD_LIBRARY
    CCOLAMD_STATIC
    CCOLAMD_LIBRARIES
)

if ( CCOLAMD_FOUND )
    message ( STATUS "CCOLAMD version: ${CCOLAMD_VERSION}" )
    message ( STATUS "CCOLAMD include: ${CCOLAMD_INCLUDE_DIR}" )
    message ( STATUS "CCOLAMD library: ${CCOLAMD_LIBRARY}" )
    message ( STATUS "CCOLAMD static:  ${CCOLAMD_STATIC}" )
else ( )
    message ( STATUS "CCOLAMD not found" )
    set ( CCOLAMD_INCLUDE_DIR "" )
    set ( CCOLAMD_LIBRARIES "" )
    set ( CCOLAMD_LIBRARY "" )
    set ( CCOLAMD_STATIC "" )
endif ( )

