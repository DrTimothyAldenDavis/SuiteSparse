#-------------------------------------------------------------------------------
# SuiteSparse/CHOLMOD/cmake_modules/FindCHOLMOD.cmake
#-------------------------------------------------------------------------------

# The following copyright and license applies to just this file only, not to
# the library itself:
# FindCHOLMOD.cmake, Copyright (c) 2022-2023, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

#-------------------------------------------------------------------------------

# Finds the CHOLMOD include file and compiled library and sets:

# CHOLMOD_INCLUDE_DIR - where to find cholmod.h
# CHOLMOD_LIBRARY     - compiled CHOLMOD library
# CHOLMOD_LIBRARIES   - libraries when using CHOLMOD
# CHOLMOD_FOUND       - true if CHOLMOD found

# set ``CHOLMOD_ROOT`` to a CHOLMOD installation root to
# tell this module where to look.

# All the Find*.cmake files in SuiteSparse are installed by 'make install' into
# /usr/local/lib/cmake/SuiteSparse (where '/usr/local' is the
# ${CMAKE_INSTALL_PREFIX}).  To access this file, place the following commands
# in your CMakeLists.txt file.  See also SuiteSparse/Example/CMakeLists.txt:
#
#   set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
#       ${CMAKE_INSTALL_PREFIX}/lib/cmake/SuiteSparse )

#-------------------------------------------------------------------------------

# include files for CHOLMOD
find_path ( CHOLMOD_INCLUDE_DIR
    NAMES cholmod.h
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/CHOLMOD
    HINTS ${CMAKE_SOURCE_DIR}/../CHOLMOD
    PATH_SUFFIXES include Include
)

# dynamic CHOLMOD library (or static if no dynamic library was built)
find_library ( CHOLMOD_LIBRARY
    NAMES cholmod cholmod_static
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/CHOLMOD
    HINTS ${CMAKE_SOURCE_DIR}/../CHOLMOD
    PATH_SUFFIXES lib build build/Release build/Debug
)

if ( MSVC )
    set ( STATIC_NAME cholmod_static )
else ( )
    set ( STATIC_NAME cholmod )
    set ( save ${CMAKE_FIND_LIBRARY_SUFFIXES} )
    set ( CMAKE_FIND_LIBRARY_SUFFIXES
        ${CMAKE_STATIC_LIBRARY_SUFFIX} ${CMAKE_FIND_LIBRARY_SUFFIXES} )
endif ( )

# static CHOLMOD library
find_library ( CHOLMOD_STATIC
    NAMES ${STATIC_NAME}
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/CHOLMOD
    HINTS ${CMAKE_SOURCE_DIR}/../CHOLMOD
    PATH_SUFFIXES lib build build/Release build/Debug
)

if ( NOT MSVC )
    # restore the CMAKE_FIND_LIBRARY_SUFFIXES variable
    set ( CMAKE_FIND_LIBRARY_SUFFIXES ${save} )
endif ( )

# get version of the library from the dynamic library name
get_filename_component ( CHOLMOD_LIBRARY  ${CHOLMOD_LIBRARY} REALPATH )
get_filename_component ( CHOLMOD_FILENAME ${CHOLMOD_LIBRARY} NAME )
string (
    REGEX MATCH "[0-9]+.[0-9]+.[0-9]+"
    CHOLMOD_VERSION
    ${CHOLMOD_FILENAME}
)

# set ( CHOLMOD_VERSION "" )
if ( EXISTS "${CHOLMOD_INCLUDE_DIR}" AND NOT CHOLMOD_VERSION )
    # if the version does not appear in the filename, read the include file
    file ( STRINGS ${CHOLMOD_INCLUDE_DIR}/cholmod.h CHOLMOD_MAJOR_STR
        REGEX "define CHOLMOD_MAIN_VERSION" )
    file ( STRINGS ${CHOLMOD_INCLUDE_DIR}/cholmod.h CHOLMOD_MINOR_STR
        REGEX "define CHOLMOD_SUB_VERSION" )
    file ( STRINGS ${CHOLMOD_INCLUDE_DIR}/cholmod.h CHOLMOD_PATCH_STR
        REGEX "define CHOLMOD_SUBSUB_VERSION" )
    message ( STATUS "major: ${CHOLMOD_MAJOR_STR}" )
    message ( STATUS "minor: ${CHOLMOD_MINOR_STR}" )
    message ( STATUS "patch: ${CHOLMOD_PATCH_STR}" )
    string ( REGEX MATCH "[0-9]+" CHOLMOD_MAJOR ${CHOLMOD_MAJOR_STR} )
    string ( REGEX MATCH "[0-9]+" CHOLMOD_MINOR ${CHOLMOD_MINOR_STR} )
    string ( REGEX MATCH "[0-9]+" CHOLMOD_PATCH ${CHOLMOD_PATCH_STR} )
    set (CHOLMOD_VERSION "${CHOLMOD_MAJOR}.${CHOLMOD_MINOR}.${CHOLMOD_PATCH}")
endif ( )

set (CHOLMOD_LIBRARIES ${CHOLMOD_LIBRARY} )

include (FindPackageHandleStandardArgs)

find_package_handle_standard_args ( CHOLMOD
    REQUIRED_VARS CHOLMOD_LIBRARY CHOLMOD_INCLUDE_DIR
    VERSION_VAR CHOLMOD_VERSION
)

mark_as_advanced (
    CHOLMOD_INCLUDE_DIR
    CHOLMOD_LIBRARY
    CHOLMOD_STATIC
    CHOLMOD_LIBRARIES
)

if ( CHOLMOD_FOUND )
    message ( STATUS "CHOLMOD version: ${CHOLMOD_VERSION}" )
    message ( STATUS "CHOLMOD include: ${CHOLMOD_INCLUDE_DIR}" )
    message ( STATUS "CHOLMOD library: ${CHOLMOD_LIBRARY}" )
    message ( STATUS "CHOLMOD static:  ${CHOLMOD_STATIC}" )
else ( )
    message ( STATUS "CHOLMOD not found" )
    set ( CHOLMOD_INCLUDE_DIR "" )
    set ( CHOLMOD_LIBRARIES "" )
    set ( CHOLMOD_LIBRARY "" )
    set ( CHOLMOD_STATIC "" )
endif ( )

