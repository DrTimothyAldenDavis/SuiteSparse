#-------------------------------------------------------------------------------
# SuiteSparse/LDL/cmake_modules/FindLDL.cmake
#-------------------------------------------------------------------------------

# The following copyright and license applies to just this file only, not to
# the library itself:
# FindLDL.cmake, Copyright (c) 2022-2023, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

#-------------------------------------------------------------------------------

# Finds the LDL include file and compiled library and sets:

# LDL_INCLUDE_DIR - where to find ldl.h
# LDL_LIBRARY     - dynamic LDL library
# LDL_STATIC      - static LDL library
# LDL_LIBRARIES   - libraries when using LDL
# LDL_FOUND       - true if LDL found

# set ``LDL_ROOT`` to a LDL installation root to
# tell this module where to look.

# All the Find*.cmake files in SuiteSparse are installed by 'make install' into
# /usr/local/lib/cmake/SuiteSparse (where '/usr/local' is the
# ${CMAKE_INSTALL_PREFIX}).  To access this file, place the following commands
# in your CMakeLists.txt file.  See also SuiteSparse/Example/CMakeLists.txt:
#
#   set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
#       ${CMAKE_INSTALL_PREFIX}/lib/cmake/SuiteSparse )

#-------------------------------------------------------------------------------

# include files for LDL
find_path ( LDL_INCLUDE_DIR
    NAMES ldl.h
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/LDL
    HINTS ${CMAKE_SOURCE_DIR}/../LDL
    PATH_SUFFIXES include Include
)

# dynamic LDL library (or static if no dynamic library was built)
find_library ( LDL_LIBRARY
    NAMES ldl ldl_static
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/LDL
    HINTS ${CMAKE_SOURCE_DIR}/../LDL
    PATH_SUFFIXES lib build build/Release build/Debug
)

if ( MSVC )
    set ( STATIC_NAME ldl_static )
else ( )
    set ( STATIC_NAME ldl )
    set ( save ${CMAKE_FIND_LIBRARY_SUFFIXES} )
    set ( CMAKE_FIND_LIBRARY_SUFFIXES
        ${CMAKE_STATIC_LIBRARY_SUFFIX} ${CMAKE_FIND_LIBRARY_SUFFIXES} )
endif ( )

# static LDL library
find_library ( LDL_STATIC
    NAMES ${STATIC_NAME}
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/LDL
    HINTS ${CMAKE_SOURCE_DIR}/../LDL
    PATH_SUFFIXES lib build build/Release build/Debug
)

if ( NOT MSVC )
    # restore the CMAKE_FIND_LIBRARY_SUFFIXES variable
    set ( CMAKE_FIND_LIBRARY_SUFFIXES ${save} )
endif ( )

# get version of the library from the dynamic library name
get_filename_component ( LDL_LIBRARY  ${LDL_LIBRARY} REALPATH )
get_filename_component ( LDL_FILENAME ${LDL_LIBRARY} NAME )
string (
    REGEX MATCH "[0-9]+.[0-9]+.[0-9]+"
    LDL_VERSION
    ${LDL_FILENAME}
)

# set ( LDL_VERSION "" )
if ( EXISTS "${LDL_INCLUDE_DIR}" AND NOT LDL_VERSION )
    # if the version does not appear in the filename, read the include file
    file ( STRINGS ${LDL_INCLUDE_DIR}/ldl.h LDL_MAJOR_STR
        REGEX "define LDL_MAIN_VERSION" )
    file ( STRINGS ${LDL_INCLUDE_DIR}/ldl.h LDL_MINOR_STR
        REGEX "define LDL_SUB_VERSION" )
    file ( STRINGS ${LDL_INCLUDE_DIR}/ldl.h LDL_PATCH_STR
        REGEX "define LDL_SUBSUB_VERSION" )
    message ( STATUS "major: ${LDL_MAJOR_STR}" )
    message ( STATUS "minor: ${LDL_MINOR_STR}" )
    message ( STATUS "patch: ${LDL_PATCH_STR}" )
    string ( REGEX MATCH "[0-9]+" LDL_MAJOR ${LDL_MAJOR_STR} )
    string ( REGEX MATCH "[0-9]+" LDL_MINOR ${LDL_MINOR_STR} )
    string ( REGEX MATCH "[0-9]+" LDL_PATCH ${LDL_PATCH_STR} )
    set (LDL_VERSION "${LDL_MAJOR}.${LDL_MINOR}.${LDL_PATCH}")
endif ( )

set ( LDL_LIBRARIES ${LDL_LIBRARY} )

include (FindPackageHandleStandardArgs)

find_package_handle_standard_args ( LDL
    REQUIRED_VARS LDL_LIBRARY LDL_INCLUDE_DIR
    VERSION_VAR LDL_VERSION
)

mark_as_advanced (
    LDL_INCLUDE_DIR
    LDL_LIBRARY
    LDL_STATIC
    LDL_LIBRARIES
)

if ( LDL_FOUND )
    message ( STATUS "LDL version: ${LDL_VERSION}" )
    message ( STATUS "LDL include: ${LDL_INCLUDE_DIR}" )
    message ( STATUS "LDL library: ${LDL_LIBRARY}" )
    message ( STATUS "LDL static:  ${LDL_STATIC}" )
else ( )
    message ( STATUS "LDL not found" )
    set ( LDL_INCLUDE_DIR "" )
    set ( LDL_LIBRARIES "" )
    set ( LDL_LIBRARY "" )
    set ( LDL_STATIC "" )
endif ( )

