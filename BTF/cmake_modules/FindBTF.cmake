#-------------------------------------------------------------------------------
# SuiteSparse/BTF/cmake_modules/FindBTF.cmake
#-------------------------------------------------------------------------------

# The following copyright and license applies to just this file only, not to
# the library itself:
# FindBTF.cmake, Copyright (c) 2022-2023, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

#-------------------------------------------------------------------------------

# Finds the BTF include file and compiled library and sets:

# BTF_INCLUDE_DIR - where to find btf.h
# BTF_LIBRARY     - dynamic BTF library
# BTF_STATIC      - static BTF library
# BTF_LIBRARIES   - libraries when using BTF
# BTF_FOUND       - true if BTF found

# set ``BTF_ROOT`` to a BTF installation root to
# tell this module where to look.

# All the Find*.cmake files in SuiteSparse are installed by 'make install' into
# /usr/local/lib/cmake/SuiteSparse (where '/usr/local' is the
# ${CMAKE_INSTALL_PREFIX}).  To access this file, place the following commands
# in your CMakeLists.txt file.  See also SuiteSparse/Example/CMakeLists.txt:
#
#   set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
#       ${CMAKE_INSTALL_PREFIX}/lib/cmake/SuiteSparse )

#-------------------------------------------------------------------------------

# include files for BTF
find_path ( BTF_INCLUDE_DIR
    NAMES btf.h
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/BTF
    HINTS ${CMAKE_SOURCE_DIR}/../BTF
    PATH_SUFFIXES include Include
)

# dynamic BTF library (or static if no dynamic library was built)
find_library ( BTF_LIBRARY
    NAMES btf btf_static
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/BTF
    HINTS ${CMAKE_SOURCE_DIR}/../BTF
    PATH_SUFFIXES lib build build/Release build/Debug
)

if ( MSVC )
    set ( STATIC_NAME btf_static )
else ( )
    set ( STATIC_NAME btf )
    set ( save ${CMAKE_FIND_LIBRARY_SUFFIXES} )
    set ( CMAKE_FIND_LIBRARY_SUFFIXES
        ${CMAKE_STATIC_LIBRARY_SUFFIX} ${CMAKE_FIND_LIBRARY_SUFFIXES} )
endif ( )

# static BTF library
find_library ( BTF_STATIC
    NAMES ${STATIC_NAME}
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/BTF
    HINTS ${CMAKE_SOURCE_DIR}/../BTF
    PATH_SUFFIXES lib build build/Release build/Debug
)

if ( NOT MSVC )
    # restore the CMAKE_FIND_LIBRARY_SUFFIXES variable
    set ( CMAKE_FIND_LIBRARY_SUFFIXES ${save} )
endif ( )

# get version of the library from the dynamic library name
get_filename_component ( BTF_LIBRARY  ${BTF_LIBRARY} REALPATH )
get_filename_component ( BTF_FILENAME ${BTF_LIBRARY} NAME )
string (
    REGEX MATCH "[0-9]+.[0-9]+.[0-9]+"
    BTF_VERSION
    ${BTF_FILENAME}
)

# set ( BTF_VERSION "" )
if ( EXISTS "${BTF_INCLUDE_DIR}" AND NOT BTF_VERSION )
    # if the version does not appear in the filename, read the include file
    file ( STRINGS ${BTF_INCLUDE_DIR}/btf.h BTF_MAJOR_STR
        REGEX "define BTF_MAIN_VERSION" )
    file ( STRINGS ${BTF_INCLUDE_DIR}/btf.h BTF_MINOR_STR
        REGEX "define BTF_SUB_VERSION" )
    file ( STRINGS ${BTF_INCLUDE_DIR}/btf.h BTF_PATCH_STR
        REGEX "define BTF_SUBSUB_VERSION" )
    message ( STATUS "major: ${BTF_MAJOR_STR}" )
    message ( STATUS "minor: ${BTF_MINOR_STR}" )
    message ( STATUS "patch: ${BTF_PATCH_STR}" )
    string ( REGEX MATCH "[0-9]+" BTF_MAJOR ${BTF_MAJOR_STR} )
    string ( REGEX MATCH "[0-9]+" BTF_MINOR ${BTF_MINOR_STR} )
    string ( REGEX MATCH "[0-9]+" BTF_PATCH ${BTF_PATCH_STR} )
    set (BTF_VERSION "${BTF_MAJOR}.${BTF_MINOR}.${BTF_PATCH}")
endif ( )

set ( BTF_LIBRARIES ${BTF_LIBRARY} )

include (FindPackageHandleStandardArgs)

find_package_handle_standard_args ( BTF
    REQUIRED_VARS BTF_LIBRARY BTF_INCLUDE_DIR
    VERSION_VAR BTF_VERSION
)

mark_as_advanced (
    BTF_INCLUDE_DIR
    BTF_LIBRARY
    BTF_STATIC
    BTF_LIBRARIES
)

if ( BTF_FOUND )
    message ( STATUS "BTF version: ${BTF_VERSION}" )
    message ( STATUS "BTF include: ${BTF_INCLUDE_DIR}" )
    message ( STATUS "BTF library: ${BTF_LIBRARY}" )
    message ( STATUS "BTF static:  ${BTF_STATIC}" )
else ( )
    message ( STATUS "BTF not found" )
    set ( BTF_INCLUDE_DIR "" )
    set ( BTF_LIBRARIES "" )
    set ( BTF_LIBRARY "" )
    set ( BTF_STATIC "" )
endif ( )

