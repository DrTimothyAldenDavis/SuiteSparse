#-------------------------------------------------------------------------------
# SuiteSparse/KLU/cmake_modules/FindKLU.cmake
#-------------------------------------------------------------------------------

# The following copyright and license applies to just this file only, not to
# the library itself:
# FindKLU.cmake, Copyright (c) 2022-2023, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

#-------------------------------------------------------------------------------

# Finds the KLU include file and compiled library and sets:

# KLU_INCLUDE_DIR - where to find klu.h
# KLU_LIBRARY     - dynamic KLU library
# KLU_STATIC      - static KLU library
# KLU_LIBRARIES   - libraries when using KLU
# KLU_FOUND       - true if KLU found

# set ``KLU_ROOT`` to a KLU installation root to
# tell this module where to look.

# All the Find*.cmake files in SuiteSparse are installed by 'make install' into
# /usr/local/lib/cmake/SuiteSparse (where '/usr/local' is the
# ${CMAKE_INSTALL_PREFIX}).  To access this file, place the following commands
# in your CMakeLists.txt file.  See also SuiteSparse/Example/CMakeLists.txt:
#
#   set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
#       ${CMAKE_INSTALL_PREFIX}/lib/cmake/SuiteSparse )

#-------------------------------------------------------------------------------

# include files for KLU
find_path ( KLU_INCLUDE_DIR
    NAMES klu.h
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/KLU
    HINTS ${CMAKE_SOURCE_DIR}/../KLU
    PATH_SUFFIXES include Include
)

# dynamic KLU library (or static if no dynamic library was built)
find_library ( KLU_LIBRARY
    NAMES klu klu_static
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/KLU
    HINTS ${CMAKE_SOURCE_DIR}/../KLU
    PATH_SUFFIXES lib build build/Release build/Debug
)

if ( MSVC )
    set ( STATIC_NAME klu_static )
else ( )
    set ( STATIC_NAME klu )
    set ( save ${CMAKE_FIND_LIBRARY_SUFFIXES} )
    set ( CMAKE_FIND_LIBRARY_SUFFIXES
        ${CMAKE_STATIC_LIBRARY_SUFFIX} ${CMAKE_FIND_LIBRARY_SUFFIXES} )
endif ( )

# static KLU library
find_library ( KLU_STATIC
    NAMES ${STATIC_NAME}
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/KLU
    HINTS ${CMAKE_SOURCE_DIR}/../KLU
    PATH_SUFFIXES lib build build/Release build/Debug
)

if ( NOT MSVC )
    # restore the CMAKE_FIND_LIBRARY_SUFFIXES variable
    set ( CMAKE_FIND_LIBRARY_SUFFIXES ${save} )
endif ( )

# get version of the library from the dynamic library name
get_filename_component ( KLU_LIBRARY  ${KLU_LIBRARY} REALPATH )
get_filename_component ( KLU_FILENAME ${KLU_LIBRARY} NAME )
string (
    REGEX MATCH "[0-9]+.[0-9]+.[0-9]+"
    KLU_VERSION
    ${KLU_FILENAME}
)

# set ( KLU_VERSION "" )
if ( EXISTS "${KLU_INCLUDE_DIR}" AND NOT KLU_VERSION )
    # if the version does not appear in the filename, read the include file
    file ( STRINGS ${KLU_INCLUDE_DIR}/klu.h KLU_MAJOR_STR
        REGEX "define KLU_MAIN_VERSION" )
    file ( STRINGS ${KLU_INCLUDE_DIR}/klu.h KLU_MINOR_STR
        REGEX "define KLU_SUB_VERSION" )
    file ( STRINGS ${KLU_INCLUDE_DIR}/klu.h KLU_PATCH_STR
        REGEX "define KLU_SUBSUB_VERSION" )
    message ( STATUS "major: ${KLU_MAJOR_STR}" )
    message ( STATUS "minor: ${KLU_MINOR_STR}" )
    message ( STATUS "patch: ${KLU_PATCH_STR}" )
    string ( REGEX MATCH "[0-9]+" KLU_MAJOR ${KLU_MAJOR_STR} )
    string ( REGEX MATCH "[0-9]+" KLU_MINOR ${KLU_MINOR_STR} )
    string ( REGEX MATCH "[0-9]+" KLU_PATCH ${KLU_PATCH_STR} )
    set (KLU_VERSION "${KLU_MAJOR}.${KLU_MINOR}.${KLU_PATCH}")
endif ( )

set ( KLU_LIBRARIES ${KLU_LIBRARY} )

include (FindPackageHandleStandardArgs)

find_package_handle_standard_args ( KLU
    REQUIRED_VARS KLU_LIBRARY KLU_INCLUDE_DIR
    VERSION_VAR KLU_VERSION
    )

mark_as_advanced (
    KLU_INCLUDE_DIR
    KLU_LIBRARY
    KLU_STATIC
    KLU_LIBRARIES
    )

if ( KLU_FOUND )
    message ( STATUS "KLU version: ${KLU_VERSION}" )
    message ( STATUS "KLU include: ${KLU_INCLUDE_DIR}" )
    message ( STATUS "KLU library: ${KLU_LIBRARY}" )
    message ( STATUS "KLU static:  ${KLU_STATIC}" )
else ( )
    message ( STATUS "KLU not found" )
    set ( KLU_INCLUDE_DIR "" )
    set ( KLU_LIBRARIES "" )
    set ( KLU_LIBRARY "" )
    set ( KLU_STATIC "" )
endif ( )

