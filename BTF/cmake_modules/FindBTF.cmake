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

# dynamic BTF library
find_library ( BTF_LIBRARY
    NAMES btf
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/BTF
    HINTS ${CMAKE_SOURCE_DIR}/../BTF
    PATH_SUFFIXES lib build
)

if ( MSVC )
    set ( STATIC_SUFFIX .lib )
else ( )
    set ( STATIC_SUFFIX .a )
endif ( )

# static BTF library
set ( save ${CMAKE_FIND_LIBRARY_SUFFIXES} )
set ( CMAKE_FIND_LIBRARY_SUFFIXES ${STATIC_SUFFIX} ${CMAKE_FIND_LIBRARY_SUFFIXES} )
find_library ( BTF_STATIC
    NAMES btf
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/BTF
    HINTS ${CMAKE_SOURCE_DIR}/../BTF
    PATH_SUFFIXES lib build
)
set ( CMAKE_FIND_LIBRARY_SUFFIXES ${save} )

# get version of the library from the dynamic library name
get_filename_component ( BTF_LIBRARY  ${BTF_LIBRARY} REALPATH )
get_filename_component ( BTF_FILENAME ${BTF_LIBRARY} NAME )
string (
    REGEX MATCH "[0-9]+.[0-9]+.[0-9]+"
    BTF_VERSION
    ${BTF_FILENAME}
)
set ( BTF_LIBRARIES ${BTF_LIBRARY} )

if ( NOT BTF_VERSION )
    foreach ( _VERSION MAIN_VERSION SUB_VERSION SUBSUB_VERSION )
        # if the version does not appear in the filename, read the include file
        file ( STRINGS ${BTF_INCLUDE_DIR}/btf.h _VERSION_LINE REGEX "define[ ]+BTF_${_VERSION}" )
        if ( _VERSION_LINE )
            string ( REGEX REPLACE ".*define[ ]+BTF_${_VERSION}[ ]+([0-9]*).*" "\\1" _BTF_${_VERSION} "${_VERSION_LINE}" )
        endif ( )
        unset ( _VERSION_LINE )
    endforeach ( )
    set ( BTF_VERSION "${_BTF_MAIN_VERSION}.${_BTF_SUB_VERSION}.${_BTF_SUBSUB_VERSION}" )
endif ( )

include (FindPackageHandleStandardArgs)

find_package_handle_standard_args ( BTF
    REQUIRED_VARS BTF_LIBRARIES BTF_INCLUDE_DIR
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
endif ( )

