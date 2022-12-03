#-------------------------------------------------------------------------------
# SuiteSparse/SuiteSparse_config/cmake_modules/FindBTF.cmake
#-------------------------------------------------------------------------------

# Copyright (c) 2022, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

# Finds the BTF include file and compiled library and sets:

# BTF_INCLUDE_DIR - where to find btf.h
# BTF_LIBRARY     - dynamic BTF library
# BTF_STATIC      - static BTF library
# BTF_LIBRARIES   - libraries when using BTF
# BTF_FOUND       - true if BTF found

# set ``BTF_ROOT`` to a BTF installation root to
# tell this module where to look.

# To use this file in your application, copy this file into MyApp/cmake_modules
# where MyApp is your application and add the following to your
# MyApp/CMakeLists.txt file:
#
#   set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake_modules")
#
# or, assuming MyApp and SuiteSparse sit side-by-side in a common folder, you
# can leave this file in place and use this command (revise as needed):
#
#   set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
#       "${CMAKE_SOURCE_DIR}/../SuiteSparse/SuiteSparse_config/cmake_modules")

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

# static BTF library
set ( save ${CMAKE_FIND_LIBRARY_SUFFIXES} )
set ( CMAKE_FIND_LIBRARY_SUFFIXES ${STATIC_SUFFIX} ${CMAKE_FIND_LIBRARY_SUFFIXES} )
find_library ( BTF_LIBRARY
    NAMES btf
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/BTF
    HINTS ${CMAKE_SOURCE_DIR}/../BTF
    PATH_SUFFIXES lib build
)
set ( ${CMAKE_FIND_LIBRARY_SUFFIXES} save )

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
    message ( STATUS "BTF version:     ${BTF_VERSION}" )
    message ( STATUS "BTF include dir: ${BTF_INCLUDE_DIR}" )
    message ( STATUS "BTF dynamic:     ${BTF_LIBRARY}" )
    message ( STATUS "BTF static:      ${BTF_STATIC}" )
else ( )
    message ( STATUS "BTF not found" )
endif ( )

