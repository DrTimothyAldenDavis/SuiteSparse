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

# dynamic LDL library
find_library ( LDL_LIBRARY
    NAMES ldl
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/LDL
    HINTS ${CMAKE_SOURCE_DIR}/../LDL
    PATH_SUFFIXES lib build
)

if ( MSVC )
    set ( STATIC_SUFFIX .lib )
else ( )
    set ( STATIC_SUFFIX .a )
endif ( )

# static LDL library
set ( save ${CMAKE_FIND_LIBRARY_SUFFIXES} )
set ( CMAKE_FIND_LIBRARY_SUFFIXES ${STATIC_SUFFIX} ${CMAKE_FIND_LIBRARY_SUFFIXES} )
find_library ( LDL_STATIC
    NAMES ldl
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/LDL
    HINTS ${CMAKE_SOURCE_DIR}/../LDL
    PATH_SUFFIXES lib build
)
set ( CMAKE_FIND_LIBRARY_SUFFIXES ${save} )

# get version of the library from the dynamic library name
get_filename_component ( LDL_LIBRARY  ${LDL_LIBRARY} REALPATH )
get_filename_component ( LDL_FILENAME ${LDL_LIBRARY} NAME )
string (
    REGEX MATCH "[0-9]+.[0-9]+.[0-9]+"
    LDL_VERSION
    ${LDL_FILENAME}
)

if ( NOT LDL_VERSION )
    # if the version does not appear in the filename, read the include file
    foreach ( _VERSION MAIN_VERSION SUB_VERSION SUBSUB_VERSION )
        file ( STRINGS ${LDL_INCLUDE_DIR}/ldl.h _VERSION_LINE REGEX "define[ ]+LDL_${_VERSION}" )
        if ( _VERSION_LINE )
            string ( REGEX REPLACE ".*define[ ]+LDL_${_VERSION}[ ]+([0-9]*).*" "\\1" _LDL_${_VERSION} "${_VERSION_LINE}" )
        endif ( )
        unset ( _VERSION_LINE )
    endforeach ( )
    set ( LDL_VERSION "${_LDL_MAIN_VERSION}.${_LDL_SUB_VERSION}.${_LDL_SUBSUB_VERSION}" )
endif ( )

set ( LDL_LIBRARIES ${LDL_LIBRARY} )

include (FindPackageHandleStandardArgs)

find_package_handle_standard_args ( LDL
    REQUIRED_VARS LDL_LIBRARIES LDL_INCLUDE_DIR
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
endif ( )

