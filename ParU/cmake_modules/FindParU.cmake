#-------------------------------------------------------------------------------
# SuiteSparse/ParU/cmake_modules/FindParU.cmake
#-------------------------------------------------------------------------------

# The following copyright and license applies to just this file only, not to
# the library itself:
# FindParU.cmake, Copyright (c) 2022, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

#-------------------------------------------------------------------------------

# Finds the ParU include file and compiled library and sets:

# PARU_INCLUDE_DIR - where to find ParU.hpp
# PARU_LIBRARY     - dynamic PARU library
# PARU_STATIC      - static PARU library
# PARU_LIBRARIES   - libraries when using PARU
# PARU_FOUND       - true if PARU found

# set ``PARU_ROOT`` or ``ParU_ROOT`` to an PARU installation root to
# tell this module where to look.

# All the Find*.cmake files in SuiteSparse are installed by 'make install' into
# /usr/local/lib/cmake/SuiteSparse (where '/usr/local' is the
# ${CMAKE_INSTALL_PREFIX}).  To access this file, place the following commands
# in your CMakeLists.txt file.  See also SuiteSparse/Example/CMakeLists.txt:
#
#   set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
#       ${CMAKE_INSTALL_PREFIX}/lib/cmake/SuiteSparse )

#-------------------------------------------------------------------------------

# include files for ParU
find_path ( PARU_INCLUDE_DIR
    NAMES ParU.hpp
    HINTS ${PARU_ROOT}
    HINTS ENV PARU_ROOT
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/ParU
    HINTS ${CMAKE_SOURCE_DIR}/../ParU
    PATH_SUFFIXES include Include
)

# dynamic ParU library
find_library ( PARU_LIBRARY
    NAMES paru
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/ParU
    HINTS ${CMAKE_SOURCE_DIR}/../ParU
    PATH_SUFFIXES lib build
)

if ( MSVC )
    set ( STATIC_SUFFIX .lib )
else ( )
    set ( STATIC_SUFFIX .a )
endif ( )

# static ParU library
set ( save ${CMAKE_FIND_LIBRARY_SUFFIXES} )
set ( CMAKE_FIND_LIBRARY_SUFFIXES ${STATIC_SUFFIX} ${CMAKE_FIND_LIBRARY_SUFFIXES} )
find_library ( PARU_STATIC
    NAMES paru
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/ParU
    HINTS ${CMAKE_SOURCE_DIR}/../ParU
    PATH_SUFFIXES lib build
)
set ( CMAKE_FIND_LIBRARY_SUFFIXES ${save} )

# get version of the library from the dynamic library name
get_filename_component ( PARU_LIBRARY  ${PARU_LIBRARY} REALPATH )
get_filename_component ( PARU_FILENAME ${PARU_LIBRARY} NAME )
string (
    REGEX MATCH "[0-9]+.[0-9]+.[0-9]+"
    PARU_VERSION
    ${PARU_FILENAME}
)

if ( NOT PARU_VERSION )
    # if the version does not appear in the filename, read the include file
    foreach ( _VERSION MAIN_VERSION MINOR_VERSION UPDATE_VERSION )
        file ( STRINGS ${PARU_INCLUDE_DIR}/ParU.h _VERSION_LINE REGEX "define[ ]+PARU_${_VERSION}" )
        if ( _VERSION_LINE )
            string ( REGEX REPLACE ".*define[ ]+PARU_${_VERSION}[ ]+([0-9]*).*" "\\1" _PARU_${_VERSION} "${_VERSION_LINE}" )
        endif ( )
        unset ( _VERSION_LINE )
    endforeach ( )
    set ( PARU_VERSION "${_PARU_MAIN_VERSION}.${_PARU_MINOR_VERSION}.${_PARU_UPDATE_VERSION}" )
endif ( )

set ( PARU_LIBRARIES ${PARU_LIBRARY} )

include (FindPackageHandleStandardArgs)

find_package_handle_standard_args ( PARU
    REQUIRED_VARS PARU_LIBRARIES PARU_INCLUDE_DIR
    VERSION_VAR PARU_VERSION
)

mark_as_advanced (
    PARU_INCLUDE_DIR
    PARU_LIBRARY
    PARU_STATIC
    PARU_LIBRARIES
)

if ( PARU_FOUND )
    message ( STATUS "ParU version: ${PARU_VERSION}" )
    message ( STATUS "ParU include: ${PARU_INCLUDE_DIR}")
    message ( STATUS "ParU library: ${PARU_LIBRARY}")
    message ( STATUS "ParU static:  ${PARU_STATIC}")
else ( )
    message ( STATUS "ParU not found" )
endif ( )

