#-------------------------------------------------------------------------------
# SuiteSparse/COLAMD/cmake_modules/FindCOLAMD.cmake
#-------------------------------------------------------------------------------

# The following copyright and license applies to just this file only, not to
# the library itself:
# Copyright (c) 2022, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

#-------------------------------------------------------------------------------

# Finds the COLAMD include file and compiled library and sets:

# COLAMD_INCLUDE_DIR - where to find colamd.h
# COLAMD_LIBRARY     - dynamic COLAMD library
# COLAMD_STATIC      - static COLAMD library
# COLAMD_LIBRARIES   - libraries when using COLAMD
# COLAMD_FOUND       - true if COLAMD found

# set ``COLAMD_ROOT`` to a COLAMD installation root to
# tell this module where to look.

# All the Find*.cmake files in SuiteSparse are installed by 'make install' into
# /usr/local/lib/cmake/SuiteSparse (where '/usr/local' is the
# ${CMAKE_INSTALL_PREFIX}).  To access this file, place the following commands
# in your CMakeLists.txt file.  See also SuiteSparse/Example/CMakeLists.txt:
#
#   set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
#       ${CMAKE_INSTALL_PREFIX}/lib/cmake/SuiteSparse )

#-------------------------------------------------------------------------------

# include files for COLAMD
find_path ( COLAMD_INCLUDE_DIR
    NAMES colamd.h
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/COLAMD
    HINTS ${CMAKE_SOURCE_DIR}/../COLAMD
    PATH_SUFFIXES include Include
)

# dynamic COLAMD library
find_library ( COLAMD_LIBRARY
    NAMES colamd
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/COLAMD
    HINTS ${CMAKE_SOURCE_DIR}/../COLAMD
    PATH_SUFFIXES lib build
)

if ( MSVC )
    set ( STATIC_SUFFIX .lib )
else ( )
    set ( STATIC_SUFFIX .a )
endif ( )

# static COLAMD library
set ( save ${CMAKE_FIND_LIBRARY_SUFFIXES} )
set ( CMAKE_FIND_LIBRARY_SUFFIXES ${STATIC_SUFFIX} ${CMAKE_FIND_LIBRARY_SUFFIXES} )
find_library ( COLAMD_STATIC
    NAMES colamd
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/COLAMD
    HINTS ${CMAKE_SOURCE_DIR}/../COLAMD
    PATH_SUFFIXES lib build
)
set ( CMAKE_FIND_LIBRARY_SUFFIXES ${save} )

# get version of the library from the dynamic library name
get_filename_component ( COLAMD_LIBRARY  ${COLAMD_LIBRARY} REALPATH )
get_filename_component ( COLAMD_FILENAME ${COLAMD_LIBRARY} NAME )
string (
    REGEX MATCH "[0-9]+.[0-9]+.[0-9]+"
    COLAMD_VERSION
    ${COLAMD_FILENAME}
)

if ( NOT COLAMD_VERSION )
    # if the version does not appear in the filename, read the include file
    foreach (_VERSION MAIN_VERSION SUB_VERSION SUBSUB_VERSION)
        file (STRINGS ${COLAMD_INCLUDE_DIR}/colamd.h _VERSION_LINE REGEX "define[ ]+COLAMD_${_VERSION}")
        if (_VERSION_LINE)
            string (REGEX REPLACE ".*define[ ]+COLAMD_${_VERSION}[ ]+([0-9]*).*" "\\1" _COLAMD_${_VERSION} "${_VERSION_LINE}")
        endif ()
        unset (_VERSION_LINE)
    endforeach ()
    set (COLAMD_VERSION "${_COLAMD_MAIN_VERSION}.${_COLAMD_SUB_VERSION}.${_COLAMD_SUBSUB_VERSION}")
endif ( )

set (COLAMD_LIBRARIES ${COLAMD_LIBRARY})

include (FindPackageHandleStandardArgs)

find_package_handle_standard_args ( COLAMD
    REQUIRED_VARS COLAMD_LIBRARIES COLAMD_INCLUDE_DIR
    VERSION_VAR COLAMD_VERSION
)

mark_as_advanced (
    COLAMD_INCLUDE_DIR
    COLAMD_LIBRARY
    COLAMD_STATIC
    COLAMD_LIBRARIES
)

if ( COLAMD_FOUND )
    message ( STATUS "COLAMD version: ${COLAMD_VERSION}" )
    message ( STATUS "COLAMD include: ${COLAMD_INCLUDE_DIR}" )
    message ( STATUS "COLAMD library: ${COLAMD_LIBRARY}" )
    message ( STATUS "COLAMD static:  ${COLAMD_STATIC}" )
else ( )
    message ( STATUS "COLAMD not found" )
endif ( )

