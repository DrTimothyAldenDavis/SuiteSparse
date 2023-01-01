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

# dynamic CCOLAMD library
find_library ( CCOLAMD_LIBRARY
    NAMES ccolamd
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/CCOLAMD
    HINTS ${CMAKE_SOURCE_DIR}/../CCOLAMD
    PATH_SUFFIXES lib build
)

if ( MSVC )
    set ( STATIC_SUFFIX .lib )
else ( )
    set ( STATIC_SUFFIX .a )
endif ( )

# static CCOLAMD library
set ( save ${CMAKE_FIND_LIBRARY_SUFFIXES} )
set ( CMAKE_FIND_LIBRARY_SUFFIXES ${STATIC_SUFFIX} ${CMAKE_FIND_LIBRARY_SUFFIXES} )
find_library ( CCOLAMD_STATIC
    NAMES ccolamd
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/CCOLAMD
    HINTS ${CMAKE_SOURCE_DIR}/../CCOLAMD
    PATH_SUFFIXES lib build
)
set ( CMAKE_FIND_LIBRARY_SUFFIXES ${save} )

# get version of the library from the dynamic library name
get_filename_component ( CCOLAMD_LIBRARY  ${CCOLAMD_LIBRARY} REALPATH )
get_filename_component ( CCOLAMD_FILENAME ${CCOLAMD_LIBRARY} NAME )
string (
    REGEX MATCH "[0-9]+.[0-9]+.[0-9]+"
    CCOLAMD_VERSION
    ${CCOLAMD_FILENAME}
)

if ( NOT CCOLAMD_VERSION )
    foreach ( _VERSION MAIN_VERSION SUB_VERSION SUBSUB_VERSION )
        # if the version does not appear in the filename, read the include file
        file ( STRINGS ${CCOLAMD_INCLUDE_DIR}/ccolamd.h _VERSION_LINE REGEX "define[ ]+CCOLAMD_${_VERSION}" )
        if ( _VERSION_LINE )
            string ( REGEX REPLACE ".*define[ ]+CCOLAMD_${_VERSION}[ ]+([0-9]*).*" "\\1" _CCOLAMD_${_VERSION} "${_VERSION_LINE}" )
        endif ( )
        unset ( _VERSION_LINE )
        endforeach ( )
    set ( CCOLAMD_VERSION "${_CCOLAMD_MAIN_VERSION}.${_CCOLAMD_SUB_VERSION}.${_CCOLAMD_SUBSUB_VERSION}" )
endif ( )

set ( CCOLAMD_LIBRARIES ${CCOLAMD_LIBRARY} )

include (FindPackageHandleStandardArgs)

find_package_handle_standard_args ( CCOLAMD
    REQUIRED_VARS CCOLAMD_LIBRARIES CCOLAMD_INCLUDE_DIR
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
endif ( )

