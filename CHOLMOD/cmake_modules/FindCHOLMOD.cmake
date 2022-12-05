#-------------------------------------------------------------------------------
# SuiteSparse/CHOLMOD/cmake_modules/FindCHOLMOD.cmake
#-------------------------------------------------------------------------------

# The following copyright and license applies to just this file only, not to
# the library itself:
# FindCHOLMOD.cmake, Copyright (c) 2022, Timothy A. Davis.  All Rights Reserved.
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

# dynamic CHOLMOD library
find_library ( CHOLMOD_LIBRARY
    NAMES cholmod
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/CHOLMOD
    HINTS ${CMAKE_SOURCE_DIR}/../CHOLMOD
    PATH_SUFFIXES lib build
)

if ( MSVC )
    set ( STATIC_SUFFIX .lib )
else ( )
    set ( STATIC_SUFFIX .a )
endif ( )

# static CHOLMOD library
set ( save ${CMAKE_FIND_LIBRARY_SUFFIXES} )
set ( CMAKE_FIND_LIBRARY_SUFFIXES ${STATIC_SUFFIX} ${CMAKE_FIND_LIBRARY_SUFFIXES} )
find_library ( CHOLMOD_STATIC
    NAMES cholmod
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/CHOLMOD
    HINTS ${CMAKE_SOURCE_DIR}/../CHOLMOD
    PATH_SUFFIXES lib build
)
set ( CMAKE_FIND_LIBRARY_SUFFIXES ${save} )

# get version of the library from the dynamic library name
get_filename_component ( CHOLMOD_LIBRARY  ${CHOLMOD_LIBRARY} REALPATH )
get_filename_component ( CHOLMOD_FILENAME ${CHOLMOD_LIBRARY} NAME )
string (
    REGEX MATCH "[0-9]+.[0-9]+.[0-9]+"
    CHOLMOD_VERSION
    ${CHOLMOD_FILENAME}
)

if ( NOT CHOLMOD_VERSION )
    # if the version does not appear in the filename, read the include file
    foreach (_VERSION MAIN_VERSION SUB_VERSION SUBSUB_VERSION)
        file (STRINGS ${CHOLMOD_INCLUDE_DIR}/cholmod.h _VERSION_LINE REGEX "define[ ]+CHOLMOD_${_VERSION}")
        if (_VERSION_LINE)
            string (REGEX REPLACE ".*define[ ]+CHOLMOD_${_VERSION}[ ]+([0-9]*).*" "\\1" _CHOLMOD_${_VERSION} "${_VERSION_LINE}")
        endif ()
        unset (_VERSION_LINE)
    endforeach ()
    set (CHOLMOD_VERSION "${_CHOLMOD_MAIN_VERSION}.${_CHOLMOD_SUB_VERSION}.${_CHOLMOD_SUBSUB_VERSION}")
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
endif ( )

