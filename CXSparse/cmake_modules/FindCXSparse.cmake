#-------------------------------------------------------------------------------
# SuiteSparse/CXSparse/cmake_modules/FindCXSparse.cmake
#-------------------------------------------------------------------------------

# The following copyright and license applies to just this file only, not to
# the library itself:
# FindCXSparse.cmake, Copyright (c) 2022, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

#-------------------------------------------------------------------------------

# Finds the CXSparse include file and compiled library and sets:

# CXSPARSE_INCLUDE_DIR - where to find cs.h
# CXSPARSE_LIBRARY     - dynamic CXSPARSE library
# CXSPARSE_STATIC      - static CXSPARSE library
# CXSPARSE_LIBRARIES   - libraries when using CXSPARSE
# CXSPARSE_FOUND       - true if CXSPARSE found

# set ``CXSparse_ROOT`` or ``CXSPARSE_ROOT`` to a CXSPARSE installation root to
# tell this module where to look.

# All the Find*.cmake files in SuiteSparse are installed by 'make install' into
# /usr/local/lib/cmake/SuiteSparse (where '/usr/local' is the
# ${CMAKE_INSTALL_PREFIX}).  To access this file, place the following commands
# in your CMakeLists.txt file.  See also SuiteSparse/Example/CMakeLists.txt:
#
#   set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
#       ${CMAKE_INSTALL_PREFIX}/lib/cmake/SuiteSparse )

#-------------------------------------------------------------------------------

# include files for CXSPARSE
find_path ( CXSPARSE_INCLUDE_DIR
    NAMES cs.h
    HINTS ${CXSPARSE_ROOT}
    HINTS ENV CXSPARSE_ROOT
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/CXSparse
    HINTS ${CMAKE_SOURCE_DIR}/../CXSparse
    PATH_SUFFIXES include Include
)

# dynamic CXSPARSE library
find_library ( CXSPARSE_LIBRARY
    NAMES cxsparse
    HINTS ${CXSPARSE_ROOT}
    HINTS ENV CXSPARSE_ROOT
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/CXSparse
    HINTS ${CMAKE_SOURCE_DIR}/../CXSparse
    PATH_SUFFIXES lib build
)

if ( MSVC )
    set ( STATIC_SUFFIX .lib )
else ( )
    set ( STATIC_SUFFIX .a )
endif ( )

# static CXSPARSE library
set ( save ${CMAKE_FIND_LIBRARY_SUFFIXES} )
set ( CMAKE_FIND_LIBRARY_SUFFIXES ${STATIC_SUFFIX} ${CMAKE_FIND_LIBRARY_SUFFIXES} )
find_library ( CXSPARSE_STATIC
    NAMES cxsparse
    HINTS ${CXSPARSE_ROOT}
    HINTS ENV CXSPARSE_ROOT
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/CXSparse
    HINTS ${CMAKE_SOURCE_DIR}/../CXSparse
    PATH_SUFFIXES lib build
)
set ( CMAKE_FIND_LIBRARY_SUFFIXES ${save} )

# get version of the library from the dynamic library name
get_filename_component ( CXSPARSE_LIBRARY  ${CXSPARSE_LIBRARY} REALPATH )
get_filename_component ( CXSPARSE_FILENAME ${CXSPARSE_LIBRARY} NAME )
string (
    REGEX MATCH "[0-9]+.[0-9]+.[0-9]+"
    CXSPARSE_VERSION
    ${CXSPARSE_FILENAME}
)

if ( NOT CXSPARSE_VERSION )
    # if the version does not appear in the filename, read the include file
    foreach (_VERSION MAIN_VERSION SUB_VERSION SUBSUB_VERSION)
        file (STRINGS ${CXSPARSE_INCLUDE_DIR}/cs.h _VERSION_LINE REGEX "define[ ]+CXSPARSE_${_VERSION}")
        if (_VERSION_LINE)
            string (REGEX REPLACE ".*define[ ]+CXSPARSE_${_VERSION}[ ]+([0-9]*).*" "\\1" _CXSPARSE_${_VERSION} "${_VERSION_LINE}")
        endif ()
        unset (_VERSION_LINE)
    endforeach ()
    set (CXSPARSE_VERSION "${_CXSPARSE_MAIN_VERSION}.${_CXSPARSE_SUB_VERSION}.${_CXSPARSE_SUBSUB_VERSION}")
endif ( )

set ( CXSPARSE_LIBRARIES ${CXSPARSE_LIBRARY} )

include (FindPackageHandleStandardArgs)

find_package_handle_standard_args ( CXSparse
    REQUIRED_VARS CXSPARSE_LIBRARIES CXSPARSE_INCLUDE_DIR
    VERSION_VAR CXSPARSE_VERSION
)

mark_as_advanced (
    CXSPARSE_INCLUDE_DIR
    CXSPARSE_LIBRARY
    CXSPARSE_STATIC
    CXSPARSE_LIBRARIES
)

if ( CXSPARSE_FOUND )
    message ( STATUS "CXSparse version: ${CXSPARSE_VERSION}" )
    message ( STATUS "CXSparse include: ${CXSPARSE_INCLUDE_DIR}" )
    message ( STATUS "CXSparse library: ${CXSPARSE_LIBRARY}" )
    message ( STATUS "CXSparse static:  ${CXSPARSE_STATIC}" )
else ( )
    message ( STATUS "CXSparse not found" )
endif ( )

