#[=======================================================================[.rst:
FindGraphBLAS
--------

The following copyright and license applies to just this file only, not to
the GraphBLAS library itself:
LAGraph, (c) 2019-2022 by The LAGraph Contributors, All Rights Reserved.
SPDX-License-Identifier: BSD-2-Clause
See additional acknowledgments in the LICENSE file,
or contact permission@sei.cmu.edu for the full terms.

Find the native GRAPHBLAS includes and library.

IMPORTED Targets
^^^^^^^^^^^^^^^^

This module defines :prop_tgt:`IMPORTED` target ``GRAPHBLAS::GRAPHBLAS``, if
GRAPHBLAS has been found.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

::

  GRAPHBLAS_INCLUDE_DIR    - where to find GraphBLAS.h, etc.
  GRAPHBLAS_LIBRARY        - dynamic GraphBLAS library
  GRAPHBLAS_STATIC         - static GraphBLAS library
  GRAPHBLAS_LIBRARIES      - List of libraries when using GraphBLAS.
  GRAPHBLAS_FOUND          - True if GraphBLAS found.

::

Hints
^^^^^

A user may set ``GRAPHBLAS_ROOT`` or ``GraphBLAS_ROOT`` to a GraphBLAS
installation root to tell this module where to look.

Otherwise, the first place searched is in ../GraphBLAS, relative to the LAGraph
source directory.  That is, if GraphBLAS and LAGraph reside in the same parent
folder, side-by-side, and if it contains GraphBLAS/Include/GraphBLAS.h file and
GraphBLAS/build/libgraphblas.so (or dylib, etc), then that version is used.
This takes precedence over the system-wide installation of GraphBLAS, which
might be an older version.  This method gives the user the ability to compile
LAGraph with their own copy of GraphBLAS, ignoring the system-wide version.

If SuiteSparse:GraphBLAS is the GraphBLAS library being utilized,
all the Find*.cmake files in SuiteSparse are installed by 'make install' into
/usr/local/lib/cmake/SuiteSparse (where '/usr/local' is the
${CMAKE_INSTALL_PREFIX}).  To access this file, place the following commands
in your CMakeLists.txt file.  See also SuiteSparse/Example/CMakeLists.txt:

    set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
        ${CMAKE_INSTALL_PREFIX}/lib/cmake/SuiteSparse )

#]=======================================================================]

# NB: this is built around assumptions about one particular GraphBLAS
# installation (SuiteSparse:GraphBLAS). As other installations become available
# changes to this will likely be required.

# "Include" for SuiteSparse:GraphBLAS
find_path ( GRAPHBLAS_INCLUDE_DIR
  NAMES GraphBLAS.h
  HINTS ${GRAPHBLAS_ROOT}
  HINTS ENV GRAPHBLAS_ROOT
  HINTS ${CMAKE_SOURCE_DIR}/..
  HINTS ${CMAKE_SOURCE_DIR}/../GraphBLAS
  HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/GraphBLAS
  PATH_SUFFIXES include Include
  )

# dynamic SuiteSparse:GraphBLAS library
find_library ( GRAPHBLAS_LIBRARY
  NAMES graphblas
  HINTS ${GRAPHBLAS_ROOT}
  HINTS ENV GRAPHBLAS_ROOT
  HINTS ${CMAKE_SOURCE_DIR}/..
  HINTS ${CMAKE_SOURCE_DIR}/../GraphBLAS
  HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/GraphBLAS
  PATH_SUFFIXES lib build alternative
  )

if ( MSVC )
    set ( STATIC_SUFFIX .lib )
else ( )
    set ( STATIC_SUFFIX .a )
endif ( )

# static SuiteSparse:GraphBLAS library
set ( save ${CMAKE_FIND_LIBRARY_SUFFIXES} )
set ( CMAKE_FIND_LIBRARY_SUFFIXES ${STATIC_SUFFIX} ${CMAKE_FIND_LIBRARY_SUFFIXES} )
find_library ( GRAPHBLAS_STATIC
  NAMES graphblas
  HINTS ${GRAPHBLAS_ROOT}
  HINTS ENV GRAPHBLAS_ROOT
  HINTS ${CMAKE_SOURCE_DIR}/..
  HINTS ${CMAKE_SOURCE_DIR}/../GraphBLAS
  HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/GraphBLAS
  PATH_SUFFIXES lib build alternative
  )
set ( CMAKE_FIND_LIBRARY_SUFFIXES ${save} )

# get version of the library from the dynamic library name
get_filename_component ( GRAPHBLAS_LIBRARY  ${GRAPHBLAS_LIBRARY} REALPATH )
get_filename_component ( GRAPHBLAS_FILENAME ${GRAPHBLAS_LIBRARY} NAME )
string (
    REGEX MATCH "[0-9]+.[0-9]+.[0-9]+"
    GRAPHBLAS_VERSION
    ${GRAPHBLAS_FILENAME}
  )

if ( NOT GRAPHBLAS_VERSION )
    # if the version does not appear in the filename, read the include file
    foreach ( _VERSION MAJOR MINOR SUB )
        file ( STRINGS ${GRAPHBLAS_INCLUDE_DIR}/GraphBLAS.h _VERSION_LINE REGEX "define[ ]+GxB_IMPLEMENTATION_${_VERSION}" )
        if ( _VERSION_LINE )
            string (REGEX REPLACE ".*define[ ]+GxB_IMPLEMENTATION_${_VERSION}[ ]+([0-9]*).*" "\\1" _GRAPHBLAS_${_VERSION} "${_VERSION_LINE}")
        endif ( )
        unset ( _VERSION_LINE )
    endforeach ( )
    set (GRAPHBLAS_VERSION "${_GRAPHBLAS_MAJOR}.${_GRAPHBLAS_MINOR}.${_GRAPHBLAS_SUB}")
endif ( )

set ( GRAPHBLAS_LIBRARIES ${GRAPHBLAS_LIBRARY} )

include ( FindPackageHandleStandardArgs )

find_package_handle_standard_args(
  GraphBLAS
  REQUIRED_VARS GRAPHBLAS_LIBRARIES GRAPHBLAS_INCLUDE_DIR
  VERSION_VAR GRAPHBLAS_VERSION
  )

mark_as_advanced(
  GRAPHBLAS_INCLUDE_DIR
  GRAPHBLAS_LIBRARY
  GRAPHBLAS_STATIC
  GRAPHBLAS_LIBRARIES
  )

if ( GRAPHBLAS_FOUND )
    message ( STATUS "GraphBLAS version: ${GRAPHBLAS_VERSION}" )
    message ( STATUS "GraphBLAS include: ${GRAPHBLAS_INCLUDE_DIR}" )
    message ( STATUS "GraphBLAS library: ${GRAPHBLAS_LIBRARY}" )
    message ( STATUS "GraphBLAS static:  ${GRAPHBLAS_STATIC}" )
else ( )
    message ( STATUS "GraphBLAS not found" )
endif ( )

