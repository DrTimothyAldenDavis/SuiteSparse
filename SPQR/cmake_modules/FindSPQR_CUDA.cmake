#-------------------------------------------------------------------------------
# SuiteSparse/SPQR/cmake_modules/FindSPQR_CUDA.cmake
#-------------------------------------------------------------------------------

# The following copyright and license applies to just this file only, not to
# the library itself:
# FindSPQR_CUDA.cmake, Copyright (c) 2022-2023, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

#-------------------------------------------------------------------------------

# Finds the SPQR_CUDA compiled library and sets:

# SPQR_CUDA_LIBRARIES   - libraries when using SPQR_CUDA
# SPQR_CUDA_LIBRARY     - dynamic SPQR_CUDA library
# SPQR_CUDA_STATIC      - static SPQR_CUDA library
# SPQR_CUDA_FOUND       - true if SPQR_CUDA found

# set ``SPQR_CUDA_ROOT`` to a SPQR_CUDA installation root to
# tell this module where to look.

# All the Find*.cmake files in SuiteSparse are installed by 'make install' into
# /usr/local/lib/cmake/SuiteSparse (where '/usr/local' is the
# ${CMAKE_INSTALL_PREFIX}).  To access this file, place the following commands
# in your CMakeLists.txt file.  See also SuiteSparse/Example/CMakeLists.txt:
#
#   set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
#       ${CMAKE_INSTALL_PREFIX}/lib/cmake/SuiteSparse )

#-------------------------------------------------------------------------------

# dynamic SPQR_CUDA library
find_library ( SPQR_CUDA_LIBRARY
    NAMES spqr_cuda
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse
    HINTS ${CMAKE_SOURCE_DIR}/../SPQR/
    HINTS ${CMAKE_SOURCE_DIR}/../SPQR/build/SPQRGPU
    PATH_SUFFIXES lib build
)

if ( MSVC )
    set ( STATIC_SUFFIX .lib )
else ( )
    set ( STATIC_SUFFIX .a )
endif ( )

# static SPQR_CUDA library
set ( save ${CMAKE_FIND_LIBRARY_SUFFIXES} )
set ( CMAKE_FIND_LIBRARY_SUFFIXES ${STATIC_SUFFIX} ${CMAKE_FIND_LIBRARY_SUFFIXES} )
find_library ( SPQR_CUDA_STATIC
    NAMES spqr_cuda_static
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse
    HINTS ${CMAKE_SOURCE_DIR}/../SPQR/
    HINTS ${CMAKE_SOURCE_DIR}/../SPQR/build/SPQRGPU
    PATH_SUFFIXES lib build
)
set ( CMAKE_FIND_LIBRARY_SUFFIXES ${save} )

# get version of the library from the dynamic library name
get_filename_component ( SPQR_CUDA_LIBRARY  ${SPQR_CUDA_LIBRARY} REALPATH )
get_filename_component ( SPQR_CUDA_FILENAME ${SPQR_CUDA_LIBRARY} NAME )
string (
    REGEX MATCH "[0-9]+.[0-9]+.[0-9]+"
    SPQR_CUDA_VERSION
    ${SPQR_CUDA_FILENAME}
)

if ( NOT SPQR_CUDA_VERSION )
    # get version of the library from SPQR
    find_package ( SPQR )
    set ( SPQR_CUDA_VERSION "${SPQR_VERSION}" )
endif ( )

set ( SPQR_CUDA_LIBRARIES ${SPQR_CUDA_LIBRARY} )

include (FindPackageHandleStandardArgs)

find_package_handle_standard_args ( SPQR_CUDA
    REQUIRED_VARS SPQR_CUDA_LIBRARY
    VERSION_VAR SPQR_CUDA_VERSION
)

mark_as_advanced (
    SPQR_CUDA_LIBRARY
    SPQR_CUDA_STATIC
    SPQR_CUDA_LIBRARIES
)

if ( SPQR_CUDA_FOUND )
    message ( STATUS "SPQR_CUDA version: ${SPQR_CUDA_VERSION}" )
    message ( STATUS "SPQR_CUDA library: ${SPQR_CUDA_LIBRARY}" )
    message ( STATUS "SPQR_CUDA static:  ${SPQR_CUDA_STATIC}" )
else ( )
    message ( STATUS "SPQR_CUDA not found" )
    set ( SPQR_CUDA_LIBRARIES "" )
    set ( SPQR_CUDA_LIBRARY "" )
    set ( SPQR_CUDA_STATIC "" )
endif ( )

