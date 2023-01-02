#-------------------------------------------------------------------------------
# SuiteSparse/CHOLMOD/cmake_modules/FindCHOLMOD_CUDA.cmake
#-------------------------------------------------------------------------------

# The following copyright and license applies to just this file only, not to
# the library itself:
# FindCHOLMOD_CUDA.cmake, Copyright (c) 2022-2023, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

#-------------------------------------------------------------------------------

# Finds the CHOLMOD_CUDA compiled library and sets:

# CHOLMOD_CUDA_LIBRARY     - dynamic CHOLMOD_CUDA library
# CHOLMOD_CUDA_STATIC      - static CHOLMOD_CUDA library
# CHOLMOD_CUDA_LIBRARIES   - libraries when using CHOLMOD_CUDA
# CHOLMOD_CUDA_FOUND       - true if CHOLMOD_CUDA found

# set ``CHOLMOD_CUDA_ROOT`` to a CHOLMOD_CUDA installation root to
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

# dynamic CHOLMOD_CUDA library
find_library ( CHOLMOD_CUDA_LIBRARY
    NAMES cholmod_cuda
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse
    HINTS ${CMAKE_SOURCE_DIR}/../CHOLMOD/
    HINTS ${CMAKE_SOURCE_DIR}/../CHOLMOD/build/GPU
    PATH_SUFFIXES lib build
)

if ( MSVC )
    set ( STATIC_SUFFIX .lib )
else ( )
    set ( STATIC_SUFFIX .a )
endif ( )

# static CHOLMOD_CUDA library
set ( save ${CMAKE_FIND_LIBRARY_SUFFIXES} )
set ( CMAKE_FIND_LIBRARY_SUFFIXES ${STATIC_SUFFIX} ${CMAKE_FIND_LIBRARY_SUFFIXES} )
find_library ( CHOLMOD_CUDA_STATIC
    NAMES cholmod_cuda
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse
    HINTS ${CMAKE_SOURCE_DIR}/../CHOLMOD/
    HINTS ${CMAKE_SOURCE_DIR}/../CHOLMOD/build/GPU
    PATH_SUFFIXES lib build
)
set ( CMAKE_FIND_LIBRARY_SUFFIXES ${save} )

# get version of the library from the dynamic library name
get_filename_component ( CHOLMOD_CUDA_LIBRARY  ${CHOLMOD_CUDA_LIBRARY} REALPATH )
get_filename_component ( CHOLMOD_CUDA_FILENAME ${CHOLMOD_CUDA_LIBRARY} NAME )
string (
    REGEX MATCH "[0-9]+.[0-9]+.[0-9]+"
    CHOLMOD_CUDA_VERSION
    ${CHOLMOD_CUDA_FILENAME}
)

# set ( CHOLMOD_CUDA_VERSION "" )
if ( EXISTS "${CHOLMOD_INCLUDE_DIR}" AND NOT CHOLMOD_CUDA_VERSION )
    # if the version does not appear in the filename, read the include file
    file ( STRINGS ${CHOLMOD_INCLUDE_DIR}/cholmod.h CHOLMOD_CUDA_MAJOR_STR
        REGEX "define CHOLMOD_MAIN_VERSION" )
    file ( STRINGS ${CHOLMOD_INCLUDE_DIR}/cholmod.h CHOLMOD_CUDA_MINOR_STR
        REGEX "define CHOLMOD_SUB_VERSION" )
    file ( STRINGS ${CHOLMOD_INCLUDE_DIR}/cholmod.h CHOLMOD_CUDA_PATCH_STR
        REGEX "define CHOLMOD_SUBSUB_VERSION" )
    message ( STATUS "major: ${CHOLMOD_CUDA_MAJOR_STR}" )
    message ( STATUS "minor: ${CHOLMOD_CUDA_MINOR_STR}" )
    message ( STATUS "patch: ${CHOLMOD_CUDA_PATCH_STR}" )
    string ( REGEX MATCH "[0-9]+" CHOLMOD_CUDA_MAJOR ${CHOLMOD_CUDA_MAJOR_STR} )
    string ( REGEX MATCH "[0-9]+" CHOLMOD_CUDA_MINOR ${CHOLMOD_CUDA_MINOR_STR} )
    string ( REGEX MATCH "[0-9]+" CHOLMOD_CUDA_PATCH ${CHOLMOD_CUDA_PATCH_STR} )
    set (CHOLMOD_CUDA_VERSION "${CHOLMOD_CUDA_MAJOR}.${CHOLMOD_CUDA_MINOR}.${CHOLMOD_CUDA_PATCH}")
endif ( )

set (CHOLMOD_CUDA_LIBRARIES ${CHOLMOD_CUDA_LIBRARY})

include (FindPackageHandleStandardArgs)

find_package_handle_standard_args ( CHOLMOD_CUDA
    REQUIRED_VARS CHOLMOD_CUDA_LIBRARIES
    VERSION_VAR CHOLMOD_CUDA_VERSION
)

mark_as_advanced (
    CHOLMOD_CUDA_LIBRARY
    CHOLMOD_CUDA_STATIC
    CHOLMOD_CUDA_LIBRARIES
)

if ( CHOLMOD_CUDA_FOUND )
    message ( STATUS "CHOLMOD_CUDA version: ${CHOLMOD_CUDA_VERSION}" )
    message ( STATUS "CHOLMOD_CUDA library: ${CHOLMOD_CUDA_LIBRARY}" )
    message ( STATUS "CHOLMOD_CUDA static:  ${CHOLMOD_CUDA_STATIC}" )
else ( )
    message ( STATUS "CHOLMOD_CUDA not found" )
    set ( CHOLMOD_CUDA_LIBRARIES "" )
    set ( CHOLMOD_CUDA_LIBRARY "" )
    set ( CHOLMOD_CUDA_STATIC "" )
endif ( )

