#-------------------------------------------------------------------------------
# SuiteSparse_GPURuntime/cmake_modules/FindSuiteSparse_GPURuntime.cmake
#-------------------------------------------------------------------------------

# The following copyright and license applies to just this file only, not to
# the library itself:
# FindSuiteSparse_GPURuntime.cmake, Copyright (c) 2022-2023, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

#-------------------------------------------------------------------------------

# Finds the SuiteSparse_GPURuntime compiled library and sets:

# SUITESPARSE_GPURUNTIME_INCLUDE_DIR - include directory for SuiteSparse_GPURuntime
# SUITESPARSE_GPURUNTIME_LIBRARIES - libraries when using SuiteSparse_GPURuntime
# SUITESPARSE_GPURUNTIME_LIBRARY   - dynamic SuiteSparse_GPURuntime library
# SUITESPARSE_GPURUNTIME_STATIC    - static SuiteSparse_GPURuntime library
# SUITESPARSE_GPURUNTIME_FOUND     - true if SuiteSparse_GPURuntime found

# set ``SUITESPARSE_GPURUNTIME_ROOT`` or ``SuiteSparse_GPURuntime_ROOT`` to a
# SuiteSparse_GPURuntime installation root to tell this module where to look.

# All the Find*.cmake files in SuiteSparse are installed by 'make install' into
# /usr/local/lib/cmake/SuiteSparse (where '/usr/local' is the
# ${CMAKE_INSTALL_PREFIX}).  To access this file, place the following commands
# in your CMakeLists.txt file.  See also SuiteSparse/Example/CMakeLists.txt:
#
#   set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
#       ${CMAKE_INSTALL_PREFIX}/lib/cmake/SuiteSparse )

#-------------------------------------------------------------------------------

# include files for SuiteSparse_GPURuntime
find_path ( SUITESPARSE_GPURUNTIME_INCLUDE_DIR
    NAMES SuiteSparse_GPURuntime.hpp
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/SuiteSparse_GPURuntime
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse_GPURuntime
    PATH_SUFFIXES include Include
)

# dynamic SuiteSparse_GPURuntime library
find_library ( SUITESPARSE_GPURUNTIME_LIBRARY
    NAMES suitesparse_gpuruntime
    HINTS ${SUITESPARSE_GPURUNTIME_ROOT}
    HINTS ENV SUITESPARSE_GPURUNTIME_ROOT
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/SuiteSparse_GPURuntime
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse_GPURuntime
    PATH_SUFFIXES lib build
)

if ( MSVC )
    set ( STATIC_SUFFIX .lib )
else ( )
    set ( STATIC_SUFFIX .a )
endif ( )

# static SuiteSparse_GPURuntime library
set ( save ${CMAKE_FIND_LIBRARY_SUFFIXES} )
set ( CMAKE_FIND_LIBRARY_SUFFIXES ${STATIC_SUFFIX} ${CMAKE_FIND_LIBRARY_SUFFIXES} )
find_library ( SUITESPARSE_GPURUNTIME_STATIC
    NAMES suitesparse_gpuruntime_static
    HINTS ${SUITESPARSE_GPURUNTIME_ROOT}
    HINTS ENV SUITESPARSE_GPURUNTIME_ROOT
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/SuiteSparse_GPURuntime
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse_GPURuntime
    PATH_SUFFIXES lib build
)
set ( CMAKE_FIND_LIBRARY_SUFFIXES ${save} )

# get version of the library from the dynamic library name
get_filename_component ( SUITESPARSE_GPURUNTIME_LIBRARY  ${SUITESPARSE_GPURUNTIME_LIBRARY} REALPATH )
get_filename_component ( SUITESPARSE_GPURUNTIME_FILENAME ${SUITESPARSE_GPURUNTIME_LIBRARY} NAME )
string (
    REGEX MATCH "[0-9]+.[0-9]+.[0-9]+"
    SUITESPARSE_GPURUNTIME_VERSION
    ${SUITESPARSE_GPURUNTIME_FILENAME}
)

if ( NOT SUITESPARSE_GPURUNTIME_VERSION )
    # if the version does not appear in the filename, read the include file
    foreach ( _VERSION MAIN_VERSION SUB_VERSION SUBSUB_VERSION )
        file ( STRINGS ${SUITESPARSE_GPURUNTIME_INCLUDE_DIR}/amd.h _VERSION_LINE REGEX "define[ ]+SUITESPARSE_GPURUNTIME_${_VERSION}" )
        if ( _VERSION_LINE )
            string ( REGEX REPLACE ".*define[ ]+SUITESPARSE_GPURUNTIME_${_VERSION}[ ]+([0-9]*).*" "\\1" _SUITESPARSE_GPURUNTIME_${_VERSION} "${_VERSION_LINE}" )
        endif ( )
        unset ( _VERSION_LINE )
    endforeach ( )
    set ( SUITESPARSE_GPURUNTIME_VERSION "${_SUITESPARSE_GPURUNTIME_MAIN_VERSION}.${_SUITESPARSE_GPURUNTIME_SUB_VERSION}.${_SUITESPARSE_GPURUNTIME_SUBSUB_VERSION}" )
endif ( )

set ( SUITESPARSE_GPURUNTIME_LIBRARIES ${SUITESPARSE_GPURUNTIME_LIBRARY} )

include (FindPackageHandleStandardArgs)

find_package_handle_standard_args ( SuiteSparse_GPURuntime
    REQUIRED_VARS SUITESPARSE_GPURUNTIME_LIBRARIES
    VERSION_VAR SUITESPARSE_GPURUNTIME_VERSION
)

mark_as_advanced (
    SUITESPARSE_GPURUNTIME_INCLUDE_DIR
    SUITESPARSE_GPURUNTIME_LIBRARY
    SUITESPARSE_GPURUNTIME_STATIC
    SUITESPARSE_GPURUNTIME_LIBRARIES
)

if ( SUITESPARSE_GPURUNTIME_FOUND )
    message ( STATUS "SuiteSparse_GPURuntime version: ${SUITESPARSE_GPURUNTIME_VERSION}" )
    message ( STATUS "SuiteSparse_GPURuntime include: ${SUITESPARSE_GPURUNTIME_INCLUDE_DIR}" )
    message ( STATUS "SuiteSparse_GPURuntime library: ${SUITESPARSE_GPURUNTIME_LIBRARY}" )
    message ( STATUS "SuiteSparse_GPURuntime static:  ${SUITESPARSE_GPURUNTIME_STATIC}" )
else ( )
    message ( STATUS "SuiteSparse_GPURuntime not found" )
endif ( )

