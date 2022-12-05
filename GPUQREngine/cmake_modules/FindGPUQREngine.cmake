#-------------------------------------------------------------------------------
# SuiteSparse/GPUQREngine/cmake_modules/FindGPUQREngine.cmake
#-------------------------------------------------------------------------------

# The following copyright and license applies to just this file only, not to
# the library itself:
# Copyright (c) 2022, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

#-------------------------------------------------------------------------------

# Finds the GPUQREngine compiled library and sets:

# GPUQRENGINE_INCLUDE_DIR - where to find GPUQREngine.hpp
# GPUQRENGINE_LIBRARY     - dynamic GPUQREngine library
# GPUQRENGINE_STATIC      - static GPUQREngine library
# GPUQRENGINE_LIBRARIES   - libraries when using GPUQREngine
# GPUQRENGINE_FOUND       - true if GPUQREngine found

# set ``GPUQREngine_ROOT`` or ``GPUQRENGINE_ROOT`` to a GPUQREngine
# installation root to tell this module where to look.

# All the Find*.cmake files in SuiteSparse are installed by 'make install' into
# /usr/local/lib/cmake/SuiteSparse (where '/usr/local' is the
# ${CMAKE_INSTALL_PREFIX}).  To access this file, place the following commands
# in your CMakeLists.txt file.  See also SuiteSparse/Example/CMakeLists.txt:
#
#   set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
#       ${CMAKE_INSTALL_PREFIX}/lib/cmake/SuiteSparse )

#-------------------------------------------------------------------------------

# include files for GPUQREngine
find_path ( GPUQRENGINE_INCLUDE_DIR
    NAMES GPUQREngine.hpp
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/GPUQREngine
    HINTS ${CMAKE_SOURCE_DIR}/../GPUQREngine
    PATH_SUFFIXES include Include
)

# dynamic GPUQREngine library
find_library ( GPUQRENGINE_LIBRARY
    NAMES gpuqrengine
    HINTS ${GPUQRENGINE_ROOT}
    HINTS ENV GPUQRENGINE_ROOT
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/GPUQREngine
    HINTS ${CMAKE_SOURCE_DIR}/../GPUQREngine
    PATH_SUFFIXES lib build
)

if ( MSVC )
    set ( STATIC_SUFFIX .lib )
else ( )
    set ( STATIC_SUFFIX .a )
endif ( )

# static GPUQREngine library
set ( save ${CMAKE_FIND_LIBRARY_SUFFIXES} )
set ( CMAKE_FIND_LIBRARY_SUFFIXES ${STATIC_SUFFIX} ${CMAKE_FIND_LIBRARY_SUFFIXES} )
find_library ( GPUQRENGINE_STATIC
    NAMES gpuqrengine_static
    HINTS ${GPUQRENGINE_ROOT}
    HINTS ENV GPUQRENGINE_ROOT
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/GPUQREngine
    HINTS ${CMAKE_SOURCE_DIR}/../GPUQREngine
    PATH_SUFFIXES lib build
)
set ( CMAKE_FIND_LIBRARY_SUFFIXES ${save} )

# get version of the library from the dynamic library name
get_filename_component ( GPUQRENGINE_LIBRARY  ${GPUQRENGINE_LIBRARY} REALPATH )
get_filename_component ( GPUQRENGINE_FILENAME ${GPUQRENGINE_LIBRARY} NAME )
string (
    REGEX MATCH "[0-9]+.[0-9]+.[0-9]+"
    GPUQRENGINE_VERSION
    ${GPUQRENGINE_FILENAME}
)

if ( NOT GPUQRENGINE_VERSION )
    # if the version does not appear in the filename, read the include file
    foreach ( _VERSION MAIN_VERSION SUB_VERSION SUBSUB_VERSION )
        file ( STRINGS ${GPUQRENGINE_INCLUDE_DIR}/amd.h _VERSION_LINE REGEX "define[ ]+GPUQRENGINE_${_VERSION}" )
        if ( _VERSION_LINE )
            string ( REGEX REPLACE ".*define[ ]+GPUQRENGINE_${_VERSION}[ ]+([0-9]*).*" "\\1" _GPUQRENGINE_${_VERSION} "${_VERSION_LINE}" )
        endif ( )
        unset ( _VERSION_LINE )
    endforeach ( )
    set ( GPUQRENGINE_VERSION "${_GPUQRENGINE_MAIN_VERSION}.${_GPUQRENGINE_SUB_VERSION}.${_GPUQRENGINE_SUBSUB_VERSION}" )
endif ( )

# libaries when using GPUQREngine
set (GPUQRENGINE_LIBRARIES ${GPUQRENGINE_LIBRARY})

include (FindPackageHandleStandardArgs)

find_package_handle_standard_args ( GPUQREngine
    REQUIRED_VARS GPUQRENGINE_LIBRARIES
    VERSION_VAR GPUQRENGINE_VERSION
)

mark_as_advanced (
    GPUQRENGINE_INCLUDE_DIR
    GPUQRENGINE_LIBRARY
    GPUQRENGINE_STATIC
    GPUQRENGINE_LIBRARIES
)

if ( GPUQRENGINE_FOUND )
    message ( STATUS "GPUQREngine version: ${GPUQRENGINE_VERSION}" )
    message ( STATUS "GPUQREngine include: ${GPUQRENGINE_INCLUDE_DIR}" )
    message ( STATUS "GPUQREngine library: ${GPUQRENGINE_LIBRARY}" )
    message ( STATUS "GPUQREngine static:  ${GPUQRENGINE_STATIC}" )
else ( )
    message ( STATUS "GPUQREngine not found" )
endif ( )

