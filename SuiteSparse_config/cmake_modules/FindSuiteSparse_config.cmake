#-------------------------------------------------------------------------------
# SuiteSparse/SuiteSparse_config/cmake_modules/FindSuiteSparse_config.cmake
#-------------------------------------------------------------------------------

# The following copyright and license applies to just this file only, not to
# the library itself:
# FindSuiteSparse_config.cmake, Copyright (c) 2022-2023, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

#-------------------------------------------------------------------------------

# Finds the SuiteSparse_config include file and compiled library and sets:

# SUITESPARSE_CONFIG_INCLUDE_DIR - where to find SuiteSparse_config.h
# SUITESPARSE_CONFIG_LIBRARY     - dynamic SuiteSparse_config library
# SUITESPARSE_CONFIG_STATIC      - static SuiteSparse_config library
# SUITESPARSE_CONFIG_LIBRARIES   - libraries when using SuiteSparse_config
# SUITESPARSE_CONFIG_FOUND       - true if SuiteSparse_config found

# set ``SUITESPARSE_CONFIG_ROOT`` or ``SuiteSparse_config_ROOT`` to a
# SuiteSparse_config installation root to tell this module where to look.

# All the Find*.cmake files in SuiteSparse are installed by 'make install' into
# /usr/local/lib/cmake/SuiteSparse (where '/usr/local' is the
# ${CMAKE_INSTALL_PREFIX}).  To access this file, place the following commands
# in your CMakeLists.txt file.  See also SuiteSparse/Example/CMakeLists.txt:
#
#   set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
#       ${CMAKE_INSTALL_PREFIX}/lib/cmake/SuiteSparse )

#-------------------------------------------------------------------------------

# include files for SuiteSparse_config
find_path ( SUITESPARSE_CONFIG_INCLUDE_DIR
    NAMES SuiteSparse_config.h
    HINTS ${SUITESPARSE_CONFIG_ROOT}
    HINTS ENV SUITESPARSE_CONFIG_ROOT
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/SuiteSparse_config
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse_config
    PATH_SUFFIXES include Include
)

# dynamic libraries for SuiteSparse_config
find_library ( SUITESPARSE_CONFIG_LIBRARY
    NAMES suitesparseconfig
    HINTS ${SUITESPARSE_CONFIG_ROOT}
    HINTS ENV SUITESPARSE_CONFIG_ROOT
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/SuiteSparse_config
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse_config
    PATH_SUFFIXES lib build
)

if ( MSVC )
    set ( STATIC_SUFFIX .lib )
else ( )
    set ( STATIC_SUFFIX .a )
endif ( )

# static libraries for SuiteSparse_config
set ( save ${CMAKE_FIND_LIBRARY_SUFFIXES} )
set ( CMAKE_FIND_LIBRARY_SUFFIXES ${STATIC_SUFFIX} ${CMAKE_FIND_LIBRARY_SUFFIXES} )
find_library ( SUITESPARSE_CONFIG_STATIC
    NAMES suitesparseconfig
    HINTS ${SUITESPARSE_CONFIG_ROOT}
    HINTS ENV SUITESPARSE_CONFIG_ROOT
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/SuiteSparse_config
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse_config
    PATH_SUFFIXES lib build
)
set ( CMAKE_FIND_LIBRARY_SUFFIXES ${save} )

# get version of the library from the dynamic library filename, if present
get_filename_component ( SUITESPARSE_CONFIG_LIBRARY  ${SUITESPARSE_CONFIG_LIBRARY} REALPATH )
get_filename_component ( SUITESPARSE_CONFIG_FILENAME ${SUITESPARSE_CONFIG_LIBRARY} NAME )
string (
    REGEX MATCH "[0-9]+.[0-9]+.[0-9]+"
    SUITESPARSE_CONFIG_VERSION
    ${SUITESPARSE_CONFIG_FILENAME}
)

if ( NOT SUITESPARSE_CONFIG_VERSION )
    # version not found in the filename; get it from the include file
    foreach ( _VERSION MAIN_VERSION SUB_VERSION SUBSUB_VERSION )
        file ( STRINGS ${SUITESPARSE_CONFIG_INCLUDE_DIR}/SuiteSparse_config.h
            _VERSION_LINE REGEX "define[ ]+SUITESPARSE_${_VERSION}")
        if (_VERSION_LINE)
            string ( REGEX REPLACE ".*define[ ]+SUITESPARSE_${_VERSION}[ ]+([0-9]*).*" "\\1" _SP_${_VERSION} "${_VERSION_LINE}" )
        endif ( )
        unset ( _VERSION_LINE )
    endforeach ( )
    set (SUITESPARSE_CONFIG_VERSION "${_SP_MAIN_VERSION}.${_SP_SUB_VERSION}.${_SP_SUBSUB_VERSION}")
endif ( )

# libaries when using SuiteSparse_config
set (SUITESPARSE_CONFIG_LIBRARIES ${SUITESPARSE_CONFIG_LIBRARY})

include ( FindPackageHandleStandardArgs )

find_package_handle_standard_args ( SuiteSparse_config
    REQUIRED_VARS SUITESPARSE_CONFIG_LIBRARIES SUITESPARSE_CONFIG_INCLUDE_DIR
    VERSION_VAR SUITESPARSE_CONFIG_VERSION
)

mark_as_advanced (
    SUITESPARSE_CONFIG_INCLUDE_DIR
    SUITESPARSE_CONFIG_LIBRARY
    SUITESPARSE_CONFIG_STATIC
    SUITESPARSE_CONFIG_LIBRARIES
)

if ( SUITESPARSE_CONFIG_FOUND )
    message ( STATUS "SuiteSparse_config version: ${SUITESPARSE_CONFIG_VERSION}" )
    message ( STATUS "SuiteSparse_config include: ${SUITESPARSE_CONFIG_INCLUDE_DIR}" )
    message ( STATUS "SuiteSparse_config library: ${SUITESPARSE_CONFIG_LIBRARY}" )
    message ( STATUS "SuiteSparse_config static:  ${SUITESPARSE_CONFIG_STATIC}" )
else ( )
    message ( STATUS "SuiteSparse_config not found" )
endif ( )

