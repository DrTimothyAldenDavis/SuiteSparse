#-------------------------------------------------------------------------------
# SuiteSparse/KLU/cmake_modules/FindKLU_CHOLMOD.cmake
#-------------------------------------------------------------------------------

# The following copyright and license applies to just this file only, not to
# the library itself:
# FindKLU_CHOLMOD.cmake, Copyright (c) 2022-2023, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

#-------------------------------------------------------------------------------

# Finds the KLU_CHOLMOD include file and compiled library and sets:

# KLU_CHOLMOD_INCLUDE_DIR - where to find klu_cholmod.h
# KLU_CHOLMOD_LIBRARY     - compiled KLU_CHOLMOD library
# KLU_CHOLMOD_LIBRARIES   - libraries when using KLU_CHOLMOD
# KLU_CHOLMOD_FOUND       - true if KLU_CHOLMOD found

# set ``KLU_CHOLMOD_ROOT`` to a KLU_CHOLMOD installation root to
# tell this module where to look.

# All the Find*.cmake files in SuiteSparse are installed by 'make install' into
# /usr/local/lib/cmake/SuiteSparse (where '/usr/local' is the
# ${CMAKE_INSTALL_PREFIX}).  To access this file, place the following commands
# in your CMakeLists.txt file.  See also SuiteSparse/Example/CMakeLists.txt:
#
#   set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
#       ${CMAKE_INSTALL_PREFIX}/lib/cmake/SuiteSparse )

#-------------------------------------------------------------------------------

# include files for KLU_CHOLMOD
find_path ( KLU_CHOLMOD_INCLUDE_DIR
    NAMES klu_cholmod.h
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/KLU/User
    HINTS ${CMAKE_SOURCE_DIR}/../KLU/User
    PATH_SUFFIXES include Include
)

# dynamic KLU_CHOLMOD library
find_library ( KLU_CHOLMOD_LIBRARY
    NAMES klu_cholmod
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/KLU/User
    HINTS ${CMAKE_SOURCE_DIR}/../KLU/User
    PATH_SUFFIXES lib build
)

if ( MSVC )
    set ( STATIC_SUFFIX .lib )
else ( )
    set ( STATIC_SUFFIX .a )
endif ( )

# static KLU_CHOLMOD library
set ( save ${CMAKE_FIND_LIBRARY_SUFFIXES} )
set ( CMAKE_FIND_LIBRARY_SUFFIXES ${STATIC_SUFFIX} ${CMAKE_FIND_LIBRARY_SUFFIXES} )
find_library ( KLU_CHOLMOD_STATIC
    NAMES klu_cholmod
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/KLU/User
    HINTS ${CMAKE_SOURCE_DIR}/../KLU/User
    PATH_SUFFIXES lib build
)
set ( CMAKE_FIND_LIBRARY_SUFFIXES ${save} )

# get version of the library from the dynamic library name
get_filename_component ( KLU_CHOLMOD_LIBRARY  ${KLU_CHOLMOD_LIBRARY} REALPATH )
get_filename_component ( KLU_CHOLMOD_FILENAME ${KLU_CHOLMOD_LIBRARY} NAME )
string (
    REGEX MATCH "[0-9]+.[0-9]+.[0-9]+"
    KLU_CHOLMOD_VERSION
    ${KLU_CHOLMOD_FILENAME}
)

if ( NOT KLU_CHOLMOD_VERSION )
    # if the version does not appear in the filename, read the include file
    foreach (_VERSION MAIN_VERSION SUB_VERSION SUBSUB_VERSION)
        file (STRINGS ${KLU_CHOLMOD_INCLUDE_DIR}/klu.h _VERSION_LINE REGEX "define[ ]+KLU_${_VERSION}")
        if (_VERSION_LINE)
        string (REGEX REPLACE ".*define[ ]+KLU_${_VERSION}[ ]+([0-9]*).*" "\\1" _KLU_${_VERSION} "${_VERSION_LINE}")
        endif ()
        unset (_VERSION_LINE)
    endforeach ()
    set (KLU_CHOLMOD_VERSION "${_KLU_MAIN_VERSION}.${_KLU_SUB_VERSION}.${_KLU_SUBSUB_VERSION}")
endif ( )

set ( KLU_CHOLMOD_LIBRARIES ${KLU_CHOLMOD_LIBRARY} )

include (FindPackageHandleStandardArgs)

find_package_handle_standard_args ( KLU_CHOLMOD
    REQUIRED_VARS KLU_CHOLMOD_LIBRARIES KLU_CHOLMOD_INCLUDE_DIR
    VERSION_VAR KLU_CHOLMOD_VERSION
)

mark_as_advanced (
    KLU_CHOLMOD_INCLUDE_DIR
    KLU_CHOLMOD_LIBRARY
    KLU_CHOLMOD_STATIC
    KLU_CHOLMOD_LIBRARIES
)

if ( KLU_CHOLMOD_FOUND )
    message ( STATUS "KLU_CHOLMOD version: ${KLU_CHOLMOD_VERSION}" )
    message ( STATUS "KLU_CHOLMOD include: ${KLU_CHOLMOD_INCLUDE_DIR}" )
    message ( STATUS "KLU_CHOLMOD library: ${KLU_CHOLMOD_LIBRARY}" )
    message ( STATUS "KLU_CHOLMOD static:  ${KLU_CHOLMOD_STATIC}" )
else ( )
    message ( STATUS "KLU_CHOLMOD not found" )
endif ( )

