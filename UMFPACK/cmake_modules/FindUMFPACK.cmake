#-------------------------------------------------------------------------------
# SuiteSparse/UMFPACK/cmake_modules/FindUMFPACK.cmake
#-------------------------------------------------------------------------------

# The following copyright and license applies to just this file only, not to
# the library itself:
# FindUMFPACK.cmake, Copyright (c) 2022-2023, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

#-------------------------------------------------------------------------------

# Finds the UMFPACK include file and compiled library and sets:

# UMFPACK_INCLUDE_DIR - where to find umfpack.h
# UMFPACK_LIBRARY     - dynamic UMFPACK library
# UMFPACK_STATIC      - static UMFPACK library
# UMFPACK_LIBRARIES   - libraries when using UMFPACK
# UMFPACK_FOUND       - true if UMFPACK found

# set ``UMFPACK_ROOT`` to an UMFPACK installation root to tell this module
# where to look (this can be done as a cmake variable or as an evironment
# variable).

# All the Find*.cmake files in SuiteSparse are installed by 'make install' into
# /usr/local/lib/cmake/SuiteSparse (where '/usr/local' is the
# ${CMAKE_INSTALL_PREFIX}).  To access this file, place the following commands
# in your CMakeLists.txt file.  See also SuiteSparse/Example/CMakeLists.txt:
#
#   set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
#       ${CMAKE_INSTALL_PREFIX}/lib/cmake/SuiteSparse )

#-------------------------------------------------------------------------------

# include files for UMFPACK
find_path ( UMFPACK_INCLUDE_DIR
    NAMES umfpack.h
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/UMFPACK
    HINTS ${CMAKE_SOURCE_DIR}/../UMFPACK
    PATH_SUFFIXES include Include
)

# dynamic UMFPACK library (or static if no dynamic library was built)
find_library ( UMFPACK_LIBRARY
    NAMES umfpack umfpack_static
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/UMFPACK
    HINTS ${CMAKE_SOURCE_DIR}/../UMFPACK
    PATH_SUFFIXES lib build
)

if ( MSVC )
    set ( STATIC_NAME umfpack_static )
else ( )
    set ( STATIC_NAME umfpack )
    set ( save ${CMAKE_FIND_LIBRARY_SUFFIXES} )
    set ( CMAKE_FIND_LIBRARY_SUFFIXES
        ${CMAKE_STATIC_LIBRARY_SUFFIX} ${CMAKE_FIND_LIBRARY_SUFFIXES} )
endif ( )

# static UMFPACK library
find_library ( UMFPACK_STATIC
    NAMES ${STATIC_NAME}
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/UMFPACK
    HINTS ${CMAKE_SOURCE_DIR}/../UMFPACK
    PATH_SUFFIXES lib build
)

if ( NOT MSVC )
    # restore the CMAKE_FIND_LIBRARY_SUFFIXES variable
    set ( CMAKE_FIND_LIBRARY_SUFFIXES ${save} )
endif ( )

# get version of the library from the dynamic library name
get_filename_component ( UMFPACK_LIBRARY  ${UMFPACK_LIBRARY} REALPATH )
get_filename_component ( UMFPACK_FILENAME ${UMFPACK_LIBRARY} NAME )
string (
    REGEX MATCH "[0-9]+.[0-9]+.[0-9]+"
    UMFPACK_VERSION
    ${UMFPACK_FILENAME}
)

# set ( UMFPACK_VERSION "" )
if ( EXISTS "${UMFPACK_INCLUDE_DIR}" AND NOT UMFPACK_VERSION )
    # if the version does not appear in the filename, read the include file
    file ( STRINGS ${UMFPACK_INCLUDE_DIR}/umfpack.h UMFPACK_MAJOR_STR
        REGEX "define UMFPACK_MAIN_VERSION" )
    file ( STRINGS ${UMFPACK_INCLUDE_DIR}/umfpack.h UMFPACK_MINOR_STR
        REGEX "define UMFPACK_SUB_VERSION" )
    file ( STRINGS ${UMFPACK_INCLUDE_DIR}/umfpack.h UMFPACK_PATCH_STR
        REGEX "define UMFPACK_SUBSUB_VERSION" )
    message ( STATUS "major: ${UMFPACK_MAJOR_STR}" )
    message ( STATUS "minor: ${UMFPACK_MINOR_STR}" )
    message ( STATUS "patch: ${UMFPACK_PATCH_STR}" )
    string ( REGEX MATCH "[0-9]+" UMFPACK_MAJOR ${UMFPACK_MAJOR_STR} )
    string ( REGEX MATCH "[0-9]+" UMFPACK_MINOR ${UMFPACK_MINOR_STR} )
    string ( REGEX MATCH "[0-9]+" UMFPACK_PATCH ${UMFPACK_PATCH_STR} )
    set (UMFPACK_VERSION "${UMFPACK_MAJOR}.${UMFPACK_MINOR}.${UMFPACK_PATCH}")
endif ( )

set ( UMFPACK_LIBRARIES ${UMFPACK_LIBRARY} )

include (FindPackageHandleStandardArgs)

find_package_handle_standard_args ( UMFPACK
    REQUIRED_VARS UMFPACK_LIBRARY UMFPACK_INCLUDE_DIR
    VERSION_VAR UMFPACK_VERSION
)

mark_as_advanced (
    UMFPACK_INCLUDE_DIR
    UMFPACK_LIBRARY
    UMFPACK_STATIC
    UMFPACK_LIBRARIES
)

if ( UMFPACK_FOUND )
    message ( STATUS "UMFPACK version: ${UMFPACK_VERSION}" )
    message ( STATUS "UMFPACK include: ${UMFPACK_INCLUDE_DIR}" )
    message ( STATUS "UMFPACK library: ${UMFPACK_LIBRARY}" )
    message ( STATUS "UMFPACK static:  ${UMFPACK_STATIC}" )
else ( )
    message ( STATUS "UMFPACK not found" )
    set ( UMFPACK_INCLUDE_DIR "" )
    set ( UMFPACK_LIBRARIES "" )
    set ( UMFPACK_LIBRARY "" )
    set ( UMFPACK_STATIC "" )
endif ( )

