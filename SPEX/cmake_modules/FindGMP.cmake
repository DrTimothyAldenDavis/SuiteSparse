#-------------------------------------------------------------------------------
# SuiteSparse/SPEX/cmake_modules/FindGMP.cmake
#-------------------------------------------------------------------------------

# The following copyright and license applies to just this file only, not to
# the library itself:
# FindGMP.cmake, Copyright (c) 2022-2024, Timothy A. Davis. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

#-------------------------------------------------------------------------------

# Finds the gmp include file and compiled library and sets:

# GMP_INCLUDE_DIR - where to find gmp.h
# GMP_LIBRARY     - dynamic gmp library
# GMP_STATIC      - static gmp library
# GMP_LIBRARIES   - libraries when using gmp
# GMP_FOUND       - true if gmp found

# set ``GMP_ROOT`` to a gmp installation root to
# tell this module where to look.

# Since this file searches for a non-SuiteSparse library, it is not installed
# with 'make install' when installing SPEX.

#-------------------------------------------------------------------------------

if ( DEFINED ENV{CMAKE_PREFIX_PATH} )
    # import CMAKE_PREFIX_PATH, typically created by spack
    list ( APPEND CMAKE_PREFIX_PATH $ENV{CMAKE_PREFIX_PATH} )
endif ( )

# include files for gmp
find_path ( GMP_INCLUDE_DIR
    NAMES gmp.h
    PATH_SUFFIXES include Include
)

# dynamic gmp library
find_library ( GMP_LIBRARY
    NAMES gmp
    PATH_SUFFIXES lib build
)

if ( MSVC )
    set ( STATIC_SUFFIX .lib )
else ( )
    set ( STATIC_SUFFIX .a )
endif ( )

# static gmp library
set ( save ${CMAKE_FIND_LIBRARY_SUFFIXES} )
set ( CMAKE_FIND_LIBRARY_SUFFIXES ${STATIC_SUFFIX} ${CMAKE_FIND_LIBRARY_SUFFIXES} )
find_library ( GMP_STATIC
    NAMES gmp
    PATH_SUFFIXES lib build
)
set ( CMAKE_FIND_LIBRARY_SUFFIXES ${save} )

# get version of the library from the filename
get_filename_component ( GMP_LIBRARY ${GMP_LIBRARY} REALPATH )

# look in the middle for 6.2.1 (/spackstuff/gmp-6.2.1-morestuff/libgmp.10.4.1)
string ( REGEX MATCH "gmp-[0-9]+.[0-9]+.[0-9]+" GMP_VERSION1 ${GMP_LIBRARY} )

if ( GMP_VERSION1 STREQUAL "" )
    # gmp has been found, but not as a spack library.  Hunt for the version
    # number in gmp.h.  The gmp.h file includes the following lines:
    #   #define __GNU_MP_VERSION            6
    #   #define __GNU_MP_VERSION_MINOR      2
    #   #define __GNU_MP_VERSION_PATCHLEVEL 0
    file ( STRINGS ${GMP_INCLUDE_DIR}/gmp.h GMP_VER_MAJOR_STRING
        REGEX "define __GNU_MP_VERSION " )
    file ( STRINGS ${GMP_INCLUDE_DIR}/gmp.h GMP_VER_MINOR_STRING
        REGEX "define __GNU_MP_VERSION_MINOR" )
    file ( STRINGS ${GMP_INCLUDE_DIR}/gmp.h GMP_VER_PATCH_STRING
        REGEX "define __GNU_MP_VERSION_PATCH" )
    message ( STATUS "major from gmp.h: ${GMP_VER_MAJOR_STRING}" )
    message ( STATUS "minor from gmp.h: ${GMP_VER_MINOR_STRING}" )
    message ( STATUS "patch from gmp.h: ${GMP_VER_PATCH_STRING}" )
    if ( GMP_VER_MAJOR_STRING STREQUAL "")
        # look at the end of the filename for the version number
        string (
            REGEX MATCH "[0-9]+.[0-9]+.[0-9]+"
            GMP_VERSION ${GMP_LIBRARY} )
    else ( )
        # get the version number from inside the gmp.h file itself
        string ( REGEX MATCH "[0-9]+" GMP_VER_MAJOR ${GMP_VER_MAJOR_STRING} )
        string ( REGEX MATCH "[0-9]+" GMP_VER_MINOR ${GMP_VER_MINOR_STRING} )
        string ( REGEX MATCH "[0-9]+" GMP_VER_PATCH ${GMP_VER_PATCH_STRING} )
        set ( GMP_VERSION "${GMP_VER_MAJOR}.${GMP_VER_MINOR}.${GMP_VER_PATCH}")
    endif ( )
else ( )
    # look at gmp-6.2.1 for the version number (spack library)
    string ( REGEX MATCH "[0-9]+.[0-9]+.[0-9]" GMP_VERSION ${GMP_VERSION1} )
endif ( )

set ( GMP_LIBRARIES ${GMP_LIBRARY} )

include (FindPackageHandleStandardArgs)

find_package_handle_standard_args ( GMP
    REQUIRED_VARS GMP_LIBRARIES GMP_INCLUDE_DIR
    VERSION_VAR GMP_VERSION
)

mark_as_advanced (
    GMP_INCLUDE_DIR
    GMP_LIBRARY
    GMP_STATIC
    GMP_LIBRARIES
)

if ( GMP_FOUND )
    message ( STATUS "gmp version: ${GMP_VERSION}" )
    message ( STATUS "gmp include: ${GMP_INCLUDE_DIR}" )
    message ( STATUS "gmp library: ${GMP_LIBRARY}" )
    message ( STATUS "gmp static:  ${GMP_STATIC}" )
else ( )
    message ( STATUS "gmp not found" )
endif ( )

