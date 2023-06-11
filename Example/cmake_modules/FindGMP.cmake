#-------------------------------------------------------------------------------
# SuiteSparse/SPEX/cmake_modules/FindGMP.cmake
#-------------------------------------------------------------------------------

# The following copyright and license applies to just this file only, not to
# the library itself:
# FindGMP.cmake, Copyright (c) 2022-2023, Timothy A. Davis.  All Rights Reserved.
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
    list ( PREPEND CMAKE_PREFIX_PATH $ENV{CMAKE_PREFIX_PATH} )
endif ( )

# Try to get information from pkg-config file first.
find_package ( PkgConfig )
if ( PKG_CONFIG_FOUND )
    set ( GMP_PC_OPTIONS "" )
    if ( GMP_FIND_VERSION )
        set ( GMP_PC_OPTIONS "gmp>=${GMP_FIND_VERSION}" )
    else ( )
        set ( GMP_PC_OPTIONS "gmp" )
    endif ( )
    if ( GMP_FIND_REQUIRED )
        # FIXME: Are there installations without pkg-config file?
        # list ( APPEND GMP_PC_OPTIONS REQUIRED )
    endif ( )
    pkg_check_modules ( GMP ${GMP_PC_OPTIONS} )

    if ( GMP_FOUND )
        # assume first is the actual library
        # FIXME: Would it be possible to return all libraries in that variable?
        list ( GET GMP_LINK_LIBRARIES 0 GMP_LIBRARY )
        set ( GMP_INCLUDE_DIR ${GMP_INCLUDEDIR} )
    endif ( )
    if (GMP_STATIC_FOUND)
        # assume first is the actual library
        list ( GET GMP_STATIC_LINK_LIBRARIES 0 GMP_STATIC )
        set ( GMP_INCLUDE_DIR ${GMP_INCLUDEDIR} )
    endif ( )
endif ( )

if ( NOT GMP_FOUND )
    # Manual search if pkg-config couldn't be used.

    # include files for gmp
    find_path ( GMP_INCLUDE_DIR
        NAMES gmp.h
        PATH_SUFFIXES include Include
    )

    # dynamic gmp library (or possibly static if no GMP dynamic library exists)
    find_library ( GMP_LIBRARY
        NAMES gmp
        PATH_SUFFIXES lib build
    )

    # static gmp library
    if ( NOT MSVC )
        set ( save_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES} )
        set ( CMAKE_FIND_LIBRARY_SUFFIXES
            ${CMAKE_STATIC_LIBRARY_SUFFIX} ${CMAKE_FIND_LIBRARY_SUFFIXES} )
    endif ( )

    find_library ( GMP_STATIC
        NAMES gmp
        PATH_SUFFIXES lib build
    )

    if ( NOT MSVC )
        # restore the CMAKE_FIND_LIBRARY_SUFFIXES variable
        set ( CMAKE_FIND_LIBRARY_SUFFIXES ${save_CMAKE_FIND_LIBRARY_SUFFIXES} )
    endif ( )

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
            REGEX "define __GNU_MP_VERSION_MINOR " )
        file ( STRINGS ${GMP_INCLUDE_DIR}/gmp.h GMP_VER_PATCH_STRING
            REGEX "define __GNU_MP_VERSION_PATCHLEVEL " )
        message ( STATUS "major: ${GMP_VER_MAJOR_STRING}" )
        message ( STATUS "minor: ${GMP_VER_MINOR_STRING}" )
        message ( STATUS "patch: ${GMP_VER_PATCH_STRING}" )
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
endif ( )

if ( NOT GMP_STATIC )
    set ( GMP_STATIC ${GMP_LIBRARY} )
endif ( )

set ( GMP_LIBRARIES ${GMP_LIBRARY} )

include (FindPackageHandleStandardArgs)

find_package_handle_standard_args ( GMP
    REQUIRED_VARS GMP_LIBRARY GMP_INCLUDE_DIR
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
    set ( GMP_INCLUDE_DIR "" )
    set ( GMP_LIBRARIES "" )
    set ( GMP_LIBRARY "" )
    set ( GMP_STATIC "" )
endif ( )

