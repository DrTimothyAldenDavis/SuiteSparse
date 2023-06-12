#-------------------------------------------------------------------------------
# SuiteSparse/SPEX/cmake_modules/FindMPFR.cmake
#-------------------------------------------------------------------------------

# The following copyright and license applies to just this file only, not to
# the library itself:
# FindMPFR.cmake, Copyright (c) 2022-2023, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

#-------------------------------------------------------------------------------

# Finds the mpfr include file and compiled library and sets:

# MPFR_INCLUDE_DIR - where to find mpfr.h
# MPFR_LIBRARY     - dynamic mpfr library
# MPFR_STATIC      - static mpfr library
# MPFR_LIBRARIES   - libraries when using mpfr
# MPFR_FOUND       - true if mpfr found

# set ``MPFR_ROOT`` to a mpfr installation root to
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
    set ( MPFR_PC_OPTIONS "" )
    if ( MPFR_FIND_VERSION )
        set ( MPFR_PC_OPTIONS "mpfr>=${MPFR_FIND_VERSION}" )
    else ( )
        set ( MPFR_PC_OPTIONS "mpfr" )
    endif ( )
    if ( MPFR_FIND_REQUIRED )
        # FIXME: Are there installations without pkg-config file?
        # list ( APPEND MPFR_PC_OPTIONS REQUIRED )
    endif ( )
    pkg_check_modules ( MPFR ${MPFR_PC_OPTIONS} )

    if ( MPFR_FOUND )
        # assume first is the actual library
        list ( GET MPFR_LINK_LIBRARIES 0 MPFR_LIBRARY )
        set ( MPFR_INCLUDE_DIR ${MPFR_INCLUDEDIR} )
    endif ( )
    if (MPFR_STATIC_FOUND)
        # assume first is the actual library
        list ( GET MPFR_STATIC_LINK_LIBRARIES 0 MPFR_STATIC )
        set ( MPFR_INCLUDE_DIR ${MPFR_INCLUDEDIR} )
    endif ( )
endif ( )

if ( NOT MPFR_FOUND )
    # Manual search if pkg-config couldn't be used.
    # include files for mpfr
    find_path ( MPFR_INCLUDE_DIR
        NAMES mpfr.h
        PATH_SUFFIXES include Include
    )

    # dynamic mpfr library (or possibly static if no mpfr dynamic library exists)
    find_library ( MPFR_LIBRARY
        NAMES mpfr
        PATH_SUFFIXES lib build
    )

    # check if found
    if ( MPFR_LIBRARY MATCHES ".*NOTFOUND" OR MPFR_INCLUDE_DIR MATCHES ".*NOTFOUND" )
        set ( FOUND_IT false )
    else ( )
        set ( FOUND_IT true )
    endif ( )

    # static mpfr library
    if ( NOT MSVC )
        set ( save_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES} )
        set ( CMAKE_FIND_LIBRARY_SUFFIXES
            ${CMAKE_STATIC_LIBRARY_SUFFIX} ${CMAKE_FIND_LIBRARY_SUFFIXES} )
    endif ( )

    find_library ( MPFR_STATIC
        NAMES mpfr
        PATH_SUFFIXES lib build
    )

    if ( NOT MSVC )
        # restore the CMAKE_FIND_LIBRARY_SUFFIXES variable
        set ( CMAKE_FIND_LIBRARY_SUFFIXES ${save_CMAKE_FIND_LIBRARY_SUFFIXES} )
    endif ( )

    # get version of the library from the filename
    get_filename_component ( MPFR_LIBRARY ${MPFR_LIBRARY} REALPATH )

    # look in the middle for 4.1.0 (/spackstuff/mpfr-4.1.0-morestuff/libmpfr.10.4.1)
    string ( REGEX MATCH "mpfr-[0-9]+.[0-9]+.[0-9]+" MPFR_VERSION1 ${MPFR_LIBRARY} )

    if ( NOT FOUND_IT )
        # mpfr has not been found
        set ( MPFR_VERSION 0.0.0 )
        message ( WARNING "MPFR not found")
    elseif ( MPFR_VERSION1 STREQUAL "" )
        # mpfr has been found, but not a spack library.  Hunt for the version
        # number in mpfr.h.  The mpfr.h file includes the following line:
        #       #define MPFR_VERSION_STRING "4.0.2"
        file ( STRINGS ${MPFR_INCLUDE_DIR}/mpfr.h MPFR_VER_STRING
            REGEX "MPFR_VERSION_STRING" )
        message ( STATUS "major/minor/patch: ${MPFR_VER_STRING}" )
        if ( MPFR_VER_STRING STREQUAL "")
            # look at the end of the filename for the version number
            string (
                REGEX MATCH "[0-9]+.[0-9]+.[0-9]+"
                MPFR_VERSION ${MPFR_LIBRARY} )
        else ( )
            # get the version number from inside the mpfr.h file itself
            string ( REGEX MATCH "[0-9]+.[0-9]+.[0-9]+" MPFR_VERSION ${MPFR_VER_STRING} )
        endif ( )
    else ( )
        # look at mpfr-4.1.0 for the version number (spack library)
        string ( REGEX MATCH "[0-9]+.[0-9]+.[0-9]+" MPFR_VERSION ${MPFR_VERSION1} )
    endif ( )
endif ( )

if ( NOT MPFR_STATIC )
    set ( MPFR_STATIC ${MPFR_LIBRARY} )
endif ( )

set ( MPFR_LIBRARIES ${MPFR_LIBRARY} )

include (FindPackageHandleStandardArgs)

find_package_handle_standard_args ( MPFR
    REQUIRED_VARS MPFR_LIBRARY MPFR_INCLUDE_DIR
    VERSION_VAR MPFR_VERSION
)

mark_as_advanced (
    MPFR_INCLUDE_DIR
    MPFR_LIBRARY
    MPFR_STATIC
    MPFR_LIBRARIES
)

if ( MPFR_FOUND )
    message ( STATUS "mpfr version: ${MPFR_VERSION}" )
    message ( STATUS "mpfr include: ${MPFR_INCLUDE_DIR}" )
    message ( STATUS "mpfr library: ${MPFR_LIBRARY}" )
    message ( STATUS "mpfr static:  ${MPFR_STATIC}" )
else ( )
    message ( STATUS "mpfr not found" )
    set ( MPFR_INCLUDE_DIR "" )
    set ( MPFR_LIBRARIES "" )
    set ( MPFR_LIBRARY "" )
    set ( MPFR_STATIC "" )
endif ( )

