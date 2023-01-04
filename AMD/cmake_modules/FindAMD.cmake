#-------------------------------------------------------------------------------
# SuiteSparse/AMD/cmake_modules/FindAMD.cmake
#-------------------------------------------------------------------------------

# The following copyright and license applies to just this file only, not to
# the library itself:
# FindAMD.cmake, Copyright (c) 2022-2023, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

#-------------------------------------------------------------------------------

# Finds the AMD include file and compiled library and sets:

# AMD_INCLUDE_DIR - where to find amd.h
# AMD_LIBRARY     - dynamic AMD library
# AMD_STATIC      - static AMD library
# AMD_LIBRARIES   - libraries when using AMD
# AMD_FOUND       - true if AMD found

# set ``AMD_ROOT`` to an AMD installation root to
# tell this module where to look.

# All the Find*.cmake files in SuiteSparse are installed by 'make install' into
# /usr/local/lib/cmake/SuiteSparse (where '/usr/local' is the
# ${CMAKE_INSTALL_PREFIX}).  To access this file, place the following commands
# in your CMakeLists.txt file.  See also SuiteSparse/Example/CMakeLists.txt:
#
#   set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
#       ${CMAKE_INSTALL_PREFIX}/lib/cmake/SuiteSparse )

#-------------------------------------------------------------------------------

# include files for AMD
find_path ( AMD_INCLUDE_DIR
    NAMES amd.h
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/AMD
    HINTS ${CMAKE_SOURCE_DIR}/../AMD
    PATH_SUFFIXES include Include
)

# dynamic AMD library (or static if no dynamic library was built)
find_library ( AMD_LIBRARY
    NAMES amd amd_static
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/AMD
    HINTS ${CMAKE_SOURCE_DIR}/../AMD
    PATH_SUFFIXES lib build build/Release build/Debug
)

if ( MSVC )
    set ( STATIC_NAME amd_static )
else ( )
    set ( STATIC_NAME amd )
    set ( save ${CMAKE_FIND_LIBRARY_SUFFIXES} )
    set ( CMAKE_FIND_LIBRARY_SUFFIXES
        ${CMAKE_STATIC_LIBRARY_SUFFIX} ${CMAKE_FIND_LIBRARY_SUFFIXES} )
endif ( )

# static AMD library
find_library ( AMD_STATIC
    NAMES ${STATIC_NAME}
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/AMD
    HINTS ${CMAKE_SOURCE_DIR}/../AMD
    PATH_SUFFIXES lib build build/Release build/Debug
)

if ( NOT MSVC )
    # restore the CMAKE_FIND_LIBRARY_SUFFIXES variable
    set ( CMAKE_FIND_LIBRARY_SUFFIXES ${save} )
endif ( )

# get version of the library from the dynamic library name
get_filename_component ( AMD_LIBRARY  ${AMD_LIBRARY} REALPATH )
get_filename_component ( AMD_FILENAME ${AMD_LIBRARY} NAME )
string (
    REGEX MATCH "[0-9]+.[0-9]+.[0-9]+"
    AMD_VERSION
    ${AMD_FILENAME}
)

# set ( AMD_VERSION "" )
if ( EXISTS "${AMD_INCLUDE_DIR}" AND NOT AMD_VERSION )
    # if the version does not appear in the filename, read the include file
    file ( STRINGS ${AMD_INCLUDE_DIR}/amd.h AMD_MAJOR_STR
        REGEX "define AMD_MAIN_VERSION" )
    file ( STRINGS ${AMD_INCLUDE_DIR}/amd.h AMD_MINOR_STR
        REGEX "define AMD_SUB_VERSION" )
    file ( STRINGS ${AMD_INCLUDE_DIR}/amd.h AMD_PATCH_STR
        REGEX "define AMD_SUBSUB_VERSION" )
    message ( STATUS "major: ${AMD_MAJOR_STR}" )
    message ( STATUS "minor: ${AMD_MINOR_STR}" )
    message ( STATUS "patch: ${AMD_PATCH_STR}" )
    string ( REGEX MATCH "[0-9]+" AMD_MAJOR ${AMD_MAJOR_STR} )
    string ( REGEX MATCH "[0-9]+" AMD_MINOR ${AMD_MINOR_STR} )
    string ( REGEX MATCH "[0-9]+" AMD_PATCH ${AMD_PATCH_STR} )
    set (AMD_VERSION "${AMD_MAJOR}.${AMD_MINOR}.${AMD_PATCH}")
endif ( )

set ( AMD_LIBRARIES ${AMD_LIBRARY} )

include (FindPackageHandleStandardArgs)

find_package_handle_standard_args ( AMD
    REQUIRED_VARS AMD_LIBRARY AMD_INCLUDE_DIR
    VERSION_VAR AMD_VERSION
)

mark_as_advanced (
    AMD_INCLUDE_DIR
    AMD_LIBRARY
    AMD_STATIC
    AMD_LIBRARIES
)

if ( AMD_FOUND )
    message ( STATUS "AMD version: ${AMD_VERSION}" )
    message ( STATUS "AMD include: ${AMD_INCLUDE_DIR}")
    message ( STATUS "AMD library: ${AMD_LIBRARY}")
    message ( STATUS "AMD static:  ${AMD_STATIC}")
else ( )
    message ( STATUS "AMD not found" )
    set ( AMD_INCLUDE_DIR "" )
    set ( AMD_LIBRARIES "" )
    set ( AMD_LIBRARY "" )
    set ( AMD_STATIC "" )
endif ( )

