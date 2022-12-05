#-------------------------------------------------------------------------------
# SuiteSparse/Mongoose/cmake_modules/FindMongoose.cmake
#-------------------------------------------------------------------------------

# The following copyright and license applies to just this file only, not to
# the library itself:
# Copyright (c) 2022, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

#-------------------------------------------------------------------------------

# Finds the Mongoose include file and compiled library and sets:

# MONGOOSE_INCLUDE_DIR - where to find Mongoose.hpp
# MONGOOSE_LIBRARY     - dynamic Mongoose library
# MONGOOSE_STATIC      - static Mongoose library
# MONGOOSE_LIBRARIES   - libraries when using Mongoose
# MONGOOSE_FOUND       - true if Mongoose found

# set ``MONGOOSE_ROOT`` or ``Mongoose_ROOT`` to a MONGOOSE installation root to
# tell this module where to look.

# All the Find*.cmake files in SuiteSparse are installed by 'make install' into
# /usr/local/lib/cmake/SuiteSparse (where '/usr/local' is the
# ${CMAKE_INSTALL_PREFIX}).  To access this file, place the following commands
# in your CMakeLists.txt file.  See also SuiteSparse/Example/CMakeLists.txt:
#
#   set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
#       ${CMAKE_INSTALL_PREFIX}/lib/cmake/SuiteSparse )

#-------------------------------------------------------------------------------

# include files for Mongoose
find_path ( MONGOOSE_INCLUDE_DIR
    NAMES Mongoose.hpp
    HINTS ${MONGOOSE_ROOT}
    HINTS ENV ${MONGOOSE_ROOT}
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/Mongoose
    HINTS ${CMAKE_SOURCE_DIR}/../Mongoose
    PATH_SUFFIXES include Include
)

# dynamic Mongoose library
find_library ( MONGOOSE_LIBRARY
    NAMES mongoose
    HINTS ${MONGOOSE_ROOT}
    HINTS ENV ${MONGOOSE_ROOT}
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/Mongoose
    HINTS ${CMAKE_SOURCE_DIR}/../Mongoose
    PATH_SUFFIXES lib build
)

if ( MSVC )
    set ( STATIC_SUFFIX .lib )
else ( )
    set ( STATIC_SUFFIX .a )
endif ( )

# static Mongoose library
set ( save ${CMAKE_FIND_LIBRARY_SUFFIXES} )
set ( CMAKE_FIND_LIBRARY_SUFFIXES ${STATIC_SUFFIX} ${CMAKE_FIND_LIBRARY_SUFFIXES} )
find_library ( MONGOOSE_STATIC
    NAMES mongoose
    HINTS ${MONGOOSE_ROOT}
    HINTS ENV ${MONGOOSE_ROOT}
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/Mongoose
    HINTS ${CMAKE_SOURCE_DIR}/../Mongoose
    PATH_SUFFIXES lib build
)
set ( CMAKE_FIND_LIBRARY_SUFFIXES ${save} )

# get version of the library from the dynamic library name
get_filename_component ( MONGOOSE_LIBRARY  ${MONGOOSE_LIBRARY} REALPATH )
get_filename_component ( MONGOOSE_FILENAME ${MONGOOSE_LIBRARY} NAME )
string (
    REGEX MATCH "[0-9]+.[0-9]+.[0-9]+"
    MONGOOSE_VERSION
    ${MONGOOSE_FILENAME}
)

if ( NOT MONGOOSE_VERSION )
    # if the version does not appear in the filename, read the include file
    foreach ( _VERSION VERSION_MAJOR VERSION_MINOR VERSION_PATCH )
        file ( STRINGS ${MONGOOSE_INCLUDE_DIR}/Mongoose_Version.hpp _VERSION_LINE REGEX "define[ ]+Mongoose_${_VERSION}" )
        if ( _VERSION_LINE )
            string ( REGEX REPLACE ".*define[ ]+Mongoose_${_VERSION}[ ]+([0-9]*).*" "\\1" _MONGOOSE_${_VERSION} "${_VERSION_LINE}" )
        endif ( )
        unset ( _VERSION_LINE )
    endforeach ( )
    set ( MONGOOSE_VERSION "${_MONGOOSE_VERSION_MAJOR}.${_MONGOOSE_VERSION_MINOR}.${_MONGOOSE_VERSION_PATCH}" )
endif ( )

set ( MONGOOSE_LIBRARIES ${MONGOOSE_LIBRARY} )

include (FindPackageHandleStandardArgs)

find_package_handle_standard_args ( Mongoose
    REQUIRED_VARS MONGOOSE_LIBRARIES MONGOOSE_INCLUDE_DIR
    VERSION_VAR MONGOOSE_VERSION
)

mark_as_advanced (
    MONGOOSE_INCLUDE_DIR
    MONGOOSE_LIBRARY
    MONGOOSE_STATIC
    MONGOOSE_LIBRARIES
)

if ( MONGOOSE_FOUND )
    message ( STATUS "Mongoose version: ${MONGOOSE_VERSION}" )
    message ( STATUS "Mongoose include: ${MONGOOSE_INCLUDE_DIR}" )
    message ( STATUS "Mongoose library: ${MONGOOSE_LIBRARY}" )
    message ( STATUS "Mongoose static:  ${MONGOOSE_STATIC}" )
else ( )
    message ( STATUS "Mongoose not found" )
endif ( )

