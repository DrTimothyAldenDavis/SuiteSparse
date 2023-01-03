#-------------------------------------------------------------------------------
# SuiteSparse/Mongoose/cmake_modules/FindMongoose.cmake
#-------------------------------------------------------------------------------

# The following copyright and license applies to just this file only, not to
# the library itself:
# FindMongoose.cmake, Copyright (c) 2022-2023, Timothy A. Davis.  All Rights Reserved.
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

# save the CMAKE_FIND_LIBRARY_SUFFIXES variable
set ( save ${CMAKE_FIND_LIBRARY_SUFFIXES} )

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
set ( CMAKE_FIND_LIBRARY_SUFFIXES
    ${CMAKE_SHARED_LIBRARY_SUFFIX} ${CMAKE_FIND_LIBRARY_SUFFIXES} )
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
    set ( STATIC_NAME mongoose_static )
else ( )
    set ( STATIC_NAME mongoose )
endif ( )

# static Mongoose library
set ( CMAKE_FIND_LIBRARY_SUFFIXES
    ${CMAKE_STATIC_LIBRARY_SUFFIX} ${CMAKE_FIND_LIBRARY_SUFFIXES} )
find_library ( MONGOOSE_STATIC
    NAMES ${STATIC_NAME}
    HINTS ${MONGOOSE_ROOT}
    HINTS ENV ${MONGOOSE_ROOT}
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/Mongoose
    HINTS ${CMAKE_SOURCE_DIR}/../Mongoose
    PATH_SUFFIXES lib build
)

# restore the CMAKE_FIND_LIBRARY_SUFFIXES variable
set ( CMAKE_FIND_LIBRARY_SUFFIXES ${save} )

# get version of the library from the dynamic library name
get_filename_component ( MONGOOSE_LIBRARY  ${MONGOOSE_LIBRARY} REALPATH )
get_filename_component ( MONGOOSE_FILENAME ${MONGOOSE_LIBRARY} NAME )
string (
    REGEX MATCH "[0-9]+.[0-9]+.[0-9]+"
    MONGOOSE_VERSION
    ${MONGOOSE_FILENAME}
)

# set ( MONGOOSE_VERSION "" )
if ( EXISTS "${MONGOOSE_INCLUDE_DIR}" AND NOT MONGOOSE_VERSION )
    # if the version does not appear in the filename, read the include file
    file ( STRINGS ${MONGOOSE_INCLUDE_DIR}/Mongoose.hpp MONGOOSE_MAJOR_STR
        REGEX "define Mongoose_VERSION_MAJOR" )
    file ( STRINGS ${MONGOOSE_INCLUDE_DIR}/Mongoose.hpp MONGOOSE_MINOR_STR
        REGEX "define Mongoose_VERSION_MINOR" )
    file ( STRINGS ${MONGOOSE_INCLUDE_DIR}/Mongoose.hpp MONGOOSE_PATCH_STR
        REGEX "define Mongoose_VERSION_PATCH" )
    message ( STATUS "major: ${MONGOOSE_MAJOR_STR}" )
    message ( STATUS "minor: ${MONGOOSE_MINOR_STR}" )
    message ( STATUS "patch: ${MONGOOSE_PATCH_STR}" )
    string ( REGEX MATCH "[0-9]+" MONGOOSE_MAJOR ${MONGOOSE_MAJOR_STR} )
    string ( REGEX MATCH "[0-9]+" MONGOOSE_MINOR ${MONGOOSE_MINOR_STR} )
    string ( REGEX MATCH "[0-9]+" MONGOOSE_PATCH ${MONGOOSE_PATCH_STR} )
    set (MONGOOSE_VERSION "${MONGOOSE_MAJOR}.${MONGOOSE_MINOR}.${MONGOOSE_PATCH}")
endif ( )

set ( MONGOOSE_LIBRARIES ${MONGOOSE_LIBRARY} )

include (FindPackageHandleStandardArgs)

find_package_handle_standard_args ( Mongoose
    REQUIRED_VARS MONGOOSE_LIBRARY MONGOOSE_INCLUDE_DIR
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
    set ( MONGOOSE_INCLUDE_DIR "" )
    set ( MONGOOSE_LIBRARIES "" )
    set ( MONGOOSE_LIBRARY "" )
    set ( MONGOOSE_STATIC "" )
endif ( )

