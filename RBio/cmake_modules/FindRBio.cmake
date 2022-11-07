#-------------------------------------------------------------------------------
# SuiteSparse/SuiteSparse_config/cmake_modules/FindRBio.cmake
#-------------------------------------------------------------------------------

# Copyright (c) 2022, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

# Finds the RBio include file and compiled library and sets:

# RBIO_INCLUDE_DIR - where to find RBio.h
# RBIO_LIBRARY     - compiled RBio library
# RBIO_LIBRARIES   - libraries when using RBio
# RBIO_FOUND       - true if RBio found

# set ``RBIO_ROOT`` to a RBio installation root to
# tell this module where to look.

# To use this file in your application, copy this file into MyApp/cmake_modules
# where MyApp is your application and add the following to your
# MyApp/CMakeLists.txt file:
#
#   set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake_modules")
#
# or, assuming MyApp and SuiteSparse sit side-by-side in a common folder, you
# can leave this file in place and use this command (revise as needed):
#
#   set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
#       "${CMAKE_SOURCE_DIR}/../SuiteSparse/SuiteSparse_config/cmake_modules")

#-------------------------------------------------------------------------------

# include files for RBio
find_path ( RBIO_INCLUDE_DIR
    NAMES RBio.h
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/RBio
    HINTS ${CMAKE_SOURCE_DIR}/../RBio
    PATHS RBIO_ROOT ENV RBIO_ROOT
    PATH_SUFFIXES include Include
)

# compiled libraries RBio
find_library ( RBIO_LIBRARY
    NAMES rbio${CMAKE_RELEASE_POSTFIX}
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/RBio
    HINTS ${CMAKE_SOURCE_DIR}/../RBio
    PATHS RBIO_ROOT ENV RBIO_ROOT
    PATH_SUFFIXES lib build alternative
)

# get version of the library
get_filename_component ( RBIO_LIBRARY  ${RBIO_LIBRARY} REALPATH )
get_filename_component ( RBIO_FILENAME ${SPEX_LIBRARY} NAME )
string (
    REGEX MATCH "[0-9]+.[0-9]+.[0-9]+"
    RBIO_VERSION
    ${RBIO_FILENAME}
)
set (RBIO_LIBRARIES ${RBIO_LIBRARY})

include (FindPackageHandleStandardArgs)

find_package_handle_standard_args ( RBio
    REQUIRED_VARS RBIO_LIBRARIES RBIO_INCLUDE_DIR
    VERSION_VAR RBIO_VERSION
)

mark_as_advanced (
    RBIO_INCLUDE_DIR
    RBIO_LIBRARY
    RBIO_LIBRARIES
)

if ( RBIO_FOUND )
    message ( STATUS "RBio include dir: ${RBIO_INCLUDE_DIR}")
    message ( STATUS "RBio library:     ${RBIO_LIBRARY}" )
    message ( STATUS "RBio version:     ${RBIO_VERSION}" )
else ( )
    message ( STATUS "RBio not found" )
endif ( )

