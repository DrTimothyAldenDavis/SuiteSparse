#-------------------------------------------------------------------------------
# SuiteSparse/SuiteSparse_config/cmake_modules/FindSPEX.cmake
#-------------------------------------------------------------------------------

# Copyright (c) 2022, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

# Finds the SPEX include file and compiled library and sets:

# SPEX_INCLUDE_DIR - where to find SPEX.h
# SPEX_LIBRARY     - compiled SPEX library
# SPEX_LIBRARIES   - libraries when using SPEX
# SPEX_FOUND       - true if SPEX found

# set ``SPEX_ROOT`` to a SPEX installation root to
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

# include files for SPEX
find_path ( SPEX_INCLUDE_DIR
    NAMES SPEX.h
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/SPEX
    HINTS ${CMAKE_SOURCE_DIR}/../SPEX
    PATHS SPEX_ROOT ENV SPEX_ROOT
    PATH_SUFFIXES include Include
)

# compiled libraries SPEX
find_library ( SPEX_LIBRARY
    NAMES spex
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/SPEX
    HINTS ${CMAKE_SOURCE_DIR}/../SPEX
    PATHS SPEX_ROOT ENV SPEX_ROOT
    PATH_SUFFIXES lib build alternative
)

# get version of the library
get_filename_component (SPEX_LIBRARY ${SPEX_LIBRARY} REALPATH)
string (
    REGEX MATCH "[0-9]+.[0-9]+.[0-9]+"
    SPEX_VERSION
    ${SPEX_LIBRARY}
)
set (SPEX_LIBRARIES ${SPEX_LIBRARY})

include (FindPackageHandleStandardArgs)

find_package_handle_standard_args ( SPEX
    REQUIRED_VARS SPEX_LIBRARIES SPEX_INCLUDE_DIR
    VERSION_VAR SPEX_VERSION
)

mark_as_advanced (
    SPEX_INCLUDE_DIR
    SPEX_LIBRARY
    SPEX_LIBRARIES
)

if ( SPEX_FOUND )
    message ( STATUS "SPEX include dir: " ${SPEX_INCLUDE_DIR} )
    message ( STATUS "SPEX library:     " ${SPEX_LIBRARY} )
    message ( STATUS "SPEX version:     " ${SPEX_VERSION} )
    message ( STATUS "SPEX libraries:   " ${SPEX_LIBRARIES} )
else ( )
    message ( STATUS "SPEX not found" )
endif ( )

