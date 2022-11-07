#-------------------------------------------------------------------------------
# SuiteSparse/SuiteSparse_config/cmake_modules/FindCCOLAMD.cmake
#-------------------------------------------------------------------------------

# Copyright (c) 2022, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

# Finds the CCOLAMD include file and compiled library and sets:

# CCOLAMD_INCLUDE_DIR - where to find ccolamd.h
# CCOLAMD_LIBRARY     - compiled CCOLAMD library
# CCOLAMD_LIBRARIES   - libraries when using CCOLAMD
# CCOLAMD_FOUND       - true if CCOLAMD found

# set ``CCOLAMD_ROOT`` to a CCOLAMD installation root to
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

# include files for CCOLAMD
find_path ( CCOLAMD_INCLUDE_DIR
    NAMES ccolamd.h
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/CCOLAMD
    HINTS ${CMAKE_SOURCE_DIR}/../CCOLAMD
    PATHS CCOLAMD_ROOT ENV CCOLAMD_ROOT
    PATH_SUFFIXES include Include
)

# compiled libraries CCOLAMD
find_library ( CCOLAMD_LIBRARY
    NAMES ccolamd${CMAKE_RELEASE_POSTFIX}
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/CCOLAMD
    HINTS ${CMAKE_SOURCE_DIR}/../CCOLAMD
    PATHS CCOLAMD_ROOT ENV CCOLAMD_ROOT
    PATH_SUFFIXES lib build alternative
)

# get version of the library
get_filename_component ( CCOLAMD_LIBRARY  ${CCOLAMD_LIBRARY} REALPATH )
get_filename_component ( CCOLAMD_FILENAME ${CCOLAMD_LIBRARY} NAME )
string (
    REGEX MATCH "[0-9]+.[0-9]+.[0-9]+"
    CCOLAMD_VERSION
    ${CCOLAMD_FILENAME}
)
set (CCOLAMD_LIBRARIES ${CCOLAMD_LIBRARY})

include (FindPackageHandleStandardArgs)

find_package_handle_standard_args ( CCOLAMD
    REQUIRED_VARS CCOLAMD_LIBRARIES CCOLAMD_INCLUDE_DIR
    VERSION_VAR CCOLAMD_VERSION
)

mark_as_advanced (
    CCOLAMD_INCLUDE_DIR
    CCOLAMD_LIBRARY
    CCOLAMD_LIBRARIES
)

if ( CCOLAMD_FOUND )
    message ( STATUS "CCOLAMD include dir: ${CCOLAMD_INCLUDE_DIR}" )
    message ( STATUS "CCOLAMD library:     ${CCOLAMD_LIBRARY}" )
    message ( STATUS "CCOLAMD version:     ${CCOLAMD_VERSION}" )
else ( )
    message ( STATUS "CCOLAMD not found" )
endif ( )

