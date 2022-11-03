#-------------------------------------------------------------------------------
# SuiteSparse/SuiteSparse_config/cmake_modules/FindSLIPLU.cmake
#-------------------------------------------------------------------------------

# Copyright (c) 2022, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

# Finds the SLIPLU include file and compiled library and sets:

# SLIPLU_INCLUDE_DIR - where to find SLIP_LU.h
# SLIPLU_LIBRARY     - compiled SLIPLU library
# SLIPLU_LIBRARIES   - libraries when using SLIPLU
# SLIPLU_FOUND       - true if SLIPLU found

# set ``SLIPLU_ROOT`` to a SLIPLU installation root to
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

# include files for SLIPLU
find_path ( SLIPLU_INCLUDE_DIR
    NAMES SLIP_LU.h
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/SLIP_LU
    HINTS ${CMAKE_SOURCE_DIR}/../SLIP_LU
    PATHS SLIPLU_ROOT ENV SLIPLU_ROOT
    PATH_SUFFIXES include Include
)

# compiled libraries SLIPLU
find_library ( SLIPLU_LIBRARY
    NAMES sliplu
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/SLIP_LU
    HINTS ${CMAKE_SOURCE_DIR}/../SLIP_LU
    PATHS SLIPLU_ROOT ENV SLIPLU_ROOT
    PATH_SUFFIXES lib build alternative
)

# get version of the library
get_filename_component (SLIPLU_LIBRARY ${SLIPLU_LIBRARY} REALPATH)
string (
    REGEX MATCH "[0-9]+.[0-9]+.[0-9]+"
    SLIPLU_VERSION
    ${SLIPLU_LIBRARY}
)
set (SLIPLU_LIBRARIES ${SLIPLU_LIBRARY})

include (FindPackageHandleStandardArgs)

find_package_handle_standard_args ( SLIPLU
    REQUIRED_VARS SLIPLU_LIBRARIES SLIPLU_INCLUDE_DIR
    VERSION_VAR SLIPLU_VERSION
)

mark_as_advanced (
    SLIPLU_INCLUDE_DIR
    SLIPLU_LIBRARY
    SLIPLU_LIBRARIES
)

if ( SLIPLU_FOUND )
    message ( STATUS "SLIPLU include dir: " ${SLIPLU_INCLUDE_DIR} )
    message ( STATUS "SLIPLU library:     " ${SLIPLU_LIBRARY} )
    message ( STATUS "SLIPLU version:     " ${SLIPLU_VERSION} )
    message ( STATUS "SLIPLU libraries:   " ${SLIPLU_LIBRARIES} )
else ( )
    message ( STATUS "SLIPLU not found" )
endif ( )

