#-------------------------------------------------------------------------------
# SuiteSparse/SuiteSparse_config/cmake_modules/FindSPQR.cmake
#-------------------------------------------------------------------------------

# Copyright (c) 2022, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

# Finds the SPQR include file and compiled library and sets:

# SPQR_INCLUDE_DIR - where to find SuiteSparseQR.hpp and other headers
# SPQR_LIBRARY     - compiled SPQR library
# SPQR_LIBRARIES   - libraries when using SPQR
# SPQR_FOUND       - true if SPQR found

# set ``SPQR_ROOT`` to a SPQR installation root to
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

# include files for SPQR
find_path ( SPQR_INCLUDE_DIR
    NAMES SuiteSparseQR.hpp
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/SPQR
    HINTS ${CMAKE_SOURCE_DIR}/../SPQR
    PATH_SUFFIXES include Include
)

# compiled libraries SPQR
find_library ( SPQR_LIBRARY
    NAMES spqr
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/SPQR
    HINTS ${CMAKE_SOURCE_DIR}/../SPQR
    PATH_SUFFIXES lib build
)

# get version of the library
foreach (_VERSION MAIN_VERSION SUB_VERSION SUBSUB_VERSION)
  file (STRINGS ${SPQR_INCLUDE_DIR}/SuiteSparseQR_definitions.h _VERSION_LINE REGEX "define[ ]+SPQR_${_VERSION}")
  if (_VERSION_LINE)
    string (REGEX REPLACE ".*define[ ]+SPQR_${_VERSION}[ ]+([0-9]*).*" "\\1" _SPQR_${_VERSION} "${_VERSION_LINE}")
  endif ()
  unset (_VERSION_LINE)
endforeach ()
set (SPQR_VERSION "${_SPQR_MAIN_VERSION}.${_SPQR_SUB_VERSION}.${_SPQR_SUBSUB_VERSION}")
set (SPQR_LIBRARIES ${SPQR_LIBRARY})

include (FindPackageHandleStandardArgs)

find_package_handle_standard_args ( SPQR
    REQUIRED_VARS SPQR_LIBRARIES SPQR_INCLUDE_DIR
    VERSION_VAR SPQR_VERSION
)

mark_as_advanced (
    SPQR_INCLUDE_DIR
    SPQR_LIBRARY
    SPQR_LIBRARIES
)

if ( SPQR_FOUND )
    message ( STATUS "SPQR include dir: ${SPQR_INCLUDE_DIR}" )
    message ( STATUS "SPQR library:     ${SPQR_LIBRARY}" )
    message ( STATUS "SPQR version:     ${SPQR_VERSION}" )
else ( )
    message ( STATUS "SPQR not found" )
endif ( )

