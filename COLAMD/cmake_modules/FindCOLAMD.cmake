#-------------------------------------------------------------------------------
# SuiteSparse/SuiteSparse_config/cmake_modules/FindCOLAMD.cmake
#-------------------------------------------------------------------------------

# Copyright (c) 2022, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

# Finds the COLAMD include file and compiled library and sets:

# COLAMD_INCLUDE_DIR - where to find colamd.h
# COLAMD_LIBRARY     - compiled COLAMD library
# COLAMD_LIBRARIES   - libraries when using COLAMD
# COLAMD_FOUND       - true if COLAMD found

# set ``COLAMD_ROOT`` to a COLAMD installation root to
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

# include files for COLAMD
find_path ( COLAMD_INCLUDE_DIR
    NAMES colamd.h
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/COLAMD
    HINTS ${CMAKE_SOURCE_DIR}/../COLAMD
    PATH_SUFFIXES include Include
)

# compiled libraries COLAMD
find_library ( COLAMD_LIBRARY
    NAMES colamd
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/COLAMD
    HINTS ${CMAKE_SOURCE_DIR}/../COLAMD
    PATH_SUFFIXES lib build
)

# get version of the library
foreach (_VERSION MAIN_VERSION SUB_VERSION SUBSUB_VERSION)
  file (STRINGS ${COLAMD_INCLUDE_DIR}/colamd.h _VERSION_LINE REGEX "define[ ]+COLAMD_${_VERSION}")
  if (_VERSION_LINE)
    string (REGEX REPLACE ".*define[ ]+COLAMD_${_VERSION}[ ]+([0-9]*).*" "\\1" _COLAMD_${_VERSION} "${_VERSION_LINE}")
  endif ()
  unset (_VERSION_LINE)
endforeach ()
set (COLAMD_VERSION "${_COLAMD_MAIN_VERSION}.${_COLAMD_SUB_VERSION}.${_COLAMD_SUBSUB_VERSION}")
set (COLAMD_LIBRARIES ${COLAMD_LIBRARY})

include (FindPackageHandleStandardArgs)

find_package_handle_standard_args ( COLAMD
    REQUIRED_VARS COLAMD_LIBRARIES COLAMD_INCLUDE_DIR
    VERSION_VAR COLAMD_VERSION
)

mark_as_advanced (
    COLAMD_INCLUDE_DIR
    COLAMD_LIBRARY
    COLAMD_LIBRARIES
)

if ( COLAMD_FOUND )
    message ( STATUS "COLAMD include dir: ${COLAMD_INCLUDE_DIR}" )
    message ( STATUS "COLAMD library:     ${COLAMD_LIBRARY}" )
    message ( STATUS "COLAMD version:     ${COLAMD_VERSION}" )
else ( )
    message ( STATUS "COLAMD not found" )
endif ( )

