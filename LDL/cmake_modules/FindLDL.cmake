#-------------------------------------------------------------------------------
# SuiteSparse/SuiteSparse_config/cmake_modules/FindLDL.cmake
#-------------------------------------------------------------------------------

# Copyright (c) 2022, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

# Finds the LDL include file and compiled library and sets:

# LDL_INCLUDE_DIR - where to find ldl.h
# LDL_LIBRARY     - compiled LDL library
# LDL_LIBRARIES   - libraries when using LDL
# LDL_FOUND       - true if LDL found

# set ``LDL_ROOT`` to a LDL installation root to
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

# include files for LDL
find_path ( LDL_INCLUDE_DIR
    NAMES ldl.h
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/LDL
    HINTS ${CMAKE_SOURCE_DIR}/../LDL
    PATHS LDL_ROOT ENV LDL_ROOT
    PATH_SUFFIXES include Include
)

# compiled libraries LDL
find_library ( LDL_LIBRARY
    NAMES ldl
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/LDL
    HINTS ${CMAKE_SOURCE_DIR}/../LDL
    PATHS LDL_ROOT ENV LDL_ROOT
    PATH_SUFFIXES lib build alternative
)

# get version of the library
foreach (_VERSION MAIN_VERSION SUB_VERSION SUBSUB_VERSION)
  file (STRINGS ${LDL_INCLUDE_DIR}/ldl.h _VERSION_LINE REGEX "define[ ]+LDL_${_VERSION}")
  if (_VERSION_LINE)
    string (REGEX REPLACE ".*define[ ]+LDL_${_VERSION}[ ]+([0-9]*).*" "\\1" _LDL_${_VERSION} "${_VERSION_LINE}")
  endif ()
  unset (_VERSION_LINE)
endforeach ()
set (LDL_VERSION "${_LDL_MAIN_VERSION}.${_LDL_SUB_VERSION}.${_LDL_SUBSUB_VERSION}")
set (LDL_LIBRARIES ${LDL_LIBRARY})

include (FindPackageHandleStandardArgs)

find_package_handle_standard_args ( LDL
    REQUIRED_VARS LDL_LIBRARIES LDL_INCLUDE_DIR
    VERSION_VAR LDL_VERSION
)

mark_as_advanced (
    LDL_INCLUDE_DIR
    LDL_LIBRARY
    LDL_LIBRARIES
)

if ( LDL_FOUND )
    message ( STATUS "LDL include dir: ${LDL_INCLUDE_DIR}" )
    message ( STATUS "LDL library:     ${LDL_LIBRARY}" )
    message ( STATUS "LDL version:     ${LDL_VERSION}" )
else ( )
    message ( STATUS "LDL not found" )
endif ( )

