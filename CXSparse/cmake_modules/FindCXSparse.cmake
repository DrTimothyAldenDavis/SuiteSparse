#-------------------------------------------------------------------------------
# SuiteSparse/SuiteSparse_config/cmake_modules/FindCXSparse.cmake
#-------------------------------------------------------------------------------

# Copyright (c) 2022, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

# Finds the CXSparse include file and compiled library and sets:

# CXSPARSE_INCLUDE_DIR - where to find cs.h
# CXSPARSE_LIBRARY     - compiled CXSPARSE library
# CXSPARSE_LIBRARIES   - libraries when using CXSPARSE
# CXSPARSE_FOUND       - true if CXSPARSE found

# set ``CXSparse_ROOT`` or ``CXSPARSE_ROOT`` to a CXSPARSE installation root to
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

# include files for CXSPARSE
find_path ( CXSPARSE_INCLUDE_DIR
    NAMES cs.h
    HINTS ${CXSPARSE_ROOT}
    HINTS ENV CXSPARSE_ROOT
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/CXSparse
    HINTS ${CMAKE_SOURCE_DIR}/../CXSparse
    PATH_SUFFIXES include Include
)

# compiled libraries CXSPARSE
find_library ( CXSPARSE_LIBRARY
    NAMES cxsparse
    HINTS ${CXSPARSE_ROOT}
    HINTS ENV CXSPARSE_ROOT
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/CXSparse
    HINTS ${CMAKE_SOURCE_DIR}/../CXSparse
    PATH_SUFFIXES lib build
)

# get version of the library
foreach (_VERSION MAIN_VERSION SUB_VERSION SUBSUB_VERSION)
  file (STRINGS ${CXSPARSE_INCLUDE_DIR}/cs.h _VERSION_LINE REGEX "define[ ]+CXSPARSE_${_VERSION}")
  if (_VERSION_LINE)
    string (REGEX REPLACE ".*define[ ]+CXSPARSE_${_VERSION}[ ]+([0-9]*).*" "\\1" _CXSPARSE_${_VERSION} "${_VERSION_LINE}")
  endif ()
  unset (_VERSION_LINE)
endforeach ()
set (CXSPARSE_VERSION "${_CXSPARSE_MAIN_VERSION}.${_CXSPARSE_SUB_VERSION}.${_CXSPARSE_SUBSUB_VERSION}")
set (CXSPARSE_LIBRARIES ${CXSPARSE_LIBRARY})

include (FindPackageHandleStandardArgs)

find_package_handle_standard_args ( CXSparse
    REQUIRED_VARS CXSPARSE_LIBRARIES CXSPARSE_INCLUDE_DIR
    VERSION_VAR CXSPARSE_VERSION
)

mark_as_advanced (
    CXSPARSE_INCLUDE_DIR
    CXSPARSE_LIBRARY
    CXSPARSE_LIBRARIES
)

if ( CXSPARSE_FOUND )
    message ( STATUS "CXSparse include dir: ${CXSPARSE_INCLUDE_DIR}" )
    message ( STATUS "CXSparse library:     ${CXSPARSE_LIBRARY}" )
    message ( STATUS "CXSparse version:     ${CXSPARSE_VERSION}" )
else ( )
    message ( STATUS "CXSparse not found" )
endif ( )

