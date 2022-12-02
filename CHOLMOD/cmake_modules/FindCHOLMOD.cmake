#-------------------------------------------------------------------------------
# SuiteSparse/SuiteSparse_config/cmake_modules/FindCHOLMOD.cmake
#-------------------------------------------------------------------------------

# Copyright (c) 2022, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

# Finds the CHOLMOD include file and compiled library and sets:

# CHOLMOD_INCLUDE_DIR - where to find cholmod.h
# CHOLMOD_LIBRARY     - compiled CHOLMOD library
# CHOLMOD_LIBRARIES   - libraries when using CHOLMOD
# CHOLMOD_FOUND       - true if CHOLMOD found

# set ``CHOLMOD_ROOT`` to a CHOLMOD installation root to
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

# include files for CHOLMOD
find_path ( CHOLMOD_INCLUDE_DIR
    NAMES cholmod.h
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/CHOLMOD
    HINTS ${CMAKE_SOURCE_DIR}/../CHOLMOD
    PATH_SUFFIXES include Include
)

# compiled libraries CHOLMOD
find_library ( CHOLMOD_LIBRARY
    NAMES cholmod
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/CHOLMOD
    HINTS ${CMAKE_SOURCE_DIR}/../CHOLMOD
    PATH_SUFFIXES lib build
)

# get version of the library
foreach (_VERSION MAIN_VERSION SUB_VERSION SUBSUB_VERSION)
  file (STRINGS ${CHOLMOD_INCLUDE_DIR}/cholmod.h _VERSION_LINE REGEX "define[ ]+CHOLMOD_${_VERSION}")
  if (_VERSION_LINE)
    string (REGEX REPLACE ".*define[ ]+CHOLMOD_${_VERSION}[ ]+([0-9]*).*" "\\1" _CHOLMOD_${_VERSION} "${_VERSION_LINE}")
  endif ()
  unset (_VERSION_LINE)
endforeach ()
set (CHOLMOD_VERSION "${_CHOLMOD_MAIN_VERSION}.${_CHOLMOD_SUB_VERSION}.${_CHOLMOD_SUBSUB_VERSION}")
set (CHOLMOD_LIBRARIES ${CHOLMOD_LIBRARY} )

include (FindPackageHandleStandardArgs)

find_package_handle_standard_args ( CHOLMOD
    REQUIRED_VARS CHOLMOD_LIBRARY CHOLMOD_INCLUDE_DIR
    VERSION_VAR CHOLMOD_VERSION
)

mark_as_advanced (
    CHOLMOD_INCLUDE_DIR
    CHOLMOD_LIBRARY
    CHOLMOD_LIBRARIES
)

if ( CHOLMOD_FOUND )
    message ( STATUS "CHOLMOD include dir: ${CHOLMOD_INCLUDE_DIR}" )
    message ( STATUS "CHOLMOD library:     ${CHOLMOD_LIBRARY}" )
    message ( STATUS "CHOLMOD version:     ${CHOLMOD_VERSION}" )
else ( )
    message ( STATUS "CHOLMOD not found" )
endif ( )

