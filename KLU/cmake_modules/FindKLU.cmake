#-------------------------------------------------------------------------------
# SuiteSparse/SuiteSparse_config/cmake_modules/FindKLU.cmake
#-------------------------------------------------------------------------------

# Copyright (c) 2022, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

# Finds the KLU include file and compiled library and sets:

# KLU_INCLUDE_DIR - where to find klu.h
# KLU_LIBRARY     - compiled KLU library
# KLU_LIBRARIES   - libraries when using KLU
# KLU_FOUND       - true if KLU found

# set ``KLU_ROOT`` to a KLU installation root to
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

# include files for KLU
find_path ( KLU_INCLUDE_DIR
    NAMES klu.h
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/KLU
    HINTS ${CMAKE_SOURCE_DIR}/../KLU
    PATH_SUFFIXES include Include
)

# compiled libraries KLU
find_library ( KLU_LIBRARY
    NAMES klu
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/KLU
    HINTS ${CMAKE_SOURCE_DIR}/../KLU
    PATH_SUFFIXES lib build alternative
)

# get version of the library
foreach (_VERSION MAIN_VERSION SUB_VERSION SUBSUB_VERSION)
  file (STRINGS ${KLU_INCLUDE_DIR}/klu.h _VERSION_LINE REGEX "define[ ]+KLU_${_VERSION}")
  if (_VERSION_LINE)
    string (REGEX REPLACE ".*define[ ]+KLU_${_VERSION}[ ]+([0-9]*).*" "\\1" _KLU_${_VERSION} "${_VERSION_LINE}")
  endif ()
  unset (_VERSION_LINE)
endforeach ()
set (KLU_VERSION "${_KLU_MAIN_VERSION}.${_KLU_SUB_VERSION}.${_KLU_SUBSUB_VERSION}")
set (KLU_LIBRARIES ${KLU_LIBRARY})

include (FindPackageHandleStandardArgs)

find_package_handle_standard_args ( KLU
    REQUIRED_VARS KLU_LIBRARIES KLU_INCLUDE_DIR
    VERSION_VAR KLU_VERSION
)

mark_as_advanced (
    KLU_INCLUDE_DIR
    KLU_LIBRARY
    KLU_LIBRARIES
)

if ( KLU_FOUND )
    message ( STATUS "KLU include dir: ${KLU_INCLUDE_DIR}" )
    message ( STATUS "KLU library:     ${KLU_LIBRARY}" )
    message ( STATUS "KLU version:     ${KLU_VERSION}" )
else ( )
    message ( STATUS "KLU not found" )
endif ( )

