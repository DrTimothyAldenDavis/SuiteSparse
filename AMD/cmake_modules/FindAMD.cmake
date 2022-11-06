#-------------------------------------------------------------------------------
# SuiteSparse/SuiteSparse_config/cmake_modules/FindAMD.cmake
#-------------------------------------------------------------------------------

# Copyright (c) 2022, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

# Finds the AMD include file and compiled library and sets:

# AMD_INCLUDE_DIR - where to find amd.h
# AMD_LIBRARY     - compiled AMD library
# AMD_LIBRARIES   - libraries when using AMD
# AMD_FOUND       - true if AMD found

# set ``AMD_ROOT`` to a AMD installation root to
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

# include files for AMD
find_path ( AMD_INCLUDE_DIR
    NAMES amd.h
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/AMD
    HINTS ${CMAKE_SOURCE_DIR}/../AMD
    PATHS AMD_ROOT ENV AMD_ROOT
    PATH_SUFFIXES include Include
)

# compiled libraries AMD
find_library ( AMD_LIBRARY
    NAMES amd
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/AMD
    HINTS ${CMAKE_SOURCE_DIR}/../AMD
    PATHS AMD_ROOT ENV AMD_ROOT
    PATH_SUFFIXES lib build alternative
)

# get version of the library
get_filename_component ( AMD_LIBRARY  ${AMD_LIBRARY} REALPATH )
get_filename_component ( AMD_FILENAME ${AMD_LIBRARY} NAME )
string (
    REGEX MATCH "[0-9]+.[0-9]+.[0-9]+"
    AMD_VERSION
    ${AMD_FILENAME}
)
set (AMD_LIBRARIES ${AMD_LIBRARY})

include (FindPackageHandleStandardArgs)

find_package_handle_standard_args ( AMD
    REQUIRED_VARS AMD_LIBRARIES AMD_INCLUDE_DIR
    VERSION_VAR AMD_VERSION
)

mark_as_advanced (
    AMD_INCLUDE_DIR
    AMD_LIBRARY
    AMD_LIBRARIES
)

if ( AMD_FOUND )
    message ( STATUS "AMD include dir: ${AMD_INCLUDE_DIR}")
    message ( STATUS "AMD library:     ${AMD_LIBRARY}")
    message ( STATUS "AMD version:     ${AMD_VERSION}" )
else ( )
    message ( STATUS "AMD not found" )
endif ( )

