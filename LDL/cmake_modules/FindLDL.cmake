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
get_filename_component ( LDL_LIBRARY  ${LDL_LIBRARY} REALPATH )
get_filename_component ( LDL_FILENAME ${LDL_LIBRARY} NAME )
string (
    REGEX MATCH "[0-9]+.[0-9]+.[0-9]+"
    LDL_VERSION
    ${LDL_FILENAME}
)
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
    message ( STATUS "LDL include dir: " ${LDL_INCLUDE_DIR} )
    message ( STATUS "LDL library:     " ${LDL_LIBRARY} )
    message ( STATUS "LDL version:     " ${LDL_VERSION} )
    message ( STATUS "LDL libraries:   " ${LDL_LIBRARIES} )
else ( )
    message ( STATUS "LDL not found" )
endif ( )

