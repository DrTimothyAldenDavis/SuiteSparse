#-------------------------------------------------------------------------------
# SuiteSparse/SuiteSparse_config/cmake_modules/FindSuiteSparse_config.cmake
#-------------------------------------------------------------------------------

# Copyright (c) 2022, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

# Finds the SuiteSparse_config include file and compiled library and sets:

# SUITESPARSE_CONFIG_INCLUDE_DIR - where to find SuiteSparse_config.h
# SUITESPARSE_CONFIG_LIBRARY     - compiled SuiteSparse_config library
# SUITESPARSE_CONFIG_LIBRARIES   - libraries when using SuiteSparse_config
# SUITESPARSE_CONFIG_FOUND       - true if SuiteSparse_config found

# set ``SUITESPARSE_CONFIG_ROOT`` to a SuiteSparse_config installation root to
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
#   set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/../SuiteSparse/SuiteSparse_config/cmake_modules")

#-------------------------------------------------------------------------------

# include files for SuiteSparse_config
find_path ( SUITESPARSE_CONFIG_INCLUDE_DIR
    NAMES SuiteSparse_config.h
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/SuiteSparse_config
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse_config
    PATHS SUITESPARSE_CONFIG_ROOT ENV SUITESPARSE_CONFIG_ROOT
    PATH_SUFFIXES include Include
)

# compiled libraries SuiteSparse_config
find_library ( SUITESPARSE_CONFIG_LIBRARY
    NAMES suitesparseconfig
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/SuiteSparse_config
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse_config
    PATHS SUITESPARSE_CONFIG_ROOT ENV SUITESPARSE_CONFIG_ROOT
    PATH_SUFFIXES lib build alternative
)

# get version of the library
get_filename_component ( SUITESPARSE_CONFIG_LIBRARY  ${SUITESPARSE_CONFIG_LIBRARY} REALPATH )
get_filename_component ( SUITESPARSE_CONFIG_FILENAME ${SUITESPARSE_CONFIG_LIBRARY} NAME )
string (
    REGEX MATCH "[0-9]+.[0-9]+.[0-9]+"
    SUITESPARSE_CONFIG_VERSION
    ${SUITESPARSE_CONFIG_FILENAME}
)

# libaries when using SuiteSparse_config
set (SUITESPARSE_CONFIG_LIBRARIES ${SUITESPARSE_CONFIG_LIBRARY})

include (FindPackageHandleStandardArgs)

find_package_handle_standard_args ( SuiteSparse_config
    REQUIRED_VARS SUITESPARSE_CONFIG_LIBRARIES SUITESPARSE_CONFIG_INCLUDE_DIR
    VERSION_VAR SUITESPARSE_CONFIG_VERSION
)

mark_as_advanced (
    SUITESPARSE_CONFIG_INCLUDE_DIR
    SUITESPARSE_CONFIG_LIBRARY
    SUITESPARSE_CONFIG_LIBRARIES
)

if ( SUITESPARSE_CONFIG_FOUND )
    message ( STATUS "SuiteSparse_config include dir: ${SUITESPARSE_CONFIG_INCLUDE_DIR}" )
    message ( STATUS "SuiteSparse_config version:     ${SUITESPARSE_CONFIG_VERSION}" )
else ( )
    message ( STATUS "SuiteSparse_config not found" )
endif ( )

