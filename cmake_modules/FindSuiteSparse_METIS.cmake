#-------------------------------------------------------------------------------
# SuiteSparse/cmake_modules/FindSuiteSparse_metis.cmake
#-------------------------------------------------------------------------------

# Copyright (c) 2022, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

# Finds the SuiteSparse_metis include file and compiled library and sets:

# SUITESPARSE_METIS_INCLUDE_DIR - where to find metis.h
# SUITESPARSE_METIS_LIBRARY     - compiled SuiteSparse_metis library
# SUITESPARSE_METIS_LIBRARIES   - libraries when using SuiteSparse_metis
# SUITESPARSE_METIS_FOUND       - true if SuiteSparse_metis found

# set ``SUITESPARSE_METIS_ROOT`` to a SuiteSparse_metis installation root to
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
#       "${CMAKE_SOURCE_DIR}/../SuiteSparse/cmake_modules")

#-------------------------------------------------------------------------------

# include files for SuiteSparse_metis
find_path ( SUITESPARSE_METIS_INCLUDE_DIR
    NAMES suitesparse_metis.h
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/SuiteSparse_metis
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse_metis
    PATHS SUITESPARSE_METIS_ROOT ENV SUITESPARSE_METIS_ROOT
    PATH_SUFFIXES include Include
)

# compiled libraries SuiteSparse_metis
find_library ( SUITESPARSE_METIS_LIBRARY
    NAMES suitesparse_metis
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/SuiteSparse_metis
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse_metis
    PATHS SUITESPARSE_METIS_ROOT ENV SUITESPARSE_METIS_ROOT
    PATH_SUFFIXES lib build alternative
)

# get version of the library
get_filename_component (SUITESPARSE_METIS_LIBRARY ${SUITESPARSE_METIS_LIBRARY} REALPATH)
string (
    REGEX MATCH "[0-9]+.[0-9]+.[0-9]+"
    SUITESPARSE_METIS_VERSION
    ${SUITESPARSE_METIS_LIBRARY}
)
set (SUITESPARSE_METIS_LIBRARIES ${SUITESPARSE_METIS_LIBRARY})

include (FindPackageHandleStandardArgs)

find_package_handle_standard_args ( SuiteSparse_metis
    REQUIRED_VARS SUITESPARSE_METIS_LIBRARIES SUITESPARSE_METIS_INCLUDE_DIR
    VERSION_VAR SUITESPARSE_METIS_VERSION
)

mark_as_advanced (
    SUITESPARSE_METIS_INCLUDE_DIR
    SUITESPARSE_METIS_LIBRARY
    SUITESPARSE_METIS_LIBRARIES
)

if ( SUITESPARSE_METIS_FOUND )
    message ( STATUS "SuiteSparse_metis include dir: " ${SUITESPARSE_METIS_INCLUDE_DIR} )
    message ( STATUS "SuiteSparse_metis library:     " ${SUITESPARSE_METIS_LIBRARY} )
    message ( STATUS "SuiteSparse_metis version:     " ${SUITESPARSE_METIS_VERSION} )
    message ( STATUS "SuiteSparse_metis libraries:   " ${SUITESPARSE_METIS_LIBRARIES} )
else ( )
    message ( STATUS "SuiteSparse_metis not found" )
endif ( )

