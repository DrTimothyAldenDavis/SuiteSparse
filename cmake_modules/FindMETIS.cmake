#-------------------------------------------------------------------------------
# SuiteSparse/cmake_modules/FindMETIS.cmake
#-------------------------------------------------------------------------------

# Copyright (c) 2022, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

# Finds the METIS include file and compiled library and sets:

# METIS_INCLUDE_DIR - where to find metis.h
# METIS_LIBRARY     - compiled METIS library
# METIS_LIBRARIES   - libraries when using METIS
# METIS_FOUND       - true if METIS found

# set ``METIS_ROOT`` to a METIS installation root to
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

# include files for METIS
find_path ( METIS_INCLUDE_DIR
    NAMES metis.h
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/metis-5.1.0
    HINTS ${CMAKE_SOURCE_DIR}/../metis-5.1.0
    PATHS METIS_ROOT ENV METIS_ROOT
    PATH_SUFFIXES include Include
)

# compiled libraries METIS
find_library ( METIS_LIBRARY
    NAMES metis
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/metis-5.1.0
    HINTS ${CMAKE_SOURCE_DIR}/../metis-5.1.0
    PATHS METIS_ROOT ENV METIS_ROOT
    PATH_SUFFIXES lib build alternative
)

# get version of the library
get_filename_component (METIS_LIBRARY ${METIS_LIBRARY} REALPATH)
string (
    REGEX MATCH "[0-9]+.[0-9]+.[0-9]+"
    METIS_VERSION
    ${METIS_LIBRARY}
)
set (METIS_LIBRARIES ${METIS_LIBRARY})

include (FindPackageHandleStandardArgs)

find_package_handle_standard_args ( METIS
    REQUIRED_VARS METIS_LIBRARIES METIS_INCLUDE_DIR
    VERSION_VAR METIS_VERSION
)

mark_as_advanced (
    METIS_INCLUDE_DIR
    METIS_LIBRARY
    METIS_LIBRARIES
)

if ( METIS_FOUND )
    message ( STATUS "METIS include dir: " ${METIS_INCLUDE_DIR} )
    message ( STATUS "METIS library:     " ${METIS_LIBRARY} )
    message ( STATUS "METIS version:     " ${METIS_VERSION} )
    message ( STATUS "METIS libraries:   " ${METIS_LIBRARIES} )
else ( )
    message ( STATUS "METIS not found" )
endif ( )

