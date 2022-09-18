#-------------------------------------------------------------------------------
# SuiteSparse/cmake_modules/FindCHOLMOD.cmake
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
#       "${CMAKE_SOURCE_DIR}/../SuiteSparse/cmake_modules")

#-------------------------------------------------------------------------------

# include files for CHOLMOD
find_path ( CHOLMOD_INCLUDE_DIR
    NAMES cholmod.h
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/CHOLMOD
    HINTS ${CMAKE_SOURCE_DIR}/../CHOLMOD
    PATHS CHOLMOD_ROOT ENV CHOLMOD_ROOT
    PATH_SUFFIXES include Include
)

# compiled libraries CHOLMOD
find_library ( CHOLMOD_LIBRARY
    NAMES cholmod
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/CHOLMOD
    HINTS ${CMAKE_SOURCE_DIR}/../CHOLMOD
    PATHS CHOLMOD_ROOT ENV CHOLMOD_ROOT
    PATH_SUFFIXES lib build alternative
)

# get version of the library
get_filename_component (CHOLMOD_LIBRARY ${CHOLMOD_LIBRARY} REALPATH)
string (
    REGEX MATCH "[0-9]+.[0-9]+.[0-9]+"
    CHOLMOD_VERSION
    ${CHOLMOD_LIBRARY}
)
set (CHOLMOD_LIBRARIES ${CHOLMOD_LIBRARY})

include (FindPackageHandleStandardArgs)

find_package_handle_standard_args ( CHOLMOD
    REQUIRED_VARS CHOLMOD_LIBRARIES CHOLMOD_INCLUDE_DIR
    VERSION_VAR CHOLMOD_VERSION
)

mark_as_advanced (
    CHOLMOD_INCLUDE_DIR
    CHOLMOD_LIBRARY
    CHOLMOD_LIBRARIES
)

if ( CHOLMOD_FOUND )
    message ( STATUS "CHOLMOD include dir: " ${CHOLMOD_INCLUDE_DIR} )
    message ( STATUS "CHOLMOD library:     " ${CHOLMOD_LIBRARY} )
    message ( STATUS "CHOLMOD version:     " ${CHOLMOD_VERSION} )
    message ( STATUS "CHOLMOD libraries:   " ${CHOLMOD_LIBRARIES} )
else ( )
    message ( STATUS "CHOLMOD not found" )
endif ( )

