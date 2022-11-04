#-------------------------------------------------------------------------------
# SuiteSparse/SuiteSparse_config/cmake_modules/FindUMFPACK.cmake
#-------------------------------------------------------------------------------

# Copyright (c) 2022, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

# Finds the UMFPACK include file and compiled library and sets:

# UMFPACK_INCLUDE_DIR - where to find umfpack.h
# UMFPACK_LIBRARY     - compiled UMFPACK library
# UMFPACK_LIBRARIES   - libraries when using UMFPACK
# UMFPACK_FOUND       - true if UMFPACK found

# set ``UMFPACK_ROOT`` to a UMFPACK installation root to
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

# include files for UMFPACK
find_path ( UMFPACK_INCLUDE_DIR
    NAMES umfpack.h
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/UMFPACK
    HINTS ${CMAKE_SOURCE_DIR}/../UMFPACK
    PATHS UMFPACK_ROOT ENV UMFPACK_ROOT
    PATH_SUFFIXES include Include
)

# compiled libraries UMFPACK
find_library ( UMFPACK_LIBRARY
    NAMES umfpack
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/UMFPACK
    HINTS ${CMAKE_SOURCE_DIR}/../UMFPACK
    PATHS UMFPACK_ROOT ENV UMFPACK_ROOT
    PATH_SUFFIXES lib build alternative
)

# get version of the library
get_filename_component ( UMFPACK_LIBRARY  ${UMFPACK_LIBRARY} REALPATH )
get_filename_component ( UMFPACK_FILENAME ${UMFPACK_LIBRARY} NAME )
string (
    REGEX MATCH "[0-9]+.[0-9]+.[0-9]+"
    UMFPACK_VERSION
    ${UMFPACK_FILENAME}
)
set (UMFPACK_LIBRARIES ${UMFPACK_LIBRARY})

include (FindPackageHandleStandardArgs)

find_package_handle_standard_args ( UMFPACK
    REQUIRED_VARS UMFPACK_LIBRARIES UMFPACK_INCLUDE_DIR
    VERSION_VAR UMFPACK_VERSION
)

mark_as_advanced (
    UMFPACK_INCLUDE_DIR
    UMFPACK_LIBRARY
    UMFPACK_LIBRARIES
)

if ( UMFPACK_FOUND )
    message ( STATUS "UMFPACK include dir: " ${UMFPACK_INCLUDE_DIR} )
    message ( STATUS "UMFPACK library:     " ${UMFPACK_LIBRARY} )
    message ( STATUS "UMFPACK version:     " ${UMFPACK_VERSION} )
    message ( STATUS "UMFPACK libraries:   " ${UMFPACK_LIBRARIES} )
else ( )
    message ( STATUS "UMFPACK not found" )
endif ( )

