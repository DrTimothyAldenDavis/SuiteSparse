#-------------------------------------------------------------------------------
# SuiteSparse/SuiteSparse_config/cmake_modules/FindKLU_CHOLMOD.cmake
#-------------------------------------------------------------------------------

# Copyright (c) 2022, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

# Finds the KLU_CHOLMOD include file and compiled library and sets:

# KLU_CHOLMOD_INCLUDE_DIR - where to find klu_cholmod.h
# KLU_CHOLMOD_LIBRARY     - compiled KLU_CHOLMOD library
# KLU_CHOLMOD_LIBRARIES   - libraries when using KLU_CHOLMOD
# KLU_CHOLMOD_FOUND       - true if KLU_CHOLMOD found

# set ``KLU_CHOLMOD_ROOT`` to a KLU_CHOLMOD installation root to
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

# include files for KLU_CHOLMOD
find_path ( KLU_CHOLMOD_INCLUDE_DIR
    NAMES klu_cholmod.h
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/KLU/User
    HINTS ${CMAKE_SOURCE_DIR}/../KLU/User
    PATH_SUFFIXES include Include
)

# compiled libraries KLU_CHOLMOD
find_library ( KLU_CHOLMOD_LIBRARY
    NAMES klu_cholmod
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/KLU/User
    HINTS ${CMAKE_SOURCE_DIR}/../KLU/User
    PATH_SUFFIXES lib build
)

# get version of the library
find_package (KLU)
set (KLU_CHOLMOD_VERSION "${KLU_VERSION}")
set (KLU_CHOLMOD_LIBRARIES ${KLU_CHOLMOD_LIBRARY})

include (FindPackageHandleStandardArgs)

find_package_handle_standard_args ( KLU_CHOLMOD
    REQUIRED_VARS KLU_CHOLMOD_LIBRARIES KLU_CHOLMOD_INCLUDE_DIR
    VERSION_VAR KLU_CHOLMOD_VERSION
)

mark_as_advanced (
    KLU_CHOLMOD_INCLUDE_DIR
    KLU_CHOLMOD_LIBRARY
    KLU_CHOLMOD_LIBRARIES
)

if ( KLU_CHOLMOD_FOUND )
    message ( STATUS "KLU_CHOLMOD include dir: ${KLU_CHOLMOD_INCLUDE_DIR}" )
    message ( STATUS "KLU_CHOLMOD library:     ${KLU_CHOLMOD_LIBRARY}" )
    message ( STATUS "KLU_CHOLMOD version:     ${KLU_CHOLMOD_VERSION}" )
else ( )
    message ( STATUS "KLU_CHOLMOD not found" )
endif ( )

