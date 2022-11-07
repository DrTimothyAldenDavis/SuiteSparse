#-------------------------------------------------------------------------------
# SuiteSparse/SuiteSparse_config/cmake_modules/FindCHOLMOD_CUDA.cmake
#-------------------------------------------------------------------------------

# Copyright (c) 2022, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

# Finds the CHOLMOD_CUDA compiled library and sets:

# CHOLMOD_CUDA_LIBRARY     - compiled CHOLMOD_CUDA library
# CHOLMOD_CUDA_LIBRARIES   - libraries when using CHOLMOD_CUDA
# CHOLMOD_CUDA_FOUND       - true if CHOLMOD_CUDA found

# set ``CHOLMOD_CUDA_ROOT`` to a CHOLMOD_CUDA installation root to
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

# compiled libraries CHOLMOD_CUDA for CUDA
find_library ( CHOLMOD_CUDA_LIBRARY
    NAMES cholmod_cuda${CMAKE_RELEASE_POSTFIX}
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse
    HINTS ${CMAKE_SOURCE_DIR}/../CHOLMOD/
    HINTS ${CMAKE_SOURCE_DIR}/../CHOLMOD/build/GPU
    PATHS CHOLMOD_CUDA_ROOT ENV CHOLMOD_CUDA_ROOT
    PATH_SUFFIXES lib build alternative
)

# get version of the library
get_filename_component ( CHOLMOD_CUDA_LIBRARY  ${CHOLMOD_CUDA_LIBRARY} REALPATH )
get_filename_component ( CHOLMOD_CUDA_FILENAME ${CHOLMOD_CUDA_LIBRARY} NAME )
string (
    REGEX MATCH "[0-9]+.[0-9]+.[0-9]+"
    CHOLMOD_CUDA_VERSION
    ${CHOLMOD_CUDA_FILENAME}
)
set (CHOLMOD_CUDA_LIBRARIES ${CHOLMOD_CUDA_LIBRARY})

include (FindPackageHandleStandardArgs)

find_package_handle_standard_args ( CHOLMOD_CUDA
    REQUIRED_VARS CHOLMOD_CUDA_LIBRARIES
    VERSION_VAR CHOLMOD_CUDA_VERSION
)

mark_as_advanced (
    CHOLMOD_CUDA_LIBRARY
    CHOLMOD_CUDA_LIBRARIES
)

if ( CHOLMOD_CUDA_FOUND )
    message ( STATUS "CHOLMOD_CUDA library:     ${CHOLMOD_CUDA_LIBRARY}" )
    message ( STATUS "CHOLMOD_CUDA version:     ${CHOLMOD_CUDA_VERSION}" )
else ( )
    message ( STATUS "CHOLMOD_CUDA not found" )
endif ( )

