#-------------------------------------------------------------------------------
# SuiteSparse/SuiteSparse_config/cmake_modules/FindSPQR_CUDA.cmake
#-------------------------------------------------------------------------------

# Copyright (c) 2022, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

# Finds the SPQR_CUDA compiled library and sets:

# SPQR_CUDA_LIBRARY     - compiled SPQR_CUDA library
# SPQR_CUDA_LIBRARIES   - libraries when using SPQR_CUDA
# SPQR_CUDA_FOUND       - true if SPQR_CUDA found

# set ``SPQR_CUDA_ROOT`` to a SPQR_CUDA installation root to
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

# compiled libraries SPQR_CUDA for CUDA
find_library ( SPQR_CUDA_LIBRARY
    NAMES spqr_cuda
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse
    HINTS ${CMAKE_SOURCE_DIR}/../SPQR/
    HINTS ${CMAKE_SOURCE_DIR}/../SPQR/build/SPQRGPU
    PATHS SPQR_CUDA_ROOT ENV SPQR_CUDA_ROOT
    PATH_SUFFIXES lib build alternative
)

# get version of the library
get_filename_component ( SPQR_CUDA_LIBRARY  ${SPQR_CUDA_LIBRARY} REALPATH )
get_filename_component ( SPQR_CUDA_FILENAME ${SPQR_CUDA_LIBRARY} NAME )
string (
    REGEX MATCH "[0-9]+.[0-9]+.[0-9]+"
    SPQR_CUDA_VERSION
    ${SPQR_CUDA_FILENAME}
)
set (SPQR_CUDA_LIBRARIES ${SPQR_CUDA_LIBRARY})

include (FindPackageHandleStandardArgs)

find_package_handle_standard_args ( SPQR_CUDA
    REQUIRED_VARS SPQR_CUDA_LIBRARIES
    VERSION_VAR SPQR_CUDA_VERSION
)

mark_as_advanced (
    SPQR_CUDA_LIBRARY
    SPQR_CUDA_LIBRARIES
)

if ( SPQR_CUDA_FOUND )
    message ( STATUS "SPQR_CUDA library:     ${SPQR_CUDA_LIBRARY}" )
    message ( STATUS "SPQR_CUDA version:     ${SPQR_CUDA_VERSION}" )
else ( )
    message ( STATUS "SPQR_CUDA not found" )
endif ( )

