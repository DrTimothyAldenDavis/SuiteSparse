#-------------------------------------------------------------------------------
# SuiteSparse/SuiteSparse_config/cmake_modules/FindSuiteSparse_GPURuntime.cmake
#-------------------------------------------------------------------------------

# Copyright (c) 2022, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

# Finds the SuiteSparse_GPURuntime compiled library and sets:

# SUITESPARSE_GPURUNTIME_LIBRARY     - compiled SuiteSparse_GPURuntime library
# SUITESPARSE_GPURUNTIME_LIBRARIES   - libraries when using SuiteSparse_GPURuntime
# SUITESPARSE_GPURUNTIME_FOUND       - true if SuiteSparse_GPURuntime found

# set ``SUITESPARSE_GPURUNTIME_ROOT`` to a SuiteSparse_GPURuntime installation root to
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

# compiled libraries SuiteSparse_GPURuntime
find_library ( SUITESPARSE_GPURUNTIME_LIBRARY
    NAMES suitesparse_gpuruntime
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/SuiteSparse_GPURuntime
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse_GPURuntime
    PATHS SUITESPARSE_GPURUNTIME_ROOT ENV SUITESPARSE_GPURUNTIME_ROOT
    PATH_SUFFIXES lib build alternative
)

# get version of the library
get_filename_component (SUITESPARSE_GPURUNTIME_LIBRARY ${SUITESPARSE_GPURUNTIME_LIBRARY} REALPATH)
string (
    REGEX MATCH "[0-9]+.[0-9]+.[0-9]+"
    SUITESPARSE_GPURUNTIME_VERSION
    ${SUITESPARSE_GPURUNTIME_LIBRARY}
)

# libaries when using SuiteSparse_GPURuntime
set (SUITESPARSE_GPURUNTIME_LIBRARIES ${SUITESPARSE_GPURUNTIME_LIBRARY})

include (FindPackageHandleStandardArgs)

find_package_handle_standard_args ( SuiteSparse_GPURuntime
    REQUIRED_VARS SUITESPARSE_GPURUNTIME_LIBRARIES
    VERSION_VAR SUITESPARSE_GPURUNTIME_VERSION
)

mark_as_advanced (
    SUITESPARSE_GPURUNTIME_LIBRARY
    SUITESPARSE_GPURUNTIME_LIBRARIES
)

if ( SUITESPARSE_GPURUNTIME_FOUND )
    message ( STATUS "SuiteSparse_GPURuntime version:     ${SUITESPARSE_GPURUNTIME_VERSION} ")
    message ( STATUS "SuiteSparse_GPURuntime libraries:   ${SUITESPARSE_GPURUNTIME_LIBRARIES} ")
else ( )
    message ( STATUS "SuiteSparse_GPURuntime not found" )
endif ( )

