#-------------------------------------------------------------------------------
# SuiteSparse/SuiteSparse_config/cmake_modules/FindGPUQREngine.cmake
#-------------------------------------------------------------------------------

# Copyright (c) 2022, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

# Finds the GPUQREngine compiled library and sets:

# GPUQRENGINE_LIBRARY     - compiled GPUQREngine library
# GPUQRENGINE_LIBRARIES   - libraries when using GPUQREngine
# GPUQRENGINE_FOUND       - true if GPUQREngine found

# set ``GPUQRENGINE_ROOT`` to a GPUQREngine installation root to
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

# compiled libraries GPUQREngine
find_library ( GPUQRENGINE_LIBRARY
    NAMES gpuqrengine
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/GPUQREngine
    HINTS ${CMAKE_SOURCE_DIR}/../GPUQREngine
    PATHS GPUQRENGINE_ROOT ENV GPUQRENGINE_ROOT
    PATH_SUFFIXES lib build alternative
)

# get version of the library
get_filename_component ( GPUQRENGINE_LIBRARY  ${GPUQRENGINE_LIBRARY} REALPATH )
get_filename_component ( GPUQRENGINE_FILENAME ${GPUQRENGINE_LIBRARY} NAME )
string (
    REGEX MATCH "[0-9]+.[0-9]+.[0-9]+"
    GPUQRENGINE_VERSION
    ${GPUQRENGINE_FILENAME}
)

# libaries when using GPUQREngine
set (GPUQRENGINE_LIBRARIES ${GPUQRENGINE_LIBRARY})

include (FindPackageHandleStandardArgs)

find_package_handle_standard_args ( GPUQREngine
    REQUIRED_VARS GPUQRENGINE_LIBRARIES
    VERSION_VAR GPUQRENGINE_VERSION
)

mark_as_advanced (
    GPUQRENGINE_LIBRARY
    GPUQRENGINE_LIBRARIES
)

if ( GPUQRENGINE_FOUND )
    message ( STATUS "GPUQREngine version:     ${GPUQRENGINE_VERSION} ")
    message ( STATUS "GPUQREngine libraries:   ${GPUQRENGINE_LIBRARIES} ")
else ( )
    message ( STATUS "GPUQREngine not found" )
endif ( )

