#-------------------------------------------------------------------------------
# SuiteSparse/SPQR/SPQRGPU/CMakeLists.txt:  cmake for SPQR/SPQRGPU module
#-------------------------------------------------------------------------------

# SPQR/SPQRGPU Module.  Copyright (C) 2005-2022, Timothy A. Davis.
# All Rights Reserved.
# SPDX-License-Identifier: GPL-2.0+

#-------------------------------------------------------------------------------
# get the version
#-------------------------------------------------------------------------------

# cmake 3.22 is required to find the BLAS/LAPACK
cmake_minimum_required ( VERSION 3.19 )

message ( STATUS "Building SPQR_CUDA version: v"
    ${SPQR_VERSION_MAJOR}.
    ${SPQR_VERSION_MINOR}.
    ${SPQR_VERSION_SUB} " (" ${SPQR_DATE} ")" )

include ( SuiteSparsePolicy )

project ( spqr_cuda 
    VERSION "${SPQR_VERSION_MAJOR}.${SPQR_VERSION_MINOR}.${SPQR_VERSION_SUB}"
    LANGUAGES C CXX CUDA )

set ( CMAKE_CUDA_FLAGS "-cudart=static -lineinfo -DSUITESPARSE_CUDA" )
set ( CMAKE_CXX_STANDARD 17 )
# set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++17 -fPIC -DSUITESPARSE_CUDA" )
set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSUITESPARSE_CUDA" )
set ( CMAKE_C_STANDARD 11 )

message ( STATUS "C++ flags for CUDA:  ${CMAKE_CXX_FLAGS}" )
message ( STATUS "nvcc flags for CUDA: ${CMAKE_CUDA_FLAGS}" )

file ( GLOB SPQR_CUDA_SOURCES "spqrgpu_*.cpp" )

add_library ( spqr_cuda SHARED ${SPQR_CUDA_SOURCES} )

set_target_properties ( spqr_cuda PROPERTIES
    VERSION ${SPQR_VERSION_MAJOR}.${SPQR_VERSION_MINOR}.${SPQR_VERSION_SUB}
    CXX_STANDARD_REQUIRED 17
    SOVERSION ${SPQR_VERSION_MAJOR} )

add_library ( spqr_cuda_static STATIC ${SPQR_CUDA_SOURCES} )

set_target_properties ( spqr_cuda_static PROPERTIES
    VERSION ${SPQR_VERSION_MAJOR}.${SPQR_VERSION_MINOR}.${SPQR_VERSION_SUB}
    CXX_STANDARD_REQUIRED 17
    SOVERSION ${SPQR_VERSION_MAJOR} )

set ( SPQR_CUDA_INCLUDES
        ../../GPUQREngine/Include
        ../../SuiteSparse_GPURuntime/Include
        ../../SuiteSparse_config
        ../../CHOLMOD/Include
        ../Include )

include_directories ( ${SPQR_CUDA_INCLUDES} ${CUDAToolkit_INCLUDE_DIRS} )

target_include_directories ( spqr_cuda PUBLIC
        ${CUDAToolkit_INCLUDE_DIRS}
        ${SPQR_CUDA_INCLUDES} )
set_target_properties ( spqr_cuda PROPERTIES POSITION_INDEPENDENT_CODE ON )
set_target_properties ( spqr_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON )

target_include_directories ( spqr_cuda_static PUBLIC
        ${CUDAToolkit_INCLUDE_DIRS}
        ${SPQR_CUDA_INCLUDES} )
set_target_properties ( spqr_cuda_static PROPERTIES CUDA_SEPARABLE_COMPILATION on )
set_target_properties ( spqr_cuda_static PROPERTIES POSITION_INDEPENDENT_CODE on )

target_link_libraries ( spqr_cuda CUDA::nvrtc CUDA::cudart_static
    CUDA::nvToolsExt CUDA::cublas )

target_link_libraries ( spqr_cuda_static CUDA::nvrtc CUDA::cudart_static
    CUDA::nvToolsExt CUDA::cublas )

#-------------------------------------------------------------------------------
# installation location
#-------------------------------------------------------------------------------

if ( GLOBAL_INSTALL )
    # install in /usr/local/lib and /usr/local/include.
    # requires "sudo make install"
    message ( STATUS "Installation will be system-wide (requires 'sudo make install')" )
    install ( TARGETS spqr_cuda
        LIBRARY       DESTINATION ${CMAKE_INSTALL_LIBDIR} )
    install ( TARGETS spqr_cuda_static
        ARCHIVE       DESTINATION ${CMAKE_INSTALL_LIBDIR} )
endif ( )

if ( INSIDE_SUITESPARSE )
    # also install in SuiteSparse/lib and SuiteSparse/include;
    # does not require "sudo make install", just "make install"
    message ( STATUS "Installation in ../../lib and ../../include," )
    message ( STATUS "  with 'make local ; make install'. No 'sudo' required." )
    install ( TARGETS spqr_cuda
        LIBRARY       DESTINATION ${SUITESPARSE_LIBDIR} )
    install ( TARGETS spqr_cuda_static
        ARCHIVE       DESTINATION ${SUITESPARSE_LIBDIR} )
endif ( )

