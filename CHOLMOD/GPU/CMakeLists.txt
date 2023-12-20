#-------------------------------------------------------------------------------
# SuiteSparse/CHOLMOD/GPU/CMakeLists.txt:  cmake for CHOLMOD/GPU module
#-------------------------------------------------------------------------------

# CHOLMOD/GPU Module.  Copyright (C) 2005-2022, Timothy A. Davis.
# All Rights Reserved.
# SPDX-License-Identifier: GPL-2.0+

#-------------------------------------------------------------------------------
# get the version
#-------------------------------------------------------------------------------

# cmake 3.22 is required to find the BLAS/LAPACK
cmake_minimum_required ( VERSION 3.22 )

message ( STATUS "Building CHOLMOD_CUDA version: v"
    ${CHOLMOD_VERSION_MAJOR}.
    ${CHOLMOD_VERSION_MINOR}.
    ${CHOLMOD_VERSION_SUB} " (" ${CHOLMOD_DATE} ")" )

include ( SuiteSparsePolicy )

enable_language ( CUDA )
set ( CMAKE_CUDA_FLAGS "-cudart=static -lineinfo -DCHOLMOD_HAS_CUDA" )
set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCHOLMOD_HAS_CUDA" )
message ( STATUS "C++ flags for CUDA:  ${CMAKE_CXX_FLAGS}" )
message ( STATUS "nvcc flags for CUDA: ${CMAKE_CUDA_FLAGS}" )
file ( GLOB CHOLMOD_CUDA_SOURCES "cholmod_gpu.c" "cholmod_l_gpu.c"
    "cholmod_gpu_kernels.cu" )

if ( BUILD_SHARED_LIBS )
    target_sources ( CHOLMOD PRIVATE ${CHOLMOD_CUDA_SOURCES} )
endif ( )

if ( BUILD_STATIC_LIBS )
    target_sources ( CHOLMOD_static PRIVATE ${CHOLMOD_CUDA_SOURCES} )
endif ( )

set ( CHOLMOD_CUDA_INCLUDES
        ../Check
        ../../SuiteSparse_config
        ../Cholesky
        ../Utility
        ../Supernodal
        ../Include
        ../GPU )

include_directories ( ${CHOLMOD_CUDA_INCLUDES} ${CUDAToolkit_INCLUDE_DIRS} )

if ( BUILD_SHARED_LIBS )
    target_include_directories ( CHOLMOD PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
        ${CHOLMOD_CUDA_INCLUDES} )
    set_target_properties ( CHOLMOD PROPERTIES POSITION_INDEPENDENT_CODE ON )
    set_target_properties ( CHOLMOD PROPERTIES CUDA_SEPARABLE_COMPILATION ON )
    target_compile_definitions ( CHOLMOD PUBLIC "CHOLMOD_HAS_CUDA" )
endif ( )

if ( BUILD_STATIC_LIBS )
    target_include_directories ( CHOLMOD_static PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
        ${CHOLMOD_CUDA_INCLUDES} )
    set_target_properties ( CHOLMOD_static PROPERTIES CUDA_SEPARABLE_COMPILATION on )
    set_target_properties ( CHOLMOD_static PROPERTIES POSITION_INDEPENDENT_CODE on )
    target_compile_definitions ( CHOLMOD_static PUBLIC "CHOLMOD_HAS_CUDA" )
endif ( )

if ( BUILD_SHARED_LIBS )
    target_link_libraries ( CHOLMOD PRIVATE
        CUDA::nvrtc CUDA::cudart_static CUDA::cublas )
endif ( )
if ( BUILD_STATIC_LIBS )
    target_link_libraries ( CHOLMOD_static PRIVATE
        CUDA::nvrtc CUDA::cudart_static CUDA::cublas )
endif ( )
