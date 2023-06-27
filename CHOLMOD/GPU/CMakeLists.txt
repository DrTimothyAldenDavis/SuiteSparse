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
cmake_minimum_required ( VERSION 3.19 )

message ( STATUS "Building CHOLMOD_CUDA version: v"
    ${CHOLMOD_VERSION_MAJOR}.
    ${CHOLMOD_VERSION_MINOR}.
    ${CHOLMOD_VERSION_SUB} " (" ${CHOLMOD_DATE} ")" )

include ( SuiteSparsePolicy )

if ( SUITESPARSE_CUDA )
    project ( cholmod_cuda
        VERSION "${CHOLMOD_VERSION_MAJOR}.${CHOLMOD_VERSION_MINOR}.${CHOLMOD_VERSION_SUB}"
        LANGUAGES C CXX CUDA )
    set ( CMAKE_CUDA_FLAGS "-cudart=static -lineinfo -DSUITESPARSE_CUDA" )
    set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSUITESPARSE_CUDA" )
    message ( STATUS "C++ flags for CUDA:  ${CMAKE_CXX_FLAGS}" )
    message ( STATUS "nvcc flags for CUDA: ${CMAKE_CUDA_FLAGS}" )
    file ( GLOB CHOLMOD_CUDA_SOURCES "cholmod_gpu.c" "cholmod_l_gpu.c"
        "cholmod_gpu_kernels.cu" )
else ( )
    project ( cholmod_cuda
        VERSION "${CHOLMOD_VERSION_MAJOR}.${CHOLMOD_VERSION_MINOR}.${CHOLMOD_VERSION_SUB}"
        LANGUAGES C CXX )
    file ( GLOB CHOLMOD_CUDA_SOURCES "cholmod_gpu.c" "cholmod_l_gpu.c" )
endif ( )

set ( CMAKE_CXX_STANDARD 17 )
set ( CMAKE_C_STANDARD 11 )

add_library ( cholmod_cuda SHARED ${CHOLMOD_CUDA_SOURCES} )

set_target_properties ( cholmod_cuda PROPERTIES
    VERSION ${CHOLMOD_VERSION_MAJOR}.${CHOLMOD_VERSION_MINOR}.${CHOLMOD_VERSION_SUB}
    CXX_STANDARD_REQUIRED 17
    SOVERSION ${CHOLMOD_VERSION_MAJOR} )

if ( NOT NSTATIC )
    add_library ( cholmod_cuda_static STATIC ${CHOLMOD_CUDA_SOURCES} )

    set_target_properties ( cholmod_cuda_static PROPERTIES
        VERSION ${CHOLMOD_VERSION_MAJOR}.${CHOLMOD_VERSION_MINOR}.${CHOLMOD_VERSION_SUB}
        CXX_STANDARD_REQUIRED 17
        OUTPUT_NAME cholmod_cuda
        SOVERSION ${CHOLMOD_VERSION_MAJOR} )

    if ( MSVC )
        set_target_properties ( cholmod_cuda_static PROPERTIES
            OUTPUT_NAME cholmod_cuda_static )
    endif ( )
endif ( )

set ( CHOLMOD_CUDA_INCLUDES
        ../Check
        ../../SuiteSparse_config
        ../Cholesky
        ../Core
        ../Supernodal
        ../Include
        ../GPU )

include_directories ( ${CHOLMOD_CUDA_INCLUDES} ${CUDAToolkit_INCLUDE_DIRS} )

target_include_directories ( cholmod_cuda PUBLIC
        ${CUDAToolkit_INCLUDE_DIRS}
        ${CHOLMOD_CUDA_INCLUDES} )
set_target_properties ( cholmod_cuda PROPERTIES POSITION_INDEPENDENT_CODE ON )
set_target_properties ( cholmod_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON )

if ( NOT NSTATIC )
    target_include_directories ( cholmod_cuda_static PUBLIC
            ${CUDAToolkit_INCLUDE_DIRS}
            ${CHOLMOD_CUDA_INCLUDES} )
    set_target_properties ( cholmod_cuda_static PROPERTIES CUDA_SEPARABLE_COMPILATION on )
    set_target_properties ( cholmod_cuda_static PROPERTIES POSITION_INDEPENDENT_CODE on )
endif ( )

if ( SUITESPARSE_CUDA )
    # this must be a PUBLIC link, not PRIVATE, so the CHOLMOD library
    # itself can link against these libraries
    target_link_libraries ( cholmod_cuda PUBLIC CUDA::nvrtc CUDA::cudart_static
        CUDA::nvToolsExt CUDA::cublas )
    if ( NOT NSTATIC )
        target_link_libraries ( cholmod_cuda_static PUBLIC CUDA::nvrtc CUDA::cudart_static
            CUDA::nvToolsExt CUDA::cublas )
    endif ( )
endif ( )

#-------------------------------------------------------------------------------
# installation location
#-------------------------------------------------------------------------------

install ( TARGETS cholmod_cuda
    LIBRARY DESTINATION ${SUITESPARSE_LIBDIR}
    ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR}
    RUNTIME DESTINATION ${SUITESPARSE_BINDIR}
    PUBLIC_HEADER DESTINATION ${SUITESPARSE_INCLUDEDIR} )
if ( NOT NSTATIC )
    install ( TARGETS cholmod_cuda_static
        ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR} )
endif ( )
