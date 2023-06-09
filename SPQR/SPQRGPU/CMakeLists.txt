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

if ( SUITESPARSE_CUDA )
    project ( spqr_cuda 
        VERSION "${SPQR_VERSION_MAJOR}.${SPQR_VERSION_MINOR}.${SPQR_VERSION_SUB}"
        LANGUAGES C CXX CUDA )
    set ( CMAKE_CUDA_FLAGS "-cudart=static -lineinfo -DSUITESPARSE_CUDA" )
    set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSUITESPARSE_CUDA" )
    message ( STATUS "C++ flags for CUDA:  ${CMAKE_CXX_FLAGS}" )
    message ( STATUS "nvcc flags for CUDA: ${CMAKE_CUDA_FLAGS}" )
else ( )
    project ( spqr_cuda 
        VERSION "${SPQR_VERSION_MAJOR}.${SPQR_VERSION_MINOR}.${SPQR_VERSION_SUB}"
        LANGUAGES C CXX )
endif ( )

set ( CMAKE_CXX_STANDARD 17 )
set ( CMAKE_C_STANDARD 11 )

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
    OUTPUT_NAME spqr_cuda
    SOVERSION ${SPQR_VERSION_MAJOR} )

if ( MSVC )
    set_target_properties ( spqr_cuda_static PROPERTIES
        OUTPUT_NAME spqr_cuda_static )
endif ( )

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

target_link_libraries ( spqr_cuda PRIVATE ${CHOLMOD_LIBRARIES} )
target_link_libraries ( spqr_cuda_static PUBLIC ${CHOLMOD_LIBRARIES} )
if ( SUITESPARSE_CUDA )
    target_link_libraries ( spqr_cuda PRIVATE CUDA::nvrtc CUDA::cudart_static
        CUDA::nvToolsExt CUDA::cublas )
    target_link_libraries ( spqr_cuda_static PUBLIC CUDA::nvrtc CUDA::cudart_static
        CUDA::nvToolsExt CUDA::cublas )
endif ( )

#-------------------------------------------------------------------------------
# installation location
#-------------------------------------------------------------------------------

install ( TARGETS spqr_cuda
    LIBRARY DESTINATION ${SUITESPARSE_LIBDIR}
    ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR}
    RUNTIME DESTINATION ${SUITESPARSE_BINDIR} )
install ( TARGETS spqr_cuda_static
    ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR} )
