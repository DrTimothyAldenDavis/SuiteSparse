#-------------------------------------------------------------------------------
# SPQR/SPQRGPU/CMakeLists.txt:  cmake for SPQR/SPQRGPU module
#-------------------------------------------------------------------------------

# SPQR/SPQRGPU Module.  Copyright (C) 2005-2022, Timothy A. Davis.
# All Rights Reserved.
# SPDX-License-Identifier: GPL-2.0+

#-------------------------------------------------------------------------------
# get the version
#-------------------------------------------------------------------------------

# cmake 3.22 is required to find the BLAS/LAPACK
cmake_minimum_required ( VERSION 3.22 )

message ( STATUS "Building SPQR_CUDA version: v"
    ${SPQR_VERSION_MAJOR}.
    ${SPQR_VERSION_MINOR}.
    ${SPQR_VERSION_SUB} " (" ${SPQR_DATE} ")" )

include ( SuiteSparsePolicy )

enable_language ( CXX )

message ( STATUS "C++ flags for CUDA:  ${CMAKE_CXX_FLAGS}" )

file ( GLOB SPQR_CUDA_SOURCES "spqrgpu_*.cpp" )

if ( BUILD_SHARED_LIBS )
    target_sources ( SPQR PRIVATE ${SPQR_CUDA_SOURCES} )
endif ( )

if ( BUILD_STATIC_LIBS )
    target_sources ( SPQR_static PRIVATE ${SPQR_CUDA_SOURCES} )
endif ( )

if ( BUILD_SHARED_LIBS )
    set_target_properties ( SPQR PROPERTIES POSITION_INDEPENDENT_CODE ON )
    set_target_properties ( SPQR PROPERTIES CUDA_SEPARABLE_COMPILATION ON )
    target_compile_definitions ( SPQR PRIVATE "SPQR_HAS_CUDA" )

    target_link_libraries ( SPQR PRIVATE
        CUDA::nvrtc CUDA::cudart_static CUDA::cublas )
endif ( )

if ( BUILD_STATIC_LIBS )
    set_target_properties ( SPQR_static PROPERTIES CUDA_SEPARABLE_COMPILATION ON )
    set_target_properties ( SPQR_static PROPERTIES POSITION_INDEPENDENT_CODE ON )
    target_compile_definitions ( SPQR_static PRIVATE "SPQR_HAS_CUDA" )

    target_link_libraries ( SPQR_static PRIVATE
        CUDA::nvrtc CUDA::cudart_static CUDA::cublas )
endif ( )
