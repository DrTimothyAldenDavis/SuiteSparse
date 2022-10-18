cmake_minimum_required ( VERSION 3.19 )

message ( STATUS "Building CHOLMOD_CUDA version: v"
    ${CHOLMOD_VERSION_MAJOR}.
    ${CHOLMOD_VERSION_MINOR}.
    ${CHOLMOD_VERSION_SUB} " (" ${CHOLMOD_DATE} ")" )

# FIXME: use CHOLMOD_VERSION_*
project ( cholmod_cuda 
    VERSION "${CHOLMOD_VERSION_MAJOR}.${CHOLMOD_VERSION_MINOR}.${CHOLMOD_VERSION_SUB}"
    LANGUAGES C CXX CUDA )

set ( CMAKE_CUDA_FLAGS "-cudart=static -lineinfo " )
set ( CMAKE_CXX_STANDARD 17 )
set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++17 -fPIC -DGPU_BLAS" )
set ( CMAKE_C_STANDARD 11 )

message ( STATUS "C++ flags for CUDA:" "${CMAKE_CXX_FLAGS}" )

file ( GLOB CHOLMOD_CUDA_SOURCES "cholmod_gpu.c" "cholmod_gpu_kernels.cu" "cholmod_l_gpu.c" )

# include: "cholmod_gpu_kernels.h"

add_library ( cholmod_cuda SHARED ${CHOLMOD_CUDA_SOURCES} )

SET_TARGET_PROPERTIES ( cholmod_cuda PROPERTIES
    VERSION ${CHOLMOD_VERSION_MAJOR}.${CHOLMOD_VERSION_MINOR}.${CHOLMOD_VERSION_SUB}
    CXX_STANDARD_REQUIRED 17
    SOVERSION ${CHOLMOD_VERSION_MAJOR} )

add_library ( cholmod_cuda_static STATIC ${CHOLMOD_CUDA_SOURCES} )

SET_TARGET_PROPERTIES ( cholmod_cuda_static PROPERTIES
    VERSION ${CHOLMOD_VERSION_MAJOR}.${CHOLMOD_VERSION_MINOR}.${CHOLMOD_VERSION_SUB}
    CXX_STANDARD_REQUIRED 17
    SOVERSION ${CHOLMOD_VERSION_MAJOR} )

set ( CHOLMOD_CUDA_INCLUDES
        ../Check
        ../../SuiteSparse_config
        ../Cholesky
        ../Core
        ../Supernodal
        ../Include
        ../GPU )

include_directories ( ${CHOLMOD_CUDA_INCLUDES} ${CUDAToolkit_INCLUDE_DIRS} )

message ( STATUS "CHOLMOD CUDA includes: ${CHOLMOD_CUDA_INCLUDES}" )
message ( STATUS "CUDA Toolkit includes: ${CUDAToolkit_INCLUDE_DIRS}" )

target_include_directories ( cholmod_cuda PUBLIC
        ${CUDAToolkit_INCLUDE_DIRS}
        ${CHOLMOD_CUDA_INCLUDES} )
set_target_properties ( cholmod_cuda PROPERTIES POSITION_INDEPENDENT_CODE ON )
set_target_properties ( cholmod_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON )
set_target_properties ( cholmod_cuda PROPERTIES CUDA_ARCHITECTURES "75" )

target_include_directories ( cholmod_cuda_static PUBLIC
        ${CUDAToolkit_INCLUDE_DIRS}
        ${CHOLMOD_CUDA_INCLUDES} )
set_target_properties ( cholmod_cuda_static PROPERTIES CUDA_SEPARABLE_COMPILATION on )
set_target_properties ( cholmod_cuda_static PROPERTIES POSITION_INDEPENDENT_CODE on )
set_target_properties ( cholmod_cuda_static PROPERTIES CUDA_ARCHITECTURES "75" )

target_link_libraries ( cholmod_cuda CUDA::nvrtc CUDA::cudart_static
    CUDA::nvToolsExt CUDA::cublas )

target_link_libraries ( cholmod_cuda_static CUDA::nvrtc CUDA::cudart_static
    CUDA::nvToolsExt CUDA::cublas )

#-------------------------------------------------------------------------------
# installation location
#-------------------------------------------------------------------------------

# use "cmake -DSUITESPARSE_LOCAL=1" to install only in ../lib and
# ../include.  

if ( NOT SUITESPARSE_LOCAL )
    # install in /usr/local/lib and /usr/local/include.
    # requires "sudo make install"
    message ( STATUS "Installation will be system-wide (requires 'sudo make install')" )
    install ( TARGETS cholmod_cuda
        LIBRARY       DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} )
    install ( TARGETS cholmod_cuda_static
       ARCHIVE       DESTINATION ${CMAKE_INSTALL_LIBDIR} )
endif ( )

if ( INSIDE_SUITESPARSE )
    # also install in SuiteSparse/lib and SuiteSparse/include;
    # does not require "sudo make install", just "make install"
    message ( STATUS "Installation in ../../lib and ../../include," )
    message ( STATUS "  with 'make local ; make install'. No 'sudo' required." )
    install ( TARGETS cholmod_cuda
        LIBRARY       DESTINATION ${SUITESPARSE_LIBDIR}
        PUBLIC_HEADER DESTINATION ${SUITESPARSE_INCLUDEDIR} )
    install ( TARGETS cholmod_cuda_static
        ARCHIVE       DESTINATION ${SUITESPARSE_LIBDIR} )
endif ( )

