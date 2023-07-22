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

file ( GLOB SPQR_CUDA_SOURCES "spqrgpu_*.cpp" )

add_library ( spqr_cuda SHARED ${SPQR_CUDA_SOURCES} )

set_target_properties ( spqr_cuda PROPERTIES
    VERSION ${SPQR_VERSION_MAJOR}.${SPQR_VERSION_MINOR}.${SPQR_VERSION_SUB}
    CXX_STANDARD_REQUIRED ON
    CXX_STANDARD 17
    C_STANDARD_REQUIRED ON
    C_STANDARD 11
    SOVERSION ${SPQR_VERSION_MAJOR} )

if ( NOT NSTATIC )
    add_library ( spqr_cuda_static STATIC ${SPQR_CUDA_SOURCES} )

    set_target_properties ( spqr_cuda_static PROPERTIES
        VERSION ${SPQR_VERSION_MAJOR}.${SPQR_VERSION_MINOR}.${SPQR_VERSION_SUB}
        CXX_STANDARD_REQUIRED ON
        CXX_STANDARD 17
        C_STANDARD_REQUIRED ON
        C_STANDARD 11
        OUTPUT_NAME spqr_cuda
        SOVERSION ${SPQR_VERSION_MAJOR} )

    if ( MSVC )
        set_target_properties ( spqr_cuda_static PROPERTIES
            OUTPUT_NAME spqr_cuda_static )
    endif ( )
endif ( )

set ( SPQR_CUDA_INCLUDES ../Include )

include_directories ( ${SPQR_CUDA_INCLUDES} ${CUDAToolkit_INCLUDE_DIRS} )

if ( TARGET SuiteSparse::GPUQREngine )
    target_include_directories ( spqr_cuda PRIVATE
        $<TARGET_PROPERTY:SuiteSparse::GPUQREngine,INTERFACE_INCLUDE_DIRECTORIES> )
    target_include_directories ( spqr_cuda_static PRIVATE
        $<TARGET_PROPERTY:SuiteSparse::GPUQREngine,INTERFACE_INCLUDE_DIRECTORIES> )
endif ( )

if ( TARGET suitesparse_gpuruntime )
    target_include_directories ( spqr_cuda PRIVATE
        $<TARGET_PROPERTY:suitesparse_gpuruntime,INTERFACE_INCLUDE_DIRECTORIES> )
    target_include_directories ( spqr_cuda_static PRIVATE
        $<TARGET_PROPERTY:suitesparse_gpuruntime,INTERFACE_INCLUDE_DIRECTORIES> )
endif ( )

target_include_directories ( spqr_cuda PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
        ${SPQR_CUDA_INCLUDES}
        $<TARGET_PROPERTY:SuiteSparse::SuiteSparseConfig,INTERFACE_INCLUDE_DIRECTORIES> )
set_target_properties ( spqr_cuda PROPERTIES POSITION_INDEPENDENT_CODE ON )
set_target_properties ( spqr_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON )

if ( NOT NSTATIC )
target_include_directories ( spqr_cuda_static PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
        ${SPQR_CUDA_INCLUDES}
        $<TARGET_PROPERTY:SuiteSparse::SuiteSparseConfig,INTERFACE_INCLUDE_DIRECTORIES> )
set_target_properties ( spqr_cuda_static PROPERTIES CUDA_SEPARABLE_COMPILATION on )
set_target_properties ( spqr_cuda_static PROPERTIES POSITION_INDEPENDENT_CODE on )

if ( TARGET SuiteSparse::CHOLMOD_static )
    target_link_libraries ( spqr_cuda_static PUBLIC SuiteSparse::CHOLMOD_static )
else ( )
    target_link_libraries ( spqr_cuda_static PUBLIC SuiteSparse::CHOLMOD )
endif ( )
endif ( )

target_link_libraries ( spqr_cuda PRIVATE SuiteSparse::CHOLMOD )

if ( SUITESPARSE_CUDA )
    target_link_libraries ( spqr_cuda PRIVATE CUDA::nvrtc CUDA::cudart_static
        CUDA::nvToolsExt CUDA::cublas )
    if ( NOT NSTATIC )
        target_link_libraries ( spqr_cuda_static PUBLIC CUDA::nvrtc CUDA::cudart_static
            CUDA::nvToolsExt CUDA::cublas )
    endif ( )
endif ( )

#-------------------------------------------------------------------------------
# installation location
#-------------------------------------------------------------------------------

include ( CMakePackageConfigHelpers )

install ( TARGETS spqr_cuda
    EXPORT SPQR_CUDATargets
    LIBRARY DESTINATION ${SUITESPARSE_LIBDIR}
    ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR}
    RUNTIME DESTINATION ${SUITESPARSE_BINDIR} )

if ( NOT NSTATIC )
install ( TARGETS spqr_cuda_static
    EXPORT SPQR_CUDATargets
    ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR} )
endif ( )

# create (temporary) export target file during build
export ( EXPORT SPQR_CUDATargets
    FILE ${CMAKE_CURRENT_BINARY_DIR}/../SPQR_CUDATargets.cmake )

# install export target, config and version files for find_package
install ( EXPORT SPQR_CUDATargets
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SPQR_CUDA )

configure_package_config_file (
    ../Config/SPQR_CUDAConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/../SPQR_CUDAConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SPQR_CUDA )

write_basic_package_version_file (
    ${CMAKE_CURRENT_BINARY_DIR}/../SPQR_CUDAConfigVersion.cmake
    COMPATIBILITY SameMajorVersion )

install ( FILES
    ${CMAKE_CURRENT_BINARY_DIR}/../SPQR_CUDAConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/../SPQR_CUDAConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SPQR_CUDA )

