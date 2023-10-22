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
cmake_minimum_required ( VERSION 3.20 )

message ( STATUS "Building CHOLMOD_CUDA version: v"
    ${CHOLMOD_VERSION_MAJOR}.
    ${CHOLMOD_VERSION_MINOR}.
    ${CHOLMOD_VERSION_SUB} " (" ${CHOLMOD_DATE} ")" )

include ( SuiteSparsePolicy )

project ( cholmod_cuda
    VERSION "${CHOLMOD_VERSION_MAJOR}.${CHOLMOD_VERSION_MINOR}.${CHOLMOD_VERSION_SUB}"
    LANGUAGES C CXX CUDA )
set ( CMAKE_CUDA_FLAGS "-cudart=static -lineinfo -DSUITESPARSE_CUDA" )
set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSUITESPARSE_CUDA" )
message ( STATUS "C++ flags for CUDA:  ${CMAKE_CXX_FLAGS}" )
message ( STATUS "nvcc flags for CUDA: ${CMAKE_CUDA_FLAGS}" )
file ( GLOB CHOLMOD_CUDA_SOURCES "cholmod_gpu.c" "cholmod_l_gpu.c"
    "cholmod_gpu_kernels.cu" )

add_library ( CHOLMOD_CUDA SHARED ${CHOLMOD_CUDA_SOURCES} )

set_target_properties ( CHOLMOD_CUDA PROPERTIES
    VERSION ${CHOLMOD_VERSION_MAJOR}.${CHOLMOD_VERSION_MINOR}.${CHOLMOD_VERSION_SUB}
    CXX_STANDARD_REQUIRED ON
    CXX_STANDARD 17
    C_STANDARD_REQUIRED ON
    C_STANDARD 11
    OUTPUT_NAME cholmod_cuda
    SOVERSION ${CHOLMOD_VERSION_MAJOR} )

if ( NOT NSTATIC )
    add_library ( CHOLMOD_CUDA_static STATIC ${CHOLMOD_CUDA_SOURCES} )

    set_target_properties ( CHOLMOD_CUDA_static PROPERTIES
        VERSION ${CHOLMOD_VERSION_MAJOR}.${CHOLMOD_VERSION_MINOR}.${CHOLMOD_VERSION_SUB}
        CXX_STANDARD_REQUIRED ON
        CXX_STANDARD 17
        C_STANDARD_REQUIRED ON
        C_STANDARD 11
        OUTPUT_NAME cholmod_cuda
        SOVERSION ${CHOLMOD_VERSION_MAJOR} )

    if ( MSVC )
        set_target_properties ( CHOLMOD_CUDA_static PROPERTIES
            OUTPUT_NAME cholmod_cuda_static )
    endif ( )
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

target_include_directories ( CHOLMOD_CUDA PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
        ${CHOLMOD_CUDA_INCLUDES} )
set_target_properties ( CHOLMOD_CUDA PROPERTIES POSITION_INDEPENDENT_CODE ON )
set_target_properties ( CHOLMOD_CUDA PROPERTIES CUDA_SEPARABLE_COMPILATION ON )
target_compile_definitions ( CHOLMOD_CUDA PUBLIC "SUITESPARSE_CUDA" )

if ( NOT NSTATIC )
    target_include_directories ( CHOLMOD_CUDA_static PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
        ${CHOLMOD_CUDA_INCLUDES} )
    set_target_properties ( CHOLMOD_CUDA_static PROPERTIES CUDA_SEPARABLE_COMPILATION on )
    set_target_properties ( CHOLMOD_CUDA_static PROPERTIES POSITION_INDEPENDENT_CODE on )
    target_compile_definitions ( CHOLMOD_CUDA_static PUBLIC "SUITESPARSE_CUDA" )
endif ( )

target_link_libraries ( CHOLMOD_CUDA PRIVATE CUDA::nvrtc CUDA::cudart_static
    CUDA::nvToolsExt CUDA::cublas )
if ( NOT NSTATIC )
    target_link_libraries ( CHOLMOD_CUDA_static PUBLIC CUDA::nvrtc CUDA::cudart_static
        CUDA::nvToolsExt CUDA::cublas )
endif ( )

#-------------------------------------------------------------------------------
# installation location
#-------------------------------------------------------------------------------

include ( CMakePackageConfigHelpers )

install ( TARGETS CHOLMOD_CUDA
    EXPORT CHOLMOD_CUDATargets
    LIBRARY DESTINATION ${SUITESPARSE_LIBDIR}
    ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR}
    RUNTIME DESTINATION ${SUITESPARSE_BINDIR}
    PUBLIC_HEADER DESTINATION ${SUITESPARSE_INCLUDEDIR} )
if ( NOT NSTATIC )
    install ( TARGETS CHOLMOD_CUDA_static
        EXPORT CHOLMOD_CUDATargets
        ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR} )
endif ( )


# create (temporary) export target file during build
export ( EXPORT CHOLMOD_CUDATargets
    NAMESPACE SuiteSparse::
    FILE ${CMAKE_CURRENT_BINARY_DIR}/../CHOLMOD_CUDATargets.cmake )

# install export target, config and version files for find_package
install ( EXPORT CHOLMOD_CUDATargets
    NAMESPACE SuiteSparse::
    DESTINATION ${SUITESPARSE_LIBDIR}/cmake/CHOLMOD_CUDA )

configure_package_config_file (
    ../Config/CHOLMOD_CUDAConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/../CHOLMOD_CUDAConfig.cmake
    INSTALL_DESTINATION ${SUITESPARSE_LIBDIR}/cmake/CHOLMOD_CUDA )

write_basic_package_version_file (
    ${CMAKE_CURRENT_BINARY_DIR}/../CHOLMOD_CUDAConfigVersion.cmake
    COMPATIBILITY SameMajorVersion )

install ( FILES
    ${CMAKE_CURRENT_BINARY_DIR}/../CHOLMOD_CUDAConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/../CHOLMOD_CUDAConfigVersion.cmake
    DESTINATION ${SUITESPARSE_LIBDIR}/cmake/CHOLMOD_CUDA )

#-------------------------------------------------------------------------------
# create pkg-config file
#-------------------------------------------------------------------------------

if ( NOT MSVC )
    set ( prefix "${CMAKE_INSTALL_PREFIX}" )
    set ( exec_prefix "\${prefix}" )
    cmake_path ( IS_ABSOLUTE SUITESPARSE_LIBDIR SUITESPARSE_LIBDIR_IS_ABSOLUTE )
    if (SUITESPARSE_LIBDIR_IS_ABSOLUTE)
        set ( libdir "${SUITESPARSE_LIBDIR}")
    else ( )
        set ( libdir "\${exec_prefix}/${SUITESPARSE_LIBDIR}")
    endif ( )
    cmake_path ( IS_ABSOLUTE SUITESPARSE_INCLUDEDIR SUITESPARSE_INCLUDEDIR_IS_ABSOLUTE )
    if (SUITESPARSE_INCLUDEDIR_IS_ABSOLUTE)
        set ( includedir "${SUITESPARSE_INCLUDEDIR}")
    else ( )
        set ( includedir "\${prefix}/${SUITESPARSE_INCLUDEDIR}")
    endif ( )
    configure_file (
        ../Config/CHOLMOD_CUDA.pc.in
        CHOLMOD_CUDA.pc
        @ONLY
        NEWLINE_STYLE LF )
    install ( FILES
        ${CMAKE_CURRENT_BINARY_DIR}/CHOLMOD_CUDA.pc
        DESTINATION ${SUITESPARSE_LIBDIR}/pkgconfig )
endif ( )
