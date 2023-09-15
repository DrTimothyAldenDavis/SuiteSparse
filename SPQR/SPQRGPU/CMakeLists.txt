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
cmake_minimum_required ( VERSION 3.20 )

message ( STATUS "Building SPQR_CUDA version: v"
    ${SPQR_VERSION_MAJOR}.
    ${SPQR_VERSION_MINOR}.
    ${SPQR_VERSION_SUB} " (" ${SPQR_DATE} ")" )

include ( SuiteSparsePolicy )

project ( spqr_cuda 
    VERSION "${SPQR_VERSION_MAJOR}.${SPQR_VERSION_MINOR}.${SPQR_VERSION_SUB}"
    LANGUAGES CXX )
message ( STATUS "C++ flags for CUDA:  ${CMAKE_CXX_FLAGS}" )

file ( GLOB SPQR_CUDA_SOURCES "spqrgpu_*.cpp" )

add_library ( SPQR_CUDA SHARED ${SPQR_CUDA_SOURCES} )

set_target_properties ( SPQR_CUDA PROPERTIES
    VERSION ${SPQR_VERSION_MAJOR}.${SPQR_VERSION_MINOR}.${SPQR_VERSION_SUB}
    CXX_STANDARD_REQUIRED ON
    CXX_STANDARD 17
    C_STANDARD_REQUIRED ON
    C_STANDARD 11
    OUTPUT_NAME spqr_cuda
    SOVERSION ${SPQR_VERSION_MAJOR} )

if ( NOT NSTATIC )
    add_library ( SPQR_CUDA_static STATIC ${SPQR_CUDA_SOURCES} )

    set_target_properties ( SPQR_CUDA_static PROPERTIES
        VERSION ${SPQR_VERSION_MAJOR}.${SPQR_VERSION_MINOR}.${SPQR_VERSION_SUB}
        CXX_STANDARD_REQUIRED ON
        CXX_STANDARD 17
        C_STANDARD_REQUIRED ON
        C_STANDARD 11
        OUTPUT_NAME spqr_cuda
        SOVERSION ${SPQR_VERSION_MAJOR} )

    if ( MSVC )
        set_target_properties ( SPQR_CUDA_static PROPERTIES
            OUTPUT_NAME SPQR_CUDA_static )
    endif ( )
endif ( )

set ( SPQR_CUDA_INCLUDES ../Include )

if ( TARGET SuiteSparse::GPUQREngine )
    target_include_directories ( SPQR_CUDA PRIVATE
        "$<TARGET_PROPERTY:SuiteSparse::GPUQREngine,INTERFACE_INCLUDE_DIRECTORIES>" )
    target_include_directories ( SPQR_CUDA_static PRIVATE
        "$<TARGET_PROPERTY:SuiteSparse::GPUQREngine,INTERFACE_INCLUDE_DIRECTORIES>" )
endif ( )

if ( TARGET SuiteSparse::GPURuntime )
    target_include_directories ( SPQR_CUDA PRIVATE
        "$<TARGET_PROPERTY:SuiteSparse::GPURuntime,INTERFACE_INCLUDE_DIRECTORIES>" )
    target_include_directories ( SPQR_CUDA_static PRIVATE
        "$<TARGET_PROPERTY:SuiteSparse::GPURuntime,INTERFACE_INCLUDE_DIRECTORIES>" )
endif ( )

target_include_directories ( SPQR_CUDA PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
        ${SPQR_CUDA_INCLUDES}
        "$<TARGET_PROPERTY:SuiteSparse::SuiteSparseConfig,INTERFACE_INCLUDE_DIRECTORIES>" )
set_target_properties ( SPQR_CUDA PROPERTIES POSITION_INDEPENDENT_CODE ON )
set_target_properties ( SPQR_CUDA PROPERTIES CUDA_SEPARABLE_COMPILATION ON )
target_compile_definitions ( SPQR_CUDA PRIVATE "SUITESPARSE_CUDA" )

if ( NOT NSTATIC )
    target_include_directories ( SPQR_CUDA_static PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
        ${SPQR_CUDA_INCLUDES}
        "$<TARGET_PROPERTY:SuiteSparse::SuiteSparseConfig,INTERFACE_INCLUDE_DIRECTORIES>" )
    set_target_properties ( SPQR_CUDA_static PROPERTIES CUDA_SEPARABLE_COMPILATION ON )
    set_target_properties ( SPQR_CUDA_static PROPERTIES POSITION_INDEPENDENT_CODE ON )
    target_compile_definitions ( SPQR_CUDA_static PRIVATE "SUITESPARSE_CUDA" )


    if ( TARGET SuiteSparse::CHOLMOD_static )
        target_link_libraries ( SPQR_CUDA_static PUBLIC SuiteSparse::CHOLMOD_static )
    else ( )
        target_link_libraries ( SPQR_CUDA_static PUBLIC SuiteSparse::CHOLMOD )
    endif ( )
endif ( )

target_link_libraries ( SPQR_CUDA PRIVATE SuiteSparse::CHOLMOD )

target_link_libraries ( SPQR_CUDA PRIVATE CUDA::nvrtc CUDA::cudart_static
    CUDA::nvToolsExt CUDA::cublas )
if ( NOT NSTATIC )
    target_link_libraries ( SPQR_CUDA_static PUBLIC CUDA::nvrtc CUDA::cudart_static
        CUDA::nvToolsExt CUDA::cublas )
endif ( )

#-------------------------------------------------------------------------------
# installation location
#-------------------------------------------------------------------------------

include ( CMakePackageConfigHelpers )

install ( TARGETS SPQR_CUDA
    EXPORT SPQR_CUDATargets
    LIBRARY DESTINATION ${SUITESPARSE_LIBDIR}
    ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR}
    RUNTIME DESTINATION ${SUITESPARSE_BINDIR} )

if ( NOT NSTATIC )
install ( TARGETS SPQR_CUDA_static
    EXPORT SPQR_CUDATargets
    ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR} )
endif ( )

# create (temporary) export target file during build
export ( EXPORT SPQR_CUDATargets
    NAMESPACE SuiteSparse::
    FILE ${CMAKE_CURRENT_BINARY_DIR}/../SPQR_CUDATargets.cmake )

# install export target, config and version files for find_package
install ( EXPORT SPQR_CUDATargets
    NAMESPACE SuiteSparse::
    DESTINATION ${SUITESPARSE_LIBDIR}/cmake/SPQR_CUDA )

configure_package_config_file (
    ../Config/SPQR_CUDAConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/../SPQR_CUDAConfig.cmake
    INSTALL_DESTINATION ${SUITESPARSE_LIBDIR}/cmake/SPQR_CUDA )

write_basic_package_version_file (
    ${CMAKE_CURRENT_BINARY_DIR}/../SPQR_CUDAConfigVersion.cmake
    COMPATIBILITY SameMajorVersion )

install ( FILES
    ${CMAKE_CURRENT_BINARY_DIR}/../SPQR_CUDAConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/../SPQR_CUDAConfigVersion.cmake
    DESTINATION ${SUITESPARSE_LIBDIR}/cmake/SPQR_CUDA )

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
        ../Config/SPQR_CUDA.pc.in
        SPQR_CUDA.pc
        @ONLY
        NEWLINE_STYLE LF )
    install ( FILES
        ${CMAKE_CURRENT_BINARY_DIR}/SPQR_CUDA.pc
        DESTINATION ${SUITESPARSE_LIBDIR}/pkgconfig )
endif ( )
