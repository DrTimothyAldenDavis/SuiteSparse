#-------------------------------------------------------------------------------
# SPQR/GPURuntime/CMakeLists.txt
#-------------------------------------------------------------------------------

# SPQR/GPURuntime, Copyright (c) 2013-2023, Timothy A Davis,
# Sencer Nuri Yeralan, and Sanjay Ranka.  All Rights Reserved.
# SPDX-License-Identifier: GPL-2.0+

#-------------------------------------------------------------------------------
# get the version
#-------------------------------------------------------------------------------

cmake_minimum_required ( VERSION 3.20 )

message ( STATUS "Building SPQR/GPURUNTIME version: v"
    ${SPQR_VERSION_MAJOR}.
    ${SPQR_VERSION_MINOR}.
    ${SPQR_VERSION_SUB} " (" ${SPQR_DATE} ")" )

#-------------------------------------------------------------------------------
# define the project
#-------------------------------------------------------------------------------

project ( suitesparse_gpuruntime 
    VERSION "${SPQR_VERSION_MAJOR}.${SPQR_VERSION_MINOR}.${SPQR_VERSION_SUB}"
    LANGUAGES C CXX CUDA )

#-------------------------------------------------------------------------------
# SuiteSparse policies
#-------------------------------------------------------------------------------

set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    ${PROJECT_SOURCE_DIR}/../../SuiteSparse_config/cmake_modules )

option ( ENABLE_CUDA "Enable CUDA acceleration" ON )

include ( SuiteSparsePolicy )

#-------------------------------------------------------------------------------
# configure files
#-------------------------------------------------------------------------------

configure_file ( "Config/SuiteSparse_GPURuntime.hpp.in"
    "${PROJECT_SOURCE_DIR}/Include/SuiteSparse_GPURuntime.hpp"
    NEWLINE_STYLE LF )

#-------------------------------------------------------------------------------

file ( GLOB SUITESPARSE_GPURUNTIME_SOURCES "Source/*.cpp" )

set ( SUITESPARSE_GPURUNTIME_INCLUDES Include )

include_directories ( ${SUITESPARSE_GPURUNTIME_INCLUDES} ${CUDAToolkit_INCLUDE_DIRS} )

#-------------------------------------------------------------------------------
# dynamic suitesparse_gpuruntime library properties
#-------------------------------------------------------------------------------

if ( BUILD_SHARED_LIBS )
    add_library ( GPURuntime SHARED ${SUITESPARSE_GPURUNTIME_SOURCES} )

    set_target_properties ( GPURuntime PROPERTIES
        VERSION ${SPQR_VERSION_MAJOR}.${SPQR_VERSION_MINOR}.${SPQR_VERSION_SUB}
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        OUTPUT_NAME suitesparse_gpuruntime
        SOVERSION ${SPQR_VERSION_MAJOR}
        PUBLIC_HEADER "Include/SuiteSparse_GPURuntime.hpp"
        WINDOWS_EXPORT_ALL_SYMBOLS ON )

    target_include_directories ( GPURuntime PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
        ${SUITESPARSE_GPURUNTIME_INCLUDES}
        $<TARGET_PROPERTY:SuiteSparse::SuiteSparseConfig,INTERFACE_INCLUDE_DIRECTORIES> )

    set_target_properties ( GPURuntime PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RUNTIME_LIBRARY Static )
    target_link_libraries ( GPURuntime PRIVATE
        CUDA::nvrtc CUDA::cudart_static CUDA::nvToolsExt CUDA::cublas )
    target_compile_definitions ( GPURuntime PRIVATE "SUITESPARSE_CUDA" )

    target_include_directories ( GPURuntime 
        INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>
                  $<INSTALL_INTERFACE:${SUITESPARSE_INCLUDEDIR}> )
endif ( )

#-------------------------------------------------------------------------------
# static suitesparse_gpuruntime library properties
#-------------------------------------------------------------------------------

if ( BUILD_STATIC_LIBS )
    add_library ( GPURuntime_static STATIC ${SUITESPARSE_GPURUNTIME_SOURCES} )

    set_target_properties ( GPURuntime_static PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        OUTPUT_NAME suitesparse_gpuruntime
        PUBLIC_HEADER "Include/SuiteSparse_GPURuntime.hpp" )

    if ( MSVC )
        set_target_properties ( GPURuntime_static PROPERTIES
            OUTPUT_NAME suitesparse_gpuruntime_static )
    endif ( )

    target_include_directories ( GPURuntime_static PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
        ${SUITESPARSE_GPURUNTIME_INCLUDES}
        $<TARGET_PROPERTY:SuiteSparse::SuiteSparseConfig,INTERFACE_INCLUDE_DIRECTORIES> )

    set_target_properties ( GPURuntime_static PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RUNTIME_LIBRARY Static )
    target_link_libraries ( GPURuntime_static PUBLIC
        CUDA::nvrtc CUDA::cudart_static CUDA::nvToolsExt CUDA::cublas )
    target_compile_definitions ( GPURuntime_static PRIVATE "SUITESPARSE_CUDA" )

    target_include_directories ( GPURuntime_static 
        INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>
                  $<INSTALL_INTERFACE:${SUITESPARSE_INCLUDEDIR}> )
endif ( )

#-------------------------------------------------------------------------------
# SuiteSparse_GPURuntime installation location
#-------------------------------------------------------------------------------

include ( CMakePackageConfigHelpers )

if ( BUILD_SHARED_LIBS )
    install ( TARGETS GPURuntime
        EXPORT SuiteSparse_GPURuntimeTargets
        LIBRARY DESTINATION ${SUITESPARSE_LIBDIR}
        ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR}
        RUNTIME DESTINATION ${SUITESPARSE_BINDIR}
        PUBLIC_HEADER DESTINATION ${SUITESPARSE_INCLUDEDIR} )
endif ( )
if ( BUILD_STATIC_LIBS )
    install ( TARGETS GPURuntime_static
        EXPORT SuiteSparse_GPURuntimeTargets
        ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR}
        PUBLIC_HEADER DESTINATION ${SUITESPARSE_INCLUDEDIR} )
endif ( )

# create (temporary) export target file during build
export ( EXPORT SuiteSparse_GPURuntimeTargets
    NAMESPACE SuiteSparse::
    FILE ${CMAKE_CURRENT_BINARY_DIR}/SuiteSparse_GPURuntimeTargets.cmake )

# install export target, config and version files for find_package
install ( EXPORT SuiteSparse_GPURuntimeTargets
    NAMESPACE SuiteSparse::
    DESTINATION ${SUITESPARSE_PKGFILEDIR}/cmake/SuiteSparse_GPURuntime )

configure_package_config_file (
    Config/SuiteSparse_GPURuntimeConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/SuiteSparse_GPURuntimeConfig.cmake
    INSTALL_DESTINATION ${SUITESPARSE_PKGFILEDIR}/cmake/SuiteSparse_GPURuntime )

write_basic_package_version_file (
    ${CMAKE_CURRENT_BINARY_DIR}/SuiteSparse_GPURuntimeConfigVersion.cmake
    COMPATIBILITY SameMajorVersion )

install ( FILES
    ${CMAKE_CURRENT_BINARY_DIR}/SuiteSparse_GPURuntimeConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/SuiteSparse_GPURuntimeConfigVersion.cmake
    DESTINATION ${SUITESPARSE_PKGFILEDIR}/cmake/SuiteSparse_GPURuntime )

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
        Config/SuiteSparse_GPURuntime.pc.in
        SuiteSparse_GPURuntime.pc
        @ONLY
        NEWLINE_STYLE LF )
    install ( FILES
        ${CMAKE_CURRENT_BINARY_DIR}/SuiteSparse_GPURuntime.pc
        DESTINATION ${SUITESPARSE_PKGFILEDIR}/pkgconfig )
endif ( )
