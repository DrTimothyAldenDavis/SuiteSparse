#-------------------------------------------------------------------------------
# SuiteSparse/SuiteSparse_GPURuntime/CMakeLists.txt
#-------------------------------------------------------------------------------

# SuiteSparse_GPURuntime, Copyright (c) 2013-2022, Timothy A Davis,
# Sencer Nuri Yeralan, and Sanjay Ranka.  All Rights Reserved.
# SPDX-License-Identifier: GPL-2.0+

#-------------------------------------------------------------------------------
# get the version
#-------------------------------------------------------------------------------

cmake_minimum_required ( VERSION 3.19 )

set ( SUITESPARSE_GPURUNTIME_DATE "June 16, 2023" )
set ( SUITESPARSE_GPURUNTIME_VERSION_MAJOR 2 )
set ( SUITESPARSE_GPURUNTIME_VERSION_MINOR 0 )
set ( SUITESPARSE_GPURUNTIME_VERSION_SUB   4 )

message ( STATUS "Building SUITESPARSE_GPURUNTIME version: v"
    ${SUITESPARSE_GPURUNTIME_VERSION_MAJOR}.
    ${SUITESPARSE_GPURUNTIME_VERSION_MINOR}.
    ${SUITESPARSE_GPURUNTIME_VERSION_SUB} " (" ${SUITESPARSE_GPURUNTIME_DATE} ")" )

#-------------------------------------------------------------------------------
# SuiteSparse policies
#-------------------------------------------------------------------------------

set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    ${CMAKE_SOURCE_DIR}/../SuiteSparse_config/cmake_modules )

option ( ENABLE_CUDA "Enable CUDA acceleration" on )

include ( SuiteSparsePolicy )

#-------------------------------------------------------------------------------
# define the project
#-------------------------------------------------------------------------------

if ( SUITESPARSE_CUDA )
    project ( suitesparse_gpuruntime 
        VERSION "${SUITESPARSE_GPURUNTIME_VERSION_MAJOR}.${SUITESPARSE_GPURUNTIME_VERSION_MINOR}.${SUITESPARSE_GPURUNTIME_VERSION_SUB}"
        LANGUAGES C CXX CUDA )
else ( )
    # CUDA not available
    project ( suitesparse_gpuruntime 
        VERSION "${SUITESPARSE_GPURUNTIME_VERSION_MAJOR}.${SUITESPARSE_GPURUNTIME_VERSION_MINOR}.${SUITESPARSE_GPURUNTIME_VERSION_SUB}"
        LANGUAGES C CXX )
    message ( STATUS "CUDA not enabled:  library will be empty" )
endif ( )

#-------------------------------------------------------------------------------
# find library dependencies
#-------------------------------------------------------------------------------

find_package ( SuiteSparse_config 7.1.0
    PATHS ${CMAKE_SOURCE_DIR}/../SuiteSparse_config/build NO_DEFAULT_PATH )
if ( NOT TARGET suitesparseconfig )
    find_package ( SuiteSparse_config 7.1.0 REQUIRED )
endif ( )

#-------------------------------------------------------------------------------
# configure files
#-------------------------------------------------------------------------------

configure_file ( "Config/SuiteSparse_GPURuntime.hpp.in"
    "${PROJECT_SOURCE_DIR}/Include/SuiteSparse_GPURuntime.hpp"
    NEWLINE_STYLE LF )

#-------------------------------------------------------------------------------

if ( SUITESPARSE_CUDA )
    set ( CMAKE_CUDA_FLAGS "-cudart=static -lineinfo -DSUITESPARSE_CUDA" )
    set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSUITESPARSE_CUDA" )
    message ( STATUS "C++ flags for CUDA:  ${CMAKE_CXX_FLAGS}" )
    message ( STATUS "nvcc flags for CUDA: ${CMAKE_CUDA_FLAGS}" )
endif ( )

file ( GLOB SUITESPARSE_GPURUNTIME_SOURCES "Source/*.cpp" )

set ( SUITESPARSE_GPURUNTIME_INCLUDES Include )

include_directories ( ${SUITESPARSE_GPURUNTIME_INCLUDES} ${CUDAToolkit_INCLUDE_DIRS} )

#-------------------------------------------------------------------------------
# dynamic suitesparse_gpuruntime library properties
#-------------------------------------------------------------------------------

add_library ( suitesparse_gpuruntime SHARED ${SUITESPARSE_GPURUNTIME_SOURCES} )

set_target_properties ( suitesparse_gpuruntime PROPERTIES
    VERSION ${SUITESPARSE_GPURUNTIME_VERSION_MAJOR}.${SUITESPARSE_GPURUNTIME_VERSION_MINOR}.${SUITESPARSE_GPURUNTIME_VERSION_SUB}
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    SOVERSION ${SUITESPARSE_GPURUNTIME_VERSION_MAJOR}
    PUBLIC_HEADER "Include/SuiteSparse_GPURuntime.hpp"
    WINDOWS_EXPORT_ALL_SYMBOLS ON )

target_include_directories ( suitesparse_gpuruntime PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
    ${SUITESPARSE_GPURUNTIME_INCLUDES} )

target_include_directories ( suitesparse_gpuruntime PUBLIC
    $<TARGET_PROPERTY:suitesparseconfig,INTERFACE_INCLUDE_DIRECTORIES> )

if ( SUITESPARSE_CUDA )
    set_target_properties ( suitesparse_gpuruntime PROPERTIES POSITION_INDEPENDENT_CODE ON )
    set_target_properties ( suitesparse_gpuruntime PROPERTIES CUDA_SEPARABLE_COMPILATION ON )
    target_link_libraries ( suitesparse_gpuruntime PRIVATE CUDA::nvrtc CUDA::cudart_static
        CUDA::nvToolsExt CUDA::cublas )
endif ( )

target_include_directories ( suitesparse_gpuruntime 
    INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>"
              "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>" )

#-------------------------------------------------------------------------------
# static suitesparse_gpuruntime library properties
#-------------------------------------------------------------------------------

if ( NOT NSTATIC )
    add_library ( suitesparse_gpuruntime_static STATIC ${SUITESPARSE_GPURUNTIME_SOURCES} )

    set_target_properties ( suitesparse_gpuruntime_static PROPERTIES
        VERSION ${SUITESPARSE_GPURUNTIME_VERSION_MAJOR}.${SUITESPARSE_GPURUNTIME_VERSION_MINOR}.${SUITESPARSE_GPURUNTIME_VERSION_SUB}
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        OUTPUT_NAME suitesparse_gpuruntime
        SOVERSION ${SUITESPARSE_GPURUNTIME_VERSION_MAJOR} )

    if ( MSVC )
        set_target_properties ( suitesparse_gpuruntime_static PROPERTIES
            OUTPUT_NAME suitesparse_gpuruntime_static )
    endif ( )

    target_include_directories ( suitesparse_gpuruntime_static PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
        ${SUITESPARSE_GPURUNTIME_INCLUDES} )

    target_include_directories ( suitesparse_gpuruntime_static PUBLIC
        $<TARGET_PROPERTY:suitesparseconfig,INTERFACE_INCLUDE_DIRECTORIES> )

    if ( SUITESPARSE_CUDA )
        set_target_properties ( suitesparse_gpuruntime_static PROPERTIES CUDA_SEPARABLE_COMPILATION on )
        set_target_properties ( suitesparse_gpuruntime_static PROPERTIES POSITION_INDEPENDENT_CODE on )
        target_link_libraries ( suitesparse_gpuruntime_static PUBLIC CUDA::nvrtc CUDA::cudart_static
            CUDA::nvToolsExt CUDA::cublas )
    endif ( )

    target_include_directories ( suitesparse_gpuruntime_static 
        INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>"
                  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>" )

endif ( )

#-------------------------------------------------------------------------------
# SuiteSparse_GPURuntime installation location
#-------------------------------------------------------------------------------

include ( CMakePackageConfigHelpers )

install ( TARGETS suitesparse_gpuruntime
    EXPORT SuiteSparse_GPURuntimeTargets
    LIBRARY DESTINATION ${SUITESPARSE_LIBDIR}
    ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR}
    RUNTIME DESTINATION ${SUITESPARSE_BINDIR}
    PUBLIC_HEADER DESTINATION ${SUITESPARSE_INCLUDEDIR} )
if ( NOT NSTATIC )
    install ( TARGETS suitesparse_gpuruntime_static
        EXPORT SuiteSparse_GPURuntimeTargets
        ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR} )
endif ( )

# create (temporary) export target file during build
export ( EXPORT SuiteSparse_GPURuntimeTargets
    FILE ${CMAKE_CURRENT_BINARY_DIR}/SuiteSparse_GPURuntimeTargets.cmake )

# install export target, config and version files for find_package
install ( EXPORT SuiteSparse_GPURuntimeTargets
    DESTINATION ${SUITESPARSE_LIBDIR}/cmake/SuiteSparse_GPURuntime )

configure_package_config_file (
    Config/SuiteSparse_GPURuntimeConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/SuiteSparse_GPURuntimeConfig.cmake
    INSTALL_DESTINATION ${SUITESPARSE_LIBDIR}/cmake/SuiteSparse_GPURuntime )

write_basic_package_version_file (
    ${CMAKE_CURRENT_BINARY_DIR}/SuiteSparse_GPURuntimeConfigVersion.cmake
    COMPATIBILITY SameMajorVersion )

install ( FILES
    ${CMAKE_CURRENT_BINARY_DIR}/SuiteSparse_GPURuntimeConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/SuiteSparse_GPURuntimeConfigVersion.cmake
    DESTINATION ${SUITESPARSE_LIBDIR}/cmake/SuiteSparse_GPURuntime )

#-------------------------------------------------------------------------------
# report status
#-------------------------------------------------------------------------------

include ( SuiteSparseReport )

