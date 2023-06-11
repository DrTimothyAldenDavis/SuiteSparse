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

set ( SUITESPARSE_GPURUNTIME_DATE "June 12, 2023" )
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
    ${CMAKE_SOURCE_DIR}/cmake_modules
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

find_package ( SuiteSparse_config 7.1.0 REQUIRED )

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

set ( SUITESPARSE_GPURUNTIME_INCLUDES Include 
    ${SUITESPARSE_CONFIG_INCLUDE_DIR} )

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

target_include_directories ( suitesparse_gpuruntime PUBLIC
        ${CUDAToolkit_INCLUDE_DIRS}
        ${SUITESPARSE_GPURUNTIME_INCLUDES} )

if ( SUITESPARSE_CUDA )
    set_target_properties ( suitesparse_gpuruntime PROPERTIES POSITION_INDEPENDENT_CODE ON )
    set_target_properties ( suitesparse_gpuruntime PROPERTIES CUDA_SEPARABLE_COMPILATION ON )
    target_link_libraries ( suitesparse_gpuruntime PRIVATE CUDA::nvrtc CUDA::cudart_static
        CUDA::nvToolsExt CUDA::cublas )
endif ( )

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

    target_include_directories ( suitesparse_gpuruntime_static PUBLIC
            ${CUDAToolkit_INCLUDE_DIRS}
            ${SUITESPARSE_GPURUNTIME_INCLUDES} )

    if ( SUITESPARSE_CUDA )
        set_target_properties ( suitesparse_gpuruntime_static PROPERTIES CUDA_SEPARABLE_COMPILATION on )
        set_target_properties ( suitesparse_gpuruntime_static PROPERTIES POSITION_INDEPENDENT_CODE on )
        target_link_libraries ( suitesparse_gpuruntime_static PUBLIC CUDA::nvrtc CUDA::cudart_static
            CUDA::nvToolsExt CUDA::cublas )
    endif ( )
endif ( )

#-------------------------------------------------------------------------------
# SuiteSparse_GPURuntime installation location
#-------------------------------------------------------------------------------

install ( TARGETS suitesparse_gpuruntime
    LIBRARY DESTINATION ${SUITESPARSE_LIBDIR}
    ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR}
    RUNTIME DESTINATION ${SUITESPARSE_BINDIR}
    PUBLIC_HEADER DESTINATION ${SUITESPARSE_INCLUDEDIR} )
install ( FILES
    ${CMAKE_SOURCE_DIR}/cmake_modules/FindSuiteSparse_GPURuntime.cmake
    DESTINATION ${SUITESPARSE_LIBDIR}/cmake/SuiteSparse 
    COMPONENT Development )
if ( NOT NSTATIC )
    install ( TARGETS suitesparse_gpuruntime_static
        ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR} )
endif ( )

#-------------------------------------------------------------------------------
# report status
#-------------------------------------------------------------------------------

include ( SuiteSparseReport )

