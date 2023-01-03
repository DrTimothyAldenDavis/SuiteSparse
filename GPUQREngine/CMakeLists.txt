#-------------------------------------------------------------------------------
# SuiteSparse/GPUQREngine/CMakeLists.txt
#-------------------------------------------------------------------------------

# GPUQREngine, Copyright (c) 2013-2023, Timothy A Davis, Sencer Nuri Yeralan,
# and Sanjay Ranka.  All Rights Reserved.
# SPDX-License-Identifier: GPL-2.0+

#-------------------------------------------------------------------------------
# get the version
#-------------------------------------------------------------------------------

# cmake 3.22 is required to find the BLAS/LAPACK
cmake_minimum_required ( VERSION 3.19 )

set ( GPUQRENGINE_DATE "Jan 1, 2023" )
set ( GPUQRENGINE_VERSION_MAJOR 2 )
set ( GPUQRENGINE_VERSION_MINOR 0 )
set ( GPUQRENGINE_VERSION_SUB   3 )

message ( STATUS "Building GPUQRENGINE version: v"
    ${GPUQRENGINE_VERSION_MAJOR}.
    ${GPUQRENGINE_VERSION_MINOR}.
    ${GPUQRENGINE_VERSION_SUB} " (" ${GPUQRENGINE_DATE} ")" )

#-------------------------------------------------------------------------------
# SuiteSparse policies
#-------------------------------------------------------------------------------

set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    ${CMAKE_SOURCE_DIR}/cmake_modules
    ${CMAKE_SOURCE_DIR}/../COLAMD/cmake_modules
    ${CMAKE_SOURCE_DIR}/../AMD/cmake_modules
    ${CMAKE_SOURCE_DIR}/../CAMD/cmake_modules
    ${CMAKE_SOURCE_DIR}/../CCOLAMD/cmake_modules
    ${CMAKE_SOURCE_DIR}/../CHOLMOD/cmake_modules
    ${CMAKE_SOURCE_DIR}/../SuiteSparse_GPURuntime/cmake_modules
    ${CMAKE_SOURCE_DIR}/../SuiteSparse_config/cmake_modules )

option ( ENABLE_CUDA "Enable CUDA acceleration" on )

include ( SuiteSparsePolicy )

#-------------------------------------------------------------------------------
# define the project
#-------------------------------------------------------------------------------

if ( SUITESPARSE_CUDA )
    project ( gpuqrengine 
        VERSION "${GPUQRENGINE_VERSION_MAJOR}.${GPUQRENGINE_VERSION_MINOR}.${GPUQRENGINE_VERSION_SUB}"
        LANGUAGES C CXX CUDA )
else ( )
    # CUDA not available
    project ( gpuqrengine 
        VERSION "${GPUQRENGINE_VERSION_MAJOR}.${GPUQRENGINE_VERSION_MINOR}.${GPUQRENGINE_VERSION_SUB}"
        LANGUAGES C CXX )
    message ( STATUS "CUDA not enabled:  library will be empty" )
endif ( )

#-------------------------------------------------------------------------------
# find library dependencies
#-------------------------------------------------------------------------------

# for the library itself
find_package ( SuiteSparse_config 7.0.0 REQUIRED )

if ( SUITESPARSE_CUDA )
    find_package ( SuiteSparse_GPURuntime 2.0.3 REQUIRED )
endif ( )

# Demo disabled for GPUQREngine v2.x
set ( DEMO_OK false )

if ( DEMO AND DEMO_OK )
    # for the demo only:
    find_package ( AMD 3.0.3 )
    find_package ( COLAMD 3.0.3 )
    find_package ( CAMD 3.0.3 )
    find_package ( CCOLAMD 3.0.3 )
    find_package ( CHOLMOD 4.0.3 )
    find_package ( CHOLMOD_CUDA 4.0.3 )
endif ( )

if ( SUITESPARSE_CUDA AND CHOLMOD_FOUND AND AMD_FOUND AND COLAMD_FOUND AND 
    CAMD_FOUND AND CCOLAMD_FOUND )
    set ( DEMO_OK true )
else ( )
    set ( DEMO_OK false )
endif ( )

#-------------------------------------------------------------------------------
# configure files
#-------------------------------------------------------------------------------

configure_file ( "Config/GPUQREngine.hpp.in"
    "${PROJECT_SOURCE_DIR}/Include/GPUQREngine.hpp"
    NEWLINE_STYLE LF )

#-------------------------------------------------------------------------------

if ( SUITESPARSE_CUDA )
    set ( CMAKE_CUDA_FLAGS "-cudart=static -lineinfo -DSUITESPARSE_CUDA" )
    set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSUITESPARSE_CUDA" )
    message ( STATUS "C++ flags for CUDA:  ${CMAKE_CXX_FLAGS}" )
    message ( STATUS "nvcc flags for CUDA: ${CMAKE_CUDA_FLAGS}" )
endif ( )

set ( CMAKE_CXX_STANDARD 17 )
set ( CMAKE_C_STANDARD 11 )

if ( SUITESPARSE_CUDA )
    file ( GLOB GPUQRENGINE_SOURCES "Source/*.cpp" "Source/*.cu" "Source/*/*.cpp" )
else ( )
    file ( GLOB GPUQRENGINE_SOURCES "Source/*.cpp" )
endif ( )

set ( GPUQRENGINE_INCLUDES Include Include/Kernel
    Include/Kernel/Apply Include/Kernel/Assemble Include/Kernel/Factorize
    ${CMAKE_SOURCE_DIR}/../SuiteSparse_GPURuntime/Include
    ${SUITESPARSE_CONFIG_INCLUDE_DIR} )

include_directories ( ${GPUQRENGINE_INCLUDES} ${CUDAToolkit_INCLUDE_DIRS} )

#-------------------------------------------------------------------------------
# dynamic gpuqrengine library properties
#-------------------------------------------------------------------------------

add_library ( gpuqrengine SHARED ${GPUQRENGINE_SOURCES} )

set_target_properties ( gpuqrengine PROPERTIES
    VERSION ${GPUQRENGINE_VERSION_MAJOR}.${GPUQRENGINE_VERSION_MINOR}.${GPUQRENGINE_VERSION_SUB}
    CXX_STANDARD_REQUIRED 17
    SOVERSION ${GPUQRENGINE_VERSION_MAJOR}
    PUBLIC_HEADER "Include/GPUQREngine.hpp"
    WINDOWS_EXPORT_ALL_SYMBOLS ON )

target_include_directories ( gpuqrengine PUBLIC
        ${CUDAToolkit_INCLUDE_DIRS}
        ${GPUQRENGINE_INCLUDES} )

if ( SUITESPARSE_CUDA )
    set_target_properties ( gpuqrengine PROPERTIES POSITION_INDEPENDENT_CODE ON )
    set_target_properties ( gpuqrengine PROPERTIES CUDA_SEPARABLE_COMPILATION ON )
    target_link_libraries ( gpuqrengine CUDA::nvrtc CUDA::cudart_static
        CUDA::nvToolsExt CUDA::cublas )
endif ( )

#-------------------------------------------------------------------------------
# static gpuqrengine library properties
#-------------------------------------------------------------------------------

if ( NOT NSTATIC )
    add_library ( gpuqrengine_static STATIC ${GPUQRENGINE_SOURCES} )

    set_target_properties ( gpuqrengine_static PROPERTIES
        VERSION ${GPUQRENGINE_VERSION_MAJOR}.${GPUQRENGINE_VERSION_MINOR}.${GPUQRENGINE_VERSION_SUB}
        CXX_STANDARD_REQUIRED 17
        OUTPUT_NAME gpuqrengine
        SOVERSION ${GPUQRENGINE_VERSION_MAJOR} )

    if ( MSVC )
        set_target_properties ( gpuqrengine_static PROPERTIES
            OUTPUT_NAME gpuqrengine_static )
    endif ( )

    target_include_directories ( gpuqrengine_static PUBLIC
            ${CUDAToolkit_INCLUDE_DIRS}
            ${GPUQRENGINE_INCLUDES} )

    if ( SUITESPARSE_CUDA )
        set_target_properties ( gpuqrengine_static PROPERTIES CUDA_SEPARABLE_COMPILATION on )
        set_target_properties ( gpuqrengine_static PROPERTIES POSITION_INDEPENDENT_CODE on )
        target_link_libraries ( gpuqrengine_static CUDA::nvrtc CUDA::cudart_static
            CUDA::nvToolsExt CUDA::cublas )
    endif ( )
endif ( )

#-------------------------------------------------------------------------------
# installation location
#-------------------------------------------------------------------------------

if ( GLOBAL_INSTALL )
    # install in /usr/local/lib and /usr/local/include.
    # requires "sudo make install"
    message ( STATUS "Installation will be system-wide (requires 'sudo make install')" )
    install ( TARGETS gpuqrengine
        LIBRARY       DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} )
    install ( FILES ${CMAKE_SOURCE_DIR}/cmake_modules/FindGPUQREngine.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SuiteSparse 
        COMPONENT Development )
    if ( NOT NSTATIC )
    install ( TARGETS gpuqrengine_static
        ARCHIVE       DESTINATION ${CMAKE_INSTALL_LIBDIR} )
    endif ( )
endif ( )

if ( INSIDE_SUITESPARSE )
    # also install in SuiteSparse/lib and SuiteSparse/include;
    # does not require "sudo make install", just "make install"
    message ( STATUS "Installation in ../../lib and ../../include," )
    message ( STATUS "  with 'make local ; make install'. No 'sudo' required." )
    install ( TARGETS gpuqrengine
        LIBRARY       DESTINATION ${SUITESPARSE_LIBDIR}
        PUBLIC_HEADER DESTINATION ${SUITESPARSE_INCLUDEDIR} )
    install ( FILES ${CMAKE_SOURCE_DIR}/cmake_modules/FindGPUQREngine.cmake
        DESTINATION ${SUITESPARSE_LIBDIR}/cmake/SuiteSparse 
        COMPONENT Development )
    if ( NOT NSTATIC )
    install ( TARGETS gpuqrengine_static
        ARCHIVE       DESTINATION ${SUITESPARSE_LIBDIR} )
    endif ( )
endif ( )

#-------------------------------------------------------------------------------
# Demo library and programs
#-------------------------------------------------------------------------------

if ( DEMO AND DEMO_OK )

    #---------------------------------------------------------------------------
    # demo library
    #---------------------------------------------------------------------------

    message ( STATUS "Also compiling the demos in GPUQREngine/Demo" )

    #---------------------------------------------------------------------------
    # Demo programs
    #---------------------------------------------------------------------------

    add_executable ( gpuqrengine_demo "Demo/gpuqrengine_demo.cpp" )

    # Libraries required for Demo programs
    target_link_libraries ( gpuqrengine_demo PUBLIC gpuqrengine
        ${CHOLMOD_LIBRARIES} ${AMD_LIBRARIES} ${COLAMD_LIBRARIES}
        ${CAMD_LIBRARIES} ${CCOLAMD_LIBRARIES} ${CHOLMOD_CUDA}
        ${SUITESPARSE_GPURUNTIME_LIBRARIES} ${SUITESPARSE_CONFIG_LIBRARIES} )

else ( )

    message ( STATUS "Skipping the demos in GPUQREngine/Demo" )

endif ( )

#-------------------------------------------------------------------------------
# report status
#-------------------------------------------------------------------------------

include ( SuiteSparseReport )

