#-------------------------------------------------------------------------------
# SuiteSparse/GPUQREngine/CMakeLists.txt
#-------------------------------------------------------------------------------

# GPUQREngine, Copyright (c) 2013-2022, Timothy A Davis, Sencer Nuri Yeralan,
# and Sanjay Ranka.  All Rights Reserved.
# SPDX-License-Identifier: GPL-2.0+

#-------------------------------------------------------------------------------
# get the version
#-------------------------------------------------------------------------------

# cmake 3.22 is required to find the BLAS/LAPACK
cmake_minimum_required ( VERSION 3.19 )
cmake_policy ( SET CMP0104 NEW )

set ( GPUQRENGINE_DATE "Oct 31, 2022" )  # FIXME
set ( GPUQRENGINE_VERSION_MAJOR 2 )
set ( GPUQRENGINE_VERSION_MINOR 0 )
set ( GPUQRENGINE_VERSION_SUB   0 )

message ( STATUS "Building GPUQRENGINE version: v"
    ${GPUQRENGINE_VERSION_MAJOR}.
    ${GPUQRENGINE_VERSION_MINOR}.
    ${GPUQRENGINE_VERSION_SUB} " (" ${GPUQRENGINE_DATE} ")" )

#-------------------------------------------------------------------------------
# SuiteSparse policies
#-------------------------------------------------------------------------------

set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    ${CMAKE_SOURCE_DIR}/cmake_modules
    ${CMAKE_SOURCE_DIR}/../SuiteSparse_config/cmake_modules
    ${CMAKE_SOURCE_DIR}/../SuiteSparse/SuiteSparse_config/cmake_modules
    )

if ( NOT DEFINED ENABLE_CUDA )
    # default is to use CUDA, if available
    set ( ENABLE_CUDA true )
endif ( )

include ( SuiteSparsePolicy )

# FIXME: handle the case when CUDA is not available

#-------------------------------------------------------------------------------
# define the project
#-------------------------------------------------------------------------------

project ( gpuqrengine 
    VERSION "${GPUQRENGINE_VERSION_MAJOR}.${GPUQRENGINE_VERSION_MINOR}.${GPUQRENGINE_VERSION_SUB}"
    LANGUAGES C CXX CUDA )

#-------------------------------------------------------------------------------
# find library dependencies
#-------------------------------------------------------------------------------

# for the library itself
find_package ( SuiteSparse_config 6.0.0 REQUIRED )
find_package ( SuiteSparse_GPURuntime 2.0.0 REQUIRED )

# for the demo only:
find_package ( CHOLMOD 4.0.0 )
find_package ( COLAMD 3.0.0 )
find_package ( AMD 3.0.0 )
find_package ( CAMD 3.0.0 )
find_package ( CCOLAMD 3.0.0 )
find_package ( SuiteSparse_metis 5.1.0 )

if ( SUITESPARSE_CUDA AND CHOLMOD_FOUND AND AMD_FOUND AND COLAMD_FOUND AND 
    CAMD_FOUND AND CCOLAMD_FOUND AND SUITESPARSE_METIS_FOUND )
    set ( DEMO_OK true )
else ( )
    set ( DEMO_OK false )
endif ( )

#-------------------------------------------------------------------------------
# configure files
#-------------------------------------------------------------------------------

configure_file ( "Config/GPUQREngine.hpp.in" "${PROJECT_SOURCE_DIR}/Include/GPUQREngine.hpp")

#-------------------------------------------------------------------------------

set ( CMAKE_CUDA_FLAGS "-cudart=static -lineinfo -DSUITESPARSE_CUDA" )
set ( CMAKE_CXX_STANDARD 17 )
# set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++17 -fPIC -DSUITESPARSE_CUDA" )
set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSUITESPARSE_CUDA" )
set ( CMAKE_C_STANDARD 11 )

message ( STATUS "C++ flags for CUDA:  ${CMAKE_CXX_FLAGS}" )
message ( STATUS "nvcc flags for CUDA: ${CMAKE_CUDA_FLAGS}" )

file ( GLOB GPUQRENGINE_SOURCES "Source/*.cpp" "Source/*.cu" "Source/*/*.cpp" )
file ( GLOB GPUQRENGINE_HEADERS "Include/GPUQREngine.hpp" )
# message ( STATUS "Sources: ${GPUQRENGINE_SOURCES}" )

set ( GPUQRENGINE_INCLUDES Include Include/Kernel
    Include/Kernel/Apply Include/Kernel/Assemble Include/Kernel/Factorize
    ${SUITESPARSE_GPURUNTIME_INCLUDE_DIR} 
    ${SUITESPARSE_CONFIG_INCLUDE_DIR} )

include_directories ( ${GPUQRENGINE_INCLUDES} ${CUDAToolkit_INCLUDE_DIRS} )

# message ( STATUS "GPUQRENGINE CUDA includes:
#     ${GPUQRENGINE_INCLUDES}" )

# message ( STATUS "CUDA Toolkit includes: ${CUDAToolkit_INCLUDE_DIRS}" )

#-------------------------------------------------------------------------------
# dynamic gpuqrengine library properties
#-------------------------------------------------------------------------------

add_library ( gpuqrengine SHARED ${GPUQRENGINE_SOURCES} )

set_target_properties ( gpuqrengine PROPERTIES
    VERSION ${GPUQRENGINE_VERSION_MAJOR}.${GPUQRENGINE_VERSION_MINOR}.${GPUQRENGINE_VERSION_SUB}
    CXX_STANDARD_REQUIRED 17
    SOVERSION ${GPUQRENGINE_VERSION_MAJOR}
    PUBLIC_HEADER "${GPUQRENGINE_HEADERS}" )

target_include_directories ( gpuqrengine PUBLIC
        ${CUDAToolkit_INCLUDE_DIRS}
        ${GPUQRENGINE_INCLUDES} )
set_target_properties ( gpuqrengine PROPERTIES POSITION_INDEPENDENT_CODE ON )
set_target_properties ( gpuqrengine PROPERTIES CUDA_SEPARABLE_COMPILATION ON )
set_target_properties ( gpuqrengine PROPERTIES CUDA_ARCHITECTURES "75" )

target_link_libraries ( gpuqrengine CUDA::nvrtc CUDA::cudart_static
    CUDA::nvToolsExt CUDA::cublas )

#-------------------------------------------------------------------------------
# static gpuqrengine library properties
#-------------------------------------------------------------------------------

add_library ( gpuqrengine_static STATIC ${GPUQRENGINE_SOURCES} )

set_target_properties ( gpuqrengine_static PROPERTIES
    VERSION ${GPUQRENGINE_VERSION_MAJOR}.${GPUQRENGINE_VERSION_MINOR}.${GPUQRENGINE_VERSION_SUB}
    CXX_STANDARD_REQUIRED 17
    SOVERSION ${GPUQRENGINE_VERSION_MAJOR} )
#   PUBLIC_HEADER "${GPUQRENGINE_HEADERS}" )

target_include_directories ( gpuqrengine_static PUBLIC
        ${CUDAToolkit_INCLUDE_DIRS}
        ${GPUQRENGINE_INCLUDES} )
set_target_properties ( gpuqrengine_static PROPERTIES CUDA_SEPARABLE_COMPILATION on )
set_target_properties ( gpuqrengine_static PROPERTIES POSITION_INDEPENDENT_CODE on )
set_target_properties ( gpuqrengine_static PROPERTIES CUDA_ARCHITECTURES "75" )

target_link_libraries ( gpuqrengine_static CUDA::nvrtc CUDA::cudart_static
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
    install ( TARGETS gpuqrengine
        LIBRARY       DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} )
    install ( TARGETS gpuqrengine_static
        ARCHIVE       DESTINATION ${CMAKE_INSTALL_LIBDIR} )
endif ( )

if ( INSIDE_SUITESPARSE )
    # also install in SuiteSparse/lib and SuiteSparse/include;
    # does not require "sudo make install", just "make install"
    message ( STATUS "Installation in ../../lib and ../../include," )
    message ( STATUS "  with 'make local ; make install'. No 'sudo' required." )
    install ( TARGETS gpuqrengine
        LIBRARY       DESTINATION ${SUITESPARSE_LIBDIR}
        PUBLIC_HEADER DESTINATION ${SUITESPARSE_INCLUDEDIR} )
    install ( TARGETS gpuqrengine_static
        ARCHIVE       DESTINATION ${SUITESPARSE_LIBDIR} )
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
        ${CAMD_LIBRARIES} ${CCOLAMD_LIBRARIES} ${SUITESPARSE_METIS_LIBRARIES}
        ${SUITESPARSE_GPURUNTIME_LIBRARIES} ${SUITESPARSE_CONFIG_LIBRARIES} )

else ( )

    message ( STATUS "Skipping the demos in GPUQREngine/Demo" )

endif ( )

#-------------------------------------------------------------------------------
# report status
#-------------------------------------------------------------------------------

include ( SuiteSparseReport )



