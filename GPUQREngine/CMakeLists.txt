#-------------------------------------------------------------------------------
# SuiteSparse/GPUQREngine/CMakeLists.txt
#-------------------------------------------------------------------------------

# GPUQREngine, Copyright (c) 2013-2023, Timothy A Davis, Sencer Nuri Yeralan,
# and Sanjay Ranka.  All Rights Reserved.
# SPDX-License-Identifier: GPL-2.0+

#-------------------------------------------------------------------------------
# get the version
#-------------------------------------------------------------------------------

cmake_minimum_required ( VERSION 3.20 )

set ( GPUQRENGINE_DATE "Oct 23, 2023" )
set ( GPUQRENGINE_VERSION_MAJOR 3 )
set ( GPUQRENGINE_VERSION_MINOR 2 )
set ( GPUQRENGINE_VERSION_SUB   2 )

message ( STATUS "Building GPUQRENGINE version: v"
    ${GPUQRENGINE_VERSION_MAJOR}.
    ${GPUQRENGINE_VERSION_MINOR}.
    ${GPUQRENGINE_VERSION_SUB} " (" ${GPUQRENGINE_DATE} ")" )

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
find_package ( SuiteSparse_config 7.3.0
    PATHS ${CMAKE_SOURCE_DIR}/../SuiteSparse_config/build NO_DEFAULT_PATH )
if ( NOT TARGET SuiteSparse::SuiteSparseConfig )
    find_package ( SuiteSparse_config 7.3.0 REQUIRED )
endif ( )

if ( SUITESPARSE_CUDA )
    find_package ( SuiteSparse_GPURuntime 3.2.1
        PATHS ${CMAKE_SOURCE_DIR}/../SuiteSparse_GPURuntime/build NO_DEFAULT_PATH )
    if ( NOT TARGET SuiteSparse::GPURuntime )
        find_package ( SuiteSparse_GPURuntime 3.2.1 REQUIRED )
    endif ( )
endif ( )

# Demo disabled for GPUQREngine v2.x
set ( DEMO_OK false )

if ( DEMO AND DEMO_OK )
    # for the demo only:
    find_package ( AMD 3.2.1
        PATHS ${CMAKE_SOURCE_DIR}/../AMD/build NO_DEFAULT_PATH )
    if ( NOT TARGET SuiteSparse::AMD )
        find_package ( AMD 3.2.1 )
    endif ( )

    find_package ( COLAMD 3.2.1
        PATHS ${CMAKE_SOURCE_DIR}/../COLAMD/build NO_DEFAULT_PATH )
    if ( NOT TARGET SuiteSparse::COLAMD )
        find_package ( COLAMD 3.2.1 )
    endif ( )

    find_package ( CAMD 3.2.1
        PATHS ${CMAKE_SOURCE_DIR}/../CAMD/build NO_DEFAULT_PATH )
    if ( NOT TARGET SuiteSparse::CAMD )
        find_package ( CAMD 3.2.1 )
    endif ( )

    find_package ( CCOLAMD 3.2.1
        PATHS ${CMAKE_SOURCE_DIR}/../CCOLAMD/build NO_DEFAULT_PATH )
    if ( NOT TARGET SuiteSparse::CCOLAMD )
        find_package ( CCOLAMD 3.2.1 )
    endif ( )

    find_package ( CHOLMOD 5.0.0
        PATHS ${CMAKE_SOURCE_DIR}/../CHOLMOD/build NO_DEFAULT_PATH )
    if ( NOT TARGET SuiteSparse::CHOLMOD )
        find_package ( CHOLMOD 5.0.0 )
    endif ( )

    find_package ( CHOLMOD_CUDA 5.0.0
        PATHS ${CMAKE_SOURCE_DIR}/../CHOLMOD/build NO_DEFAULT_PATH )
    if ( NOT TARGET SuiteSparse::CHOLMOD_CUDA )
        find_package ( CHOLMOD_CUDA 5.0.0 )
    endif ( )
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
    Include/Kernel/Apply Include/Kernel/Assemble Include/Kernel/Factorize )

include_directories ( ${GPUQRENGINE_INCLUDES} ${CUDAToolkit_INCLUDE_DIRS} )

#-------------------------------------------------------------------------------
# dynamic gpuqrengine library properties
#-------------------------------------------------------------------------------

add_library ( GPUQREngine SHARED ${GPUQRENGINE_SOURCES} )

set_target_properties ( GPUQREngine PROPERTIES
    VERSION ${GPUQRENGINE_VERSION_MAJOR}.${GPUQRENGINE_VERSION_MINOR}.${GPUQRENGINE_VERSION_SUB}
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    OUTPUT_NAME gpuqrengine
    SOVERSION ${GPUQRENGINE_VERSION_MAJOR}
    PUBLIC_HEADER "Include/GPUQREngine.hpp"
    WINDOWS_EXPORT_ALL_SYMBOLS ON )

target_include_directories ( GPUQREngine PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
    ${GPUQRENGINE_INCLUDES}
    "$<TARGET_PROPERTY:SuiteSparse::SuiteSparseConfig,INTERFACE_INCLUDE_DIRECTORIES>" )

if ( SUITESPARSE_CUDA )
    target_include_directories ( GPUQREngine PRIVATE
        "$<TARGET_PROPERTY:SuiteSparse::GPURuntime,INTERFACE_INCLUDE_DIRECTORIES>" )

    set_target_properties ( GPUQREngine PROPERTIES POSITION_INDEPENDENT_CODE ON )
    set_target_properties ( GPUQREngine PROPERTIES CUDA_SEPARABLE_COMPILATION ON )
    target_link_libraries ( GPUQREngine PRIVATE CUDA::nvrtc CUDA::cudart_static
        CUDA::nvToolsExt CUDA::cublas )
endif ( )

target_include_directories ( GPUQREngine
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>
              $<INSTALL_INTERFACE:${SUITESPARSE_INCLUDEDIR}> )

#-------------------------------------------------------------------------------
# static gpuqrengine library properties
#-------------------------------------------------------------------------------

if ( NOT NSTATIC )
    add_library ( GPUQREngine_static STATIC ${GPUQRENGINE_SOURCES} )

    set_target_properties ( GPUQREngine_static PROPERTIES
        VERSION ${GPUQRENGINE_VERSION_MAJOR}.${GPUQRENGINE_VERSION_MINOR}.${GPUQRENGINE_VERSION_SUB}
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        OUTPUT_NAME gpuqrengine
        SOVERSION ${GPUQRENGINE_VERSION_MAJOR} )

    if ( MSVC )
        set_target_properties ( GPUQREngine_static PROPERTIES
            OUTPUT_NAME gpuqrengine_static )
    endif ( )

    target_include_directories ( GPUQREngine_static PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
        ${GPUQRENGINE_INCLUDES}
        "$<TARGET_PROPERTY:SuiteSparse::SuiteSparseConfig,INTERFACE_INCLUDE_DIRECTORIES>" )

    if ( SUITESPARSE_CUDA )
        target_include_directories ( GPUQREngine_static PRIVATE
            "$<TARGET_PROPERTY:SuiteSparse::GPURuntime,INTERFACE_INCLUDE_DIRECTORIES>" )

        set_target_properties ( GPUQREngine_static PROPERTIES CUDA_SEPARABLE_COMPILATION on )
        set_target_properties ( GPUQREngine_static PROPERTIES POSITION_INDEPENDENT_CODE on )
        target_link_libraries ( GPUQREngine_static PUBLIC CUDA::nvrtc CUDA::cudart_static
            CUDA::nvToolsExt CUDA::cublas )
    endif ( )

    target_include_directories ( GPUQREngine_static
        INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>
                  $<INSTALL_INTERFACE:${SUITESPARSE_INCLUDEDIR}> )
endif ( )

#-------------------------------------------------------------------------------
# installation location
#-------------------------------------------------------------------------------

include ( CMakePackageConfigHelpers )

install ( TARGETS GPUQREngine
    EXPORT GPUQREngineTargets
    LIBRARY DESTINATION ${SUITESPARSE_LIBDIR}
    ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR}
    RUNTIME DESTINATION ${SUITESPARSE_BINDIR}
    PUBLIC_HEADER DESTINATION ${SUITESPARSE_INCLUDEDIR} )
if ( NOT NSTATIC )
    install ( TARGETS GPUQREngine_static
        EXPORT GPUQREngineTargets
        ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR} )
endif ( )

# create (temporary) export target file during build
export ( EXPORT GPUQREngineTargets
    NAMESPACE SuiteSparse::
    FILE ${CMAKE_CURRENT_BINARY_DIR}/GPUQREngineTargets.cmake )

# install export target, config and version files for find_package
install ( EXPORT GPUQREngineTargets
    NAMESPACE SuiteSparse::
    DESTINATION ${SUITESPARSE_LIBDIR}/cmake/GPUQREngine )

configure_package_config_file (
    Config/GPUQREngineConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/GPUQREngineConfig.cmake
    INSTALL_DESTINATION ${SUITESPARSE_LIBDIR}/cmake/GPUQREngine )

write_basic_package_version_file (
    ${CMAKE_CURRENT_BINARY_DIR}/GPUQREngineConfigVersion.cmake
    COMPATIBILITY SameMajorVersion )

install ( FILES
    ${CMAKE_CURRENT_BINARY_DIR}/GPUQREngineConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/GPUQREngineConfigVersion.cmake
    DESTINATION ${SUITESPARSE_LIBDIR}/cmake/GPUQREngine )

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
        Config/GPUQREngine.pc.in
        GPUQREngine.pc
        @ONLY
        NEWLINE_STYLE LF )
    install ( FILES
        ${CMAKE_CURRENT_BINARY_DIR}/GPUQREngine.pc
        DESTINATION ${SUITESPARSE_LIBDIR}/pkgconfig )
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
    target_link_libraries ( gpuqrengine_demo PUBLIC GPUQREngine
        SuiteSparse::CHOLMOD SuiteSparse::AMD SuiteSparse::COLAMD
        SuiteSparse::CAMD SuiteSparse::CCOLAMD SuiteSparse::CHOLMOD_CUDA
        SuiteSparse::GPURuntime SuiteSparse::SuiteSparseConfig )

else ( )

    message ( STATUS "Skipping the demos in GPUQREngine/Demo" )

endif ( )

#-------------------------------------------------------------------------------
# report status
#-------------------------------------------------------------------------------

include ( SuiteSparseReport )

