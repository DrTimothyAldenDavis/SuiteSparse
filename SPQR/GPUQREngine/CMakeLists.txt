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

set ( GPUQRENGINE_DATE "Sept 18, 2023" )
set ( GPUQRENGINE_VERSION_MAJOR 3 )
set ( GPUQRENGINE_VERSION_MINOR 2 )
set ( GPUQRENGINE_VERSION_SUB   1 )

message ( STATUS "Building GPUQRENGINE version: v"
    ${GPUQRENGINE_VERSION_MAJOR}.
    ${GPUQRENGINE_VERSION_MINOR}.
    ${GPUQRENGINE_VERSION_SUB} " (" ${GPUQRENGINE_DATE} ")" )

set ( GPUQRENGINE_VERSION_MAJOR ${GPUQRENGINE_VERSION_MAJOR} PARENT_SCOPE )
set ( GPUQRENGINE_VERSION_MINOR ${GPUQRENGINE_VERSION_MINOR} PARENT_SCOPE )
set ( GPUQRENGINE_VERSION_SUB   ${GPUQRENGINE_VERSION_SUB} PARENT_SCOPE )

#-------------------------------------------------------------------------------
# define the project
#-------------------------------------------------------------------------------

project ( gpuqrengine 
    VERSION "${GPUQRENGINE_VERSION_MAJOR}.${GPUQRENGINE_VERSION_MINOR}.${GPUQRENGINE_VERSION_SUB}"
    LANGUAGES CXX CUDA )

#-------------------------------------------------------------------------------
# SuiteSparse policies
#-------------------------------------------------------------------------------

set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    ${PROJECT_SOURCE_DIR}/../../SuiteSparse_config/cmake_modules )

include ( SuiteSparsePolicy )

#-------------------------------------------------------------------------------
# find library dependencies
#-------------------------------------------------------------------------------

# for the library itself
find_package ( SuiteSparse_config 7.2.0
    PATHS ${CMAKE_SOURCE_DIR}/../../SuiteSparse_config/build NO_DEFAULT_PATH )
if ( NOT TARGET SuiteSparse::SuiteSparseConfig )
    find_package ( SuiteSparse_config 7.2.0 REQUIRED )
endif ( )

# Demo disabled for GPUQREngine v2.x
set ( DEMO_OK false )

if ( DEMO AND DEMO_OK )
    # for the demo only:
    find_package ( CHOLMOD 4.2.0
        PATHS ${CMAKE_SOURCE_DIR}/../../CHOLMOD/build NO_DEFAULT_PATH )
    if ( NOT TARGET SuiteSparse::CHOLMOD )
        find_package ( CHOLMOD 4.2.0 )
    endif ( )
endif ( )

if ( SUITESPARSE_CUDA AND CHOLMOD_FOUND )
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

file ( GLOB GPUQRENGINE_SOURCES "Source/*.cpp" "Source/*.cu" "Source/*/*.cpp" )

set ( GPUQRENGINE_INCLUDES Include Include/Kernel
    Include/Kernel/Apply Include/Kernel/Assemble Include/Kernel/Factorize )

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
    ${GPUQRENGINE_INCLUDES}
    "$<TARGET_PROPERTY:SuiteSparse::SuiteSparseConfig,INTERFACE_INCLUDE_DIRECTORIES>"
    "$<TARGET_PROPERTY:GPURuntime,INTERFACE_INCLUDE_DIRECTORIES>" )

set_target_properties ( GPUQREngine PROPERTIES POSITION_INDEPENDENT_CODE ON )
set_target_properties ( GPUQREngine PROPERTIES CUDA_SEPARABLE_COMPILATION ON )
target_link_libraries ( GPUQREngine PRIVATE
    CUDA::nvrtc CUDA::cudart_static CUDA::nvToolsExt CUDA::cublas )
target_compile_definitions ( GPUQREngine PRIVATE "SUITESPARSE_CUDA" )

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

    target_include_directories ( GPUQREngine_static PRIVATE
        "$<TARGET_PROPERTY:GPURuntime,INTERFACE_INCLUDE_DIRECTORIES>" )

    set_target_properties ( GPUQREngine_static PROPERTIES CUDA_SEPARABLE_COMPILATION ON )
    set_target_properties ( GPUQREngine_static PROPERTIES POSITION_INDEPENDENT_CODE ON )
    target_link_libraries ( GPUQREngine_static PUBLIC
        CUDA::nvrtc CUDA::cudart_static CUDA::nvToolsExt CUDA::cublas )
    target_compile_definitions ( GPUQREngine_static PRIVATE "SUITESPARSE_CUDA" )

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
        SuiteSparse::CHOLMOD SuiteSparse::CHOLMOD_CUDA
        GPURuntime SuiteSparse::SuiteSparseConfig )

else ( )

    message ( STATUS "Skipping the demos in GPUQREngine/Demo" )

endif ( )
