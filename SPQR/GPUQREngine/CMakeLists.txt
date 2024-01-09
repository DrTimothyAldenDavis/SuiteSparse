#-------------------------------------------------------------------------------
# SPQR/GPUQREngine/CMakeLists.txt
#-------------------------------------------------------------------------------

# SPQR/GPUQREngine, Copyright (c) 2013-2023, Timothy A Davis, Sencer Nuri
# Yeralan, and Sanjay Ranka.  All Rights Reserved.
# SPDX-License-Identifier: GPL-2.0+

#-------------------------------------------------------------------------------
# get the version
#-------------------------------------------------------------------------------

cmake_minimum_required ( VERSION 3.22 )

message ( STATUS "Building SPQR/GPUQRENGINE version: v"
    ${SPQR_VERSION_MAJOR}.
    ${SPQR_VERSION_MINOR}.
    ${SPQR_VERSION_SUB} " (" ${SPQR_DATE} ")" )

#-------------------------------------------------------------------------------
# define the project
#-------------------------------------------------------------------------------

project ( gpuqrengine 
    VERSION "${SPQR_VERSION_MAJOR}.${SPQR_VERSION_MINOR}.${SPQR_VERSION_SUB}"
    LANGUAGES CXX CUDA )

#-------------------------------------------------------------------------------
# SuiteSparse policies
#-------------------------------------------------------------------------------

set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    ${PROJECT_SOURCE_DIR}/../../SuiteSparse_config/cmake_modules )

# include ( SuiteSparsePolicy )

#-------------------------------------------------------------------------------
# find library dependencies
#-------------------------------------------------------------------------------

# Demo disabled for GPUQREngine v2.x
set ( DEMO_OK false )

if ( NOT SUITESPARSE_ROOT_CMAKELISTS AND SUITESPARSE_DEMOS AND DEMO_OK )
    # for the demo only:
    find_package ( CHOLMOD 5.1.1
        PATHS ${CMAKE_SOURCE_DIR}/../../CHOLMOD/build NO_DEFAULT_PATH )
    if ( NOT TARGET SuiteSparse::CHOLMOD )
        find_package ( CHOLMOD 5.1.1 )
    endif ( )
endif ( )

if ( CHOLMOD_FOUND )
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

file ( GLOB GPUQRENGINE_SOURCES "Source/*.cpp"
    "Source/*.cu"
    "Source/BucketList/BucketList.cpp"
    "Source/LLBundle/LLBundle.cpp"
    "Source/Scheduler/Scheduler.cpp"
    "Source/TaskDescriptor/*.cpp" )

set ( GPUQRENGINE_INCLUDES Include Include/Kernel
    Include/Kernel/Apply Include/Kernel/Assemble Include/Kernel/Factorize )

#-------------------------------------------------------------------------------
# dynamic gpuqrengine library properties
#-------------------------------------------------------------------------------

if ( BUILD_SHARED_LIBS )
    add_library ( GPUQREngine SHARED ${GPUQRENGINE_SOURCES} )

    set_target_properties ( GPUQREngine PROPERTIES
        VERSION ${SPQR_VERSION_MAJOR}.${SPQR_VERSION_MINOR}.${SPQR_VERSION_SUB}
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        OUTPUT_NAME gpuqrengine
        SOVERSION ${SPQR_VERSION_MAJOR}
        PUBLIC_HEADER "Include/GPUQREngine.hpp"
        WINDOWS_EXPORT_ALL_SYMBOLS ON )

    target_include_directories ( GPUQREngine PRIVATE
        ${GPUQRENGINE_INCLUDES} )

    target_link_libraries ( GPUQREngine PRIVATE
        GPURuntime SuiteSparse::SuiteSparseConfig )

    set_target_properties ( GPUQREngine PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_RUNTIME_LIBRARY Static )
    target_link_libraries ( GPUQREngine PRIVATE
        CUDA::nvrtc CUDA::cudart_static CUDA::cublas )
    target_compile_definitions ( GPUQREngine PRIVATE "SPQR_HAS_CUDA" )

    target_include_directories ( GPUQREngine
        INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>
                  $<INSTALL_INTERFACE:${SUITESPARSE_INCLUDEDIR}> )
endif ( )

#-------------------------------------------------------------------------------
# static gpuqrengine library properties
#-------------------------------------------------------------------------------

if ( BUILD_STATIC_LIBS )
    add_library ( GPUQREngine_static STATIC ${GPUQRENGINE_SOURCES} )

    set_target_properties ( GPUQREngine_static PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        OUTPUT_NAME gpuqrengine
        PUBLIC_HEADER "Include/GPUQREngine.hpp" )

    if ( MSVC )
        set_target_properties ( GPUQREngine_static PROPERTIES
            OUTPUT_NAME gpuqrengine_static )
    endif ( )

    target_include_directories ( GPUQREngine_static PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
        ${GPUQRENGINE_INCLUDES} )

    target_link_libraries ( GPUQREngine_static PRIVATE GPURuntime_static )

    if ( TARGET SuiteSparse::SuiteSparseConfig_static )
        target_link_libraries ( GPUQREngine_static PRIVATE SuiteSparse::SuiteSparseConfig_static )
    else ( )
        target_link_libraries ( GPUQREngine_static PRIVATE SuiteSparse::SuiteSparseConfig )
    endif ( )

    set_target_properties ( GPUQREngine_static PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_RUNTIME_LIBRARY Static )
    target_link_libraries ( GPUQREngine_static PRIVATE CUDA::nvrtc CUDA::cudart_static )
    target_compile_definitions ( GPUQREngine_static PRIVATE "SPQR_HAS_CUDA" )
    # FIXME: Ok to hardcode CUDA library names like this?
    set ( GPUQRENGINE_STATIC_LIBS "-L${CUDAToolkit_LIBRARY_DIR} -lcuda -lcudart_static" )
    if ( TARGET CUDA::cublas_static )
        target_link_libraries ( GPUQREngine_static PRIVATE CUDA::cublas_static )
        set ( GPUQRENGINE_STATIC_LIBS "${GPUQRENGINE_STATIC_LIBS} -lcublas_static -lcublasLt_static -lculibos" )
    else ( )
        target_link_libraries ( GPUQREngine_static PRIVATE CUDA::cublas )
        set ( GPUQRENGINE_STATIC_LIBS "${GPUQRENGINE_STATIC_LIBS} -lcublas" )
    endif ( )

    target_include_directories ( GPUQREngine_static
        INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>
                  $<INSTALL_INTERFACE:${SUITESPARSE_INCLUDEDIR}> )
endif ( )

#-------------------------------------------------------------------------------
# installation location
#-------------------------------------------------------------------------------

include ( CMakePackageConfigHelpers )

if ( BUILD_SHARED_LIBS )
    install ( TARGETS GPUQREngine
        EXPORT GPUQREngineTargets
        LIBRARY DESTINATION ${SUITESPARSE_LIBDIR}
        ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR}
        RUNTIME DESTINATION ${SUITESPARSE_BINDIR}
        PUBLIC_HEADER DESTINATION ${SUITESPARSE_INCLUDEDIR} )
endif ( )
if ( BUILD_STATIC_LIBS )
    install ( TARGETS GPUQREngine_static
        EXPORT GPUQREngineTargets
        ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR}
        PUBLIC_HEADER DESTINATION ${SUITESPARSE_INCLUDEDIR} )
endif ( )

# create (temporary) export target file during build
export ( EXPORT GPUQREngineTargets
    NAMESPACE SuiteSparse::
    FILE ${CMAKE_CURRENT_BINARY_DIR}/GPUQREngineTargets.cmake )

# install export target, config and version files for find_package
install ( EXPORT GPUQREngineTargets
    NAMESPACE SuiteSparse::
    DESTINATION ${SUITESPARSE_PKGFILEDIR}/cmake/GPUQREngine )

configure_package_config_file (
    Config/GPUQREngineConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/GPUQREngineConfig.cmake
    INSTALL_DESTINATION ${SUITESPARSE_PKGFILEDIR}/cmake/GPUQREngine )

write_basic_package_version_file (
    ${CMAKE_CURRENT_BINARY_DIR}/GPUQREngineConfigVersion.cmake
    COMPATIBILITY SameMajorVersion )

install ( FILES
    ${CMAKE_CURRENT_BINARY_DIR}/GPUQREngineConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/GPUQREngineConfigVersion.cmake
    DESTINATION ${SUITESPARSE_PKGFILEDIR}/cmake/GPUQREngine )

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
    if ( BUILD_SHARED_LIBS )
        set ( SUITESPARSE_LIB_BASE_NAME $<TARGET_FILE_BASE_NAME:GPUQREngine> )
    else ( )
        set ( SUITESPARSE_LIB_BASE_NAME $<TARGET_FILE_BASE_NAME:GPUQREngine_static> )
    endif ( )
    configure_file (
        Config/GPUQREngine.pc.in
        GPUQREngine.pc.out
        @ONLY
        NEWLINE_STYLE LF )
    file ( GENERATE
        OUTPUT GPUQREngine.pc
        INPUT ${CMAKE_CURRENT_BINARY_DIR}/GPUQREngine.pc.out
        NEWLINE_STYLE LF )
    install ( FILES
        ${CMAKE_CURRENT_BINARY_DIR}/GPUQREngine.pc
        DESTINATION ${SUITESPARSE_PKGFILEDIR}/pkgconfig )
endif ( )

#-------------------------------------------------------------------------------
# Demo library and programs
#-------------------------------------------------------------------------------

if ( SUITESPARSE_DEMOS AND DEMO_OK )

    #---------------------------------------------------------------------------
    # demo library
    #---------------------------------------------------------------------------

    message ( STATUS "Also compiling the demos in GPUQREngine/Demo" )

    #---------------------------------------------------------------------------
    # Demo programs
    #---------------------------------------------------------------------------

    add_executable ( gpuqrengine_demo "Demo/gpuqrengine_demo.cpp" )

    # Libraries required for Demo programs
    if ( BUILD_SHARED_LIBS )
        target_link_libraries ( gpuqrengine_demo
            PUBLIC GPUQREngine GPURuntime )
    else ( )
        target_link_libraries ( gpuqrengine_demo
            PUBLIC GPUQREngine_static GPURuntime_static )
    endif ( )
    target_link_libraries ( gpuqrengine_demo
        PUBLIC SuiteSparse::CHOLMOD SuiteSparse::SuiteSparseConfig )

else ( )

    message ( STATUS "Skipping the demos in GPUQREngine/Demo" )

endif ( )
