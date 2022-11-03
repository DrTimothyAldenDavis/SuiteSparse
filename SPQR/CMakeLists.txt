#-------------------------------------------------------------------------------
# SuiteSparse/SPQR/CMakeLists.txt:  cmake for SuiteSparseQR (aka "SPQR")
#-------------------------------------------------------------------------------

# SPQR (SuiteSparseQR), Copyright (c) 2008-2022, Timothy A Davis.
# All Rights Reserved.
# SPDX-License-Identifier: GPL-2.0+

#-------------------------------------------------------------------------------
# get the version
#-------------------------------------------------------------------------------

# cmake 3.22 is required to find the BLAS in SuiteSparsePolicy.cmake
cmake_minimum_required ( VERSION 3.22 )

set ( SPQR_DATE "Nov 4, 2022" )
set ( SPQR_VERSION_MAJOR 3 )
set ( SPQR_VERSION_MINOR 0 )
set ( SPQR_VERSION_SUB   0 )

message ( STATUS "Building SPQR version: v"
    ${SPQR_VERSION_MAJOR}.
    ${SPQR_VERSION_MINOR}.
    ${SPQR_VERSION_SUB} " (" ${SPQR_DATE} ")" )

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

#-------------------------------------------------------------------------------
# define the project
#-------------------------------------------------------------------------------

if ( SUITESPARSE_CUDA )
    # SPQR with CUDA
    project ( spqr
        VERSION "${SPQR_VERSION_MAJOR}.${SPQR_VERSION_MINOR}.${SPQR_VERSION_SUB}"
        LANGUAGES C CXX CUDA )
else ( )
    # SPQR without CUDA
    project ( spqr
        VERSION "${SPQR_VERSION_MAJOR}.${SPQR_VERSION_MINOR}.${SPQR_VERSION_SUB}"
        LANGUAGES C CXX )
endif ( )

#-------------------------------------------------------------------------------
# find library dependencies
#-------------------------------------------------------------------------------

enable_language ( C Fortran )
message ( STATUS "Fortran: ${CMAKE_Fortran_COMPILER_ID} ") 
find_package ( OpenMP )
find_package ( SuiteSparse_config 6.0.0 REQUIRED )
find_package ( AMD 3.0.0 REQUIRED )
find_package ( COLAMD 3.0.0 REQUIRED )
find_package ( CHOLMOD 4.0.0 REQUIRED )

if ( SUITESPARSE_CUDA )
    find_package ( SuiteSparse_GPURuntime 2.0.0 REQUIRED )
    find_package ( GPUQREngine 2.0.0 REQUIRED )
    find_package ( CHOLMOD_CUDA 4.0.0 REQUIRED )
endif ( )

include ( SuiteSparseBLAS )     # requires cmake 3.22

if ( SUITESPARSE_CUDA )
    # with CUDA
    set ( CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -DSUITESPARSE_CUDA " )
    add_subdirectory ( SPQRGPU )
    set ( SPQR_CUDA spqr_cuda ${CUDA_LIBRARIES} ${GPUQRENGINE_LIBRARIES} ${SUITESPARSE_GPURUNTIME_LIBRARIES} ${CHOLMOD_CUDA_LIBRARIES} )
    include_directories ( SPQRGPU ${CUDAToolkit_INCLUDE_DIRS} )
    link_directories ( "SPQRGPU" "${CUDA_LIBRARIES}" "/usr/local/cuda/lib64/stubs" "/usr/local/cuda/lib64" )
else ( )
    # without CUDA and RMM
    set ( SPQR_CUDA )
endif ( )

#-------------------------------------------------------------------------------
# configure files
#-------------------------------------------------------------------------------

configure_file ( "Config/SuiteSparseQR_definitions.h.in"
    "${PROJECT_SOURCE_DIR}/Include/SuiteSparseQR_definitions.h" )
configure_file ( "Config/spqr_version.tex.in"
    "${PROJECT_SOURCE_DIR}/Doc/spqr_version.tex" )

#-------------------------------------------------------------------------------
# include directories
#-------------------------------------------------------------------------------

include_directories ( Source Include ${SUITESPARSE_CONFIG_INCLUDE_DIR}
    ${AMD_INCLUDE_DIR} ${COLAMD_INCLUDE_DIR} ${CHOLMOD_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/../CHOLMOD/Include )

if ( SUITESPARSE_CUDA )
    include_directories ( 
    ${CMAKE_SOURCE_DIR}/../SuiteSparse_GPURuntime/Include
    ${CMAKE_SOURCE_DIR}/../GPUQREngine/Include )
endif ( )

#-------------------------------------------------------------------------------
# dynamic spqr library properties
#-------------------------------------------------------------------------------

file ( GLOB SPQR_SOURCES "Source/spqr_*.cpp" "Source/SuiteSparseQR*.cpp" )

add_library ( spqr SHARED ${SPQR_SOURCES} )

set_target_properties ( spqr PROPERTIES
    VERSION ${SPQR_VERSION_MAJOR}.${SPQR_VERSION_MINOR}.${SPQR_VERSION_SUB}
    CXX_STANDARD_REQUIRED 11
    C_STANDARD_REQUIRED 11
    SOVERSION ${SPQR_VERSION_MAJOR} )

#-------------------------------------------------------------------------------
# static spqr library properties
#-------------------------------------------------------------------------------

add_library ( spqr_static STATIC ${SPQR_SOURCES} )

set_target_properties ( spqr_static PROPERTIES
    VERSION ${SPQR_VERSION_MAJOR}.${SPQR_VERSION_MINOR}.${SPQR_VERSION_SUB}
    CXX_STANDARD_REQUIRED 11
    C_STANDARD_REQUIRED 11
    OUTPUT_NAME spqr
    SOVERSION ${SPQR_VERSION_MAJOR} )

#-------------------------------------------------------------------------------
# add the library dependencies
#-------------------------------------------------------------------------------

# suitesparseconfig:
target_link_libraries ( spqr PUBLIC ${SUITESPARSE_CONFIG_LIBRARIES} )
target_link_libraries ( spqr_static PUBLIC ${SUITESPARSE_CONFIG_LIBRARIES} )

# OpenMP:
if ( OPENMP_FOUND )
    message ( STATUS "OpenMP C libraries:      ${OpenMP_C_LIBRARIES} ")
    message ( STATUS "OpenMP C include:        ${OpenMP_C_INCLUDE_DIRS} ")
    message ( STATUS "OpenMP C flags:          ${OpenMP_C_FLAGS} ")
    target_link_libraries ( spqr PUBLIC ${OpenMP_C_LIBRARIES} )
    target_link_libraries ( spqr_static PUBLIC ${OpenMP_C_LIBRARIES} )
    set ( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} " )
    include_directories ( ${OpenMP_C_INCLUDE_DIRS} )
endif ( )

# libm:
if ( NOT "${CMAKE_C_COMPILER_ID}" STREQUAL "MSVC" )
    target_link_libraries ( spqr PUBLIC m )
    target_link_libraries ( spqr_static PUBLIC m )
endif ( )

# amd:
target_link_libraries ( spqr PUBLIC ${AMD_LIBRARIES} )
target_link_libraries ( spqr_static PUBLIC ${AMD_LIBRARIES} )

# colamd:
target_link_libraries ( spqr PUBLIC ${COLAMD_LIBRARIES} )
target_link_libraries ( spqr_static PUBLIC ${COLAMD_LIBRARIES} )

# BLAS:
message ( STATUS "BLAS libraries:      ${BLAS_LIBRARIES} ")
message ( STATUS "BLAS linker flags:   ${BLAS_LINKER_FLAGS} ")
target_link_libraries ( spqr PUBLIC ${BLAS_LIBRARIES} )
target_link_libraries ( spqr_static PUBLIC ${BLAS_LIBRARIES} )
include_directories ( ${BLAS_INCLUDE_DIRS} )

# cholmod:
target_link_libraries ( spqr PUBLIC ${CHOLMOD_LIBRARIES} )
target_link_libraries ( spqr_static PUBLIC ${CHOLMOD_LIBRARIES} )

# CUDA
if ( SUITESPARSE_CUDA )
    message ( STATUS "SPQR cuda:  ${SPQR_CUDA} " )
    target_link_libraries ( spqr PUBLIC ${SPQR_CUDA} ${CUDA_LIBRARIES} )
    target_link_libraries ( spqr_static PUBLIC ${SPQR_CUDA} ${CUDA_LIBRARIES} )

#   # GPUQREngine
#   target_link_libraries ( spqr PUBLIC ${GPUQRENGINE_LIBRARIES} )
#   target_link_libraries ( spqr_static PUBLIC ${GPUQRENGINE_LIBRARIES} )

#   # SuiteSparse_GPURuntime
#   target_link_libraries ( spqr PUBLIC ${SUITESPARSE_GPURUNTIME_LIBRARIES} )
#   target_link_libraries ( spqr_static PUBLIC ${SUITESPARSE_GPURUNTIME_LIBRARIES} )
endif ( )

#-------------------------------------------------------------------------------
# SPQR installation location
#---------------------------------------------------------------Z--------------

# use "cmake -DSUITESPARSE_LOCAL=1" to install only in SuiteSparse/lib and
# SuiteSparse/include.  

if ( NOT SUITESPARSE_LOCAL )
    # install in /usr/local/lib and /usr/local/include.
    # requires "sudo make install"
    message ( STATUS "Installation will be system-wide (requires 'sudo make install')" )
    install ( TARGETS spqr
        LIBRARY       DESTINATION ${CMAKE_INSTALL_LIBDIR} )
    install ( TARGETS spqr_static
        ARCHIVE       DESTINATION ${CMAKE_INSTALL_LIBDIR} )
    install ( FILES "Include/SuiteSparseQR_C.h"
        "Include/SuiteSparseQR_definitions.h"
        "Include/SuiteSparseQR.hpp" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} )
endif ( )

if ( INSIDE_SUITESPARSE )
    # also install in SuiteSparse/lib and SuiteSparse/include;
    # does not require "sudo make install", just "make install"
    message ( STATUS "Installation in ../lib and ../include only," )
    message ( STATUS "  with 'make install'. No 'sudo' required." )
    install ( TARGETS spqr
        LIBRARY       DESTINATION ${SUITESPARSE_LIBDIR} )
    install ( TARGETS spqr_static
        ARCHIVE       DESTINATION ${SUITESPARSE_LIBDIR} )
    install ( FILES "Include/SuiteSparseQR_C.h"
        "Include/SuiteSparseQR_definitions.h"
        "Include/SuiteSparseQR.hpp" DESTINATION ${SUITESPARSE_INCLUDEDIR} )
endif ( )

#-------------------------------------------------------------------------------
# Demo library and programs
#-------------------------------------------------------------------------------

if ( DEMO )

    #---------------------------------------------------------------------------
    # demo library
    #---------------------------------------------------------------------------

    message ( STATUS "Also compiling the demos in SPQR/Demo" )

    #---------------------------------------------------------------------------
    # Demo programs
    #---------------------------------------------------------------------------

    add_executable ( qrsimple  "Demo/qrsimple.cpp" )
    add_executable ( qrsimplec "Demo/qrsimplec.c" )
    add_executable ( qrdemo    "Demo/qrdemo.cpp" )
    add_executable ( qrdemoc   "Demo/qrdemoc.c" )

    if ( SUITESPARSE_CUDA )
        add_executable ( qrdemo_gpu  "Demo/qrdemo_gpu.cpp" )
        add_executable ( qrdemo_gpu2 "Demo/qrdemo_gpu2.cpp" )
        add_executable ( qrdemo_gpu3 "Demo/qrdemo_gpu3.cpp" )
    endif ( )

    # Libraries required for Demo programs
    target_link_libraries ( qrsimple  PUBLIC spqr ${SPQR_CUDA} )
    target_link_libraries ( qrsimplec PUBLIC spqr ${SPQR_CUDA} )
    target_link_libraries ( qrdemo    PUBLIC spqr ${SPQR_CUDA} )
    target_link_libraries ( qrdemoc   PUBLIC spqr ${SPQR_CUDA} )

    if ( SUITESPARSE_CUDA )
        target_link_libraries ( qrdemo_gpu   PUBLIC spqr ${SPQR_CUDA} )
        target_link_libraries ( qrdemo_gpu2  PUBLIC spqr ${SPQR_CUDA} )
        target_link_libraries ( qrdemo_gpu3  PUBLIC spqr ${SPQR_CUDA} )
    endif ( )

else ( )

    message ( STATUS "Skipping the demos in SPQR/Demo" )

endif ( )

#-------------------------------------------------------------------------------
# report status
#-------------------------------------------------------------------------------

include ( SuiteSparseReport )

