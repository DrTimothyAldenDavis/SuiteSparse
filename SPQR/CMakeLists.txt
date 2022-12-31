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

set ( SPQR_DATE "Jan 1, 2023" )
set ( SPQR_VERSION_MAJOR 3 )
set ( SPQR_VERSION_MINOR 0 )
set ( SPQR_VERSION_SUB   3 )

message ( STATUS "Building SPQR version: v"
    ${SPQR_VERSION_MAJOR}.
    ${SPQR_VERSION_MINOR}.
    ${SPQR_VERSION_SUB} " (" ${SPQR_DATE} ")" )

#-------------------------------------------------------------------------------
# SuiteSparse policies
#-------------------------------------------------------------------------------

set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    ${CMAKE_SOURCE_DIR}/cmake_modules
    ${CMAKE_SOURCE_DIR}/../COLAMD/cmake_modules
    ${CMAKE_SOURCE_DIR}/../AMD/cmake_modules
    ${CMAKE_SOURCE_DIR}/../CHOLMOD/cmake_modules
    ${CMAKE_SOURCE_DIR}/../GPUQREngine/cmake_modules
    ${CMAKE_SOURCE_DIR}/../SuiteSparse_GPURuntime/cmake_modules
    ${CMAKE_SOURCE_DIR}/../SuiteSparse_config/cmake_modules )

option ( ENABLE_CUDA "Enable CUDA acceleration" on )

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

enable_language ( C )

option ( NOPENMP "ON: do not use OpenMP.  OFF (default): use OpenMP" off )
if ( NOPENMP )
    # OpenMP has been disabled
    set ( OPENMP_FOUND false )
else ( )
    find_package ( OpenMP )
endif ( )

find_package ( SuiteSparse_config 7.0.0 REQUIRED )
find_package ( AMD 3.0.3 REQUIRED )
find_package ( COLAMD 3.0.3 REQUIRED )
find_package ( CHOLMOD 4.0.3 REQUIRED )

if ( SUITESPARSE_CUDA )
    find_package ( SuiteSparse_GPURuntime 2.0.3 REQUIRED )
    find_package ( GPUQREngine 2.0.3 REQUIRED )
    find_package ( CHOLMOD_CUDA 4.0.3 REQUIRED )
endif ( )

include ( SuiteSparseBLAS )     # requires cmake 3.22
include ( SuiteSparseLAPACK )   # requires cmake 3.22

add_subdirectory ( SPQRGPU )
set ( SPQR_CUDA spqr_cuda ${CUDA_LIBRARIES} ${GPUQRENGINE_LIBRARIES} ${SUITESPARSE_GPURUNTIME_LIBRARIES} ${CHOLMOD_CUDA_LIBRARIES} )

if ( SUITESPARSE_CUDA )
    # with CUDA
    include_directories ( SPQRGPU ${CUDAToolkit_INCLUDE_DIRS} )
    link_directories ( "SPQRGPU" "${CUDA_LIBRARIES}" "/usr/local/cuda/lib64/stubs" "/usr/local/cuda/lib64" )
endif ( )

#-------------------------------------------------------------------------------
# configure files
#-------------------------------------------------------------------------------

configure_file ( "Config/SuiteSparseQR_definitions.h.in"
    "${PROJECT_SOURCE_DIR}/Include/SuiteSparseQR_definitions.h"
    NEWLINE_STYLE LF )
configure_file ( "Config/spqr_version.tex.in"
    "${PROJECT_SOURCE_DIR}/Doc/spqr_version.tex"
    NEWLINE_STYLE LF )

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

if ( NOT NSTATIC )
add_library ( spqr_static STATIC ${SPQR_SOURCES} )

set_target_properties ( spqr_static PROPERTIES
    VERSION ${SPQR_VERSION_MAJOR}.${SPQR_VERSION_MINOR}.${SPQR_VERSION_SUB}
    CXX_STANDARD_REQUIRED 11
    C_STANDARD_REQUIRED 11
    OUTPUT_NAME spqr
    SOVERSION ${SPQR_VERSION_MAJOR} )
endif ( )

#-------------------------------------------------------------------------------
# add the library dependencies
#-------------------------------------------------------------------------------

# suitesparseconfig:
target_link_libraries ( spqr PUBLIC ${SUITESPARSE_CONFIG_LIBRARIES} )
if ( NOT NSTATIC )
target_link_libraries ( spqr_static PUBLIC ${SUITESPARSE_CONFIG_LIBRARIES} )
endif ( )

# OpenMP:
if ( OPENMP_FOUND )
    message ( STATUS "OpenMP C libraries:      ${OpenMP_C_LIBRARIES} ")
    message ( STATUS "OpenMP C include:        ${OpenMP_C_INCLUDE_DIRS} ")
    message ( STATUS "OpenMP C flags:          ${OpenMP_C_FLAGS} ")
    target_link_libraries ( spqr PUBLIC ${OpenMP_C_LIBRARIES} )
    if ( NOT NSTATIC )
    target_link_libraries ( spqr_static PUBLIC ${OpenMP_C_LIBRARIES} )
    endif ( )
    set ( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} " )
    include_directories ( ${OpenMP_C_INCLUDE_DIRS} )
endif ( )

# libm:
if ( NOT MSVC )
    target_link_libraries ( spqr PUBLIC m )
    if ( NOT NSTATIC )
        target_link_libraries ( spqr_static PUBLIC m )
    endif ( )
endif ( )

# amd:
target_link_libraries ( spqr PUBLIC ${AMD_LIBRARIES} )
if ( NOT NSTATIC )
target_link_libraries ( spqr_static PUBLIC ${AMD_LIBRARIES} )
endif ( )

# colamd:
target_link_libraries ( spqr PUBLIC ${COLAMD_LIBRARIES} )
if ( NOT NSTATIC )
target_link_libraries ( spqr_static PUBLIC ${COLAMD_LIBRARIES} )
endif ( )

# BLAS:
message ( STATUS "BLAS libraries:      ${BLAS_LIBRARIES} ")
message ( STATUS "BLAS include:        ${BLAS_INCLUDE_DIRS} ")
message ( STATUS "BLAS linker flags:   ${BLAS_LINKER_FLAGS} ")
target_link_libraries ( spqr PUBLIC ${BLAS_LIBRARIES} )
if ( NOT NSTATIC )
target_link_libraries ( spqr_static PUBLIC ${BLAS_LIBRARIES} )
endif ( )
include_directories ( ${BLAS_INCLUDE_DIRS} )

# LAPACK:
message ( STATUS "LAPACK libraries:    ${LAPACK_LIBRARIES} ")
message ( STATUS "LAPACK include:      ${LAPACK_INCLUDE_DIRS} ")
message ( STATUS "LAPACK linker flags: ${LAPACK_LINKER_FLAGS} ")
target_link_libraries ( spqr PUBLIC ${LAPACK_LIBRARIES} )
if ( NOT NSTATIC )
target_link_libraries ( spqr_static PUBLIC ${LAPACK_LIBRARIES} )
endif ( )
include_directories ( ${LAPACK_INCLUDE_DIR} )

# cholmod:
target_link_libraries ( spqr PUBLIC ${CHOLMOD_LIBRARIES} )
if ( NOT NSTATIC )
target_link_libraries ( spqr_static PUBLIC ${CHOLMOD_LIBRARIES} )
endif ( )

# CUDA
message ( STATUS "SPQR cuda:  ${SPQR_CUDA} " )
target_link_libraries ( spqr PUBLIC ${SPQR_CUDA} ${CUDA_LIBRARIES} )
if ( NOT NSTATIC )
target_link_libraries ( spqr_static PUBLIC ${SPQR_CUDA} ${CUDA_LIBRARIES} )
endif ( )

#-------------------------------------------------------------------------------
# SPQR installation location
#-------------------------------------------------------------------------------

if ( GLOBAL_INSTALL )
    # install in /usr/local/lib and /usr/local/include.
    # requires "sudo make install"
    message ( STATUS "Installation will be system-wide (requires 'sudo make install')" )
    install ( TARGETS spqr
        LIBRARY       DESTINATION ${CMAKE_INSTALL_LIBDIR} )
    install ( FILES
        ${CMAKE_SOURCE_DIR}/cmake_modules/FindSPQR.cmake
        ${CMAKE_SOURCE_DIR}/cmake_modules/FindSPQR_CUDA.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SuiteSparse 
        COMPONENT Development )
    if ( NOT NSTATIC )
    install ( TARGETS spqr_static
        ARCHIVE       DESTINATION ${CMAKE_INSTALL_LIBDIR} )
    endif ( )
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
    install ( FILES
        ${CMAKE_SOURCE_DIR}/cmake_modules/FindSPQR.cmake
        ${CMAKE_SOURCE_DIR}/cmake_modules/FindSPQR_CUDA.cmake
        DESTINATION ${SUITESPARSE_LIBDIR}/cmake/SuiteSparse 
        COMPONENT Development )
    if ( NOT NSTATIC )
    install ( TARGETS spqr_static
        ARCHIVE       DESTINATION ${SUITESPARSE_LIBDIR} )
    endif ( )
    install ( FILES "Include/SuiteSparseQR_C.h"
        "Include/SuiteSparseQR_definitions.h"
        "Include/SuiteSparseQR.hpp" DESTINATION ${SUITESPARSE_INCLUDEDIR} )
endif ( )

#-------------------------------------------------------------------------------
# Demo library and programs
#-------------------------------------------------------------------------------

option ( DEMO "ON: Build the demo programs.  OFF (default): do not build the demo programs." off )
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

