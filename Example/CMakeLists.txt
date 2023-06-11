#-------------------------------------------------------------------------------
# SuiteSparse/Example/CMakeLists.txt:  cmake for Example user library, "my"
#-------------------------------------------------------------------------------

# Copyright (c) 2022-2023, Timothy A. Davis, All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

#-------------------------------------------------------------------------------
# get the version
#-------------------------------------------------------------------------------

cmake_minimum_required ( VERSION 3.19 )

# cmake inserts the date and version number into Include/my.h:
set ( MY_DATE "June 12, 2023" )
set ( MY_VERSION_MAJOR 1 )
set ( MY_VERSION_MINOR 2 )
set ( MY_VERSION_PATCH 5 )

message ( STATUS "Building MY library version: v"
    ${MY_VERSION_MAJOR}.
    ${MY_VERSION_MINOR}.
    ${MY_VERSION_PATCH} " (" ${MY_DATE} ")" )

#-------------------------------------------------------------------------------
# policies
#-------------------------------------------------------------------------------

cmake_policy ( SET CMP0042 NEW )    # enable MACOSX_RPATH by default
cmake_policy ( SET CMP0048 NEW )    # VERSION variable policy
cmake_policy ( SET CMP0054 NEW )    # if ( expression ) handling policy
cmake_policy ( SET CMP0104 NEW )    # initialize CUDA architectures

if ( WIN32 )
    set ( CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS true )
endif ( )

set ( CMAKE_MACOSX_RPATH TRUE )
enable_language ( C CXX )
include ( GNUInstallDirs )

# set the module path for all Find*.cmake files.
set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    ${CMAKE_SOURCE_DIR}/cmake_modules
    ${CMAKE_SOURCE_DIR}/../lib/cmake/SuiteSparse
    ${CMAKE_INSTALL_PREFIX}/lib/cmake/SuiteSparse )

#-------------------------------------------------------------------------------
# define my project
#-------------------------------------------------------------------------------

project ( my
    VERSION "${MY_VERSION_MAJOR}.${MY_VERSION_MINOR}.${MY_VERSION_PATCH}"
    LANGUAGES C CXX )

#-------------------------------------------------------------------------------
# find library dependencies
#-------------------------------------------------------------------------------

# look for all SuiteSparse packages:
find_package ( SuiteSparse_config 7.1.0 REQUIRED )
find_package ( AMD 3.0.4 REQUIRED )
find_package ( BTF 2.0.4 REQUIRED )
find_package ( CAMD 3.0.4 REQUIRED )
find_package ( CCOLAMD 3.0.4 REQUIRED )
find_package ( COLAMD 3.0.4 REQUIRED )
find_package ( CHOLMOD 4.0.4 REQUIRED )
find_package ( CHOLMOD_CUDA 4.0.4 REQUIRED )
find_package ( CXSparse 4.0.4 REQUIRED )
find_package ( GPUQREngine 2.0.4 REQUIRED )
find_package ( GraphBLAS 8.0.2 REQUIRED )
find_package ( KLU 2.0.4 REQUIRED )
find_package ( KLU_CHOLMOD 2.0.4 REQUIRED )
find_package ( LDL 3.0.4 REQUIRED )
find_package ( Mongoose 3.0.5 REQUIRED )
find_package ( RBio 4.0.0 REQUIRED )
find_package ( SPEX 2.0.4 REQUIRED )
find_package ( SuiteSparse_GPURuntime 2.0.4 REQUIRED )
find_package ( SPQR 3.0.4 REQUIRED )
find_package ( SPQR_CUDA 3.0.4 REQUIRED )
find_package ( UMFPACK 6.1.1 REQUIRED )

# look for all external libaries:
find_package ( OpenMP REQUIRED )
find_package ( GMP 6.1.2 REQUIRED )
find_package ( MPFR 4.0.2 REQUIRED )
include ( SuiteSparseBLAS )

#-------------------------------------------------------------------------------
# configure files
#-------------------------------------------------------------------------------

configure_file ( "Config/my.h.in"
    "${PROJECT_SOURCE_DIR}/Include/my.h"
    NEWLINE_STYLE LF )

#-------------------------------------------------------------------------------
# include directories
#-------------------------------------------------------------------------------

include_directories ( Source Include )

#-------------------------------------------------------------------------------
# my dynamic C library properties
#-------------------------------------------------------------------------------

file ( GLOB MY_SOURCES "Source/*.c" )
add_library ( my SHARED ${MY_SOURCES} )
set_target_properties ( my PROPERTIES
    VERSION ${MY_VERSION_MAJOR}.${MY_VERSION_MINOR}.${MY_VERSION_PATCH}
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
    SOVERSION ${MY_VERSION_MAJOR}
    PUBLIC_HEADER "Include/my.h"
    WINDOWS_EXPORT_ALL_SYMBOLS ON )

#-------------------------------------------------------------------------------
# my static C library properties
#-------------------------------------------------------------------------------

add_library ( my_static STATIC ${MY_SOURCES} )
set_target_properties ( my_static PROPERTIES
    VERSION ${MY_VERSION_MAJOR}.${MY_VERSION_MINOR}.${MY_VERSION_PATCH}
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
    OUTPUT_NAME my
    SOVERSION ${MY_VERSION_MAJOR} )

if ( MSVC )
    set_target_properties ( my_static PROPERTIES
        OUTPUT_NAME my_static )
endif ( )

#-------------------------------------------------------------------------------
# my dynamic C++ library properties
#-------------------------------------------------------------------------------

file ( GLOB MY_SOURCES "Source/*.cc" )
add_library ( my_cxx SHARED ${MY_SOURCES} )
set_target_properties ( my_cxx PROPERTIES
    VERSION ${MY_VERSION_MAJOR}.${MY_VERSION_MINOR}.${MY_VERSION_PATCH}
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
    SOVERSION ${MY_VERSION_MAJOR}
    PUBLIC_HEADER "Include/my.h"
    WINDOWS_EXPORT_ALL_SYMBOLS ON )

#-------------------------------------------------------------------------------
# my static C++ library properties
#-------------------------------------------------------------------------------

add_library ( my_cxx_static STATIC ${MY_SOURCES} )
set_target_properties ( my_cxx_static PROPERTIES
    VERSION ${MY_VERSION_MAJOR}.${MY_VERSION_MINOR}.${MY_VERSION_PATCH}
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
    OUTPUT_NAME my_cxx
    SOVERSION ${MY_VERSION_MAJOR} )

if ( MSVC )
    set_target_properties ( my_cxx_static PROPERTIES
        OUTPUT_NAME my_cxx_static )
endif ( )

#-------------------------------------------------------------------------------
# add the library dependencies
#-------------------------------------------------------------------------------

# OpenMP:
message ( STATUS "OpenMP C libraries:      ${OpenMP_C_LIBRARIES} ")
message ( STATUS "OpenMP C include:        ${OpenMP_C_INCLUDE_DIRS} ")
message ( STATUS "OpenMP C flags:          ${OpenMP_C_FLAGS} ")
target_link_libraries ( my PRIVATE ${OpenMP_C_LIBRARIES} )
target_link_libraries ( my_static PUBLIC ${OpenMP_C_LIBRARIES} )
target_link_libraries ( my_cxx PUBLIC ${OpenMP_C_LIBRARIES} )
target_link_libraries ( my_cxx_static PUBLIC ${OpenMP_C_LIBRARIES} )
set ( CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS} " )
set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_C_FLAGS} " )
include_directories ( ${OpenMP_C_INCLUDE_DIRS} )

# libm:
if ( NOT WIN32 )
    target_link_libraries ( my PRIVATE m )
    target_link_libraries ( my_static PUBLIC m )
    target_link_libraries ( my_cxx PUBLIC m )
    target_link_libraries ( my_cxx_static PUBLIC m )
endif ( )

# BLAS:
message ( STATUS "BLAS libraries:      ${BLAS_LIBRARIES} ")
message ( STATUS "BLAS include:        ${BLAS_INCLUDE_DIRS} ")
message ( STATUS "BLAS linker flags:   ${BLAS_LINKER_FLAGS} ")
target_link_libraries ( my PRIVATE ${BLAS_LIBRARIES} )
target_link_libraries ( my_static PUBLIC ${BLAS_LIBRARIES} )
target_link_libraries ( my_cxx PUBLIC ${BLAS_LIBRARIES} )
target_link_libraries ( my_cxx_static PUBLIC ${BLAS_LIBRARIES} )
include_directories ( ${BLAS_INCLUDE_DIRS} )

# LAPACK:
message ( STATUS "LAPACK libraries:    ${LAPACK_LIBRARIES} ")
message ( STATUS "LAPACK include:      ${LAPACK_INCLUDE_DIRS} ")
message ( STATUS "LAPACK linker flags: ${LAPACK_LINKER_FLAGS} ")
target_link_libraries ( my PRIVATE ${LAPACK_LIBRARIES} )
target_link_libraries ( my_static PUBLIC ${LAPACK_LIBRARIES} )
target_link_libraries ( my_cxx PUBLIC ${LAPACK_LIBRARIES} )
target_link_libraries ( my_cxx_static PUBLIC ${LAPACK_LIBRARIES} )
include_directories ( ${LAPACK_INCLUDE_DIRS} )

# gmp:
target_link_libraries ( my PRIVATE ${GMP_LIBRARIES} )
target_link_libraries ( my_static PUBLIC ${GMP_STATIC} )
target_link_libraries ( my_cxx PUBLIC ${GMP_LIBRARIES} )
target_link_libraries ( my_cxx_static PUBLIC ${GMP_STATIC} )
include_directories ( ${GMP_INCLUDE_DIR} )

# mpfr:
target_link_libraries ( my PRIVATE ${MPFR_LIBRARIES} )
target_link_libraries ( my_static PUBLIC ${MPFR_STATIC} )
target_link_libraries ( my_cxx PUBLIC ${MPFR_LIBRARIES} )
target_link_libraries ( my_cxx_static PUBLIC ${MPFR_STATIC} )
include_directories ( ${MPFR_INCLUDE_DIR} )

# amd:
target_link_libraries ( my PRIVATE ${AMD_LIBRARIES} )
target_link_libraries ( my_static PUBLIC ${AMD_STATIC} )
target_link_libraries ( my_cxx PUBLIC ${AMD_LIBRARIES} )
target_link_libraries ( my_cxx_static PUBLIC ${AMD_STATIC} )
include_directories ( ${AMD_INCLUDE_DIR} )

# btf:
target_link_libraries ( my PRIVATE ${BTF_LIBRARIES} )
target_link_libraries ( my_static PUBLIC ${BTF_STATIC} )
target_link_libraries ( my_cxx PUBLIC ${BTF_LIBRARIES} )
target_link_libraries ( my_cxx_static PUBLIC ${BTF_STATIC} )
include_directories ( ${BTF_INCLUDE_DIR} )

# camd:
target_link_libraries ( my PRIVATE ${CAMD_LIBRARIES} )
target_link_libraries ( my_static PUBLIC ${CAMD_STATIC} )
target_link_libraries ( my_cxx PUBLIC ${CAMD_LIBRARIES} )
target_link_libraries ( my_cxx_static PUBLIC ${CAMD_STATIC} )
include_directories ( ${CAMD_INCLUDE_DIR} )

# ccolamd:
target_link_libraries ( my PRIVATE ${CCOLAMD_LIBRARIES} )
target_link_libraries ( my_static PUBLIC ${CCOLAMD_STATIC} )
target_link_libraries ( my_cxx PUBLIC ${CCOLAMD_LIBRARIES} )
target_link_libraries ( my_cxx_static PUBLIC ${CCOLAMD_STATIC} )
include_directories ( ${CCOLAMD_INCLUDE_DIR} )

# cholmod:
target_link_libraries ( my PRIVATE ${CHOLMOD_LIBRARIES} )
target_link_libraries ( my_static PUBLIC ${CHOLMOD_STATIC} )
target_link_libraries ( my_cxx PUBLIC ${CHOLMOD_LIBRARIES} )
target_link_libraries ( my_cxx_static PUBLIC ${CHOLMOD_STATIC} )
include_directories ( ${CHOLMOD_INCLUDE_DIR} )

# colamd:
target_link_libraries ( my PRIVATE ${COLAMD_LIBRARIES} )
target_link_libraries ( my_static PUBLIC ${COLAMD_STATIC} )
target_link_libraries ( my_cxx PUBLIC ${COLAMD_LIBRARIES} )
target_link_libraries ( my_cxx_static PUBLIC ${COLAMD_STATIC} )
include_directories ( ${COLAMD_INCLUDE_DIR} )

# cxsparse:
target_link_libraries ( my PRIVATE ${CXSPARSE_LIBRARIES} )
target_link_libraries ( my_static PUBLIC ${CXSPARSE_STATIC} )
target_link_libraries ( my_cxx PUBLIC ${CXSPARSE_LIBRARIES} )
target_link_libraries ( my_cxx_static PUBLIC ${CXSPARSE_STATIC} )
include_directories ( ${CXSPARSE_INCLUDE_DIR} )

# gpuqrengine:
target_link_libraries ( my PRIVATE ${GPUQRENGINE_LIBRARIES} )
target_link_libraries ( my_static PUBLIC ${GPUQRENGINE_STATIC} )
target_link_libraries ( my_cxx PUBLIC ${GPUQRENGINE_LIBRARIES} )
target_link_libraries ( my_cxx_static PUBLIC ${GPUQRENGINE_STATIC} )
include_directories ( ${GPUQRENGINE_INCLUDE_DIR} )

# GraphBLAS:
target_link_libraries ( my PRIVATE ${GRAPHBLAS_LIBRARIES} )
target_link_libraries ( my_static PUBLIC ${GRAPHBLAS_STATIC} )
target_link_libraries ( my_cxx PUBLIC ${GRAPHBLAS_LIBRARIES} )
target_link_libraries ( my_cxx_static PUBLIC ${GRAPHBLAS_STATIC} )
include_directories ( ${GRAPHBLAS_INCLUDE_DIR} )

# klu:
target_link_libraries ( my PRIVATE ${KLU_LIBRARIES} )
target_link_libraries ( my_static PUBLIC ${KLU_STATIC} )
target_link_libraries ( my_cxx PUBLIC ${KLU_LIBRARIES} )
target_link_libraries ( my_cxx_static PUBLIC ${KLU_STATIC} )
include_directories ( ${KLU_INCLUDE_DIR} )

# klu_cholmod:
target_link_libraries ( my PRIVATE ${KLU_CHOLMOD_LIBRARIES} )
target_link_libraries ( my_static PUBLIC ${KLU_CHOLMOD_STATIC} )
target_link_libraries ( my_cxx PUBLIC ${KLU_CHOLMOD_LIBRARIES} )
target_link_libraries ( my_cxx_static PUBLIC ${KLU_CHOLMOD_STATIC} )
include_directories ( ${KLU_CHOLMOD_INCLUDE_DIR} )

# ldl:
target_link_libraries ( my PRIVATE ${LDL_LIBRARIES} )
target_link_libraries ( my_static PUBLIC ${LDL_STATIC} )
target_link_libraries ( my_cxx PUBLIC ${LDL_LIBRARIES} )
target_link_libraries ( my_cxx_static PUBLIC ${LDL_STATIC} )
include_directories ( ${LDL_INCLUDE_DIR} )

# mongoose:
target_link_libraries ( my PRIVATE ${MONGOOSE_LIBRARIES} )
target_link_libraries ( my_static PUBLIC ${MONGOSE_STATIC} )
target_link_libraries ( my_cxx PUBLIC ${MONGOOSE_LIBRARIES} )
target_link_libraries ( my_cxx_static PUBLIC ${MONGOSE_STATIC} )
include_directories ( ${MONGOOSE_INCLUDE_DIR} )

# rbio:
target_link_libraries ( my PRIVATE ${RBIO_LIBRARIES} )
target_link_libraries ( my_static PUBLIC ${RBIO_STATIC} )
target_link_libraries ( my_cxx PUBLIC ${RBIO_LIBRARIES} )
target_link_libraries ( my_cxx_static PUBLIC ${RBIO_STATIC} )
include_directories ( ${RBIO_INCLUDE_DIR} )

# spex:
target_link_libraries ( my PRIVATE ${SPEX_LIBRARIES} )
target_link_libraries ( my_static PUBLIC ${SPEX_STATIC} )
target_link_libraries ( my_cxx PUBLIC ${SPEX_LIBRARIES} )
target_link_libraries ( my_cxx_static PUBLIC ${SPEX_STATIC} )
include_directories ( ${SPEX_INCLUDE_DIR} )

# spqr:
target_link_libraries ( my PRIVATE ${SPQR_LIBRARIES} )
target_link_libraries ( my_static PUBLIC ${SPQR_STATIC} )
target_link_libraries ( my_cxx PUBLIC ${SPQR_LIBRARIES} )
target_link_libraries ( my_cxx_static PUBLIC ${SPQR_STATIC} )
include_directories ( ${SPQR_INCLUDE_DIR} )

# suitesparseconfig:
target_link_libraries ( my PRIVATE ${SUITESPARSE_CONFIG_LIBRARIES} )
target_link_libraries ( my_static PUBLIC ${SUITESPARSE_CONFIG_STATIC} )
target_link_libraries ( my_cxx PUBLIC ${SUITESPARSE_CONFIG_LIBRARIES} )
target_link_libraries ( my_cxx_static PUBLIC ${SUITESPARSE_CONFIG_STATIC} )
include_directories ( ${SUITESPARSE_CONFIG_INCLUDE_DIR} )

# suitesparse_gpuruntime:
target_link_libraries ( my PRIVATE ${SUITESPARSE_GPURUNTIME_LIBRARIES} )
target_link_libraries ( my_static PUBLIC ${SUITESPARSE_GPURUNTIME_STATIC} )
target_link_libraries ( my_cxx PUBLIC ${SUITESPARSE_GPURUNTIME_LIBRARIES} )
target_link_libraries ( my_cxx_static PUBLIC ${SUITESPARSE_GPURUNTIME_STATIC} )
include_directories ( ${SUITESPARSE_GPURUNTIME_INCLUDE_DIR} )

# umfpack:
target_link_libraries ( my PRIVATE ${UMFPACK_LIBRARIES} )
target_link_libraries ( my_static PUBLIC ${UMFPACK_STATIC} )
target_link_libraries ( my_cxx PUBLIC ${UMFPACK_LIBRARIES} )
target_link_libraries ( my_cxx_static PUBLIC ${UMFPACK_STATIC} )
include_directories ( ${UMFPACK_INCLUDE_DIR} )

#-------------------------------------------------------------------------------
# installation location
#-------------------------------------------------------------------------------

install ( TARGETS my
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} )
install ( TARGETS my_static
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} )
install ( TARGETS my_cxx
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} )
install ( TARGETS my_cxx_static
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} )

#-------------------------------------------------------------------------------
# Demo program
#-------------------------------------------------------------------------------

add_executable ( my_demo "Demo/my_demo.c" )
target_link_libraries ( my_demo PUBLIC my )

add_executable ( my_cxx_demo "Demo/my_demo.c" )
target_link_libraries ( my_cxx_demo PUBLIC my_cxx )
