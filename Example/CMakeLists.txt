#-------------------------------------------------------------------------------
# SuiteSparse/Example/CMakeLists.txt:  cmake for Example user library, "my"
#-------------------------------------------------------------------------------

# Copyright (c) 2022-2023, Timothy A. Davis, All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

#-------------------------------------------------------------------------------
# get the version
#-------------------------------------------------------------------------------

cmake_minimum_required ( VERSION 3.20 )

# cmake inserts the date and version number into Include/my.h:
set ( MY_DATE "Oct 23, 2023" )
set ( MY_VERSION_MAJOR 1 )
set ( MY_VERSION_MINOR 4 )
set ( MY_VERSION_PATCH 3 )

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
# add path to cmake target files (if necessary)
set ( CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH}
    ${CMAKE_SOURCE_DIR}/../lib/cmake )

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
find_package ( SuiteSparse_config 7.3.0 REQUIRED )
find_package ( AMD 3.2.1 REQUIRED )
find_package ( BTF 2.2.1 REQUIRED )
find_package ( CAMD 3.2.1 REQUIRED )
find_package ( CCOLAMD 3.2.1 REQUIRED )
find_package ( COLAMD 3.2.1 REQUIRED )
find_package ( CHOLMOD 5.0.0 REQUIRED )
find_package ( CXSparse 4.2.1 REQUIRED )
find_package ( GraphBLAS 8.2.1 )
find_package ( KLU 2.2.2 REQUIRED )
find_package ( KLU_CHOLMOD 2.2.2 REQUIRED )
find_package ( LDL 3.2.1 REQUIRED )
find_package ( Mongoose 3.2.1 REQUIRED )
find_package ( RBio 4.2.1 REQUIRED )
find_package ( SPEX 2.2.1 REQUIRED )
find_package ( SPQR 4.2.2 REQUIRED )
find_package ( UMFPACK 6.2.2 REQUIRED )

if ( TARGET CUDA::nvrtc )
    # CHOLMOD and SPQR have been compiled with CUDA enabled.
    find_package ( SuiteSparse_GPURuntime 3.2.1 REQUIRED )
    find_package ( GPUQREngine 3.2.2 REQUIRED )
endif ( )

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
target_link_libraries ( my PRIVATE OpenMP::OpenMP_C )
target_link_libraries ( my_static PUBLIC OpenMP::OpenMP_C )
target_link_libraries ( my_cxx PRIVATE OpenMP::OpenMP_CXX )
target_link_libraries ( my_cxx_static PUBLIC OpenMP::OpenMP_CXX )

# libm:
if ( NOT WIN32 )
    target_link_libraries ( my PRIVATE m )
    target_link_libraries ( my_static PUBLIC m )
    target_link_libraries ( my_cxx PRIVATE m )
    target_link_libraries ( my_cxx_static PUBLIC m )
endif ( )

# BLAS:
message ( STATUS "BLAS libraries:      ${BLAS_LIBRARIES} ")
message ( STATUS "BLAS include:        ${BLAS_INCLUDE_DIRS} ")
message ( STATUS "BLAS linker flags:   ${BLAS_LINKER_FLAGS} ")
target_link_libraries ( my PRIVATE ${BLAS_LIBRARIES} )
target_link_libraries ( my_static PUBLIC ${BLAS_LIBRARIES} )
target_link_libraries ( my_cxx PRIVATE ${BLAS_LIBRARIES} )
target_link_libraries ( my_cxx_static PUBLIC ${BLAS_LIBRARIES} )
include_directories ( ${BLAS_INCLUDE_DIRS} )

# LAPACK:
message ( STATUS "LAPACK libraries:    ${LAPACK_LIBRARIES} ")
message ( STATUS "LAPACK include:      ${LAPACK_INCLUDE_DIRS} ")
message ( STATUS "LAPACK linker flags: ${LAPACK_LINKER_FLAGS} ")
target_link_libraries ( my PRIVATE ${LAPACK_LIBRARIES} )
target_link_libraries ( my_static PUBLIC ${LAPACK_LIBRARIES} )
target_link_libraries ( my_cxx PRIVATE ${LAPACK_LIBRARIES} )
target_link_libraries ( my_cxx_static PUBLIC ${LAPACK_LIBRARIES} )
include_directories ( ${LAPACK_INCLUDE_DIRS} )

# gmp:
target_link_libraries ( my PRIVATE ${GMP_LIBRARIES} )
target_link_libraries ( my_static PUBLIC ${GMP_STATIC} )
target_link_libraries ( my_cxx PRIVATE ${GMP_LIBRARIES} )
target_link_libraries ( my_cxx_static PUBLIC ${GMP_STATIC} )
include_directories ( ${GMP_INCLUDE_DIR} )

# mpfr:
target_link_libraries ( my PRIVATE ${MPFR_LIBRARIES} )
target_link_libraries ( my_static PUBLIC ${MPFR_STATIC} )
target_link_libraries ( my_cxx PRIVATE ${MPFR_LIBRARIES} )
target_link_libraries ( my_cxx_static PUBLIC ${MPFR_STATIC} )
include_directories ( ${MPFR_INCLUDE_DIR} )

# AMD:
target_link_libraries ( my PRIVATE SuiteSparse::AMD )
target_link_libraries ( my_cxx PRIVATE SuiteSparse::AMD )
if ( TARGET SuiteSparse::AMD_static )
    target_link_libraries ( my_static PUBLIC SuiteSparse::AMD_static )
    target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::AMD_static )
else ( )
    target_link_libraries ( my_static PUBLIC SuiteSparse::AMD )
    target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::AMD )
endif ( )

# BTF:
target_link_libraries ( my PRIVATE SuiteSparse::BTF )
target_link_libraries ( my_cxx PRIVATE SuiteSparse::BTF )
if ( TARGET SuiteSparse::BTF_static )
    target_link_libraries ( my_static PUBLIC SuiteSparse::BTF_static )
    target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::BTF_static )
else ( )
    target_link_libraries ( my_static PUBLIC SuiteSparse::BTF )
    target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::BTF )
endif ( )

# CAMD:
target_link_libraries ( my PRIVATE SuiteSparse::CAMD )
target_link_libraries ( my_cxx PRIVATE SuiteSparse::CAMD )
if ( TARGET SuiteSparse::CAMD_static )
    target_link_libraries ( my_static PUBLIC SuiteSparse::CAMD_static )
    target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::CAMD_static )
else ( )
    target_link_libraries ( my_static PUBLIC SuiteSparse::CAMD )
    target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::CAMD )
endif ( )

# CCOLAMD:
target_link_libraries ( my PRIVATE SuiteSparse::CCOLAMD )
target_link_libraries ( my_cxx PRIVATE SuiteSparse::CCOLAMD )
if ( TARGET SuiteSparse::CCOLAMD_static )
    target_link_libraries ( my_static PUBLIC SuiteSparse::CCOLAMD_static )
    target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::CCOLAMD_static )
else ( )
    target_link_libraries ( my_static PUBLIC SuiteSparse::CCOLAMD )
    target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::CCOLAMD )
endif ( )

# CHOLMOD:
target_link_libraries ( my PRIVATE SuiteSparse::CHOLMOD )
target_link_libraries ( my_cxx PRIVATE SuiteSparse::CHOLMOD )
if ( TARGET SuiteSparse::CHOLMOD_static )
    target_link_libraries ( my_static PUBLIC SuiteSparse::CHOLMOD_static )
    target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::CHOLMOD_static )
else ( )
    target_link_libraries ( my_static PUBLIC SuiteSparse::CHOLMOD )
    target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::CHOLMOD )
endif ( )

# COLAMD:
target_link_libraries ( my PRIVATE SuiteSparse::COLAMD )
target_link_libraries ( my_cxx PRIVATE SuiteSparse::COLAMD )
if ( TARGET SuiteSparse::COLAMD_static )
    target_link_libraries ( my_static PUBLIC SuiteSparse::COLAMD_static )
    target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::COLAMD_static )
else ( )
    target_link_libraries ( my_static PUBLIC SuiteSparse::COLAMD )
    target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::COLAMD )
endif ( )

# CXSparse:
target_link_libraries ( my PRIVATE SuiteSparse::CXSparse )
target_link_libraries ( my_cxx PRIVATE SuiteSparse::CXSparse )
if ( TARGET SuiteSparse::CXSparse_static )
    target_link_libraries ( my_static PUBLIC SuiteSparse::CXSparse_static )
    target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::CXSparse_static )
else ( )
    target_link_libraries ( my_static PUBLIC SuiteSparse::CXSparse )
    target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::CXSparse )
endif ( )

# GPUQREngine:
if ( ENABLE_CUDA )
    target_link_libraries ( my PRIVATE SuiteSparse::GPUQREngine )
    target_link_libraries ( my_cxx PRIVATE SuiteSparse::GPUQREngine )
    if ( TARGET SuiteSparse::GPUQREngine_static )
        target_link_libraries ( my_static PUBLIC SuiteSparse::GPUQREngine_static )
        target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::GPUQREngine_static )
    else ( )
        target_link_libraries ( my_static PUBLIC SuiteSparse::GPUQREngine )
        target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::GPUQREngine )
    endif ( )
endif ( )

# GraphBLAS:
if ( TARGET SuiteSparse::GraphBLAS )
    target_link_libraries ( my PRIVATE SuiteSparse::GraphBLAS )
    target_link_libraries ( my_cxx PRIVATE SuiteSparse::GraphBLAS )
    if ( TARGET SuiteSparse::GraphBLAS_static )
        target_link_libraries ( my_static PUBLIC SuiteSparse::GraphBLAS_static )
        target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::GraphBLAS_static )
    else ( )
        target_link_libraries ( my_static PUBLIC SuiteSparse::GraphBLAS )
        target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::GraphBLAS )
    endif ( )
else ( )
    target_compile_definitions ( my PRIVATE NO_GRAPHBLAS )
    target_compile_definitions ( my_cxx PRIVATE NO_GRAPHBLAS )
    target_compile_definitions ( my_static PRIVATE NO_GRAPHBLAS )
    target_compile_definitions ( my_cxx_static PRIVATE NO_GRAPHBLAS )
endif ( )

# KLU:
target_link_libraries ( my PRIVATE SuiteSparse::KLU )
target_link_libraries ( my_cxx PRIVATE SuiteSparse::KLU )
if ( TARGET SuiteSparse::KLU_static )
    target_link_libraries ( my_static PUBLIC SuiteSparse::KLU_static )
    target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::KLU_static )
else ( )
    target_link_libraries ( my_static PUBLIC SuiteSparse::KLU )
    target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::KLU )
endif ( )

# KLU_CHOLMOD:
target_link_libraries ( my PRIVATE SuiteSparse::KLU_CHOLMOD )
target_link_libraries ( my_cxx PRIVATE SuiteSparse::KLU_CHOLMOD )
if ( TARGET SuiteSparse::KLU_CHOLMOD_static )
    target_link_libraries ( my_static PUBLIC SuiteSparse::KLU_CHOLMOD_static )
    target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::KLU_CHOLMOD_static )
else ( )
    target_link_libraries ( my_static PUBLIC SuiteSparse::KLU_CHOLMOD )
    target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::KLU_CHOLMOD )
endif ( )

# LDL:
target_link_libraries ( my PRIVATE SuiteSparse::LDL )
target_link_libraries ( my_cxx PRIVATE SuiteSparse::LDL )
if ( TARGET SuiteSparse::LDL_static )
    target_link_libraries ( my_static PUBLIC SuiteSparse::LDL_static )
    target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::LDL_static )
else ( )
    target_link_libraries ( my_static PUBLIC SuiteSparse::LDL )
    target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::LDL )
endif ( )

# Mongoose:
target_link_libraries ( my PRIVATE SuiteSparse::Mongoose )
target_link_libraries ( my_cxx PRIVATE SuiteSparse::Mongoose )
if ( TARGET SuiteSparse::Mongoose_static )
    target_link_libraries ( my_static PUBLIC SuiteSparse::Mongoose_static )
    target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::Mongoose_static )
else ( )
    target_link_libraries ( my_static PUBLIC SuiteSparse::Mongoose )
    target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::Mongoose )
endif ( )

# RBio:
target_link_libraries ( my PRIVATE SuiteSparse::RBio )
target_link_libraries ( my_cxx PRIVATE SuiteSparse::RBio )
if ( TARGET SuiteSparse::RBio_static )
    target_link_libraries ( my_static PUBLIC SuiteSparse::RBio_static )
    target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::RBio_static )
else ( )
    target_link_libraries ( my_static PUBLIC SuiteSparse::RBio )
    target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::RBio )
endif ( )

# SPEX:
target_link_libraries ( my PRIVATE SuiteSparse::SPEX )
target_link_libraries ( my_cxx PRIVATE SuiteSparse::SPEX )
if ( TARGET SuiteSparse::SPEX_static )
    target_link_libraries ( my_static PUBLIC SuiteSparse::SPEX_static )
    target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::SPEX_static )
else ( )
    target_link_libraries ( my_static PUBLIC SuiteSparse::SPEX )
    target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::SPEX )
endif ( )

# SPQR:
target_link_libraries ( my PRIVATE SuiteSparse::SPQR )
target_link_libraries ( my_cxx PRIVATE SuiteSparse::SPQR )
if ( TARGET SuiteSparse::SPQR_static )
    target_link_libraries ( my_static PUBLIC SuiteSparse::SPQR_static )
    target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::SPQR_static )
else ( )
    target_link_libraries ( my_static PUBLIC SuiteSparse::SPQR )
    target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::SPQR )
endif ( )

# SuiteSparseConfig:
target_link_libraries ( my PRIVATE SuiteSparse::SuiteSparseConfig )
target_link_libraries ( my_cxx PRIVATE SuiteSparse::SuiteSparseConfig )
if ( TARGET SuiteSparse::SuiteSparseConfig_static )
    target_link_libraries ( my_static PUBLIC SuiteSparse::SuiteSparseConfig_static )
    target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::SuiteSparseConfig_static )
else ( )
    target_link_libraries ( my_static PUBLIC SuiteSparse::SuiteSparseConfig )
    target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::SuiteSparseConfig )
endif ( )

# GPURuntime:
if ( ENABLE_CUDA )
    target_link_libraries ( my PRIVATE SuiteSparse::GPURuntime )
    target_link_libraries ( my_cxx PRIVATE SuiteSparse::GPURuntime )
    if ( TARGET SuiteSparse::GPURuntime_static )
        target_link_libraries ( my_static PUBLIC SuiteSparse::GPURuntime_static )
        target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::GPURuntime_static )
    else ( )
        target_link_libraries ( my_static PUBLIC SuiteSparse::GPURuntime )
        target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::GPURuntime )
    endif ( )
endif ( )

# UMFPACK:
target_link_libraries ( my PRIVATE SuiteSparse::UMFPACK )
target_link_libraries ( my_cxx PRIVATE SuiteSparse::UMFPACK )
if ( TARGET SuiteSparse::UMFPACK_static )
    target_link_libraries ( my_static PUBLIC SuiteSparse::UMFPACK_static )
    target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::UMFPACK_static )
else ( )
    target_link_libraries ( my_static PUBLIC SuiteSparse::UMFPACK )
    target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::UMFPACK )
endif ( )

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
