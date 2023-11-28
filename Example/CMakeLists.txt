#-------------------------------------------------------------------------------
# SuiteSparse/Example/CMakeLists.txt:  cmake for Example user library, "my"
#-------------------------------------------------------------------------------

# Copyright (c) 2022-2023, Timothy A. Davis, All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

# To build this example, add the location of the CMake config files to
# CMAKE_PREFIX_PATH if the SuiteSparse libraries aren't installed at a
# default location.

#-------------------------------------------------------------------------------
# get the version
#-------------------------------------------------------------------------------

cmake_minimum_required ( VERSION 3.20 )

# cmake inserts the date and version number into Include/my.h:
set ( MY_DATE "Dec 30, 2023" )
set ( MY_VERSION_MAJOR 1 )
set ( MY_VERSION_MINOR 5 )
set ( MY_VERSION_PATCH 0 )

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

option ( BUILD_SHARED_LIBS "OFF: do not build shared libraries.  ON (default): build shared libraries" ON )
option ( BUILD_STATIC_LIBS "OFF: do not build static libraries.  ON (default): build static libraries" ON )

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
find_package ( SuiteSparse_config 7.4.0 REQUIRED )
find_package ( AMD 3.3.0 REQUIRED )
find_package ( BTF 2.3.0 REQUIRED )
find_package ( CAMD 3.3.0 REQUIRED )
find_package ( CCOLAMD 3.3.0 REQUIRED )
find_package ( CHOLMOD 5.1.0 REQUIRED )
find_package ( COLAMD 3.3.0 REQUIRED )
find_package ( CXSparse 4.3.0 REQUIRED )
find_package ( GraphBLAS 8.3.0 )
find_package ( KLU 2.3.0 REQUIRED )
find_package ( KLU_CHOLMOD 2.3.0 REQUIRED )
find_package ( LDL 3.3.0 REQUIRED )
find_package ( LAGraph 1.1.0 )
find_package ( Mongoose 3.3.0 REQUIRED )
find_package ( ParU 0.1.0 REQUIRED )
find_package ( RBio 4.3.0 REQUIRED )
find_package ( SPEX 2.3.0 REQUIRED )
find_package ( SPQR 4.3.0 REQUIRED )
find_package ( UMFPACK 6.3.0 REQUIRED )

# look for all external libaries:
find_package ( OpenMP REQUIRED )
find_package ( BLAS REQUIRED )
find_package ( LAPACK REQUIRED )

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
if ( BUILD_SHARED_LIBS )
    add_library ( my SHARED ${MY_SOURCES} )
    set_target_properties ( my PROPERTIES
        VERSION ${MY_VERSION_MAJOR}.${MY_VERSION_MINOR}.${MY_VERSION_PATCH}
        C_STANDARD 11
        C_STANDARD_REQUIRED ON
        SOVERSION ${MY_VERSION_MAJOR}
        PUBLIC_HEADER "Include/my.h"
        WINDOWS_EXPORT_ALL_SYMBOLS ON )
endif ( )

#-------------------------------------------------------------------------------
# my static C library properties
#-------------------------------------------------------------------------------

if ( BUILD_STATIC_LIBS )
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
endif ( )

#-------------------------------------------------------------------------------
# my dynamic C++ library properties
#-------------------------------------------------------------------------------

file ( GLOB MY_SOURCES "Source/*.cc" )
if ( BUILD_SHARED_LIBS )
    add_library ( my_cxx SHARED ${MY_SOURCES} )
    set_target_properties ( my_cxx PROPERTIES
        VERSION ${MY_VERSION_MAJOR}.${MY_VERSION_MINOR}.${MY_VERSION_PATCH}
        CXX_STANDARD 11
        CXX_STANDARD_REQUIRED ON
        SOVERSION ${MY_VERSION_MAJOR}
        PUBLIC_HEADER "Include/my.h"
        WINDOWS_EXPORT_ALL_SYMBOLS ON )
endif ( )

#-------------------------------------------------------------------------------
# my static C++ library properties
#-------------------------------------------------------------------------------

if ( BUILD_STATIC_LIBS )
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
endif ( )

#-------------------------------------------------------------------------------
# add the library dependencies
#-------------------------------------------------------------------------------

# OpenMP:
message ( STATUS "OpenMP C libraries:      ${OpenMP_C_LIBRARIES} ")
message ( STATUS "OpenMP C include:        ${OpenMP_C_INCLUDE_DIRS} ")
message ( STATUS "OpenMP C flags:          ${OpenMP_C_FLAGS} ")
if ( BUILD_SHARED_LIBS )
    target_link_libraries ( my PRIVATE OpenMP::OpenMP_C )
    target_link_libraries ( my_cxx PRIVATE OpenMP::OpenMP_CXX )
endif ( )
if ( BUILD_STATIC_LIBS )
    target_link_libraries ( my_static PUBLIC OpenMP::OpenMP_C )
    target_link_libraries ( my_cxx_static PUBLIC OpenMP::OpenMP_CXX )
endif ( )

# libm:
if ( NOT WIN32 )
    if ( BUILD_SHARED_LIBS )
        target_link_libraries ( my PRIVATE m )
        target_link_libraries ( my_cxx PRIVATE m )
    endif ( )
    if ( BUILD_STATIC_LIBS )
        target_link_libraries ( my_static PUBLIC m )
        target_link_libraries ( my_cxx_static PUBLIC m )
    endif ( )
endif ( )

# BLAS:
message ( STATUS "BLAS libraries:      ${BLAS_LIBRARIES} ")
message ( STATUS "BLAS include:        ${BLAS_INCLUDE_DIRS} ")
message ( STATUS "BLAS linker flags:   ${BLAS_LINKER_FLAGS} ")
if ( BUILD_SHARED_LIBS )
    target_link_libraries ( my PRIVATE ${BLAS_LIBRARIES} )
    target_link_libraries ( my_cxx PRIVATE ${BLAS_LIBRARIES} )
endif ( )
if ( BUILD_STATIC_LIBS )
    target_link_libraries ( my_static PUBLIC ${BLAS_LIBRARIES} )
    target_link_libraries ( my_cxx_static PUBLIC ${BLAS_LIBRARIES} )
endif ( )
include_directories ( ${BLAS_INCLUDE_DIRS} )

# LAPACK:
message ( STATUS "LAPACK libraries:    ${LAPACK_LIBRARIES} ")
message ( STATUS "LAPACK include:      ${LAPACK_INCLUDE_DIRS} ")
message ( STATUS "LAPACK linker flags: ${LAPACK_LINKER_FLAGS} ")
if ( BUILD_SHARED_LIBS )
    target_link_libraries ( my PRIVATE ${LAPACK_LIBRARIES} )
    target_link_libraries ( my_cxx PRIVATE ${LAPACK_LIBRARIES} )
endif ( )
if ( BUILD_STATIC_LIBS )
    target_link_libraries ( my_static PUBLIC ${LAPACK_LIBRARIES} )
    target_link_libraries ( my_cxx_static PUBLIC ${LAPACK_LIBRARIES} )
endif ( )
include_directories ( ${LAPACK_INCLUDE_DIRS} )

# gmp:
if ( BUILD_SHARED_LIBS )
    target_link_libraries ( my PRIVATE ${GMP_LIBRARIES} )
    target_link_libraries ( my_cxx PRIVATE ${GMP_LIBRARIES} )
endif ( )
if ( BUILD_STATIC_LIBS )
    target_link_libraries ( my_static PUBLIC ${GMP_STATIC} )
    target_link_libraries ( my_cxx_static PUBLIC ${GMP_STATIC} )
endif ( )
include_directories ( ${GMP_INCLUDE_DIR} )

# mpfr:
if ( BUILD_SHARED_LIBS )
    target_link_libraries ( my PRIVATE ${MPFR_LIBRARIES} )
    target_link_libraries ( my_cxx PRIVATE ${MPFR_LIBRARIES} )
endif ( )
if ( BUILD_STATIC_LIBS )
    target_link_libraries ( my_static PUBLIC ${MPFR_STATIC} )
    target_link_libraries ( my_cxx_static PUBLIC ${MPFR_STATIC} )
endif ( )
include_directories ( ${MPFR_INCLUDE_DIR} )

# AMD:
if ( BUILD_SHARED_LIBS )
    target_link_libraries ( my PRIVATE SuiteSparse::AMD )
    target_link_libraries ( my_cxx PRIVATE SuiteSparse::AMD )
endif ( )
if ( BUILD_STATIC_LIBS )
    if ( TARGET SuiteSparse::AMD_static )
        target_link_libraries ( my_static PUBLIC SuiteSparse::AMD_static )
        target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::AMD_static )
    else ( )
        target_link_libraries ( my_static PUBLIC SuiteSparse::AMD )
        target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::AMD )
    endif ( )
endif ( )

# BTF:
if ( BUILD_SHARED_LIBS )
    target_link_libraries ( my PRIVATE SuiteSparse::BTF )
    target_link_libraries ( my_cxx PRIVATE SuiteSparse::BTF )
endif ( )
if ( BUILD_STATIC_LIBS )
    if ( TARGET SuiteSparse::BTF_static )
        target_link_libraries ( my_static PUBLIC SuiteSparse::BTF_static )
        target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::BTF_static )
    else ( )
        target_link_libraries ( my_static PUBLIC SuiteSparse::BTF )
        target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::BTF )
    endif ( )
endif ( )

# CAMD:
if ( BUILD_SHARED_LIBS )
    target_link_libraries ( my PRIVATE SuiteSparse::CAMD )
    target_link_libraries ( my_cxx PRIVATE SuiteSparse::CAMD )
endif ( )
if ( BUILD_STATIC_LIBS )
    if ( TARGET SuiteSparse::CAMD_static )
        target_link_libraries ( my_static PUBLIC SuiteSparse::CAMD_static )
        target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::CAMD_static )
    else ( )
        target_link_libraries ( my_static PUBLIC SuiteSparse::CAMD )
        target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::CAMD )
    endif ( )
endif ( )

# CCOLAMD:
if ( BUILD_SHARED_LIBS )
    target_link_libraries ( my PRIVATE SuiteSparse::CCOLAMD )
    target_link_libraries ( my_cxx PRIVATE SuiteSparse::CCOLAMD )
endif ( )
if ( BUILD_STATIC_LIBS )
    if ( TARGET SuiteSparse::CCOLAMD_static )
        target_link_libraries ( my_static PUBLIC SuiteSparse::CCOLAMD_static )
        target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::CCOLAMD_static )
    else ( )
        target_link_libraries ( my_static PUBLIC SuiteSparse::CCOLAMD )
        target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::CCOLAMD )
    endif ( )
endif ( )

# CHOLMOD:
if ( BUILD_SHARED_LIBS )
    target_link_libraries ( my PRIVATE SuiteSparse::CHOLMOD )
    target_link_libraries ( my_cxx PRIVATE SuiteSparse::CHOLMOD )
endif ( )
if ( BUILD_STATIC_LIBS )
    if ( TARGET SuiteSparse::CHOLMOD_static )
        target_link_libraries ( my_static PUBLIC SuiteSparse::CHOLMOD_static )
        target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::CHOLMOD_static )
    else ( )
        target_link_libraries ( my_static PUBLIC SuiteSparse::CHOLMOD )
        target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::CHOLMOD )
    endif ( )
endif ( )

# COLAMD:
if ( BUILD_SHARED_LIBS )
    target_link_libraries ( my PRIVATE SuiteSparse::COLAMD )
    target_link_libraries ( my_cxx PRIVATE SuiteSparse::COLAMD )
endif ( )
if ( BUILD_STATIC_LIBS )
    if ( TARGET SuiteSparse::COLAMD_static )
        target_link_libraries ( my_static PUBLIC SuiteSparse::COLAMD_static )
        target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::COLAMD_static )
    else ( )
        target_link_libraries ( my_static PUBLIC SuiteSparse::COLAMD )
        target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::COLAMD )
    endif ( )
endif ( )

# CXSparse:
if ( BUILD_SHARED_LIBS )
    target_link_libraries ( my PRIVATE SuiteSparse::CXSparse )
    target_link_libraries ( my_cxx PRIVATE SuiteSparse::CXSparse )
endif ( )
if ( BUILD_STATIC_LIBS )
    if ( TARGET SuiteSparse::CXSparse_static )
        target_link_libraries ( my_static PUBLIC SuiteSparse::CXSparse_static )
        target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::CXSparse_static )
    else ( )
        target_link_libraries ( my_static PUBLIC SuiteSparse::CXSparse )
        target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::CXSparse )
    endif ( )
endif ( )

# GraphBLAS:
if ( TARGET SuiteSparse::GraphBLAS )
    if ( BUILD_SHARED_LIBS )
        target_link_libraries ( my PRIVATE SuiteSparse::GraphBLAS )
        target_link_libraries ( my_cxx PRIVATE SuiteSparse::GraphBLAS )
    endif ( )
    if ( BUILD_STATIC_LIBS )
        if ( TARGET SuiteSparse::GraphBLAS_static )
            target_link_libraries ( my_static PUBLIC SuiteSparse::GraphBLAS_static )
            target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::GraphBLAS_static )
        else ( )
            target_link_libraries ( my_static PUBLIC SuiteSparse::GraphBLAS )
            target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::GraphBLAS )
        endif ( )
    endif ( )
else ( )
    if ( BUILD_SHARED_LIBS )
        target_compile_definitions ( my PRIVATE NO_GRAPHBLAS )
        target_compile_definitions ( my_cxx PRIVATE NO_GRAPHBLAS )
    endif ( )
    if ( BUILD_STATIC_LIBS )
        target_compile_definitions ( my_static PRIVATE NO_GRAPHBLAS )
        target_compile_definitions ( my_cxx_static PRIVATE NO_GRAPHBLAS )
    endif ( )
endif ( )

# KLU:
if ( BUILD_SHARED_LIBS )
    target_link_libraries ( my PRIVATE SuiteSparse::KLU )
    target_link_libraries ( my_cxx PRIVATE SuiteSparse::KLU )
endif ( )
if ( BUILD_STATIC_LIBS )
    if ( TARGET SuiteSparse::KLU_static )
        target_link_libraries ( my_static PUBLIC SuiteSparse::KLU_static )
        target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::KLU_static )
    else ( )
        target_link_libraries ( my_static PUBLIC SuiteSparse::KLU )
        target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::KLU )
    endif ( )
endif ( )

# KLU_CHOLMOD:
if ( BUILD_SHARED_LIBS )
    target_link_libraries ( my PRIVATE SuiteSparse::KLU_CHOLMOD )
    target_link_libraries ( my_cxx PRIVATE SuiteSparse::KLU_CHOLMOD )
endif ( )
if ( BUILD_STATIC_LIBS )
    if ( TARGET SuiteSparse::KLU_CHOLMOD_static )
        target_link_libraries ( my_static PUBLIC SuiteSparse::KLU_CHOLMOD_static )
        target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::KLU_CHOLMOD_static )
    else ( )
        target_link_libraries ( my_static PUBLIC SuiteSparse::KLU_CHOLMOD )
        target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::KLU_CHOLMOD )
    endif ( )
endif ( )

# LAGraph:
if ( TARGET SuiteSparse::LAGraph )
    if ( BUILD_SHARED_LIBS )
        target_link_libraries ( my PRIVATE SuiteSparse::LAGraph )
        target_link_libraries ( my_cxx PRIVATE SuiteSparse::LAGraph )
    endif ( )
    if ( BUILD_STATIC_LIBS )
        if ( TARGET SuiteSparse::LAGraph_static )
            target_link_libraries ( my_static PUBLIC SuiteSparse::LAGraph_static )
            target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::LAGraph_static )
        else ( )
            target_link_libraries ( my_static PUBLIC SuiteSparse::LAGraph )
            target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::LAGraph )
        endif ( )
    endif ( )
else ( )
    if ( BUILD_SHARED_LIBS )
        target_compile_definitions ( my PRIVATE NO_LAGRAPH )
        target_compile_definitions ( my_cxx PRIVATE NO_LAGRAPH )
    endif ( )
    if ( BUILD_STATIC_LIBS )
        target_compile_definitions ( my_static PRIVATE NO_LAGRAPH )
        target_compile_definitions ( my_cxx_static PRIVATE NO_LAGRAPH )
    endif ( )
endif ( )

# LDL:
if ( BUILD_SHARED_LIBS )
    target_link_libraries ( my PRIVATE SuiteSparse::LDL )
    target_link_libraries ( my_cxx PRIVATE SuiteSparse::LDL )
endif ( )
if ( BUILD_STATIC_LIBS )
    if ( TARGET SuiteSparse::LDL_static )
        target_link_libraries ( my_static PUBLIC SuiteSparse::LDL_static )
        target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::LDL_static )
    else ( )
        target_link_libraries ( my_static PUBLIC SuiteSparse::LDL )
        target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::LDL )
    endif ( )
endif ( )

# Mongoose:
if ( BUILD_SHARED_LIBS )
    target_link_libraries ( my PRIVATE SuiteSparse::Mongoose )
    target_link_libraries ( my_cxx PRIVATE SuiteSparse::Mongoose )
endif ( )
if ( BUILD_STATIC_LIBS )
    if ( TARGET SuiteSparse::Mongoose_static )
        target_link_libraries ( my_static PUBLIC SuiteSparse::Mongoose_static )
        target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::Mongoose_static )
    else ( )
        target_link_libraries ( my_static PUBLIC SuiteSparse::Mongoose )
        target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::Mongoose )
    endif ( )
endif ( )

# ParU:
if ( BUILD_SHARED_LIBS )
    target_link_libraries ( my PRIVATE SuiteSparse::ParU )
    target_link_libraries ( my_cxx PRIVATE SuiteSparse::ParU )
endif ( )
if ( BUILD_STATIC_LIBS )
    if ( TARGET SuiteSparse::ParU_static )
        target_link_libraries ( my_static PUBLIC SuiteSparse::ParU_static )
        target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::ParU_static )
    else ( )
        target_link_libraries ( my_static PUBLIC SuiteSparse::ParU )
        target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::ParU )
    endif ( )
endif ( )

# RBio:
if ( BUILD_SHARED_LIBS )
    target_link_libraries ( my PRIVATE SuiteSparse::RBio )
    target_link_libraries ( my_cxx PRIVATE SuiteSparse::RBio )
endif ( )
if ( BUILD_STATIC_LIBS )
    if ( TARGET SuiteSparse::RBio_static )
        target_link_libraries ( my_static PUBLIC SuiteSparse::RBio_static )
        target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::RBio_static )
    else ( )
        target_link_libraries ( my_static PUBLIC SuiteSparse::RBio )
        target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::RBio )
    endif ( )
endif ( )

# SPEX:
if ( BUILD_SHARED_LIBS )
    target_link_libraries ( my PRIVATE SuiteSparse::SPEX )
    target_link_libraries ( my_cxx PRIVATE SuiteSparse::SPEX )
endif ( )
if ( BUILD_STATIC_LIBS )
    if ( TARGET SuiteSparse::SPEX_static )
        target_link_libraries ( my_static PUBLIC SuiteSparse::SPEX_static )
        target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::SPEX_static )
    else ( )
        target_link_libraries ( my_static PUBLIC SuiteSparse::SPEX )
        target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::SPEX )
    endif ( )
endif ( )

# SPQR:
if ( BUILD_SHARED_LIBS )
    target_link_libraries ( my PRIVATE SuiteSparse::SPQR )
    target_link_libraries ( my_cxx PRIVATE SuiteSparse::SPQR )
endif ( )
if ( BUILD_STATIC_LIBS )
    if ( TARGET SuiteSparse::SPQR_static )
        target_link_libraries ( my_static PUBLIC SuiteSparse::SPQR_static )
        target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::SPQR_static )
    else ( )
        target_link_libraries ( my_static PUBLIC SuiteSparse::SPQR )
        target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::SPQR )
    endif ( )
endif ( )

# SuiteSparseConfig:
if ( BUILD_SHARED_LIBS )
    target_link_libraries ( my PRIVATE SuiteSparse::SuiteSparseConfig )
    target_link_libraries ( my_cxx PRIVATE SuiteSparse::SuiteSparseConfig )
endif ( )
if ( BUILD_STATIC_LIBS )
    if ( TARGET SuiteSparse::SuiteSparseConfig_static )
        target_link_libraries ( my_static PUBLIC SuiteSparse::SuiteSparseConfig_static )
        target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::SuiteSparseConfig_static )
    else ( )
        target_link_libraries ( my_static PUBLIC SuiteSparse::SuiteSparseConfig )
        target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::SuiteSparseConfig )
    endif ( )
endif ( )

# UMFPACK:
if ( BUILD_SHARED_LIBS )
    target_link_libraries ( my PRIVATE SuiteSparse::UMFPACK )
    target_link_libraries ( my_cxx PRIVATE SuiteSparse::UMFPACK )
endif ( )
if ( BUILD_STATIC_LIBS )
    if ( TARGET SuiteSparse::UMFPACK_static )
        target_link_libraries ( my_static PUBLIC SuiteSparse::UMFPACK_static )
        target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::UMFPACK_static )
    else ( )
        target_link_libraries ( my_static PUBLIC SuiteSparse::UMFPACK )
        target_link_libraries ( my_cxx_static PUBLIC SuiteSparse::UMFPACK )
    endif ( )
endif ( )

#-------------------------------------------------------------------------------
# installation location
#-------------------------------------------------------------------------------

include ( GNUInstallDirs )

if ( BUILD_SHARED_LIBS )
    install ( TARGETS my
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} )
    install ( TARGETS my_cxx
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} )
endif ( )
if ( BUILD_STATIC_LIBS )
    install ( TARGETS my_static
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} )
    install ( TARGETS my_cxx_static
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} )
endif ( )

#-------------------------------------------------------------------------------
# Demo program
#-------------------------------------------------------------------------------

if ( BUILD_SHARED_LIBS )
    add_executable ( my_demo "Demo/my_demo.c" )
    target_link_libraries ( my_demo PUBLIC my )

    add_executable ( my_cxx_demo "Demo/my_demo.c" )
    target_link_libraries ( my_cxx_demo PUBLIC my_cxx )
endif ( )

if ( BUILD_STATIC_LIBS )
    add_executable ( my_demo_static "Demo/my_demo.c" )
    target_link_libraries ( my_demo_static PUBLIC my_static )

    add_executable ( my_cxx_demo_static "Demo/my_demo.c" )
    target_link_libraries ( my_cxx_demo_static PUBLIC my_cxx_static )
endif ( )
