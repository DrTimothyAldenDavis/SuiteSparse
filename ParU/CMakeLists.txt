#-------------------------------------------------------------------------------
# SuiteSparse/ParU/CMakeLists.txt:  cmake for ParU
#-------------------------------------------------------------------------------

# ParU, Copyright (c) 2022, Mohsen Aznaveh and Timothy A. Davis,
# All Rights Reserved.
# SPDX-License-Identifier: GNU GPL 3.0

#-------------------------------------------------------------------------------
# get the version
#-------------------------------------------------------------------------------

# cmake 3.22 is required to find the BLAS in SuiteSparse_config
cmake_minimum_required ( VERSION 3.22 )

set ( PARU_DATE "Dec 20, 2022" )    # FIXME: pick a date
set ( PARU_VERSION_MAJOR 1 )
set ( PARU_VERSION_MINOR 0 )
set ( PARU_VERSION_UPDATE 0 )

message ( STATUS "Building PARU version: v"
    ${PARU_VERSION_MAJOR}.
    ${PARU_VERSION_MINOR}.
    ${PARU_VERSION_UPDATE} " (" ${PARU_DATE} ")" )

#-------------------------------------------------------------------------------
# SuiteSparse policies
#-------------------------------------------------------------------------------

set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    ${CMAKE_SOURCE_DIR}/cmake_modules
    ${CMAKE_SOURCE_DIR}/../AMD/cmake_modules
    ${CMAKE_SOURCE_DIR}/../CAMD/cmake_modules
    ${CMAKE_SOURCE_DIR}/../COLAMD/cmake_modules
    ${CMAKE_SOURCE_DIR}/../CCOLAMD/cmake_modules
    ${CMAKE_SOURCE_DIR}/../CHOLMOD/cmake_modules
    ${CMAKE_SOURCE_DIR}/../UMFPACK/cmake_modules
    ${CMAKE_SOURCE_DIR}/../SuiteSparse_config/cmake_modules
    ${CMAKE_SOURCE_DIR}/../SuiteSparse/SuiteSparse_config/cmake_modules
    )

include ( SuiteSparsePolicy )

#-------------------------------------------------------------------------------
# define the project
#-------------------------------------------------------------------------------

project ( paru
    VERSION "${PARU_VERSION_MAJOR}.${PARU_VERSION_MINOR}.${PARU_VERSION_UPDATE}"
    LANGUAGES C CXX )

#-------------------------------------------------------------------------------
# find library dependencies
#-------------------------------------------------------------------------------

enable_language ( C CXX Fortran )
message ( STATUS "Fortran: ${CMAKE_Fortran_COMPILER_ID} ") 
find_package ( OpenMP )

find_package ( SuiteSparse_config 7.2.0
    PATHS ${CMAKE_SOURCE_DIR}/../SuiteSparse_config/build NO_DEFAULT_PATH )
if ( NOT TARGET SuiteSparse::SuiteSparseConfig )
    find_package ( SuiteSparse_config 7.2.0 REQUIRED )
endif ( )

find_package ( CHOLMOD 4.2.2
    PATHS ${CMAKE_SOURCE_DIR}/../CHOLMOD/build NO_DEFAULT_PATH )
if ( NOT CHOLMOD_FOUND )
    find_package ( CHOLMOD 4.2.2 REQUIRED )
endif ( )

find_package ( UMFPACK 6.2.1
    PATHS ${CMAKE_SOURCE_DIR}/../UMFPACK/build NO_DEFAULT_PATH )
if ( NOT CAMD_FOUND )
    find_package ( UMFPACK 6.2.1 REQUIRED )
endif ( )

include ( SuiteSparseBLAS )     # requires cmake 3.22

#-------------------------------------------------------------------------------
# configure files
#-------------------------------------------------------------------------------

configure_file ( "Config/ParU_definitions.h.in"
    "${PROJECT_SOURCE_DIR}/Include/ParU_definitions.h" )
configure_file ( "Config/paru_version.tex.in"
    "${PROJECT_SOURCE_DIR}/Doc/paru_version.tex" )

#-------------------------------------------------------------------------------
# include directories
#-------------------------------------------------------------------------------

include_directories ( Source Include ${SUITESPARSE_CONFIG_INCLUDE_DIR}
    ${AMD_INCLUDE_DIR}
    ${CAMD_INCLUDE_DIR}
    ${COLAMD_INCLUDE_DIR}
    ${CCOLAMD_INCLUDE_DIR}
    ${CHOLMOD_INCLUDE_DIR}
    ${UMFPACK_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/../UMFPACK/Source
    ${CMAKE_SOURCE_DIR}/../UMFPACK/Include
    ${CMAKE_SOURCE_DIR}/../CHOLMOD/Include
    ${CMAKE_SOURCE_DIR}/../AMD/Include
    )

#-------------------------------------------------------------------------------
# dynamic paru library properties
#-------------------------------------------------------------------------------

file ( GLOB PARU_SOURCES "Source/*.cpp" )

add_library ( paru SHARED ${PARU_SOURCES} )

set_target_properties ( paru PROPERTIES
    VERSION ${PARU_VERSION_MAJOR}.${PARU_VERSION_MINOR}.${PARU_VERSION_UPDATE}
    C_STANDARD_REQUIRED 11
    CXX_STANDARD_REQUIRED 11
    SOVERSION ${PARU_VERSION_MAJOR} )

#-------------------------------------------------------------------------------
# static paru library properties
#-------------------------------------------------------------------------------

if ( NOT NSTATIC )
add_library ( paru_static STATIC ${PARU_SOURCES} )

set_target_properties ( paru_static PROPERTIES
    VERSION ${PARU_VERSION_MAJOR}.${PARU_VERSION_MINOR}.${PARU_VERSION_UPDATE}
    C_STANDARD_REQUIRED 11
    CXX_STANDARD_REQUIRED 11
    OUTPUT_NAME paru
    SOVERSION ${PARU_VERSION_MAJOR} )
endif ( )

#-------------------------------------------------------------------------------
# add the library dependencies
#-------------------------------------------------------------------------------

# suitesparseconfig:
target_link_libraries ( paru PUBLIC ${SUITESPARSE_CONFIG_LIBRARIES} )
if ( NOT NSTATIC )
target_link_libraries ( paru_static PUBLIC ${SUITESPARSE_CONFIG_STATIC} )
endif ( )

# OpenMP:
if ( OPENMP_FOUND )
    message ( STATUS "OpenMP C libraries:      ${OpenMP_C_LIBRARIES} ")
    message ( STATUS "OpenMP C include:        ${OpenMP_C_INCLUDE_DIRS} ")
    message ( STATUS "OpenMP C flags:          ${OpenMP_C_FLAGS} ")
    target_link_libraries ( paru PUBLIC ${OpenMP_C_LIBRARIES} )
    if ( NOT NSTATIC )
    target_link_libraries ( paru_static PUBLIC ${OpenMP_C_LIBRARIES} )
    endif ( )
    set ( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} " )
    set ( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} " )
    include_directories ( ${OpenMP_C_INCLUDE_DIRS} )
endif ( )

# libm:
if ( NOT MSVC )
    target_link_libraries ( paru PUBLIC m )
    if ( NOT NSTATIC )
    target_link_libraries ( paru_static PUBLIC m )
    endif ( )
endif ( )

# amd
target_link_libraries ( paru PUBLIC ${AMD_LIBRARIES} )
if ( NOT NSTATIC )
target_link_libraries ( paru_static PUBLIC ${AMD_STATIC} )
endif ( )

# camd:
target_link_libraries ( paru PUBLIC ${CAMD_LIBRARIES} )
if ( NOT NSTATIC )
target_link_libraries ( paru_static PUBLIC ${CAMD_STATIC} )
endif ( )

# colamd:
target_link_libraries ( paru PUBLIC ${COLAMD_LIBRARIES} )
if ( NOT NSTATIC )
target_link_libraries ( paru_static PUBLIC ${COLAMD_STATIC} )
endif ( )

# ccolamd:
target_link_libraries ( paru PUBLIC ${CCOLAMD_LIBRARIES} )
if ( NOT NSTATIC )
target_link_libraries ( paru_static PUBLIC ${CCOLAMD_STATIC} )
endif ( )

# umfpack:
target_link_libraries ( paru PUBLIC ${UMFPACK_LIBRARIES} )
if ( NOT NSTATIC )
target_link_libraries ( paru_static PUBLIC ${UMFPACK_STATIC} )
endif ( )

# BLAS:
message ( STATUS "BLAS libraries:      ${BLAS_LIBRARIES} ")
message ( STATUS "BLAS linker flags:   ${BLAS_LINKER_FLAGS} ")
target_link_libraries ( paru PUBLIC ${BLAS_LIBRARIES} )
if ( NOT NSTATIC )
target_link_libraries ( paru_static PUBLIC ${BLAS_LIBRARIES} )
include_directories ( ${BLAS_INCLUDE_DIRS} )
endif ( )

# cholmod:
target_link_libraries ( paru PUBLIC ${CHOLMOD_LIBRARIES} )
if ( NOT NSTATIC )
target_link_libraries ( paru_static PUBLIC ${CHOLMOD_STATIC} )
endif ( )

#-------------------------------------------------------------------------------
# PARU installation location
#---------------------------------------------------------------Z--------------

if ( GLOBAL_INSTALL )
    # install in /usr/local/lib and /usr/local/include.
    # requires "sudo make install"
    message ( STATUS "Installation will be system-wide (requires 'sudo make install')" )
    install ( TARGETS paru
        LIBRARY       DESTINATION ${CMAKE_INSTALL_LIBDIR} )
    install ( FILES ${CMAKE_SOURCE_DIR}/cmake_modules/FindParU.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SuiteSparse 
        COMPONENT Development )
    if ( NOT NSTATIC )
    install ( TARGETS paru_static
        ARCHIVE       DESTINATION ${CMAKE_INSTALL_LIBDIR} )
    endif ( )
    install ( FILES "Include/ParU_C.h"
        "Include/ParU_definitions.h"
        "Include/ParU.hpp" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} )
endif ( )

if ( INSIDE_SUITESPARSE )
    # also install in SuiteSparse/lib and SuiteSparse/include;
    # does not require "sudo make install", just "make install"
    message ( STATUS "Installation in ../lib and ../include only," )
    message ( STATUS "  with 'make install'. No 'sudo' required." )
    install ( TARGETS paru
        LIBRARY       DESTINATION ${SUITESPARSE_LIBDIR}
        PUBLIC_HEADER DESTINATION ${SUITESPARSE_INCLUDEDIR} )
    install ( FILES ${CMAKE_SOURCE_DIR}/cmake_modules/FindParU.cmake
        DESTINATION ${SUITESPARSE_LIBDIR}/cmake/SuiteSparse 
        COMPONENT Development )
    if ( NOT NSTATIC )
    install ( TARGETS paru_static
        ARCHIVE       DESTINATION ${SUITESPARSE_LIBDIR} )
    endif ( )
    install ( FILES "Include/ParU_C.h"
        "Include/ParU_definitions.h"
        "Include/ParU.hpp" DESTINATION ${SUITESPARSE_INCLUDEDIR} )
endif ( )

#-------------------------------------------------------------------------------
# Demo library and programs
#-------------------------------------------------------------------------------

if ( DEMO )

    #---------------------------------------------------------------------------
    # demo library
    #---------------------------------------------------------------------------

    message ( STATUS "Also compiling the demos in ParU/Demo" )

    #---------------------------------------------------------------------------
    # Demo programs
    #---------------------------------------------------------------------------

    add_executable ( paru_demo    "Demo/paru_demo.cpp"   )
    add_executable ( paru_democ   "Demo/paru_democ.c"    )
    add_executable ( paru_simple  "Demo/paru_simple.cpp" )
    add_executable ( paru_simplec "Demo/paru_simplec.c"  )

    # Libraries required for Demo programs
    target_link_libraries ( paru_demo    PUBLIC paru )
    target_link_libraries ( paru_democ   PUBLIC paru )
    target_link_libraries ( paru_simple  PUBLIC paru )
    target_link_libraries ( paru_simplec PUBLIC paru )

else ( )

    message ( STATUS "Skipping the demos in ParU/Demo" )

endif ( )

#-------------------------------------------------------------------------------
# report status
#-------------------------------------------------------------------------------

include ( SuiteSparseReport )

