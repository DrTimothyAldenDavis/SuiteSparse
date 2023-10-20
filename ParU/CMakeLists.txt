#-------------------------------------------------------------------------------
# SuiteSparse/ParU/CMakeLists.txt:  cmake for ParU
#-------------------------------------------------------------------------------

# ParU, Copyright (c) 2022-2023, Mohsen Aznaveh and Timothy A. Davis,
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
    ${CMAKE_SOURCE_DIR}/../SuiteSparse_config/cmake_modules
    )

include ( SuiteSparsePolicy )

#-------------------------------------------------------------------------------
# define the project
#-------------------------------------------------------------------------------

project ( paru
    VERSION "${PARU_VERSION_MAJOR}.${PARU_VERSION_MINOR}.${PARU_VERSION_UPDATE}" )

#-------------------------------------------------------------------------------
# find library dependencies
#-------------------------------------------------------------------------------

enable_language ( CXX )

if ( DEMO )
    enable_language ( C )
endif ( )

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
    "${PROJECT_SOURCE_DIR}/Include/ParU_definitions.h"
    NEWLINE_STYLE LF )
configure_file ( "Config/paru_version.tex.in"
    "${PROJECT_SOURCE_DIR}/Doc/paru_version.tex"
    NEWLINE_STYLE LF )

#-------------------------------------------------------------------------------
# include directories
#-------------------------------------------------------------------------------

include_directories ( Source Include
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
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
    SOVERSION ${PARU_VERSION_MAJOR} )

#-------------------------------------------------------------------------------
# static paru library properties
#-------------------------------------------------------------------------------

if ( NOT NSTATIC )
    add_library ( paru_static STATIC ${PARU_SOURCES} )

    set_target_properties ( paru_static PROPERTIES
        VERSION ${PARU_VERSION_MAJOR}.${PARU_VERSION_MINOR}.${PARU_VERSION_UPDATE}
        CXX_STANDARD 11
        CXX_STANDARD_REQUIRED ON
        OUTPUT_NAME paru
        SOVERSION ${PARU_VERSION_MAJOR} )
endif ( )

#-------------------------------------------------------------------------------
# add the library dependencies
#-------------------------------------------------------------------------------

# suitesparseconfig:
target_link_libraries ( paru PRIVATE SuiteSparse::SuiteSparseConfig )
target_include_directories ( paru PUBLIC
    "$<TARGET_PROPERTY:SuiteSparse::SuiteSparseConfig,INTERFACE_INCLUDE_DIRECTORIES>" )
if ( NOT NSTATIC )
    if ( TARGET SuiteSparse::SuiteSparseConfig_static )
        target_link_libraries ( paru_static PUBLIC SuiteSparse::SuiteSparseConfig_static )
    else ( )
        target_link_libraries ( paru_static PUBLIC SuiteSparse::SuiteSparseConfig )
    endif ( )
endif ( )

# OpenMP:
if ( OPENMP_CXX_FOUND )
    message ( STATUS "OpenMP C++ libraries:    ${OpenMP_CXX_LIBRARIES} ")
    message ( STATUS "OpenMP C++ include:      ${OpenMP_CXX_INCLUDE_DIRS} ")
    message ( STATUS "OpenMP C++ flags:        ${OpenMP_CXX_FLAGS} ")
    target_link_libraries ( paru PRIVATE OpenMP::OpenMP_CXX )
    if ( NOT NSTATIC )
        target_link_libraries ( paru_static PUBLIC OpenMP::OpenMP_CXX )
    endif ( )
endif ( )

# libm:
if ( NOT MSVC )
    target_link_libraries ( paru PRIVATE m )
    if ( NOT NSTATIC )
        target_link_libraries ( paru_static PUBLIC m )
    endif ( )
endif ( )

# umfpack:
target_link_libraries ( paru PRIVATE SuiteSparse::UMFPACK )
target_include_directories ( paru PUBLIC
    "$<TARGET_PROPERTY:SuiteSparse::UMFPACK,INTERFACE_INCLUDE_DIRECTORIES>" )
if ( NOT NSTATIC )
    if ( TARGET SuiteSparse::UMFPACK_static )
        target_link_libraries ( paru_static PUBLIC SuiteSparse::UMFPACK_static )
    else ( )
        target_link_libraries ( paru_static PUBLIC SuiteSparse::UMFPACK )
    endif ( )
endif ( )

# BLAS:
message ( STATUS "BLAS libraries:      ${BLAS_LIBRARIES} ")
message ( STATUS "BLAS linker flags:   ${BLAS_LINKER_FLAGS} ")
target_link_libraries ( paru PRIVATE ${BLAS_LIBRARIES} )
if ( NOT NSTATIC )
    target_link_libraries ( paru_static PUBLIC ${BLAS_LIBRARIES} )
    include_directories ( ${BLAS_INCLUDE_DIRS} )
endif ( )

# cholmod:
target_link_libraries ( paru PRIVATE SuiteSparse::CHOLMOD )
target_include_directories ( paru PUBLIC
    "$<TARGET_PROPERTY:SuiteSparse::CHOLMOD,INTERFACE_INCLUDE_DIRECTORIES>" )
if ( NOT NSTATIC )
    if ( TARGET SuiteSparse::CHOLMOD_static )
        target_link_libraries ( paru_static PUBLIC SuiteSparse::CHOLMOD_static )
    else ( )
        target_link_libraries ( paru_static PUBLIC SuiteSparse::CHOLMOD )
    endif ( )
endif ( )

#-------------------------------------------------------------------------------
# PARU installation location
#-------------------------------------------------------------------------------

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
    target_link_libraries ( paru_demo
        PUBLIC paru SuiteSparse::CHOLMOD SuiteSparse::UMFPACK OpenMP::OpenMP_CXX )
    target_link_libraries ( paru_democ
        PUBLIC paru SuiteSparse::CHOLMOD SuiteSparse::UMFPACK OpenMP::OpenMP_C )
    target_link_libraries ( paru_simple
        PUBLIC paru SuiteSparse::CHOLMOD )
    target_link_libraries ( paru_simplec
        PUBLIC paru SuiteSparse::CHOLMOD )

else ( )

    message ( STATUS "Skipping the demos in ParU/Demo" )

endif ( )

#-------------------------------------------------------------------------------
# report status
#-------------------------------------------------------------------------------

include ( SuiteSparseReport )

