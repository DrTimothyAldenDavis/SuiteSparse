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

option ( NOPENMP "ON: do not use OpenMP.  OFF (default): use OpenMP" OFF )
if ( NOPENMP )
    # OpenMP has been disabled
    set ( OpenMP_CXX_FOUND OFF )
    set ( OpenMP_C_FOUND OFF )
else ( )
    find_package ( OpenMP )
    if ( OpenMP_CXX_FOUND AND OpenMP_CXX_VERSION VERSION_LESS 4.5 )
        set ( NOPENMP ON )
    endif ( )
    if ( DEMO AND OpenMP_C_FOUND AND OpenMP_C_VERSION VERSION_LESS 4.5 )
        set ( NOPENMP ON )
    endif ( )
    if ( NOPENMP )
        message ( STATUS "Version of OpenMP is too old." )
        set ( OpenMP_CXX_FOUND OFF )
        set ( OpenMP_C_FOUND OFF )
    endif ( )
endif ( )

find_package ( SuiteSparse_config 7.3.0
    PATHS ${CMAKE_SOURCE_DIR}/../SuiteSparse_config/build NO_DEFAULT_PATH )
if ( NOT TARGET SuiteSparse::SuiteSparseConfig )
    find_package ( SuiteSparse_config 7.3.0 REQUIRED )
endif ( )

find_package ( CHOLMOD 5.0.0
    PATHS ${CMAKE_SOURCE_DIR}/../CHOLMOD/build NO_DEFAULT_PATH )
if ( NOT CHOLMOD_FOUND )
    find_package ( CHOLMOD 5.0.0 REQUIRED )
endif ( )

find_package ( UMFPACK 6.2.2
    PATHS ${CMAKE_SOURCE_DIR}/../UMFPACK/build NO_DEFAULT_PATH )
if ( NOT CAMD_FOUND )
    find_package ( UMFPACK 6.2.2 REQUIRED )
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

# FIXME: "paru_internal.hpp" includes "umf_internal.h" which is not installed.
#        That means the two libraries are interdependent and ParU cannot be
#        built as a stand-alone project outside a common build tree.
include_directories ( Source Include
    ${CMAKE_SOURCE_DIR}/../UMFPACK/Source
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
    SOVERSION ${PARU_VERSION_MAJOR}
    PUBLIC_HEADER "Include/ParU.hpp;Include/ParU_C.h;Include/ParU_definitions.h"
    WINDOWS_EXPORT_ALL_SYMBOLS ON )

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
        SOVERSION ${PARU_VERSION_MAJOR}
        PUBLIC_HEADER "Include/ParU.hpp;Include/ParU_C.h;Include/ParU_definitions.h"
        WINDOWS_EXPORT_ALL_SYMBOLS ON )

    if ( MSVC )
        set_target_properties ( paru_static PROPERTIES
            OUTPUT_NAME paru_static )
    endif ( )
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
if ( OpenMP_CXX_FOUND )
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

install ( TARGETS paru
    LIBRARY DESTINATION ${SUITESPARSE_LIBDIR}
    ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR}
    RUNTIME DESTINATION ${SUITESPARSE_BINDIR}
    PUBLIC_HEADER DESTINATION ${SUITESPARSE_INCLUDEDIR} )

install ( FILES ${CMAKE_SOURCE_DIR}/cmake_modules/FindParU.cmake
    DESTINATION ${SUITESPARSE_LIBDIR}/cmake/SuiteSparse 
    COMPONENT Development )

if ( NOT NSTATIC )
    install ( TARGETS paru_static
        ARCHIVE       DESTINATION ${SUITESPARSE_LIBDIR} )
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

    if ( OpenMP_CXX_FOUND )
        add_executable ( paru_demo    "Demo/paru_demo.cpp"   )
        target_link_libraries ( paru_demo
            PUBLIC paru SuiteSparse::CHOLMOD SuiteSparse::UMFPACK OpenMP::OpenMP_CXX )
    endif ( )

    if ( OpenMP_C_FOUND )
        add_executable ( paru_democ   "Demo/paru_democ.c"    )
        target_link_libraries ( paru_democ
            PUBLIC paru SuiteSparse::CHOLMOD SuiteSparse::UMFPACK OpenMP::OpenMP_C )
    endif ( )

    add_executable ( paru_simple  "Demo/paru_simple.cpp" )
    target_link_libraries ( paru_simple
        PUBLIC paru SuiteSparse::CHOLMOD )

    add_executable ( paru_simplec "Demo/paru_simplec.c"  )
    target_link_libraries ( paru_simplec
        PUBLIC paru SuiteSparse::CHOLMOD )

else ( )

    message ( STATUS "Skipping the demos in ParU/Demo" )

endif ( )

#-------------------------------------------------------------------------------
# report status
#-------------------------------------------------------------------------------

include ( SuiteSparseReport )

