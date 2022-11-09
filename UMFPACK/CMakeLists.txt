#-------------------------------------------------------------------------------
# SuiteSparse/UMFPACK/CMakeLists.txt:  cmake for UMFPACK
#-------------------------------------------------------------------------------

# Copyright (c) 1995-2022, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: GPL-2.0+

#-------------------------------------------------------------------------------
# get the version
#-------------------------------------------------------------------------------

# cmake 3.22 is required to find the BLAS in SuiteSparse_config
cmake_minimum_required ( VERSION 3.22 )

set ( UMFPACK_DATE "Nov 12, 2022" )
set ( UMFPACK_VERSION_MAJOR 6 )
set ( UMFPACK_VERSION_MINOR 0 )
set ( UMFPACK_VERSION_SUB   0 )

message ( STATUS "Building UMFPACK version: v"
    ${UMFPACK_VERSION_MAJOR}.
    ${UMFPACK_VERSION_MINOR}.
    ${UMFPACK_VERSION_SUB} " (" ${UMFPACK_DATE} ")" )

#-------------------------------------------------------------------------------
# SuiteSparse policies
#-------------------------------------------------------------------------------

set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    ${CMAKE_SOURCE_DIR}/cmake_modules
    ${CMAKE_SOURCE_DIR}/../AMD/cmake_modules
    ${CMAKE_SOURCE_DIR}/../CHOLMOD/cmake_modules
    ${CMAKE_SOURCE_DIR}/../SuiteSparse_config/cmake_modules )

include ( SuiteSparsePolicy )

#-------------------------------------------------------------------------------
# define the project
#-------------------------------------------------------------------------------

project ( umfpack
    VERSION "${UMFPACK_VERSION_MAJOR}.${UMFPACK_VERSION_MINOR}.${UMFPACK_VERSION_SUB}"
    LANGUAGES C )

#-------------------------------------------------------------------------------
# find library dependencies
#-------------------------------------------------------------------------------

enable_language ( C Fortran )

option ( NOPENMP "ON: do not use OpenMP.  OFF (default): use OpenMP" off )
if ( NOPENMP )
    # OpenMP has been disabled
    set ( OPENMP_FOUND false )
else ( )
    find_package ( OpenMP )
endif ( )

find_package ( SuiteSparse_config 6.0.0 REQUIRED )
find_package ( AMD 3.0.0 REQUIRED )
include ( SuiteSparseBLAS )     # requires cmake 3.22

option ( NCHOLMOD "ON: do not use CHOLMOD.  OFF (default): use CHOLMOD" off )

if ( NOT NCHOLMOD )
    # look for CHOLMOD (optional fill-reducing orderings for KLU)
    find_package ( CHOLMOD 4.0.0 )
    if ( NOT CHOLMOD_FOUND )
        # CHOLMOD not found so disable it
        set ( NCHOLMOD true )
    endif ( )
endif ( )

if ( NCHOLMOD )
    # tell KLU that CHOLMOD is not available
    message ( STATUS "CHOLMOD not found or not requested" )
    add_compile_definitions ( NCHOLMOD )
else ( )
    message ( STATUS "Using CHOLMOD for addtional pre-ordering options" )
endif ( )

#-------------------------------------------------------------------------------
# configure files
#-------------------------------------------------------------------------------

configure_file ( "Config/umfpack.h.in"
    "${PROJECT_SOURCE_DIR}/Include/umfpack.h" )
configure_file ( "Config/umfpack_version.tex.in"
    "${PROJECT_SOURCE_DIR}/Doc/umfpack_version.tex" )

#-------------------------------------------------------------------------------
# include directories
#-------------------------------------------------------------------------------

include_directories ( Source Include ${SUITESPARSE_CONFIG_INCLUDE_DIR}
    ${AMD_INCLUDE_DIR} ${CHOLMOD_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/../AMD/Include )

#-------------------------------------------------------------------------------
# dynamic umfpack library properties
#-------------------------------------------------------------------------------

file ( GLOB UMFPACK_SOURCES "Source2/*.c" )

add_library ( umfpack SHARED ${UMFPACK_SOURCES} )

set_target_properties ( umfpack PROPERTIES
    VERSION ${UMFPACK_VERSION_MAJOR}.${UMFPACK_VERSION_MINOR}.${UMFPACK_VERSION_SUB}
    C_STANDARD_REQUIRED 11
    SOVERSION ${UMFPACK_VERSION_MAJOR}
    PUBLIC_HEADER "Include/umfpack.h" )

#-------------------------------------------------------------------------------
# static umfpack library properties
#-------------------------------------------------------------------------------

if ( NOT NSTATIC )
add_library ( umfpack_static STATIC ${UMFPACK_SOURCES} )

set_target_properties ( umfpack_static PROPERTIES
    VERSION ${UMFPACK_VERSION_MAJOR}.${UMFPACK_VERSION_MINOR}.${UMFPACK_VERSION_SUB}
    C_STANDARD_REQUIRED 11
    OUTPUT_NAME umfpack
    SOVERSION ${UMFPACK_VERSION_MAJOR} )
endif ( )

#-------------------------------------------------------------------------------
# add the library dependencies
#-------------------------------------------------------------------------------

# suitesparseconfig:
target_link_libraries ( umfpack PUBLIC ${SUITESPARSE_CONFIG_LIBRARIES} )
if ( NOT NSTATIC )
target_link_libraries ( umfpack_static PUBLIC ${SUITESPARSE_CONFIG_LIBRARIES} )
endif ( )

# OpenMP:
if ( OPENMP_FOUND )
    message ( STATUS "OpenMP C libraries:      ${OpenMP_C_LIBRARIES} ")
    message ( STATUS "OpenMP C include:        ${OpenMP_C_INCLUDE_DIRS} ")
    message ( STATUS "OpenMP C flags:          ${OpenMP_C_FLAGS} ")
    target_link_libraries ( umfpack PUBLIC ${OpenMP_C_LIBRARIES} )
    if ( NOT NSTATIC )
    target_link_libraries ( umfpack_static PUBLIC ${OpenMP_C_LIBRARIES} )
    endif ( )
    set ( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} " )
    include_directories ( ${OpenMP_C_INCLUDE_DIRS} )
endif ( )

# libm:
if ( NOT "${CMAKE_C_COMPILER_ID}" STREQUAL "MSVC" )
    target_link_libraries ( umfpack PUBLIC m )
    if ( NOT NSTATIC )
    target_link_libraries ( umfpack_static PUBLIC m )
    endif ( )
endif ( )

# amd:
target_link_libraries ( umfpack PUBLIC ${AMD_LIBRARIES} )
if ( NOT NSTATIC )
target_link_libraries ( umfpack_static PUBLIC ${AMD_LIBRARIES} )
endif ( )

# BLAS:
message ( STATUS "BLAS libraries:      ${BLAS_LIBRARIES} ")
message ( STATUS "BLAS linker flags:   ${BLAS_LINKER_FLAGS} ")
target_link_libraries ( umfpack PUBLIC ${BLAS_LIBRARIES} )
if ( NOT NSTATIC )
target_link_libraries ( umfpack_static PUBLIC ${BLAS_LIBRARIES} )
endif ( )
include_directories ( ${BLAS_INCLUDE_DIRS} )

# cholmod:
if ( NOT NCHOLMOD )
    target_link_libraries ( umfpack PUBLIC ${CHOLMOD_LIBRARIES} )
    if ( NOT NSTATIC )
    target_link_libraries ( umfpack_static PUBLIC ${CHOLMOD_LIBRARIES} )
    endif ( )
endif ( )

#-------------------------------------------------------------------------------
# UMFPACK installation location
#---------------------------------------------------------------Z--------------

if ( GLOBAL_INSTALL )
    # install in /usr/local/lib and /usr/local/include.
    # requires "sudo make install"
    message ( STATUS "Installation will be system-wide (requires 'sudo make install')" )
    install ( TARGETS umfpack
        LIBRARY       DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} )
    install ( FILES ${CMAKE_SOURCE_DIR}/cmake_modules/FindUMFPACK.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SuiteSparse 
        COMPONENT Development )
    if ( NOT NSTATIC )
    install ( TARGETS umfpack_static
        ARCHIVE       DESTINATION ${CMAKE_INSTALL_LIBDIR} )
    endif ( )
endif ( )

if ( INSIDE_SUITESPARSE )
    # also install in SuiteSparse/lib and SuiteSparse/include;
    # does not require "sudo make install", just "make install"
    message ( STATUS "Installation in ../lib and ../include only," )
    message ( STATUS "  with 'make install'. No 'sudo' required." )
    install ( TARGETS umfpack
        LIBRARY       DESTINATION ${SUITESPARSE_LIBDIR}
        PUBLIC_HEADER DESTINATION ${SUITESPARSE_INCLUDEDIR} )
    install ( FILES ${CMAKE_SOURCE_DIR}/cmake_modules/FindUMFPACK.cmake
        DESTINATION ${SUITESPARSE_LIBDIR}/cmake/SuiteSparse 
        COMPONENT Development )
    if ( NOT NSTATIC )
    install ( TARGETS umfpack_static
        ARCHIVE       DESTINATION ${SUITESPARSE_LIBDIR} )
    endif ( )
endif ( )

#-------------------------------------------------------------------------------
# Demo library and programs
#-------------------------------------------------------------------------------

if ( DEMO )

    #---------------------------------------------------------------------------
    # demo library
    #---------------------------------------------------------------------------

    message ( STATUS "Also compiling the demos in UMFPACK/Demo" )

    #---------------------------------------------------------------------------
    # Demo programs
    #---------------------------------------------------------------------------

    add_executable ( umfpack_simple  "Demo/umfpack_simple.c" )
    add_executable ( umfpack_di_demo "Demo/umfpack_di_demo.c" )
    add_executable ( umfpack_dl_demo "Demo/umfpack_dl_demo.c" )
    add_executable ( umfpack_zi_demo "Demo/umfpack_zi_demo.c" )
    add_executable ( umfpack_zl_demo "Demo/umfpack_zl_demo.c" )

    if ( FDEMO )
        # Fortran demos
        add_executable ( umf4            "Demo/umf4.c" )
        add_executable ( readhb          "Demo/readhb.f" )
        add_executable ( readhb_nozeros  "Demo/readhb_nozeros.f" )
        add_executable ( readhb_size     "Demo/readhb_size.f" )
        add_executable ( umf4hb          "Demo/umf4hb.f"
                                         "Demo/umf4_f77wrapper.c" )
        add_executable ( umf4zhb         "Demo/umf4zhb.f"
                                         "Demo/umf4_f77zwrapper.c" )
        add_executable ( umf4hb64        "Demo/umf4hb64.f"
                                         "Demo/umf4_f77wrapper64.c" )
    endif ( )

    # Libraries required for Demo programs
    target_link_libraries ( umfpack_simple  PUBLIC umfpack )
    target_link_libraries ( umfpack_di_demo PUBLIC umfpack )
    target_link_libraries ( umfpack_dl_demo PUBLIC umfpack )
    target_link_libraries ( umfpack_zi_demo PUBLIC umfpack )
    target_link_libraries ( umfpack_zl_demo PUBLIC umfpack )
    if ( FDEMO )
        # Fortran demos
        target_link_libraries ( umf4            PUBLIC umfpack )
        target_link_libraries ( umf4hb          PUBLIC umfpack )
        target_link_libraries ( umf4zhb         PUBLIC umfpack )
        target_link_libraries ( umf4hb64        PUBLIC umfpack )
    endif ( )

else ( )

    message ( STATUS "Skipping the demos in UMFPACK/Demo" )

endif ( )

#-------------------------------------------------------------------------------
# report status
#-------------------------------------------------------------------------------

include ( SuiteSparseReport )

