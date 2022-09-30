#-------------------------------------------------------------------------------
# SuiteSparse/SuiteSparse_config/CMakeLists.txt:  cmake for SuiteSparse_config
#-------------------------------------------------------------------------------

# SuiteSparse_config, Copyright (c) 2012-2022, Timothy A. Davis.
# All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

#-------------------------------------------------------------------------------
# get the version
#-------------------------------------------------------------------------------

cmake_minimum_required ( VERSION 3.22 )

# version of both SuiteSparse and SuiteSparse_config
set ( SUITESPARSE_DATE "Oct 31, 2022" )  # FIXME
set ( SUITESPARSE_VERSION_MAJOR 6 )
set ( SUITESPARSE_VERSION_MINOR 0 )
set ( SUITESPARSE_VERSION_SUB   0 )

message ( STATUS "Building SuiteSparse_config version: v"
    ${SUITESPARSE_VERSION_MAJOR}.
    ${SUITESPARSE_VERSION_MINOR}.
    ${SUITESPARSE_VERSION_SUB} " (" ${SUITESPARSE_DATE} ")" )

#-------------------------------------------------------------------------------
# SuiteSparse policies
#-------------------------------------------------------------------------------

set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    ${CMAKE_SOURCE_DIR}/cmake_modules
    ${CMAKE_SOURCE_DIR}/../SuiteSparse_config/cmake_modules
    ${CMAKE_SOURCE_DIR}/../SuiteSparse/SuiteSparse_config/cmake_modules
    )

include ( SuiteSparsePolicy )

#-------------------------------------------------------------------------------
# define the project
#-------------------------------------------------------------------------------

project ( suitesparseconfig
    VERSION "${SUITESPARSE_VERSION_MAJOR}.${SUITESPARSE_VERSION_MINOR}.${SUITESPARSE_VERSION_SUB}"
    LANGUAGES C )

#-------------------------------------------------------------------------------
# find library dependencies
#-------------------------------------------------------------------------------

enable_language ( C Fortran )
message ( STATUS "Fortran: ${CMAKE_Fortran_COMPILER_ID} ") 
find_package ( OpenMP )

# SuiteSparse interface to the BLAS and LAPACK libraries
message ( STATUS "BLAS64_suffix:   ${BLAS64_SUFFIX}" )
message ( STATUS "Looking for 64-bit BLAS" )
set ( BLA_SIZEOF_INTEGER 8 )
find_package ( BLAS )
if ( BLAS_FOUND )
    # 64-bit BLAS found
    message ( STATUS "Found 64-bit BLAS" )
    add_compile_definitions ( BLAS64 ) 
    set ( SuiteSparse_BLAS_integer "int64_t" )
    if (NOT ${BLAS64_SUFFIX} STREQUAL "")
        # append BLAS64_SUFFIX to each BLAS and LAPACK name
        string ( FIND ${BLAS64_SUFFIX} "_" HAS_UNDERSCORE )
        if ( HAS_UNDERSCORE EQUAL -1 )
            message ( STATUS " BLAS64 suffix has an underscore" )
            add_compile_definitions ( BLAS64__SUFFIX=${BLAS64_SUFFIX} ) 
        else ( )
            message ( STATUS " BLAS64 suffix has no underscore" )
            add_compile_definitions ( BLAS64_SUFFIX=${BLAS64_SUFFIX} ) 
        endif ( )
    endif ( )
else ( ) 
    # look for 32-bit BLAS
    message ( STATUS "Looking for 32-bit BLAS" )
    set ( BLA_SIZEOF_INTEGER 4 )
    find_package ( BLAS )
    if ( BLAS_FOUND )
        message ( STATUS "Found 32-bit BLAS" )
    endif ( )
    set ( SuiteSparse_BLAS_integer "int32_t" )
endif ( ) 
include ( FortranCInterface )

#-------------------------------------------------------------------------------
# configure files
#-------------------------------------------------------------------------------

configure_file ( "Config/SuiteSparse_config.h.in"
    "${PROJECT_SOURCE_DIR}/SuiteSparse_config.h" )

#-------------------------------------------------------------------------------
# dynamic suitesparseconfig library properties
#-------------------------------------------------------------------------------

file ( GLOB SUITESPARSECONFIG_SOURCES "*.c" )

add_library ( suitesparseconfig SHARED ${SUITESPARSECONFIG_SOURCES} )

SET_TARGET_PROPERTIES ( suitesparseconfig PROPERTIES
    VERSION ${SUITESPARSE_VERSION_MAJOR}.${SUITESPARSE_VERSION_MINOR}.${SUITESPARSE_VERSION_SUB}
    C_STANDARD_REQUIRED 11
    SOVERSION ${SUITESPARSE_VERSION_MAJOR}
    PUBLIC_HEADER "SuiteSparse_config.h" )

#-------------------------------------------------------------------------------
# static suitesparseconfig library properties
#-------------------------------------------------------------------------------

add_library ( suitesparseconfig_static STATIC ${SUITESPARSECONFIG_SOURCES} )

SET_TARGET_PROPERTIES ( suitesparseconfig_static PROPERTIES
    VERSION ${SUITESPARSE_VERSION_MAJOR}.${SUITESPARSE_VERSION_MINOR}.${SUITESPARSE_VERSION_SUB}
    C_STANDARD_REQUIRED 11
    OUTPUT_NAME suitesparseconfig
    SOVERSION ${SUITESPARSE_VERSION_MAJOR} )

#-------------------------------------------------------------------------------
# add the library dependencies
#-------------------------------------------------------------------------------

# libm:
if ( NOT "${CMAKE_C_COMPILER_ID}" STREQUAL "MSVC" )
    target_link_libraries ( suitesparseconfig PUBLIC m )
    target_link_libraries ( suitesparseconfig_static PUBLIC m )
endif ( )

# OpenMP:
if ( OPENMP_FOUND )
    message ( STATUS "OpenMP C libraries:      ${OpenMP_C_LIBRARIES} ")
    message ( STATUS "OpenMP C include:        ${OpenMP_C_INCLUDE_DIRS} ")
    message ( STATUS "OpenMP C flags:          ${OpenMP_C_FLAGS} ")
    target_link_libraries ( suitesparseconfig PUBLIC ${OpenMP_C_LIBRARIES} )
    target_link_libraries ( suitesparseconfig_static PUBLIC
        ${OpenMP_C_LIBRARIES} )
    set ( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} " )
    include_directories ( ${OpenMP_C_INCLUDE_DIRS} )
endif ( )

# BLAS:
# if ( BLAS_FOUND )
#   message ( STATUS "BLAS libraries:      ${BLAS_LIBRARIES} ")
#   message ( STATUS "BLAS linker flags:   ${BLAS_LINKER_FLAGS} ")
#   message ( STATUS "BLAS include:        ${BLAS_INCLUDE_DIRS} ")
#   target_link_libraries ( suitesparseconfig PUBLIC ${BLAS_LIBRARIES} )
#   target_link_libraries ( suitesparseconfig_static PUBLIC ${BLAS_LIBRARIES} )
#   include_directories ( ${BLAS_INCLUDE_DIRS} )
# endif ( )

#-------------------------------------------------------------------------------
# suitesparseconfig installation location
#-------------------------------------------------------------------------------

# use "cmake -DSUITESPARSE_LOCAL=1" to install only in SuiteSparse/lib and
# SuiteSparse/include.  

if ( NOT SUITESPARSE_LOCAL )
    # install in /usr/local/lib and /usr/local/include.
    # requires "sudo make install"
    message ( STATUS "Installation will be system-wide (requires 'sudo make install')" )
    install ( TARGETS suitesparseconfig
        LIBRARY       DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} )
    install ( TARGETS suitesparseconfig_static
        ARCHIVE       DESTINATION ${CMAKE_INSTALL_LIBDIR} )
endif ( )

if ( INSIDE_SUITESPARSE )
    # also install in SuiteSparse/lib and SuiteSparse/include;
    # does not require "sudo make install", just "make install"
    message ( STATUS "Installation in ../lib and ../include," )
    message ( STATUS "  with 'make local ; make install'. No 'sudo' required." )
    install ( TARGETS suitesparseconfig
        LIBRARY       DESTINATION ${SUITESPARSE_LIBDIR}
        PUBLIC_HEADER DESTINATION ${SUITESPARSE_INCLUDEDIR} )
    install ( TARGETS suitesparseconfig_static
        ARCHIVE       DESTINATION ${SUITESPARSE_LIBDIR} )
endif ( )

include ( SuiteSparseReport )
