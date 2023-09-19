#-------------------------------------------------------------------------------
# SuiteSparse/CCOLAMD/CMakeLists.txt:  cmake for CCOLAMD
#-------------------------------------------------------------------------------

# Copyright (c) 2005-2023, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

#-------------------------------------------------------------------------------
# get the version
#-------------------------------------------------------------------------------

cmake_minimum_required ( VERSION 3.20 )

set ( CCOLAMD_DATE "Sept 18, 2023" )
set ( CCOLAMD_VERSION_MAJOR 3 )
set ( CCOLAMD_VERSION_MINOR 2 )
set ( CCOLAMD_VERSION_SUB   1 )

message ( STATUS "Building CCOLAMD version: v"
    ${CCOLAMD_VERSION_MAJOR}.
    ${CCOLAMD_VERSION_MINOR}.
    ${CCOLAMD_VERSION_SUB} " (" ${CCOLAMD_DATE} ")" )

#-------------------------------------------------------------------------------
# SuiteSparse policies
#-------------------------------------------------------------------------------

set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    ${CMAKE_SOURCE_DIR}/../SuiteSparse_config/cmake_modules )

include ( SuiteSparsePolicy )

#-------------------------------------------------------------------------------
# define the project
#-------------------------------------------------------------------------------

project ( ccolamd
    VERSION "${CCOLAMD_VERSION_MAJOR}.${CCOLAMD_VERSION_MINOR}.${CCOLAMD_VERSION_SUB}"
    LANGUAGES C )

#-------------------------------------------------------------------------------
# find library dependencies
#-------------------------------------------------------------------------------

find_package ( SuiteSparse_config 7.2.0
    PATHS ${CMAKE_SOURCE_DIR}/../SuiteSparse_config/build NO_DEFAULT_PATH )
if ( NOT TARGET SuiteSparse::SuiteSparseConfig )
    find_package ( SuiteSparse_config 7.2.0 REQUIRED )
endif ( )

#-------------------------------------------------------------------------------
# configure files
#-------------------------------------------------------------------------------

configure_file ( "Config/ccolamd.h.in"
    "${PROJECT_SOURCE_DIR}/Include/ccolamd.h"
    NEWLINE_STYLE LF )

#-------------------------------------------------------------------------------
# include directories
#-------------------------------------------------------------------------------

include_directories ( Source Include )

#-------------------------------------------------------------------------------
# dynamic ccolamd library properties
#-------------------------------------------------------------------------------

file ( GLOB CCOLAMD_SOURCES "Source/*.c" )

add_library ( CCOLAMD SHARED ${CCOLAMD_SOURCES} )

set_target_properties ( CCOLAMD PROPERTIES
    VERSION ${CCOLAMD_VERSION_MAJOR}.${CCOLAMD_VERSION_MINOR}.${CCOLAMD_VERSION_SUB}
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
    OUTPUT_NAME ccolamd
    SOVERSION ${CCOLAMD_VERSION_MAJOR}
    PUBLIC_HEADER "Include/ccolamd.h"
    WINDOWS_EXPORT_ALL_SYMBOLS ON )

target_include_directories ( CCOLAMD
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>
              $<INSTALL_INTERFACE:${SUITESPARSE_INCLUDEDIR}> )

#-------------------------------------------------------------------------------
# static ccolamd library properties
#-------------------------------------------------------------------------------

if ( NOT NSTATIC )
    add_library ( CCOLAMD_static STATIC ${CCOLAMD_SOURCES} )

    set_target_properties ( CCOLAMD_static PROPERTIES
        VERSION ${CCOLAMD_VERSION_MAJOR}.${CCOLAMD_VERSION_MINOR}.${CCOLAMD_VERSION_SUB}
        OUTPUT_NAME ccolamd
        C_STANDARD 11
        C_STANDARD_REQUIRED ON
        SOVERSION ${CCOLAMD_VERSION_MAJOR} )

    if ( MSVC )
        set_target_properties ( CCOLAMD_static PROPERTIES
            OUTPUT_NAME ccolamd_static )
    endif ( )

    target_include_directories ( CCOLAMD_static
        INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>
                  $<INSTALL_INTERFACE:${SUITESPARSE_INCLUDEDIR}> )

endif ( )

#-------------------------------------------------------------------------------
# add the library dependencies
#-------------------------------------------------------------------------------

target_link_libraries ( CCOLAMD PRIVATE SuiteSparse::SuiteSparseConfig )
target_include_directories ( CCOLAMD PUBLIC
    "$<TARGET_PROPERTY:SuiteSparse::SuiteSparseConfig,INTERFACE_INCLUDE_DIRECTORIES>" )
if ( NOT NSTATIC )
    if ( TARGET SuiteSparse::SuiteSparseConfig_static )
        target_link_libraries ( CCOLAMD_static PUBLIC SuiteSparse::SuiteSparseConfig_static )
    else ( )
        target_link_libraries ( CCOLAMD_static PUBLIC SuiteSparse::SuiteSparseConfig )
    endif ( )
endif ( )

# libm:
if ( NOT WIN32 )
    target_link_libraries ( CCOLAMD PRIVATE m )
    if ( NOT NSTATIC )
        set ( CCOLAMD_STATIC_LIBS "${CCOLAMD_STATIC_LIBS} -lm" )
        target_link_libraries ( CCOLAMD_static PUBLIC m )
    endif ( )
endif ( )

#-------------------------------------------------------------------------------
# COLAMD installation location
#-------------------------------------------------------------------------------

include ( CMakePackageConfigHelpers )

install ( TARGETS CCOLAMD
    EXPORT CCOLAMDTargets
    LIBRARY DESTINATION ${SUITESPARSE_LIBDIR}
    ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR}
    RUNTIME DESTINATION ${SUITESPARSE_BINDIR}
    PUBLIC_HEADER DESTINATION ${SUITESPARSE_INCLUDEDIR} )
if ( NOT NSTATIC )
    install ( TARGETS CCOLAMD_static
        EXPORT CCOLAMDTargets
        ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR} )
endif ( )

# create (temporary) export target file during build
export ( EXPORT CCOLAMDTargets
    NAMESPACE SuiteSparse::
    FILE ${CMAKE_CURRENT_BINARY_DIR}/CCOLAMDTargets.cmake )

# install export target, config and version files for find_package
install ( EXPORT CCOLAMDTargets
    NAMESPACE SuiteSparse::
    DESTINATION ${SUITESPARSE_LIBDIR}/cmake/CCOLAMD )

configure_package_config_file (
    Config/CCOLAMDConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/CCOLAMDConfig.cmake
    INSTALL_DESTINATION ${SUITESPARSE_LIBDIR}/cmake/CCOLAMD )

write_basic_package_version_file (
    ${CMAKE_CURRENT_BINARY_DIR}/CCOLAMDConfigVersion.cmake
    COMPATIBILITY SameMajorVersion )

install ( FILES
    ${CMAKE_CURRENT_BINARY_DIR}/CCOLAMDConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/CCOLAMDConfigVersion.cmake
    DESTINATION ${SUITESPARSE_LIBDIR}/cmake/CCOLAMD )

#-------------------------------------------------------------------------------
# create pkg-config file
#-------------------------------------------------------------------------------

if ( NOT MSVC )
    set ( prefix "${CMAKE_INSTALL_PREFIX}" )
    set ( exec_prefix "\${prefix}" )
    cmake_path ( IS_ABSOLUTE SUITESPARSE_LIBDIR SUITESPARSE_LIBDIR_IS_ABSOLUTE )
    if (SUITESPARSE_LIBDIR_IS_ABSOLUTE)
        set ( libdir "${SUITESPARSE_LIBDIR}")
    else ( )
        set ( libdir "\${exec_prefix}/${SUITESPARSE_LIBDIR}")
    endif ( )
    cmake_path ( IS_ABSOLUTE SUITESPARSE_INCLUDEDIR SUITESPARSE_INCLUDEDIR_IS_ABSOLUTE )
    if (SUITESPARSE_INCLUDEDIR_IS_ABSOLUTE)
        set ( includedir "${SUITESPARSE_INCLUDEDIR}")
    else ( )
        set ( includedir "\${prefix}/${SUITESPARSE_INCLUDEDIR}")
    endif ( )
    configure_file (
        Config/CCOLAMD.pc.in
        CCOLAMD.pc
        @ONLY
        NEWLINE_STYLE LF )
    install ( FILES
        ${CMAKE_CURRENT_BINARY_DIR}/CCOLAMD.pc
        DESTINATION ${SUITESPARSE_LIBDIR}/pkgconfig )
endif ( )

#-------------------------------------------------------------------------------
# Demo library and programs
#-------------------------------------------------------------------------------

option ( DEMO "ON: Build the demo programs.  OFF (default): do not build the demo programs." off )
if ( DEMO )

    #---------------------------------------------------------------------------
    # demo library
    #---------------------------------------------------------------------------

    message ( STATUS "Also compiling the demos in CCOLAMD/Demo" )

    #---------------------------------------------------------------------------
    # Demo programs
    #---------------------------------------------------------------------------

    add_executable ( ccolamd_example   "Demo/ccolamd_example.c" )
    add_executable ( ccolamd_l_example "Demo/ccolamd_l_example.c" )

    # Libraries required for Demo programs
    target_link_libraries ( ccolamd_example   PUBLIC CCOLAMD )
    target_link_libraries ( ccolamd_l_example PUBLIC CCOLAMD )

else ( )

    message ( STATUS "Skipping the demos in CCOLAMD/Demo" )

endif ( )

#-------------------------------------------------------------------------------
# report status
#-------------------------------------------------------------------------------

include ( SuiteSparseReport )

