#-------------------------------------------------------------------------------
# SuiteSparse/CAMD/CMakeLists.txt:  cmake for CAMD
#-------------------------------------------------------------------------------

# Copyright (c) 2007-2024, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

#-------------------------------------------------------------------------------
# get the version
#-------------------------------------------------------------------------------

cmake_minimum_required ( VERSION 3.22 )

set ( CAMD_DATE "June 20, 2024" )
set ( CAMD_VERSION_MAJOR 3 CACHE STRING "" FORCE )
set ( CAMD_VERSION_MINOR 3 CACHE STRING "" FORCE )
set ( CAMD_VERSION_SUB   3 CACHE STRING "" FORCE )

message ( STATUS "Building CAMD version: v"
    ${CAMD_VERSION_MAJOR}.
    ${CAMD_VERSION_MINOR}.
    ${CAMD_VERSION_SUB} " (" ${CAMD_DATE} ")" )

#-------------------------------------------------------------------------------
# define the project
#-------------------------------------------------------------------------------

project ( CAMD
    VERSION "${CAMD_VERSION_MAJOR}.${CAMD_VERSION_MINOR}.${CAMD_VERSION_SUB}"
    LANGUAGES C )

#-------------------------------------------------------------------------------
# SuiteSparse policies
#-------------------------------------------------------------------------------

set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    ${PROJECT_SOURCE_DIR}/../SuiteSparse_config/cmake_modules )

include ( SuiteSparsePolicy )

#-------------------------------------------------------------------------------
# find library dependencies
#-------------------------------------------------------------------------------

if ( NOT SUITESPARSE_ROOT_CMAKELISTS )
    find_package ( SuiteSparse_config 7.8.0
        PATHS ${CMAKE_SOURCE_DIR}/../SuiteSparse_config/build NO_DEFAULT_PATH )
    if ( NOT TARGET SuiteSparse::SuiteSparseConfig )
        find_package ( SuiteSparse_config 7.8.0 REQUIRED )
    endif ( )
endif ( )

#-------------------------------------------------------------------------------
# configure files
#-------------------------------------------------------------------------------

configure_file ( "Config/camd.h.in"
    "${PROJECT_SOURCE_DIR}/Include/camd.h"
    NEWLINE_STYLE LF )
configure_file ( "Config/camd_version.tex.in"
    "${PROJECT_SOURCE_DIR}/Doc/camd_version.tex"
    NEWLINE_STYLE LF )

#-------------------------------------------------------------------------------
# include directories
#-------------------------------------------------------------------------------

include_directories ( Source Include )

#-------------------------------------------------------------------------------
# dynamic camd library properties
#-------------------------------------------------------------------------------

file ( GLOB CAMD_SOURCES "Source/*.c" )

if ( BUILD_SHARED_LIBS )
    add_library ( CAMD SHARED ${CAMD_SOURCES} )

    set_target_properties ( CAMD PROPERTIES
        VERSION ${CAMD_VERSION_MAJOR}.${CAMD_VERSION_MINOR}.${CAMD_VERSION_SUB}
        C_STANDARD 11
        C_STANDARD_REQUIRED ON
        OUTPUT_NAME camd
        SOVERSION ${CAMD_VERSION_MAJOR}
        PUBLIC_HEADER "Include/camd.h"
        WINDOWS_EXPORT_ALL_SYMBOLS ON )

    if ( ${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.25" )
        set_target_properties ( CAMD PROPERTIES EXPORT_NO_SYSTEM ON )
    endif ( )

    target_include_directories ( CAMD
        INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>
                  $<INSTALL_INTERFACE:${SUITESPARSE_INCLUDEDIR}> )
endif ( )

#-------------------------------------------------------------------------------
# static camd library properties
#-------------------------------------------------------------------------------

if ( BUILD_STATIC_LIBS )
    add_library ( CAMD_static STATIC ${CAMD_SOURCES} )

    set_target_properties ( CAMD_static PROPERTIES
        C_STANDARD 11
        C_STANDARD_REQUIRED ON
        OUTPUT_NAME camd
        PUBLIC_HEADER "Include/camd.h" )

    if ( MSVC OR ("${CMAKE_C_SIMULATE_ID}" STREQUAL "MSVC") )
        set_target_properties ( CAMD_static PROPERTIES
            OUTPUT_NAME camd_static )
    endif ( )

    if ( ${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.25" )
        set_target_properties ( CAMD_static PROPERTIES EXPORT_NO_SYSTEM ON )
    endif ( )

    target_include_directories ( CAMD_static
        INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>
                  $<INSTALL_INTERFACE:${SUITESPARSE_INCLUDEDIR}> )
endif ( )

#-------------------------------------------------------------------------------
# add the library dependencies
#-------------------------------------------------------------------------------

if ( BUILD_SHARED_LIBS )
    target_link_libraries ( CAMD PRIVATE SuiteSparse::SuiteSparseConfig )
    target_include_directories ( CAMD PUBLIC
        "$<TARGET_PROPERTY:SuiteSparse::SuiteSparseConfig,INTERFACE_INCLUDE_DIRECTORIES>" )
endif ( )
if ( BUILD_STATIC_LIBS )
    if ( TARGET SuiteSparse::SuiteSparseConfig_static )
        target_link_libraries ( CAMD_static PUBLIC SuiteSparse::SuiteSparseConfig_static )
    else ( )
        target_link_libraries ( CAMD_static PUBLIC SuiteSparse::SuiteSparseConfig )
    endif ( )
endif ( )

# libm:
if ( NOT WIN32 )
    if ( BUILD_SHARED_LIBS )
        target_link_libraries ( CAMD PRIVATE m )
    endif ( )
    if ( BUILD_STATIC_LIBS )
        set ( CAMD_STATIC_LIBS "${CAMD_STATIC_LIBS} -lm" )
        target_link_libraries ( CAMD_static PUBLIC m )
    endif ( )
endif ( )

#-------------------------------------------------------------------------------
# CAMD installation location
#-------------------------------------------------------------------------------

include ( CMakePackageConfigHelpers )

if ( BUILD_SHARED_LIBS )
    install ( TARGETS CAMD
        EXPORT CAMDTargets
        LIBRARY DESTINATION ${SUITESPARSE_LIBDIR}
        ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR}
        RUNTIME DESTINATION ${SUITESPARSE_BINDIR}
        PUBLIC_HEADER DESTINATION ${SUITESPARSE_INCLUDEDIR} )
endif ( )
if ( BUILD_STATIC_LIBS )
    install ( TARGETS CAMD_static
        EXPORT CAMDTargets
        ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR}
        PUBLIC_HEADER DESTINATION ${SUITESPARSE_INCLUDEDIR} )
endif ( )

# create (temporary) export target file during build
export ( EXPORT CAMDTargets
    NAMESPACE SuiteSparse::
    FILE ${CMAKE_CURRENT_BINARY_DIR}/CAMDTargets.cmake )

# install export target, config and version files for find_package
install ( EXPORT CAMDTargets
    NAMESPACE SuiteSparse::
    DESTINATION ${SUITESPARSE_PKGFILEDIR}/cmake/CAMD )

# generate config file to be used in common build tree
set ( SUITESPARSE_IN_BUILD_TREE ON )
configure_package_config_file (
    Config/CAMDConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/CAMDConfig.cmake
    INSTALL_DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/CAMDConfig.cmake )

# generate config file to be installed
set ( SUITESPARSE_IN_BUILD_TREE OFF )
configure_package_config_file (
    Config/CAMDConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/target/CAMDConfig.cmake
    INSTALL_DESTINATION ${SUITESPARSE_PKGFILEDIR}/cmake/CAMD )

write_basic_package_version_file (
    ${CMAKE_CURRENT_BINARY_DIR}/CAMDConfigVersion.cmake
    COMPATIBILITY SameMajorVersion )

install ( FILES
    ${CMAKE_CURRENT_BINARY_DIR}/target/CAMDConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/CAMDConfigVersion.cmake
    DESTINATION ${SUITESPARSE_PKGFILEDIR}/cmake/CAMD )

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
    if ( BUILD_SHARED_LIBS )
        set ( SUITESPARSE_LIB_BASE_NAME $<TARGET_FILE_BASE_NAME:CAMD> )
    else ( )
        set ( SUITESPARSE_LIB_BASE_NAME $<TARGET_FILE_BASE_NAME:CAMD_static> )
    endif ( )
    configure_file (
        Config/CAMD.pc.in
        CAMD.pc.out
        @ONLY
        NEWLINE_STYLE LF )
    file ( GENERATE
        OUTPUT CAMD.pc
        INPUT ${CMAKE_CURRENT_BINARY_DIR}/CAMD.pc.out
        NEWLINE_STYLE LF )
    install ( FILES
        ${CMAKE_CURRENT_BINARY_DIR}/CAMD.pc
        DESTINATION ${SUITESPARSE_PKGFILEDIR}/pkgconfig )
endif ( )

#-------------------------------------------------------------------------------
# Demo library and programs
#-------------------------------------------------------------------------------

if ( SUITESPARSE_DEMOS )

    #---------------------------------------------------------------------------
    # demo library
    #---------------------------------------------------------------------------

    message ( STATUS "Also compiling the demos in CAMD/Demo" )

    #---------------------------------------------------------------------------
    # Demo programs
    #---------------------------------------------------------------------------

    add_executable ( camd_demo   "Demo/camd_demo.c" )
    add_executable ( camd_l_demo "Demo/camd_l_demo.c" )
    add_executable ( camd_demo2  "Demo/camd_demo2.c" )
    add_executable ( camd_simple "Demo/camd_simple.c" )

    # Libraries required for Demo programs
    if ( BUILD_SHARED_LIBS )
        target_link_libraries ( camd_demo   PUBLIC CAMD )
        target_link_libraries ( camd_l_demo PUBLIC CAMD )
        target_link_libraries ( camd_demo2  PUBLIC CAMD )
        target_link_libraries ( camd_simple PUBLIC CAMD )
    else ( )
        target_link_libraries ( camd_demo   PUBLIC CAMD_static )
        target_link_libraries ( camd_l_demo PUBLIC CAMD_static )
        target_link_libraries ( camd_demo2  PUBLIC CAMD_static )
        target_link_libraries ( camd_simple PUBLIC CAMD_static )
    endif ( )

else ( )

    message ( STATUS "Skipping the demos in CAMD/Demo" )

endif ( )

#-------------------------------------------------------------------------------
# report status
#-------------------------------------------------------------------------------

include ( SuiteSparseReport )
