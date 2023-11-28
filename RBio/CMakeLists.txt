#-------------------------------------------------------------------------------
# SuiteSparse/RBio/CMakeLists.txt:  cmake for RBio
#-------------------------------------------------------------------------------

# RBio, Copyright (c) 2009-2023, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: GPL-2.0+

#-------------------------------------------------------------------------------
# get the version
#-------------------------------------------------------------------------------

cmake_minimum_required ( VERSION 3.20 )

set ( RBIO_DATE "Dec 30, 2023" )
set ( RBIO_VERSION_MAJOR 4 CACHE STRING "" FORCE )
set ( RBIO_VERSION_MINOR 3 CACHE STRING "" FORCE )
set ( RBIO_VERSION_SUB   0 CACHE STRING "" FORCE )

message ( STATUS "Building RBIO version: v"
    ${RBIO_VERSION_MAJOR}.
    ${RBIO_VERSION_MINOR}.
    ${RBIO_VERSION_SUB} " (" ${RBIO_DATE} ")" )

#-------------------------------------------------------------------------------
# define the project
#-------------------------------------------------------------------------------

project ( rbio
    VERSION "${RBIO_VERSION_MAJOR}.${RBIO_VERSION_MINOR}.${RBIO_VERSION_SUB}"
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
    find_package ( SuiteSparse_config 7.4.0
        PATHS ${CMAKE_SOURCE_DIR}/../SuiteSparse_config/build NO_DEFAULT_PATH )
    if ( NOT TARGET SuiteSparse::SuiteSparseConfig )
        find_package ( SuiteSparse_config 7.4.0 REQUIRED )
    endif ( )
endif ( )

#-------------------------------------------------------------------------------
# configure files
#-------------------------------------------------------------------------------

configure_file ( "Config/RBio.h.in"
    "${PROJECT_SOURCE_DIR}/Include/RBio.h"
    NEWLINE_STYLE LF )

#-------------------------------------------------------------------------------
# include directories
#-------------------------------------------------------------------------------

include_directories ( Source Include )

#-------------------------------------------------------------------------------
# dynamic rbio library properties
#-------------------------------------------------------------------------------

file ( GLOB RBIO_SOURCES "Source/*.c" )

if ( BUILD_SHARED_LIBS )
    add_library ( RBio SHARED ${RBIO_SOURCES} )

    set_target_properties ( RBio PROPERTIES
        VERSION ${RBIO_VERSION_MAJOR}.${RBIO_VERSION_MINOR}.${RBIO_VERSION_SUB}
        C_STANDARD 11
        C_STANDARD_REQUIRED ON
        OUTPUT_NAME rbio
        SOVERSION ${RBIO_VERSION_MAJOR}
        PUBLIC_HEADER "Include/RBio.h"
        WINDOWS_EXPORT_ALL_SYMBOLS ON )

    target_include_directories ( RBio 
        INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>
                  $<INSTALL_INTERFACE:${SUITESPARSE_INCLUDEDIR}> )
endif ( )

#-------------------------------------------------------------------------------
# static rbio library properties
#-------------------------------------------------------------------------------

if ( BUILD_STATIC_LIBS )
    add_library ( RBio_static STATIC ${RBIO_SOURCES} )

    set_target_properties ( RBio_static PROPERTIES
        C_STANDARD 11
        C_STANDARD_REQUIRED ON
        OUTPUT_NAME rbio
        PUBLIC_HEADER "Include/RBio.h" )

    if ( MSVC )
        set_target_properties ( RBio_static PROPERTIES
            OUTPUT_NAME rbio_static )
    endif ( )

    target_include_directories ( RBio_static 
        INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>
                  $<INSTALL_INTERFACE:${SUITESPARSE_INCLUDEDIR}> )
endif ( )

#-------------------------------------------------------------------------------
# add the library dependencies
#-------------------------------------------------------------------------------

# SuiteSparseConfig:
if ( BUILD_SHARED_LIBS )
    target_link_libraries ( RBio PRIVATE SuiteSparse::SuiteSparseConfig )
    target_include_directories ( RBio PUBLIC
        "$<TARGET_PROPERTY:SuiteSparse::SuiteSparseConfig,INTERFACE_INCLUDE_DIRECTORIES>" )
endif ( )
if ( BUILD_STATIC_LIBS )
    if ( TARGET SuiteSparse::SuiteSparseConfig_static )
        target_link_libraries ( RBio_static PUBLIC SuiteSparse::SuiteSparseConfig_static )
    else ( )
        target_link_libraries ( RBio_static PUBLIC SuiteSparse::SuiteSparseConfig )
    endif ( )
endif ( )

# libm:
if ( NOT WIN32 )
    if ( BUILD_SHARED_LIBS )
        target_link_libraries ( RBio PRIVATE m )
    endif ( )
    if ( BUILD_STATIC_LIBS )
        set ( RBIO_STATIC_LIBS "${RBIO_STATIC_LIBS} -lm" )
        target_link_libraries ( RBio_static PUBLIC m )
    endif ( )
endif ( )

#-------------------------------------------------------------------------------
# RBIO installation location
#-------------------------------------------------------------------------------

include ( CMakePackageConfigHelpers )

if ( BUILD_SHARED_LIBS )
    install ( TARGETS RBio
        EXPORT RBioTargets
        LIBRARY DESTINATION ${SUITESPARSE_LIBDIR}
        ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR}
        RUNTIME DESTINATION ${SUITESPARSE_BINDIR}
        PUBLIC_HEADER DESTINATION ${SUITESPARSE_INCLUDEDIR} )
endif ( )
if ( BUILD_STATIC_LIBS )
    install ( TARGETS RBio_static
        EXPORT RBioTargets
        ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR}
        PUBLIC_HEADER DESTINATION ${SUITESPARSE_INCLUDEDIR} )
endif ( )

# create (temporary) export target file during build
export ( EXPORT RBioTargets
    NAMESPACE SuiteSparse::
    FILE ${CMAKE_CURRENT_BINARY_DIR}/RBioTargets.cmake )

# install export target, config and version files for find_package
install ( EXPORT RBioTargets
    NAMESPACE SuiteSparse::
    DESTINATION ${SUITESPARSE_PKGFILEDIR}/cmake/RBio )

# generate config file to be used in common build tree
set ( SUITESPARSE_IN_BUILD_TREE ON )
configure_package_config_file (
    Config/RBioConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/RBioConfig.cmake
    INSTALL_DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/RBioConfig.cmake )

# generate config file to be installed
set ( SUITESPARSE_IN_BUILD_TREE OFF )
configure_package_config_file (
    Config/RBioConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/target/RBioConfig.cmake
    INSTALL_DESTINATION ${SUITESPARSE_PKGFILEDIR}/cmake/RBio )

write_basic_package_version_file (
    ${CMAKE_CURRENT_BINARY_DIR}/RBioConfigVersion.cmake
    COMPATIBILITY SameMajorVersion )

install ( FILES
    ${CMAKE_CURRENT_BINARY_DIR}/target/RBioConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/RBioConfigVersion.cmake
    DESTINATION ${SUITESPARSE_PKGFILEDIR}/cmake/RBio )

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
        Config/RBio.pc.in
        RBio.pc
        @ONLY
        NEWLINE_STYLE LF )
    install ( FILES
        ${CMAKE_CURRENT_BINARY_DIR}/RBio.pc
        DESTINATION ${SUITESPARSE_PKGFILEDIR}/pkgconfig )
endif ( )

#-------------------------------------------------------------------------------
# Demo library and programs
#-------------------------------------------------------------------------------

option ( DEMO "ON: Build the demo programs.  OFF (default): do not build the demo programs." OFF )
if ( DEMO )

    #---------------------------------------------------------------------------
    # demo library
    #---------------------------------------------------------------------------

    message ( STATUS "Also compiling the demos in RBio/Demo" )

    #---------------------------------------------------------------------------
    # Demo programs
    #---------------------------------------------------------------------------

    add_executable ( RBdemo "Demo/RBdemo.c" )

    # Libraries required for Demo programs
    if ( BUILD_SHARED_LIBS )
        target_link_libraries ( RBdemo PUBLIC RBio SuiteSparse::SuiteSparseConfig )
    else ( )
        target_link_libraries ( RBdemo PUBLIC RBio_static SuiteSparse::SuiteSparseConfig )
    endif ( )

else ( )

    message ( STATUS "Skipping the demos in RBio/Demo" )

endif ( )

#-------------------------------------------------------------------------------
# report status
#-------------------------------------------------------------------------------

if ( NOT SUITESPARSE_ROOT_CMAKELISTS )
    include ( SuiteSparseReport )
endif ( )
