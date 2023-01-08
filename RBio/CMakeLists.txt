#-------------------------------------------------------------------------------
# SuiteSparse/RBio/CMakeLists.txt:  cmake for RBio
#-------------------------------------------------------------------------------

# RBio, Copyright (c) 2009-2022, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: GPL-2.0+

#-------------------------------------------------------------------------------
# get the version
#-------------------------------------------------------------------------------

cmake_minimum_required ( VERSION 3.19 )

set ( RBIO_DATE "June 16, 2023" )
set ( RBIO_VERSION_MAJOR 4 )
set ( RBIO_VERSION_MINOR 0 )
set ( RBIO_VERSION_SUB   0 )

message ( STATUS "Building RBIO version: v"
    ${RBIO_VERSION_MAJOR}.
    ${RBIO_VERSION_MINOR}.
    ${RBIO_VERSION_SUB} " (" ${RBIO_DATE} ")" )

#-------------------------------------------------------------------------------
# SuiteSparse policies
#-------------------------------------------------------------------------------

set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    ${CMAKE_SOURCE_DIR}/../SuiteSparse_config/cmake_modules )

include ( SuiteSparsePolicy )

#-------------------------------------------------------------------------------
# define the project
#-------------------------------------------------------------------------------

project ( rbio
    VERSION "${RBIO_VERSION_MAJOR}.${RBIO_VERSION_MINOR}.${RBIO_VERSION_SUB}"
    LANGUAGES C )

#-------------------------------------------------------------------------------
# find library dependencies
#-------------------------------------------------------------------------------

find_package ( SuiteSparse_config 7.1.0
    PATHS ${CMAKE_SOURCE_DIR}/../SuiteSparse_config/build NO_DEFAULT_PATH )
if ( NOT TARGET suitesparseconfig )
    find_package ( SuiteSparse_config 7.1.0 REQUIRED )
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

add_library ( rbio SHARED ${RBIO_SOURCES} )

set_target_properties ( rbio PROPERTIES
    VERSION ${RBIO_VERSION_MAJOR}.${RBIO_VERSION_MINOR}.${RBIO_VERSION_SUB}
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
    SOVERSION ${RBIO_VERSION_MAJOR}
    PUBLIC_HEADER "Include/RBio.h"
    WINDOWS_EXPORT_ALL_SYMBOLS ON )

target_include_directories ( rbio 
    INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>"
              "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>" )

#-------------------------------------------------------------------------------
# static rbio library properties
#-------------------------------------------------------------------------------

if ( NOT NSTATIC )
    add_library ( rbio_static STATIC ${RBIO_SOURCES} )

    set_target_properties ( rbio_static PROPERTIES
        VERSION ${RBIO_VERSION_MAJOR}.${RBIO_VERSION_MINOR}.${RBIO_VERSION_SUB}
        C_STANDARD 11
        C_STANDARD_REQUIRED ON
        OUTPUT_NAME rbio
        SOVERSION ${RBIO_VERSION_MAJOR} )

    if ( MSVC )
        set_target_properties ( rbio_static PROPERTIES
            OUTPUT_NAME rbio_static )
    endif ( )

    target_include_directories ( rbio_static 
        INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>"
                  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>" )

endif ( )

#-------------------------------------------------------------------------------
# add the library dependencies
#-------------------------------------------------------------------------------

# suitesparseconfig:
target_link_libraries ( rbio PRIVATE suitesparseconfig )
target_include_directories ( rbio PUBLIC "$<TARGET_PROPERTY:suitesparseconfig,INTERFACE_INCLUDE_DIRECTORIES>" )
if ( NOT NSTATIC )
    if ( TARGET suitesparseconfig_static )
        target_link_libraries ( rbio_static PUBLIC suitesparseconfig_static )
    else ( )
        target_link_libraries ( rbio_static PUBLIC suitesparseconfig )
    endif ( )
endif ( )

# libm:
if ( NOT WIN32 )
    target_link_libraries ( rbio PRIVATE m )
    if ( NOT NSTATIC )
        target_link_libraries ( rbio_static PUBLIC m )
    endif ( )
endif ( )

#-------------------------------------------------------------------------------
# RBIO installation location
#-------------------------------------------------------------------------------

include ( CMakePackageConfigHelpers )

install ( TARGETS rbio
    EXPORT RBioTargets
    LIBRARY DESTINATION ${SUITESPARSE_LIBDIR}
    ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR}
    RUNTIME DESTINATION ${SUITESPARSE_BINDIR}
    PUBLIC_HEADER DESTINATION ${SUITESPARSE_INCLUDEDIR} )
if ( NOT NSTATIC )
    install ( TARGETS rbio_static
        EXPORT RBioTargets
        ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR} )
endif ( )

# create (temporary) export target file during build
export ( EXPORT RBioTargets
    FILE ${CMAKE_CURRENT_BINARY_DIR}/RBioTargets.cmake )

# install export target, config and version files for find_package
install ( EXPORT RBioTargets
    DESTINATION ${SUITESPARSE_LIBDIR}/cmake/RBio )

configure_package_config_file (
    Config/RBioConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/RBioConfig.cmake
    INSTALL_DESTINATION ${SUITESPARSE_LIBDIR}/cmake/RBio )

write_basic_package_version_file (
    ${CMAKE_CURRENT_BINARY_DIR}/RBioConfigVersion.cmake
    COMPATIBILITY SameMajorVersion )

install ( FILES
    ${CMAKE_CURRENT_BINARY_DIR}/RBioConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/RBioConfigVersion.cmake
    DESTINATION ${SUITESPARSE_LIBDIR}/cmake/RBio )

#-------------------------------------------------------------------------------
# Demo library and programs
#-------------------------------------------------------------------------------

option ( DEMO "ON: Build the demo programs.  OFF (default): do not build the demo programs." off )
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
    target_link_libraries ( RBdemo PUBLIC rbio suitesparseconfig )

else ( )

    message ( STATUS "Skipping the demos in RBio/Demo" )

endif ( )

#-------------------------------------------------------------------------------
# report status
#-------------------------------------------------------------------------------

include ( SuiteSparseReport )

