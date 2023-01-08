#-------------------------------------------------------------------------------
# SuiteSparse/BTF/CMakeLists.txt:  cmake for BTF
#-------------------------------------------------------------------------------

# BTF, Copyright (c) 2004-2022, University of Florida.  All Rights Reserved.
# Author: Timothy A. Davis.
# SPDX-License-Identifier: LGPL-2.1+

#-------------------------------------------------------------------------------
# get the version
#-------------------------------------------------------------------------------

cmake_minimum_required ( VERSION 3.19 )

set ( BTF_DATE "June 16, 2023" )
set ( BTF_VERSION_MAJOR 2 )
set ( BTF_VERSION_MINOR 0 )
set ( BTF_VERSION_SUB   4 )

message ( STATUS "Building BTF version: v"
    ${BTF_VERSION_MAJOR}.
    ${BTF_VERSION_MINOR}.
    ${BTF_VERSION_SUB} " (" ${BTF_DATE} ")" )

#-------------------------------------------------------------------------------
# SuiteSparse policies
#-------------------------------------------------------------------------------

set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    ${CMAKE_SOURCE_DIR}/cmake_modules
    ${CMAKE_SOURCE_DIR}/../SuiteSparse_config/cmake_modules )

include ( SuiteSparsePolicy )

#-------------------------------------------------------------------------------
# define the project
#-------------------------------------------------------------------------------

project ( btf
    VERSION "${BTF_VERSION_MAJOR}.${BTF_VERSION_MINOR}.${BTF_VERSION_SUB}"
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

configure_file ( "Config/btf.h.in" "${PROJECT_SOURCE_DIR}/Include/btf.h"
    NEWLINE_STYLE LF )

#-------------------------------------------------------------------------------
# include directories
#-------------------------------------------------------------------------------

include_directories ( Source Include )

#-------------------------------------------------------------------------------
# dynamic btf library properties
#-------------------------------------------------------------------------------

file ( GLOB BTF_SOURCES "Source/*.c" )

add_library ( btf SHARED ${BTF_SOURCES} )

set_target_properties ( btf PROPERTIES
    VERSION ${BTF_VERSION_MAJOR}.${BTF_VERSION_MINOR}.${BTF_VERSION_SUB}
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
    SOVERSION ${BTF_VERSION_MAJOR}
    PUBLIC_HEADER "Include/btf.h" 
    WINDOWS_EXPORT_ALL_SYMBOLS ON)

target_include_directories ( btf
    INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>"
              "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>" )

#-------------------------------------------------------------------------------
# static btf library properties
#-------------------------------------------------------------------------------

if ( NOT NSTATIC )
    add_library ( btf_static STATIC ${BTF_SOURCES} )

    set_target_properties ( btf_static PROPERTIES
        VERSION ${BTF_VERSION_MAJOR}.${BTF_VERSION_MINOR}.${BTF_VERSION_SUB}
        C_STANDARD 11
        C_STANDARD_REQUIRED ON
        OUTPUT_NAME btf
        SOVERSION ${BTF_VERSION_MAJOR} )

    if ( MSVC )
        set_target_properties ( btf_static PROPERTIES
            OUTPUT_NAME btf_static )
    endif ( )

    target_include_directories ( btf_static
        INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>"
                  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>" )

endif ( )

#-------------------------------------------------------------------------------
# add the library dependencies
#-------------------------------------------------------------------------------

# suitesparseconfig:
target_link_libraries ( btf PRIVATE suitesparseconfig )
target_include_directories ( btf PUBLIC "$<TARGET_PROPERTY:suitesparseconfig,INTERFACE_INCLUDE_DIRECTORIES>" )
if ( NOT NSTATIC )
    if ( TARGET suitesparseconfig_static )
        target_link_libraries ( btf_static PUBLIC suitesparseconfig_static )
    else ( )
        target_link_libraries ( btf_static PUBLIC suitesparseconfig )
    endif ( )
endif ( )

# libm:
if ( NOT WIN32 )
    target_link_libraries ( btf PRIVATE m )
    if ( NOT NSTATIC )
        target_link_libraries ( btf_static PUBLIC m )
    endif ( )
endif ( )

#-------------------------------------------------------------------------------
# BTF installation location
#-------------------------------------------------------------------------------

include ( CMakePackageConfigHelpers )

install ( TARGETS btf
    EXPORT BTFTargets
    LIBRARY DESTINATION ${SUITESPARSE_LIBDIR}
    ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR}
    RUNTIME DESTINATION ${SUITESPARSE_BINDIR}
    PUBLIC_HEADER DESTINATION ${SUITESPARSE_INCLUDEDIR} )
if ( NOT NSTATIC )
    install ( TARGETS btf_static
        EXPORT BTFTargets
        ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR} )
endif ( )

# create (temporary) export target file during build
export ( EXPORT BTFTargets
    FILE ${CMAKE_CURRENT_BINARY_DIR}/BTFTargets.cmake )

# install export target, config and version files for find_package
install ( EXPORT BTFTargets
    DESTINATION ${SUITESPARSE_LIBDIR}/cmake/BTF )

configure_package_config_file (
    Config/BTFConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/BTFConfig.cmake
    INSTALL_DESTINATION ${SUITESPARSE_LIBDIR}/cmake/BTF )

write_basic_package_version_file (
    ${CMAKE_CURRENT_BINARY_DIR}/BTFConfigVersion.cmake
    COMPATIBILITY SameMajorVersion )

install ( FILES
    ${CMAKE_CURRENT_BINARY_DIR}/BTFConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/BTFConfigVersion.cmake
    DESTINATION ${SUITESPARSE_LIBDIR}/cmake/BTF )

#-------------------------------------------------------------------------------
# report status
#-------------------------------------------------------------------------------

include ( SuiteSparseReport )

