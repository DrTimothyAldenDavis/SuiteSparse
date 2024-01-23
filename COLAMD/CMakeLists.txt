#-------------------------------------------------------------------------------
# SuiteSparse/COLAMD/CMakeLists.txt:  cmake for COLAMD
#-------------------------------------------------------------------------------

# Copyright (c) 1998-2024, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

#-------------------------------------------------------------------------------
# get the version
#-------------------------------------------------------------------------------

cmake_minimum_required ( VERSION 3.22 )

set ( COLAMD_DATE "Jan 20, 2024" )
set ( COLAMD_VERSION_MAJOR 3 CACHE STRING "" FORCE )
set ( COLAMD_VERSION_MINOR 3 CACHE STRING "" FORCE )
set ( COLAMD_VERSION_SUB   2 CACHE STRING "" FORCE )

message ( STATUS "Building COLAMD version: v"
    ${COLAMD_VERSION_MAJOR}.
    ${COLAMD_VERSION_MINOR}.
    ${COLAMD_VERSION_SUB} " (" ${COLAMD_DATE} ")" )

#-------------------------------------------------------------------------------
# define the project
#-------------------------------------------------------------------------------

project ( COLAMD
    VERSION "${COLAMD_VERSION_MAJOR}.${COLAMD_VERSION_MINOR}.${COLAMD_VERSION_SUB}"
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
    find_package ( SuiteSparse_config 7.6.0
        PATHS ${CMAKE_SOURCE_DIR}/../SuiteSparse_config/build NO_DEFAULT_PATH )
    if ( NOT TARGET SuiteSparse::SuiteSparseConfig )
        find_package ( SuiteSparse_config 7.6.0 REQUIRED )
    endif ( )
endif ( )

#-------------------------------------------------------------------------------
# configure files
#-------------------------------------------------------------------------------

configure_file ( "Config/colamd.h.in"
    "${PROJECT_SOURCE_DIR}/Include/colamd.h"
    NEWLINE_STYLE LF )

#-------------------------------------------------------------------------------
# include directories
#-------------------------------------------------------------------------------

include_directories ( Source Include )

#-------------------------------------------------------------------------------
# dynamic colamd library properties
#-------------------------------------------------------------------------------

file ( GLOB COLAMD_SOURCES "Source/*.c" )

if ( BUILD_SHARED_LIBS )
    add_library ( COLAMD SHARED ${COLAMD_SOURCES} )

    set_target_properties ( COLAMD PROPERTIES
        VERSION ${COLAMD_VERSION_MAJOR}.${COLAMD_VERSION_MINOR}.${COLAMD_VERSION_SUB}
        C_STANDARD 11
        C_STANDARD_REQUIRED ON
        OUTPUT_NAME colamd
        SOVERSION ${COLAMD_VERSION_MAJOR}
        PUBLIC_HEADER "Include/colamd.h"
        WINDOWS_EXPORT_ALL_SYMBOLS ON )

    if ( ${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.25" )
        set_target_properties ( COLAMD PROPERTIES EXPORT_NO_SYSTEM ON )
    endif ( )

    target_include_directories ( COLAMD
        INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>
                  $<INSTALL_INTERFACE:${SUITESPARSE_INCLUDEDIR}> )
endif ( )

#-------------------------------------------------------------------------------
# static colamd library properties
#-------------------------------------------------------------------------------

if ( BUILD_STATIC_LIBS )
    add_library ( COLAMD_static STATIC ${COLAMD_SOURCES} )

    set_target_properties ( COLAMD_static PROPERTIES
        OUTPUT_NAME colamd
        C_STANDARD 11
        C_STANDARD_REQUIRED ON
        PUBLIC_HEADER "Include/colamd.h" )

    if ( MSVC )
        set_target_properties ( COLAMD_static PROPERTIES
            OUTPUT_NAME colamd_static )
    endif ( )

    if ( ${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.25" )
        set_target_properties ( COLAMD_static PROPERTIES EXPORT_NO_SYSTEM ON )
    endif ( )

    target_include_directories ( COLAMD_static
        INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>
                  $<INSTALL_INTERFACE:${SUITESPARSE_INCLUDEDIR}> )
endif ( )

#-------------------------------------------------------------------------------
# add the library dependencies
#-------------------------------------------------------------------------------

if ( BUILD_SHARED_LIBS )
    target_link_libraries ( COLAMD PRIVATE SuiteSparse::SuiteSparseConfig )
    target_include_directories ( COLAMD PUBLIC
        "$<TARGET_PROPERTY:SuiteSparse::SuiteSparseConfig,INTERFACE_INCLUDE_DIRECTORIES>" )
endif ( )
if ( BUILD_STATIC_LIBS )
    if ( TARGET SuiteSparse::SuiteSparseConfig_static )
        target_link_libraries ( COLAMD_static PUBLIC SuiteSparse::SuiteSparseConfig_static )
    else ( )
        target_link_libraries ( COLAMD_static PUBLIC SuiteSparse::SuiteSparseConfig )
    endif ( )
endif ( )

# libm:
if ( NOT WIN32 )
    if ( BUILD_SHARED_LIBS )
        target_link_libraries ( COLAMD PRIVATE m )
    endif ( )
    if ( BUILD_STATIC_LIBS )
        set ( COLAMD_STATIC_LIBS "${COLAMD_STATIC_LIBS} -lm" )
        target_link_libraries ( COLAMD_static PUBLIC m )
    endif ( )
endif ( )

#-------------------------------------------------------------------------------
# COLAMD installation location
#-------------------------------------------------------------------------------

include ( CMakePackageConfigHelpers )

if ( BUILD_SHARED_LIBS )
    install ( TARGETS COLAMD
        EXPORT COLAMDTargets
        LIBRARY DESTINATION ${SUITESPARSE_LIBDIR}
        ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR}
        RUNTIME DESTINATION ${SUITESPARSE_BINDIR}
        PUBLIC_HEADER DESTINATION ${SUITESPARSE_INCLUDEDIR} )
endif ( )
if ( BUILD_STATIC_LIBS )
    install ( TARGETS COLAMD_static
        EXPORT COLAMDTargets
        ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR}
        PUBLIC_HEADER DESTINATION ${SUITESPARSE_INCLUDEDIR} )
endif ( )

# create (temporary) export target file during build
export ( EXPORT COLAMDTargets
    NAMESPACE SuiteSparse::
    FILE ${CMAKE_CURRENT_BINARY_DIR}/COLAMDTargets.cmake )

# install export target, config and version files for find_package
install ( EXPORT COLAMDTargets
    NAMESPACE SuiteSparse::
    DESTINATION ${SUITESPARSE_PKGFILEDIR}/cmake/COLAMD )

# generate config file to be used in common build tree
set ( SUITESPARSE_IN_BUILD_TREE ON )
configure_package_config_file (
    Config/COLAMDConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/COLAMDConfig.cmake
    INSTALL_DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/COLAMDConfig.cmake )

# generate config file to be installed
set ( SUITESPARSE_IN_BUILD_TREE OFF )
configure_package_config_file (
    Config/COLAMDConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/target/COLAMDConfig.cmake
    INSTALL_DESTINATION ${SUITESPARSE_PKGFILEDIR}/cmake/COLAMD )

write_basic_package_version_file (
    ${CMAKE_CURRENT_BINARY_DIR}/COLAMDConfigVersion.cmake
    COMPATIBILITY SameMajorVersion )

install ( FILES
    ${CMAKE_CURRENT_BINARY_DIR}/target/COLAMDConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/COLAMDConfigVersion.cmake
    DESTINATION ${SUITESPARSE_PKGFILEDIR}/cmake/COLAMD )

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
        set ( SUITESPARSE_LIB_BASE_NAME $<TARGET_FILE_BASE_NAME:COLAMD> )
    else ( )
        set ( SUITESPARSE_LIB_BASE_NAME $<TARGET_FILE_BASE_NAME:COLAMD_static> )
    endif ( )
  configure_file (
      Config/COLAMD.pc.in
      COLAMD.pc.out
      @ONLY
      NEWLINE_STYLE LF )
    file ( GENERATE
        OUTPUT COLAMD.pc
        INPUT ${CMAKE_CURRENT_BINARY_DIR}/COLAMD.pc.out
        NEWLINE_STYLE LF )
  install ( FILES
      ${CMAKE_CURRENT_BINARY_DIR}/COLAMD.pc
      DESTINATION ${SUITESPARSE_PKGFILEDIR}/pkgconfig )
endif ( )

#-------------------------------------------------------------------------------
# Demo library and programs
#-------------------------------------------------------------------------------

if ( SUITESPARSE_DEMOS )

    #---------------------------------------------------------------------------
    # demo library
    #---------------------------------------------------------------------------

    message ( STATUS "Also compiling the demos in COLAMD/Demo" )

    #---------------------------------------------------------------------------
    # Demo programs
    #---------------------------------------------------------------------------

    add_executable ( colamd_example   "Demo/colamd_example.c" )
    add_executable ( colamd_l_example "Demo/colamd_l_example.c" )

    # Libraries required for Demo programs
    if ( BUILD_SHARED_LIBS )
        target_link_libraries ( colamd_example   PUBLIC COLAMD )
        target_link_libraries ( colamd_l_example PUBLIC COLAMD )
    else ( )
        target_link_libraries ( colamd_example   PUBLIC COLAMD_static )
        target_link_libraries ( colamd_l_example PUBLIC COLAMD_static )
    endif ( )

else ( )

    message ( STATUS "Skipping the demos in COLAMD/Demo" )

endif ( )

#-------------------------------------------------------------------------------
# report status
#-------------------------------------------------------------------------------

include ( SuiteSparseReport )
