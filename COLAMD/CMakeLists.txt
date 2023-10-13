#-------------------------------------------------------------------------------
# SuiteSparse/COLAMD/CMakeLists.txt:  cmake for COLAMD
#-------------------------------------------------------------------------------

# Copyright (c) 1998-2023, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

#-------------------------------------------------------------------------------
# get the version
#-------------------------------------------------------------------------------

cmake_minimum_required ( VERSION 3.20 )

set ( COLAMD_DATE "Sept 18, 2023" )
set ( COLAMD_VERSION_MAJOR 3 )
set ( COLAMD_VERSION_MINOR 2 )
set ( COLAMD_VERSION_SUB   1 )

message ( STATUS "Building COLAMD version: v"
    ${COLAMD_VERSION_MAJOR}.
    ${COLAMD_VERSION_MINOR}.
    ${COLAMD_VERSION_SUB} " (" ${COLAMD_DATE} ")" )

#-------------------------------------------------------------------------------
# SuiteSparse policies
#-------------------------------------------------------------------------------

set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    ${CMAKE_SOURCE_DIR}/../SuiteSparse_config/cmake_modules )

include ( SuiteSparsePolicy )

#-------------------------------------------------------------------------------
# define the project
#-------------------------------------------------------------------------------

project ( colamd
    VERSION "${COLAMD_VERSION_MAJOR}.${COLAMD_VERSION_MINOR}.${COLAMD_VERSION_SUB}"
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

add_library ( COLAMD SHARED ${COLAMD_SOURCES} )

set_target_properties ( COLAMD PROPERTIES
    VERSION ${COLAMD_VERSION_MAJOR}.${COLAMD_VERSION_MINOR}.${COLAMD_VERSION_SUB}
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
    OUTPUT_NAME colamd
    SOVERSION ${COLAMD_VERSION_MAJOR}
    PUBLIC_HEADER "Include/colamd.h"
    WINDOWS_EXPORT_ALL_SYMBOLS ON )

target_include_directories ( COLAMD
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>
              $<INSTALL_INTERFACE:${SUITESPARSE_INCLUDEDIR}> )

#-------------------------------------------------------------------------------
# static colamd library properties
#-------------------------------------------------------------------------------

if ( NOT NSTATIC )
    add_library ( COLAMD_static STATIC ${COLAMD_SOURCES} )

    set_target_properties ( COLAMD_static PROPERTIES
        VERSION ${COLAMD_VERSION_MAJOR}.${COLAMD_VERSION_MINOR}.${COLAMD_VERSION_SUB}
        OUTPUT_NAME colamd
        C_STANDARD 11
        C_STANDARD_REQUIRED ON
        SOVERSION ${COLAMD_VERSION_MAJOR} )

    if ( MSVC )
        set_target_properties ( COLAMD_static PROPERTIES
            OUTPUT_NAME colamd_static )
    endif ( )

    target_include_directories ( COLAMD_static
        INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>
                  $<INSTALL_INTERFACE:${SUITESPARSE_INCLUDEDIR}> )

endif ( )

#-------------------------------------------------------------------------------
# add the library dependencies
#-------------------------------------------------------------------------------

target_link_libraries ( COLAMD PRIVATE SuiteSparse::SuiteSparseConfig )
target_include_directories ( COLAMD PUBLIC
    "$<TARGET_PROPERTY:SuiteSparse::SuiteSparseConfig,INTERFACE_INCLUDE_DIRECTORIES>" )
if ( NOT NSTATIC )
    if ( TARGET SuiteSparse::SuiteSparseConfig_static )
        target_link_libraries ( COLAMD_static PUBLIC SuiteSparse::SuiteSparseConfig_static )
    else ( )
        target_link_libraries ( COLAMD_static PUBLIC SuiteSparse::SuiteSparseConfig )
    endif ( )
endif ( )

# libm:
if ( NOT WIN32 )
    target_link_libraries ( COLAMD PRIVATE m )
    if ( NOT NSTATIC )
        set ( COLAMD_STATIC_LIBS "${COLAMD_STATIC_LIBS} -lm" )
        target_link_libraries ( COLAMD_static PUBLIC m )
    endif ( )
endif ( )

#-------------------------------------------------------------------------------
# COLAMD installation location
#-------------------------------------------------------------------------------

include ( CMakePackageConfigHelpers )

install ( TARGETS COLAMD
    EXPORT COLAMDTargets
    LIBRARY DESTINATION ${SUITESPARSE_LIBDIR}
    ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR}
    RUNTIME DESTINATION ${SUITESPARSE_BINDIR}
    PUBLIC_HEADER DESTINATION ${SUITESPARSE_INCLUDEDIR} )
if ( NOT NSTATIC )
    install ( TARGETS COLAMD_static
        EXPORT COLAMDTargets
        ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR} )
endif ( )

# create (temporary) export target file during build
export ( EXPORT COLAMDTargets
    NAMESPACE SuiteSparse::
    FILE ${CMAKE_CURRENT_BINARY_DIR}/COLAMDTargets.cmake )

# install export target, config and version files for find_package
install ( EXPORT COLAMDTargets
    NAMESPACE SuiteSparse::
    DESTINATION ${SUITESPARSE_LIBDIR}/cmake/COLAMD )

configure_package_config_file (
    Config/COLAMDConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/COLAMDConfig.cmake
    INSTALL_DESTINATION ${SUITESPARSE_LIBDIR}/cmake/COLAMD )

write_basic_package_version_file (
    ${CMAKE_CURRENT_BINARY_DIR}/COLAMDConfigVersion.cmake
    COMPATIBILITY SameMajorVersion )

install ( FILES
    ${CMAKE_CURRENT_BINARY_DIR}/COLAMDConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/COLAMDConfigVersion.cmake
    DESTINATION ${SUITESPARSE_LIBDIR}/cmake/COLAMD )

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
      Config/COLAMD.pc.in
      COLAMD.pc
      @ONLY
      NEWLINE_STYLE LF )
  install ( FILES
      ${CMAKE_CURRENT_BINARY_DIR}/COLAMD.pc
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

    message ( STATUS "Also compiling the demos in COLAMD/Demo" )

    #---------------------------------------------------------------------------
    # Demo programs
    #---------------------------------------------------------------------------

    add_executable ( colamd_example   "Demo/colamd_example.c" )
    add_executable ( colamd_l_example "Demo/colamd_l_example.c" )

    # Libraries required for Demo programs
    target_link_libraries ( colamd_example   PUBLIC COLAMD )
    target_link_libraries ( colamd_l_example PUBLIC COLAMD )

else ( )

    message ( STATUS "Skipping the demos in COLAMD/Demo" )

endif ( )

#-------------------------------------------------------------------------------
# report status
#-------------------------------------------------------------------------------

include ( SuiteSparseReport )

